"""Damped Least Squares (DLS) inverse kinematics solver for MuJoCo models.

A compact, self-contained IK solver that uses MuJoCo's native Jacobian
(``mj_jacSite``) and a DLS update rule.  Inspired by the reference
implementation in ``simulation/src/sim_teleop/ik/dls.py`` (Pika → JAKA teleop).

The DLS formulation is numerically stable near kinematic singularities — the
damping term (λ²I) prevents the pseudo-inverse from diverging when the
Jacobian loses rank.  This makes it suitable for 5-DOF arms like the SO-101
where the task space (6-D pose) over-determines the joint space (5-D).

Rotation tracking is softened via ``rot_weight``: the solver scales *both*
the rotational Jacobian rows and the rotation error (not the error alone).
When ``rot_weight`` is ~0 the rotational Jacobian is dropped entirely so
folding the elbow is not penalised as angular velocity.

A secondary null-space task biases joints toward a rest posture (bent elbow)
so the solver prefers the home-side configuration instead of walking through
the full-extension singularity onto the opposite joint limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_ROT_WEIGHT_EPS = 1e-8


@dataclass(frozen=True)
class IKSolveResult:
    """Result of one DLS solve, including residual diagnostics."""

    qpos: np.ndarray
    pos_err: float
    rot_err: float
    manipulability: float
    achieved_pos: np.ndarray
    achieved_rot: np.ndarray


class DLSIKSolver:
    """Damped Least Squares IK solver using MuJoCo's Jacobian.

    Parameters
    ----------
    model
        ``mujoco.MjModel`` — the solver reads joint ranges and site IDs from it.
        A *separate* ``MjData`` is created internally so IK iterations never
        pollute the simulation state.
    site_name
        Name of the MuJoCo site whose world pose the IK targets (e.g.
        ``'gripperframe'``).
    body_dofs
        Indices of the DOFs (≡ qpos indices for hinge-only models) that the
        solver is allowed to move.  E.g. ``[0, 1, 2, 3, 4]`` for the 5 body
        joints of the SO-101 (gripper excluded).
    damping
        Base DLS damping factor λ (default 0.05).  Higher = more stable but
        slower convergence.
    rot_weight
        Rotation error weight relative to position (default 0.1).  Lower =
        position prioritised; 0.0 = position-only IK.
    max_iters
        Maximum DLS iterations per solve (default 20).
    pos_tol
        Position convergence threshold in metres (default 1e-4 = 0.1 mm).
    rot_tol
        Rotation convergence threshold in radians (default 1e-3 ≈ 0.06°).
    rest_qpos
        Preferred joint posture for the null-space task, in radians, length
        ``len(body_dofs)``.  ``None`` disables the secondary task.
    rest_gain
        Gain on ``(q_rest − q)`` projected into the Jacobian null space.
    adaptive_damping
        Extra λ added as ``adaptive_damping / (manipulability + 1e-3)`` so
        damping grows near singularities.
    max_dq_rad
        Per-iteration per-joint step clip in radians.  Guards Newton steps
        near singularities; 20 iterations still allow a large total move.
    """

    def __init__(
        self,
        model,
        site_name: str,
        body_dofs: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4),
        damping: float = 0.05,
        rot_weight: float = 0.1,
        max_iters: int = 20,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        rest_qpos: np.ndarray | None = None,
        rest_gain: float = 0.0,
        adaptive_damping: float = 0.0,
        max_dq_rad: float = 0.105,  # ~6°
        qpos_lo_override: np.ndarray | None = None,
    ):
        import mujoco

        self._mj = mujoco
        self._model = model
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in model")
        self._body_dofs = np.array(body_dofs, dtype=int)
        self._ik_data = mujoco.MjData(model)

        self._damping = float(damping)
        self._rot_weight = rot_weight
        self._max_iters = max_iters
        self._pos_tol = pos_tol
        self._rot_tol = rot_tol
        self._rest_qpos = None if rest_qpos is None else np.asarray(rest_qpos, dtype=float)
        self._rest_gain = float(rest_gain)
        self._adaptive_damping = float(adaptive_damping)
        self._max_dq_rad = float(max_dq_rad)

        # Pre-extract joint limits for the solved DOFs
        self._qpos_lo = np.array([model.jnt_range[d, 0] for d in body_dofs])
        self._qpos_hi = np.array([model.jnt_range[d, 1] for d in body_dofs])
        if qpos_lo_override is not None:
            override = np.asarray(qpos_lo_override, dtype=float)
            if override.shape != self._qpos_lo.shape:
                raise ValueError(
                    f"qpos_lo_override shape {override.shape} != {self._qpos_lo.shape}"
                )
            np.maximum(self._qpos_lo, override, out=self._qpos_lo)

        logger.info(
            "DLSIKSolver ready: site='%s' dofs=%s damping=%.3f rot_weight=%.3f "
            "rest_gain=%.3f adaptive_damping=%.3f max_iters=%d",
            site_name, body_dofs, damping, rot_weight, rest_gain, adaptive_damping, max_iters,
        )

    @property
    def qpos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) joint limits for the solved DOFs, in radians.

        Public so the follower can detect limit saturation and sample the
        reachable workspace (wayfinder #13).
        """
        return self._qpos_lo.copy(), self._qpos_hi.copy()

    @property
    def site_id(self) -> int:
        """MuJoCo id of the targeted site (``gripperframe``)."""
        return self._site_id

    def _rotation_error(self, target_rot: np.ndarray, cur_rot: np.ndarray) -> np.ndarray:
        R_err = target_rot @ cur_rot.T
        quat_err = np.zeros(4)  # MuJoCo convention: (w, x, y, z)
        self._mj.mju_mat2Quat(quat_err, R_err.flatten())
        rot_vec = np.zeros(3)
        self._mj.mju_quat2Vel(rot_vec, quat_err, 1.0)
        return rot_vec

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        seed_qpos: np.ndarray,
        rot_weight: float | None = None,
        rest_gain: float | None = None,
    ) -> IKSolveResult:
        """Solve IK for a target EE pose.

        Parameters
        ----------
        target_pos
            (3,) target EE position in metres (world frame).
        target_rot
            (3, 3) target EE rotation matrix (world frame).
        seed_qpos
            Full qpos vector (all joints) to warm-start from — typically the
            current simulation state.
        rot_weight
            Override ``self._rot_weight`` for this solve only.  ``None`` = use
            the constructor default.  Values below ``1e-8`` drop the rotational
            Jacobian (true position-only IK).
        rest_gain
            Override ``self._rest_gain`` for the post-solve null-space step.

        Returns
        -------
        IKSolveResult
            Body-joint angles in radians plus residual / FK diagnostics.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = np.asarray(target_rot, dtype=float)
        rw = self._rot_weight if rot_weight is None else rot_weight
        rg = self._rest_gain if rest_gain is None else float(rest_gain)
        use_rot = rw > _ROT_WEIGHT_EPS

        # Seed the private MjData
        self._ik_data.qpos[:] = seed_qpos

        nv = self._model.nv
        n_dof = len(self._body_dofs)
        last_w = 0.0

        for _ in range(self._max_iters):
            self._mj.mj_forward(self._model, self._ik_data)

            cur_pos = self._ik_data.site_xpos[self._site_id]
            cur_rot = self._ik_data.site_xmat[self._site_id].reshape(3, 3)
            pos_err = target_pos - cur_pos
            rot_vec = self._rotation_error(target_rot, cur_rot)

            if np.linalg.norm(pos_err) < self._pos_tol and (
                not use_rot or np.linalg.norm(rot_vec) < self._rot_tol
            ):
                break

            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            self._mj.mj_jacSite(self._model, self._ik_data, jacp, jacr, self._site_id)
            Jp = jacp[:, self._body_dofs]
            Jr = jacr[:, self._body_dofs]

            # Position-Jacobian manipulability: √det(Jp Jpᵀ)
            last_w = float(np.sqrt(max(np.linalg.det(Jp @ Jp.T), 0.0)))
            # Cap the singularity bonus — √det(Jp Jpᵀ) on this arm is ~0.01
            # even in a healthy home pose, so an uncapped 0.02/(w+ε) term
            # freezes the Newton step.
            extra = self._adaptive_damping / (last_w + 1e-3)
            lam = self._damping + min(extra, 2.0 * self._damping)
            lam_sq = lam ** 2

            if use_rot:
                J = np.vstack([Jp, rw * Jr])
                err = np.concatenate([pos_err, rw * rot_vec])
            else:
                J = Jp
                err = np.asarray(pos_err, dtype=float)

            n_task = J.shape[0]
            JJt = J @ J.T + lam_sq * np.eye(n_task)
            dq = J.T @ np.linalg.solve(JJt, err)

            if self._max_dq_rad > 0.0:
                np.clip(dq, -self._max_dq_rad, self._max_dq_rad, out=dq)

            cur = self._ik_data.qpos[self._body_dofs] + dq
            np.clip(cur, self._qpos_lo, self._qpos_hi, out=cur)
            self._ik_data.qpos[self._body_dofs] = cur

        # One null-space rest step *after* the primary solve.  Applying this
        # every Newton iteration multiplied the gain by max_iters and pinned
        # the arm at the home posture.
        if self._rest_qpos is not None and rg > 0.0:
            self._mj.mj_forward(self._model, self._ik_data)
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            self._mj.mj_jacSite(self._model, self._ik_data, jacp, jacr, self._site_id)
            Jp = jacp[:, self._body_dofs]
            Jr = jacr[:, self._body_dofs]
            last_w = float(np.sqrt(max(np.linalg.det(Jp @ Jp.T), 0.0)))
            extra = self._adaptive_damping / (last_w + 1e-3)
            lam = self._damping + min(extra, 2.0 * self._damping)
            if use_rot:
                J = np.vstack([Jp, rw * Jr])
            else:
                J = Jp
            JJt = J @ J.T + (lam ** 2) * np.eye(J.shape[0])
            j_hash = J.T @ np.linalg.inv(JJt)
            null_proj = np.eye(n_dof) - j_hash @ J
            q = self._ik_data.qpos[self._body_dofs]
            # Soft elbow bias only within ±15° of straight.  Prefer the
            # home-side fold but do not forbid crossing 0° — horizontal
            # arcs around the base need the other configuration.
            rest_err = np.zeros(n_dof)
            elbow = q[2]
            if abs(elbow) < np.radians(15.0):
                rest_err[2] = self._rest_qpos[2] - elbow
            dq_ns = null_proj @ (rg * rest_err)
            if self._max_dq_rad > 0.0:
                np.clip(dq_ns, -self._max_dq_rad, self._max_dq_rad, out=dq_ns)
            cur = q + dq_ns
            np.clip(cur, self._qpos_lo, self._qpos_hi, out=cur)
            self._ik_data.qpos[self._body_dofs] = cur

        self._mj.mj_forward(self._model, self._ik_data)
        achieved_pos = self._ik_data.site_xpos[self._site_id].copy()
        achieved_rot = self._ik_data.site_xmat[self._site_id].reshape(3, 3).copy()
        pos_err_n = float(np.linalg.norm(target_pos - achieved_pos))
        rot_err_n = float(np.linalg.norm(self._rotation_error(target_rot, achieved_rot)))
        return IKSolveResult(
            qpos=self._ik_data.qpos[self._body_dofs].copy(),
            pos_err=pos_err_n,
            rot_err=rot_err_n,
            manipulability=last_w,
            achieved_pos=achieved_pos,
            achieved_rot=achieved_rot,
        )
