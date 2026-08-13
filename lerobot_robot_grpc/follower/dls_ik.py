"""Damped Least Squares (DLS) inverse kinematics solver for MuJoCo models.

A compact, self-contained IK solver that uses MuJoCo's native Jacobian
(``mj_jacSite``) and a DLS update rule.  Inspired by the reference
implementation in ``simulation/src/sim_teleop/ik/dls.py`` (Pika → JAKA teleop).

The DLS formulation is numerically stable near kinematic singularities — the
damping term (λ²I) prevents the pseudo-inverse from diverging when the
Jacobian loses rank.  This makes it suitable for 5-DOF arms like the SO-101
where the task space (6-D pose) over-determines the joint space (5-D).

Rotation tracking is softened via ``rot_weight`` (default 0.1): the solver
biases ~10:1 toward position accuracy, accepting residual orientation error
when the two conflict.  This is the entire "decoupling" technique — no
two-stage IK, wrist-center frame, or FK compensation is needed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
        DLS damping factor λ (default 0.05).  Higher = more stable but slower
        convergence.
    rot_weight
        Rotation error weight relative to position (default 0.1).  Lower =
        position prioritised; 0.0 = position-only IK.
    max_iters
        Maximum DLS iterations per solve (default 20).
    pos_tol
        Position convergence threshold in metres (default 1e-4 = 0.1 mm).
    rot_tol
        Rotation convergence threshold in radians (default 1e-3 ≈ 0.06°).
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
    ):
        import mujoco

        self._mj = mujoco
        self._model = model
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in model")
        self._body_dofs = np.array(body_dofs, dtype=int)
        self._ik_data = mujoco.MjData(model)

        self._damping_sq = damping ** 2
        self._rot_weight = rot_weight
        self._max_iters = max_iters
        self._pos_tol = pos_tol
        self._rot_tol = rot_tol

        # Pre-extract joint limits for the solved DOFs
        self._qpos_lo = np.array([model.jnt_range[d, 0] for d in body_dofs])
        self._qpos_hi = np.array([model.jnt_range[d, 1] for d in body_dofs])

        logger.info(
            "DLSIKSolver ready: site='%s' dofs=%s damping=%.3f rot_weight=%.3f max_iters=%d",
            site_name, body_dofs, damping, rot_weight, max_iters,
        )

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        seed_qpos: np.ndarray,
        rot_weight: float | None = None,
    ) -> np.ndarray:
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
            the constructor default.  Lower values prioritise position; the
            caller can pass a low weight for pure-translation actions where
            the 5-DOF arm must rotate shoulder_pan (the only Z-axis joint) to
            achieve lateral (Y) motion, accepting some yaw drift.

        Returns
        -------
        (len(body_dofs),) joint angles in **radians** for the body DOFs.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = np.asarray(target_rot, dtype=float)
        rw = self._rot_weight if rot_weight is None else rot_weight

        # Seed the private MjData
        self._ik_data.qpos[:] = seed_qpos

        nv = self._model.nv
        for _ in range(self._max_iters):
            self._mj.mj_forward(self._model, self._ik_data)

            # --- Current EE pose from site ---
            cur_pos = self._ik_data.site_xpos[self._site_id].copy()
            cur_rot = self._ik_data.site_xmat[self._site_id].reshape(3, 3)

            # --- Errors ---
            pos_err = target_pos - cur_pos

            # Rotation error as axis-angle vector via MuJoCo quaternion utils
            R_err = target_rot @ cur_rot.T
            quat_err = np.zeros(4)  # MuJoCo convention: (w, x, y, z)
            self._mj.mju_mat2Quat(quat_err, R_err.flatten())
            rot_vec = np.zeros(3)
            self._mj.mju_quat2Vel(rot_vec, quat_err, 1.0)

            # Convergence check (unweighted rotation norm)
            if np.linalg.norm(pos_err) < self._pos_tol and np.linalg.norm(rot_vec) < self._rot_tol:
                break

            # --- Jacobian (6 × n_body_dofs) ---
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            self._mj.mj_jacSite(self._model, self._ik_data, jacp, jacr, self._site_id)
            J = np.vstack([jacp[:, self._body_dofs], jacr[:, self._body_dofs]])

            # --- Weighted error ---
            err = np.concatenate([pos_err, rot_vec * rw])

            # --- DLS update: dq = J^T (J J^T + λ²I)^{-1} err ---
            JJt = J @ J.T + self._damping_sq * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)

            # Apply and clip to joint limits
            cur = self._ik_data.qpos[self._body_dofs] + dq
            np.clip(cur, self._qpos_lo, self._qpos_hi, out=cur)
            self._ik_data.qpos[self._body_dofs] = cur

        return self._ik_data.qpos[self._body_dofs].copy()
