"""Sim/real-shared pose_delta control law (the one law).

Holds the entire assign + DLS + workspace-safety + slew + residual/stale-hold
pipeline behind a two-method interface so that the MuJoCo servicer and the real
Feetech servicer drive the *same* law.  Backend-specific concerns (where the
current joint angles come from, where solved joints go) stay in the servicer;
this module is pure kinematics + policy: given a latch-once deltaT, the current
joint vector, and a stale flag, it returns a joint-space action.

Kinematics engine: a single MuJoCo model, loaded by each servicer and passed in
here.  Both sim and real (and, later, a different arm) reuse the same
DLSIKSolver and FK via mj_forward -- only the model instance and the joint
source differ.  See wayfinder pika-sense-real #03 and research/09 section 7
("same solver + same hold; do not invent a second IK for hardware").

Workspace safety is a pluggable WorkspacePolicy: the sim keeps its proven
clearance bubble (clamps the offset to a fraction of the remaining reach), the
real arm uses a base safety sphere (clamps the absolute intent to max_reach
times a ratio to prevent the full-extension singularity).  Same interface, two
geometries.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from .dls_ik import DLSIKSolver

logger = logging.getLogger(__name__)


def _clamp_vector(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale vec so its Euclidean norm does not exceed max_norm."""
    if max_norm <= 0.0:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm:
        return vec
    return vec * (max_norm / norm)


@dataclass(frozen=True)
class JointSolution:
    """Result of one pose_delta solve, including diagnostics.

    joint_action is lerobot-normalised (body joints in degrees, gripper.pos in
    0-100) -- exactly what both backends write.
    """

    joint_action: dict[str, float]
    pos_err_m: float
    overshoot: bool
    held: bool
    stale: bool
    escaped: bool
    manipulability: float


class WorkspacePolicy:
    """Bounds the assigned intent so the leader tracking volume cannot drag a
    small arm beyond a safe, reachable shell.

    A policy answers two questions: the effective radius (a scalar for
    logging/tests) and the clamp of a raw intent pose.  Implementations differ
    in geometry -- a clearance bubble bounds motion relative to T_zero, a base
    sphere bounds the absolute intent -- but the law treats them uniformly.
    """

    def effective_radius(self, t_zero_pos, max_reach_m):
        raise NotImplementedError

    def clamp(self, raw_intent_pos, t_zero_pos, max_reach_m):
        raise NotImplementedError


class NoBubble(WorkspacePolicy):
    """Workspace safety disabled -- the intent passes through unchanged."""

    def effective_radius(self, t_zero_pos, max_reach_m):
        return 0.0

    def clamp(self, raw_intent_pos, t_zero_pos, max_reach_m):
        return raw_intent_pos


class ClearanceBubble(WorkspacePolicy):
    """Sim proven bubble: bound the offset to a fraction of the remaining reach
    around T_zero.  An arm re-latched far out gets a smaller bubble, so it cannot
    be commanded past full extension.  Recomputed implicitly from the current
    T_zero on every clamp (no stored state to go stale).
    """

    def __init__(self, ratio=0.60, floor_m=0.020):
        self._ratio = float(ratio)
        self._floor_m = float(floor_m)

    def effective_radius(self, t_zero_pos, max_reach_m):
        if t_zero_pos is None:
            return max(self._ratio * max_reach_m, self._floor_m)
        clearance = max_reach_m - float(np.linalg.norm(t_zero_pos))
        return max(self._ratio * clearance, self._floor_m)

    def clamp(self, raw_intent_pos, t_zero_pos, max_reach_m):
        radius = self.effective_radius(t_zero_pos, max_reach_m)
        delta = raw_intent_pos - t_zero_pos
        n = float(np.linalg.norm(delta))
        if n > radius:
            return t_zero_pos + delta * (radius / n)
        return raw_intent_pos


class FixedBubble(WorkspacePolicy):
    """A fixed-radius bubble around T_zero (the manual-override case)."""

    def __init__(self, radius_m):
        self._radius_m = float(radius_m)

    def effective_radius(self, t_zero_pos, max_reach_m):
        return self._radius_m

    def clamp(self, raw_intent_pos, t_zero_pos, max_reach_m):
        delta = raw_intent_pos - t_zero_pos
        n = float(np.linalg.norm(delta))
        if n > self._radius_m:
            return t_zero_pos + delta * (self._radius_m / n)
        return raw_intent_pos


class BaseSafetySphere(WorkspacePolicy):
    """Real-arm base safety sphere: bound the absolute intent to max_reach times
    ratio so the arm never approaches the full-extension singularity (which
    leaves it unable to return home).  Radius is recomputed from max_reach on
    every clamp, so a calibration-derived max_reach override flows through
    automatically.
    """

    def __init__(self, ratio=0.72):
        self._ratio = float(ratio)

    def effective_radius(self, t_zero_pos, max_reach_m):
        return self._ratio * max_reach_m

    def clamp(self, raw_intent_pos, t_zero_pos, max_reach_m):
        radius = self._ratio * max_reach_m
        n = float(np.linalg.norm(raw_intent_pos))
        if n > radius:
            return raw_intent_pos * (radius / n)
        return raw_intent_pos


def workspace_policy_from_legacy(workspace_bubble_m, ratio=0.60, floor_m=0.020):
    """Build a policy from the legacy workspace_bubble_m knob.

    - None  -> ClearanceBubble (auto, the sim default).
    - 0.0   -> NoBubble (disabled).
    - >0.0  -> FixedBubble (manual override).

    Lets the existing servicer/tests keep their constructor signature while the
    real servicer passes a BaseSafetySphere directly.
    """
    if workspace_bubble_m is None:
        return ClearanceBubble(ratio=ratio, floor_m=floor_m)
    if workspace_bubble_m <= 0.0:
        return NoBubble()
    return FixedBubble(workspace_bubble_m)


class PoseDeltaLaw:
    """The shared pose_delta control law: a deep module behind two methods.

    Owns the latch-once T_zero, the slew command, the DLS IK solver, the
    workspace policy, and the residual/stale holds.  A servicer drives it by:

    1. reading its current joint vector (sim data.qpos / real
       Present_Position to rad) and passing it to lock_reference / solve;
    2. writing the returned joint_action to its backend (MuJoCo ctrl / Feetech
       send_action).

    FK and IK both run on the same MuJoCo model passed at construction -- the
    real arm loads it purely as a kinematics oracle, seeded with measured
    joints.  Nothing in this module reads a live sim state directly: every FK
    comes from the qpos_rad argument, which is the seam.

    Parameters
    ----------
    model
        mujoco.MjModel -- the kinematics engine.  Same instance the DLSIKSolver
        uses.
    site_name
        MuJoCo site whose world pose is tracked (e.g. 'gripperframe').
    body_dofs
        qpos indices of the body joints the solver may move (gripper excluded).
    body_joint_names
        Names of those body joints, in order -- used to build the "<joint>.pos"
        keys of the returned action dict.  Arm-parametric so a different arm
        (e.g. a humanoid arm) just passes its own names.
    home_joints_deg
        Home posture (degrees) for the body joints -- null-space rest task and
        the limit-escape re-seed.
    workspace_policy
        How the assigned intent is bounded.  See WorkspacePolicy.
    residual_hold_m
        If set, a solve whose IK residual exceeds this (metres) returns the last
        joint action instead of walking toward an unreachable target.  None
        disables it (sim default -- preserves the apply+overshoot-limit
        behaviour).
    max_reach_override_m
        Override the URDF-sampled max reach (e.g. with a calibration-derived
        value).  None -> sample from the model joint ranges.
    """

    def __init__(
        self,
        model,
        *,
        site_name,
        body_dofs,
        body_joint_names,
        home_joints_deg,
        workspace_policy,
        rot_weight=0.3,
        translation_rot_weight=0.0,
        rotation_significant_rad=0.009,
        rest_gain=0.08,
        home_capture_m=0.080,
        home_rest_gain=0.40,
        home_seed_blend=0.15,
        snap_pos_err_m=0.008,
        position_deadband_m=0.001,
        rotation_deadband_rad=0.005,
        workspace_radius_m=0.015,
        max_dq_deg=6.0,
        elbow_floor_deg=None,
        gripper_max_distance_mm=60.0,
        workspace_escape=True,
        residual_hold_m=None,
        max_reach_override_m=None,
    ):
        import mujoco

        self._mj = mujoco
        self._model = model
        self._body_dofs = tuple(int(d) for d in body_dofs)
        self._body_joint_names = tuple(body_joint_names)
        self._n_body = len(self._body_dofs)
        self._gripper_max_distance_mm = float(gripper_max_distance_mm)
        self._workspace_policy = workspace_policy
        self._workspace_escape = bool(workspace_escape)
        self._residual_hold_m = None if residual_hold_m is None else float(residual_hold_m)

        # FK scratch (never the sim live data -- seeded from the qpos argument).
        self._fk_data = mujoco.MjData(model)

        # --- DLS IK solver -------------------------------------------------
        rest_qpos = np.radians(np.asarray(home_joints_deg, dtype=float))
        qpos_lo_override = None
        if elbow_floor_deg is not None:
            qpos_lo_override = np.array(
                [model.jnt_range[i, 0] for i in range(self._n_body)],
                dtype=float,
            )
            if len(qpos_lo_override) > 2:
                qpos_lo_override[2] = max(
                    qpos_lo_override[2], math.radians(elbow_floor_deg)
                )
        self._ik_solver = DLSIKSolver(
            model,
            site_name=site_name,
            body_dofs=list(self._body_dofs),
            rot_weight=rot_weight,
            rest_qpos=rest_qpos,
            rest_gain=rest_gain,
            max_dq_rad=math.radians(max_dq_deg),
            qpos_lo_override=qpos_lo_override,
        )
        self._site_id = self._ik_solver.site_id

        # --- Config (mirrors the sim servicer tuned knobs) ------------------
        self._rot_weight = float(rot_weight)
        self._translation_rot_weight = float(translation_rot_weight)
        self._rotation_significant_rad = float(rotation_significant_rad)
        self._home_capture_m = float(home_capture_m)
        self._home_rest_gain = float(home_rest_gain)
        self._home_seed_blend = float(home_seed_blend)
        self._rest_gain = float(rest_gain)
        self._snap_pos_err_m = float(snap_pos_err_m)
        self._position_deadband_m = float(position_deadband_m)
        self._rotation_deadband_rad = float(rotation_deadband_rad)
        self._workspace_radius_m = float(workspace_radius_m)
        self._home_joints_deg = tuple(float(v) for v in home_joints_deg)
        self._home_rad = np.radians(np.asarray(self._home_joints_deg, dtype=float))

        # --- Max reach (URDF-sampled, or calibration override) -------------
        self._max_reach_m = (
            float(max_reach_override_m)
            if max_reach_override_m is not None
            else self._sample_max_reach_m()
        )

        # --- Mutable solve state (reset on Connect) -------------------------
        self._reset_state()

        # Throttle helpers for info logs.
        self._last_clamp_log_ts = 0.0
        self._last_escape_log_ts = 0.0
        self._last_ik_debug_ts = 0.0

        logger.info(
            "PoseDeltaLaw ready: site=%r dofs=%s max_reach=%.0fmm policy=%s "
            "rot_weight=%.3f slew=%.0fmm residual_hold=%s",
            site_name, self._body_dofs, self._max_reach_m * 1000.0,
            type(workspace_policy).__name__, rot_weight,
            workspace_radius_m * 1000.0,
            "off" if self._residual_hold_m is None else "%.0fmm" % (self._residual_hold_m * 1000),
        )

    # ------------------------------------------------------------------
    # Public state / diagnostics (read by servicers, tests, 1 Hz logs)
    # ------------------------------------------------------------------

    @property
    def max_reach_m(self):
        return self._max_reach_m

    @property
    def ik_solver(self):
        return self._ik_solver

    @property
    def workspace_bubble_m(self):
        """Effective workspace radius (diagnostics).

        Geometry depends on the policy: ClearanceBubble returns the remaining
        reach around T_zero; BaseSafetySphere returns ratio x max_reach
        (base-relative, independent of T_zero); FixedBubble a constant; NoBubble 0.
        """
        tz = None if self._t_zero is None else self._t_zero[:3, 3]
        return self._workspace_policy.effective_radius(tz, self._max_reach_m)

    # --- Observable state: the public read surface tests/logs/servicers use
    #     instead of reaching into private fields. Each asserts its precondition
    #     (lock_reference / solve must have run) and returns a definite type, so
    #     the law's private storage is not part of its interface. ---
    @property
    def t_zero(self) -> np.ndarray:
        """Locked reference pose (4x4). Available after lock_reference()."""
        assert self._t_zero is not None, "lock_reference() first"
        return self._t_zero

    @property
    def target_pose(self) -> np.ndarray:
        """Current assigned intent T_zero (+) deltaT (4x4). After lock_reference/solve."""
        assert self._target_pose is not None, "lock_reference()/solve() first"
        return self._target_pose

    @property
    def last_ik_rad(self) -> np.ndarray:
        """Last solved body-joint vector (rad) -- the DLS warm-start seed."""
        assert self._last_ik_rad is not None, "solve()/seed_ik() first"
        return self._last_ik_rad

    @property
    def last_achieved_pos(self) -> np.ndarray:
        """FK position achieved by the last accepted solve (3,)."""
        assert self._last_achieved_pos is not None, "solve() first"
        return self._last_achieved_pos

    @property
    def escaped(self) -> bool:
        """True if the last solve re-seeded from home (limit-escape)."""
        return self._escaped

    def seed_ik(self, rad) -> None:
        """Set the DLS warm-start seed (the previous solve's joint vector).

        Diagnostic/test seam for driving the limit-escape path from a known
        posture without solving first; the real pipeline sets this every solve.
        """
        self._last_ik_rad = np.asarray(rad, dtype=float).copy()

    # ------------------------------------------------------------------
    # Reset / latch
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._t_zero = None
        self._target_pose = None
        self._cmd_pose = None
        self._last_ik_rad = None
        self._last_joint_action = None
        self._last_solved_pos = None
        self._last_solved_rot = None
        self._last_achieved_pos = None
        self._last_achieved_rot = None
        self.last_reach_err_m = 0.0
        self.last_manipulability = 0.0
        self.last_overshoot = False
        self.last_snapped = False
        self.last_home_err_m = 0.0
        self._escaped = False

    def reset(self):
        """Clear all latch/solve state (call on Connect)."""
        self._reset_state()

    def _fk(self, qpos_rad):
        """4x4 EE pose from a full joint vector via the shared model."""
        self._fk_data.qpos[:] = qpos_rad
        self._mj.mj_forward(self._model, self._fk_data)
        pose = np.eye(4)
        pose[:3, 3] = self._fk_data.site_xpos[self._site_id].copy()
        pose[:3, :3] = self._fk_data.site_xmat[self._site_id].reshape(3, 3).copy()
        return pose

    def lock_reference(self, qpos_rad):
        """Re-latch T_zero / intent / cmd at the FK of the given joint vector.

        Clutch re-engage contract: the servicer calls this (SetReference) with
        the current joints so the next deltaT=0 maps onto the arm current pose
        instead of pulling back to the Connect home.  Nothing here moves the arm
        -- it only re-anchors the reference.
        """
        pose = self._fk(qpos_rad)
        self._t_zero = pose
        self._target_pose = pose.copy()
        self._cmd_pose = pose.copy()
        tz = float(np.linalg.norm(self._t_zero[:3, 3]))
        radius = self._workspace_policy.effective_radius(self._t_zero[:3, 3], self._max_reach_m)
        logger.info(
            "PoseDeltaLaw reference locked: T_zero at %.0fmm -> workspace radius %.0fmm (%s)",
            tz * 1000.0, radius * 1000.0, type(self._workspace_policy).__name__,
        )

    # ------------------------------------------------------------------
    # Unit helpers
    # ------------------------------------------------------------------

    def _gripper_distance_to_0_100(self, distance_mm):
        return float(np.clip(distance_mm / self._gripper_max_distance_mm * 100.0, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    def _limit_overshoot_q(self, q_new, q_prev, lock_elbow=True):
        """When the intent is unreachable, do not flip configuration in one frame.

        lock_elbow=False is used by the workspace escape: a deliberate unfold
        back to the home side may cross the elbow sign (the per-joint clip still
        smooths the transition).
        """
        max_step = math.radians(8.0)
        dq = np.clip(q_new - q_prev, -max_step, max_step)
        limited = q_prev + dq
        elbow = 2
        if (
            lock_elbow
            and abs(q_prev[elbow]) > math.radians(15.0)
            and q_prev[elbow] * limited[elbow] < 0.0
        ):
            limited[elbow] = q_prev[elbow]
        return limited

    def _at_joint_limit(self, qpos, tol_deg=0.5):
        lo, hi = self._ik_solver.qpos_limits
        tol = math.radians(tol_deg)
        return bool(np.any(qpos <= lo + tol) or np.any(qpos >= hi - tol))

    def _sample_max_reach_m(self, samples=8000, seed=13):
        """Max gripper-site distance from the arm base over the joint ranges."""
        lo, hi = self._ik_solver.qpos_limits
        n = len(lo)
        rng = np.random.default_rng(seed)
        uniform = rng.uniform(lo, hi, size=(samples, n))
        corners = np.array(np.meshgrid(*[(lo[i], hi[i]) for i in range(n)])).T.reshape(-1, n)
        q = np.vstack([uniform, corners])
        scratch = self._mj.MjData(self._model)
        best = 0.0
        for i in range(len(q)):
            scratch.qpos[:n] = q[i]
            self._mj.mj_forward(self._model, scratch)
            best = max(best, float(np.linalg.norm(scratch.site_xpos[self._site_id])))
        return best

    def _slew_cmd_toward_intent(self):
        """Rate-limit T_cmd toward T_intent.  workspace_radius_m<=0 = unlimited."""
        intent = self._target_pose
        if intent is None:
            return  # solve locks the reference first; nothing to slew yet
        cmd_pose = self._cmd_pose
        if cmd_pose is None:
            self._cmd_pose = intent.copy()
            return
        cmd = cmd_pose.copy()
        dpos = intent[:3, 3] - cmd[:3, 3]
        if self._workspace_radius_m <= 0.0:
            cmd[:3, 3] = intent[:3, 3]
            cmd[:3, :3] = intent[:3, :3]
        else:
            cmd[:3, 3] = cmd[:3, 3] + _clamp_vector(dpos, self._workspace_radius_m)
            from lerobot.utils.rotation import Rotation as Rot

            r_err = cmd[:3, :3].T @ intent[:3, :3]
            rv = Rot.from_matrix(r_err).as_rotvec()
            max_drot = 0.105
            n = float(np.linalg.norm(rv))
            if n > max_drot:
                rv = rv * (max_drot / n)
            cmd[:3, :3] = cmd[:3, :3] @ Rot.from_rotvec(rv).as_matrix()
        self._cmd_pose = cmd

    # ------------------------------------------------------------------
    # The law
    # ------------------------------------------------------------------

    def solve(self, delta_action, qpos_rad, *, stale=False):
        """Convert a latch-once pose offset to a joint-space action via DLS IK.

        Pipeline: assign intent -> workspace clamp -> slew cmd -> optional IK
        skip (deadband) -> adaptive-rot DLS -> limit escape -> overshoot limit
        -> residual/stale hold.  qpos_rad is the current full joint vector (the
        only backend-injected input); FK and IK both run on it via the shared
        model.  See the sim servicer docstring for the detailed step rationale.
        """
        from lerobot.utils.rotation import Rotation as Rot

        delta_pos = np.array(
            [
                delta_action["hand.delta_pos.x"],
                delta_action["hand.delta_pos.y"],
                delta_action["hand.delta_pos.z"],
            ],
            dtype=float,
        )
        delta_quat = np.array(
            [
                delta_action["hand.delta_rot.qx"],
                delta_action["hand.delta_rot.qy"],
                delta_action["hand.delta_rot.qz"],
                delta_action["hand.delta_rot.qw"],
            ],
            dtype=float,
        )
        gripper_distance = delta_action["gripper.distance"]
        rot_angle = 2.0 * math.acos(min(abs(delta_quat[3]), 1.0))
        r_delta = Rot.from_quat(delta_quat)
        gripper_0_100 = self._gripper_distance_to_0_100(gripper_distance)

        if self._t_zero is None or self._target_pose is None:
            self.lock_reference(qpos_rad)
        t_zero = self._t_zero
        assert t_zero is not None

        # --- Stale hold: leader stream died -> freeze at the last command ---
        if stale and self._last_joint_action is not None:
            ja = dict(self._last_joint_action)
            ja["gripper.pos"] = gripper_0_100
            return JointSolution(ja, self.last_reach_err_m, self.last_overshoot, True, True, self._escaped, self.last_manipulability)

        # --- Assign intent from T_zero (do not accumulate) -----------------
        raw_intent_pos = t_zero[:3, 3] + delta_pos
        # --- Workspace policy: bound the intent to a safe reachable shell --
        clamped_intent_pos = self._workspace_policy.clamp(
            raw_intent_pos, t_zero[:3, 3], self._max_reach_m
        )
        radius = self._workspace_policy.effective_radius(t_zero[:3, 3], self._max_reach_m)
        if radius > 0.0:
            requested = float(np.linalg.norm(delta_pos))
            if requested > radius:
                now = time.time()
                if now - self._last_clamp_log_ts > 1.0:
                    self._last_clamp_log_ts = now
                    logger.info(
                        "WORKSPACE clamp: dT %.0fmm -> %.0fmm (radius %.0fmm, %s)",
                        requested * 1000.0, radius * 1000.0, radius * 1000.0,
                        type(self._workspace_policy).__name__,
                    )

        intent = np.eye(4)
        intent[:3, 3] = clamped_intent_pos
        intent[:3, :3] = t_zero[:3, :3] @ r_delta.as_matrix()
        self._target_pose = intent
        self.last_home_err_m = float(np.linalg.norm(intent[:3, 3] - t_zero[:3, 3]))
        self._slew_cmd_toward_intent()
        cmd_pose = self._cmd_pose
        assert cmd_pose is not None

        cmd_pos = cmd_pose[:3, 3]
        cmd_rot = cmd_pose[:3, :3]

        # --- Deadband: reuse last body joints, update gripper only --------
        if (
            self._last_joint_action is not None
            and self._last_solved_pos is not None
            and self._last_solved_rot is not None
            and float(np.linalg.norm(cmd_pos - self._last_solved_pos)) < self._position_deadband_m
        ):
            r_cmd_err = self._last_solved_rot.T @ cmd_rot
            cmd_rot_n = float(np.linalg.norm(Rot.from_matrix(r_cmd_err).as_rotvec()))
            if cmd_rot_n < self._rotation_deadband_rad:
                ja = dict(self._last_joint_action)
                ja["gripper.pos"] = gripper_0_100
                return JointSolution(ja, self.last_reach_err_m, self.last_overshoot, False, False, self._escaped, self.last_manipulability)

        if rot_angle < self._rotation_significant_rad:
            effective_rw = self._translation_rot_weight
        else:
            effective_rw = None
        near_home = self.last_home_err_m < self._home_capture_m
        rest_gain = self._home_rest_gain if near_home else self._rest_gain

        seed = np.array(qpos_rad, dtype=float)
        if self._last_ik_rad is not None:
            seed[: len(self._last_ik_rad)] = self._last_ik_rad
        if near_home and self._home_seed_blend > 0.0:
            home_rad = np.radians(np.asarray(self._home_joints_deg, dtype=float))
            b = self._home_seed_blend
            seed[: len(home_rad)] = (1.0 - b) * seed[: len(home_rad)] + b * home_rad

        result = self._ik_solver.solve(
            cmd_pos, cmd_rot, seed, rot_weight=effective_rw, rest_gain=rest_gain,
        )
        raw_rad = result.qpos

        # --- Workspace escape: flipped / limit-saturated -> home re-seed --
        self._escaped = False
        if self._workspace_escape and len(self._home_rad) == self._n_body:
            elbow = 2
            flipped = float(raw_rad[elbow]) < -math.radians(15.0)
            if flipped or self._at_joint_limit(raw_rad):
                home_seed = np.zeros_like(seed)
                home_seed[: len(self._home_rad)] = self._home_rad
                alt = self._ik_solver.solve(
                    cmd_pos, cmd_rot, home_seed, rot_weight=effective_rw, rest_gain=rest_gain,
                )
                if np.isfinite(alt.qpos).all() and (
                    alt.pos_err <= max(result.pos_err * 1.25, 0.003)
                ):
                    warm_err_m = result.pos_err
                    warm_elbow_deg = math.degrees(float(raw_rad[elbow]))
                    raw_rad = alt.qpos
                    result = alt
                    self._escaped = True
                    now = time.time()
                    if now - self._last_escape_log_ts > 1.0:
                        self._last_escape_log_ts = now
                        logger.info(
                            "IK escape: home re-seed accepted (pos_err %.1f->%.1fmm, elbow %.0f->%.0fdeg)",
                            warm_err_m * 1000.0, result.pos_err * 1000.0,
                            warm_elbow_deg, math.degrees(float(raw_rad[elbow])),
                        )

        self.last_reach_err_m = result.pos_err
        self.last_manipulability = result.manipulability

        # --- Residual hold: unreachable intent -> hold last joints ---------
        if (
            self._residual_hold_m is not None
            and result.pos_err > self._residual_hold_m
            and self._last_joint_action is not None
        ):
            ja = dict(self._last_joint_action)
            ja["gripper.pos"] = gripper_0_100
            self.last_overshoot = True
            self.last_snapped = True
            return JointSolution(ja, result.pos_err, True, True, False, self._escaped, result.manipulability)

        if (
            np.isfinite(raw_rad).all()
            and result.pos_err > self._snap_pos_err_m
            and self._last_ik_rad is not None
        ):
            raw_rad = self._limit_overshoot_q(raw_rad, self._last_ik_rad, lock_elbow=not self._escaped)

        if not np.isfinite(raw_rad).all():
            logger.warning("IK returned NaN -- holding last joint action.")
            if self._last_joint_action is not None:
                ja = dict(self._last_joint_action)
                ja["gripper.pos"] = gripper_0_100
                return JointSolution(ja, self.last_reach_err_m, self.last_overshoot, True, False, self._escaped, self.last_manipulability)
            raw_rad = np.zeros(self._n_body)
            self.last_overshoot = False
        else:
            self.last_overshoot = result.pos_err > self._snap_pos_err_m
            self._last_achieved_pos = result.achieved_pos.copy()
            self._last_achieved_rot = result.achieved_rot.copy()
            self._last_solved_pos = cmd_pos.copy()
            self._last_solved_rot = cmd_rot.copy()
        self.last_snapped = self.last_overshoot

        self._last_ik_rad = raw_rad.copy()

        now = time.time()
        if now - self._last_ik_debug_ts > 1.0:
            self._last_ik_debug_ts = now
            logger.info(
                "IK: reach=%.1fmm home=%.1fmm overshoot=%s manip=%.4f q_deg=[%s]",
                self.last_reach_err_m * 1000.0, self.last_home_err_m * 1000.0,
                self.last_overshoot, self.last_manipulability,
                ", ".join("%.1f" % math.degrees(q) for q in raw_rad),
            )

        joint_action = {}
        for i, joint in enumerate(self._body_joint_names):
            joint_action["%s.pos" % joint] = math.degrees(float(raw_rad[i]))
        joint_action["gripper.pos"] = gripper_0_100
        self._last_joint_action = dict(joint_action)
        return JointSolution(
            joint_action, self.last_reach_err_m, self.last_overshoot, False, False,
            self._escaped, self.last_manipulability,
        )
