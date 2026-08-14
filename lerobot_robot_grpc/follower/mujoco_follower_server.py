"""MuJoCo-backed SO-101 follower servicer with joint and pose_delta action modes.

A standalone :class:`FollowerServicer` (following ``MockFollowerServicer``) that
wraps a MuJoCo simulation of the SO-101 arm.  In ``pose_delta`` mode it receives
end-effector pose deltas (8 FLOAT32 features from the shared pose_delta_schema),
converts them to joint targets via a Damped Least Squares (DLS) IK solver that
uses MuJoCo's native Jacobian, and drives the MuJoCo physics engine.  In
``joint`` mode it accepts joint-space actions directly — the same contract as the
real ``SO101FollowerServicer`` minus the hardware.

Designed as the prototype validation harness for delta-pose teleoperation
(wayfinder ticket #08): no real robot, no serial port, no calibration — the
servicer *is* the robot.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np
from google.protobuf.empty_pb2 import Empty

from .follower_server import FollowerServicer
from .utils import encode_feature, load_feature
from lerobot_robot_grpc.pose_delta_schema import build_pose_delta_feature_info
from lerobot_robot_grpc.protos import device_pb2

logger = logging.getLogger(__name__)

# SO-101 joint names — MuJoCo actuator names match lerobot motor names exactly.
JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# Body joints (excludes gripper) — these are the DOF the IK solver controls.
BODY_JOINTS: tuple[str, ...] = JOINTS[:-1]

# MuJoCo gripper actuator range in radians (from so101_new_calib.xml).
# lerobot uses 0-100 (0 = closed, 100 = open); MuJoCo uses radians.
GRIPPER_RAD_MIN: float = -0.17453
GRIPPER_RAD_MAX: float = 1.74533

# Default full-open distance for the Pika Sense gripper sensor (millimetres).
# Placeholder — tune once the leader servicer (#07) produces real sensor data.
DEFAULT_GRIPPER_MAX_DISTANCE_MM: float = 60.0

# Physics step period (seconds) — 50 Hz matches the real SO101 observation rate.
_PHYSICS_PERIOD_S: float = 1.0 / 50.0

# Home joint angles (degrees) for the 5 body joints.  Bent elbow avoids the
# full-extension singularity; the arm starts inside the workspace with room to
# move in every direction.  Gripper stays at its MuJoCo default (near-closed).
HOME_JOINTS_DEG: tuple[float, ...] = (0.0, -20.0, 60.0, -40.0, 0.0)


def _clamp_vector(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale *vec* so its Euclidean norm does not exceed *max_norm*."""
    if max_norm <= 0.0:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm:
        return vec
    return vec * (max_norm / norm)


def _scalar_feature_info(key: str) -> device_pb2.OneFeatureInfo:
    """Builds a CRITICAL FLOAT32 scalar feature info for *key*.

    Mirrors ``mock_follower.scalar_feature_info`` but adds ``WATCH_DOG_LEVEL_A``
    to match the real servicer convention (``SO101FollowerServicer``).
    """
    return device_pb2.OneFeatureInfo(
        key=key,
        criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
        watchdog=device_pb2.WatchDogLevel.WATCH_DOG_LEVEL_A,
        type=device_pb2.DataType.FLOAT32,
        shape=device_pb2.ImageShape(H=1, W=1, C=1),
        encoding=device_pb2.Encoding.RAW,
        img_quality=100,
    )


class MuJoCoSO101Servicer(FollowerServicer):
    """MuJoCo-backed SO-101 follower with joint or pose_delta action mode.

    In *pose_delta* mode each :meth:`SendAction` receives 8 FLOAT32 features
    (``hand.delta_pos.{x,y,z}``, ``hand.delta_rot.{qx,qy,qz,qw}``,
    ``gripper.distance``).  The servicer:

    1. Treats the 8-FLOAT32 ΔT as the **current offset from SetReference**,
       not a per-frame increment: ``T_intent = T_zero ⊕ ΔT``.
    2. Slews a command pose toward that intent (rate limit only).
    3. DLS inverse-kinematics (MuJoCo Jacobian) → joint targets.
    4. Maps gripper distance (mm) → SO-101 gripper (0-100).
    5. Writes the 6 joint targets (converted to radians) into ``data.ctrl``.

    ``T_zero`` is locked at Connect (home FK).  The virtual intent is never
    snapped back to FK — overshoot is logged, correspondence is kept.

    The MuJoCo physics engine (stepped inside :meth:`GetObservation`) drives the
    PD actuators toward those targets, producing realistic motion.

    Parameters
    ----------
    xml_path
        Path to the MuJoCo scene XML (e.g. ``scene.xml`` which includes
        ``so101_new_calib.xml``).
    urdf_path
        Unused (legacy parameter, kept for API compatibility).  The DLS IK
        solver uses MuJoCo's native Jacobian and does not need a URDF.
    action_mode
        ``"pose_delta"`` (default) or ``"joint"``.
    render
        If ``True``, launch a passive MuJoCo viewer window.
    rot_weight
        DLS rotation tracking weight — lower values prioritise position;
        ``0.3`` gives ~3:1 position:rotation ratio suited to the 5-DOF arm.
    gripper_max_distance_mm
        Full-open distance in millimetres for the linear gripper-distance →
        0-100 mapping.
    """

    def __init__(
        self,
        xml_path: str,
        urdf_path: str | None = None,
        action_mode: str = "pose_delta",
        render: bool = False,
        rot_weight: float = 0.3,
        gripper_max_distance_mm: float = DEFAULT_GRIPPER_MAX_DISTANCE_MM,
        home_joints_deg: tuple[float, ...] = HOME_JOINTS_DEG,
        ctrl_smoothing_alpha: float = 0.20,
        position_deadband_m: float = 0.001,
        rotation_deadband_rad: float = 0.005,
        workspace_radius_m: float = 0.015,  # 15mm/step — safety: prevents high-speed drift
        translation_rot_weight: float = 0.0,
        rotation_significant_rad: float = 0.009,
        snap_pos_err_m: float = 0.008,
        rest_gain: float = 0.08,
        max_dq_deg: float = 6.0,
        elbow_floor_deg: float | None = None,
        home_capture_m: float = 0.080,
        home_rest_gain: float = 0.40,
        home_seed_blend: float = 0.15,
        workspace_bubble_m: float | None = None,
        workspace_bubble_ratio: float = 0.60,
        workspace_escape: bool = True,
    ):
        if action_mode not in ("joint", "pose_delta"):
            raise ValueError(
                f"action_mode must be 'joint' or 'pose_delta', got {action_mode!r}"
            )
        self._action_mode = action_mode

        # --- MuJoCo backend -------------------------------------------------
        import mujoco

        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._lock = threading.Lock()
        self._connected = False
        self._latest_action: dict[str, float] = {}
        # Latch-once poses (4×4).  ``_t_zero`` is locked at Connect (home FK).
        # ``_target_pose`` is the assigned intent ``T_zero ⊕ ΔT``.
        # ``_cmd_pose`` slews toward the intent at ``workspace_radius_m``.
        self._t_zero: np.ndarray | None = None
        self._target_pose: np.ndarray | None = None
        self._cmd_pose: np.ndarray | None = None
        # Previous IK solution (raw, pre-EMA) — used as the DLS seed so the
        # solver converges from a stable neighbourhood instead of the
        # PD-lagged, ringing actual qpos (which creates a self-excited limit
        # cycle).  Mirrors galbot's ``self.cur_lj = np.array(jl)`` pattern.
        self._last_ik_rad: np.ndarray | None = None
        # Desired ctrl targets (radians, len(JOINTS)).  Set by SendAction;
        # GetObservation blends ``data.ctrl`` toward this at 50 Hz so motion
        # is continuous between actions (no "jolt-then-freeze" at low action
        # rates).  None until first Connect.
        self._target_ctrl: np.ndarray | None = None
        # Last IK joint_action dict (degrees + gripper %) — reused on deadband
        # so the arm holds its body-joint targets while the gripper still
        # updates independently.
        self._last_joint_action: dict[str, float] | None = None
        # DLS IK solver (pose_delta only).  Uses MuJoCo's native Jacobian via
        # the 'gripperframe' site — no external IK library needed.
        self._ik_solver = None

        # --- Feature schemas ------------------------------------------------
        self._obs_ft_info: dict[str, device_pb2.OneFeatureInfo] = {
            f"{j}.pos": _scalar_feature_info(f"{j}.pos") for j in JOINTS
        }
        if action_mode == "pose_delta":
            self._act_ft_info: dict[str, device_pb2.OneFeatureInfo] = (
                build_pose_delta_feature_info()
            )
        else:
            self._act_ft_info = {
                f"{j}.pos": _scalar_feature_info(f"{j}.pos") for j in JOINTS
            }

        # --- IK (pose_delta only) -------------------------------------------
        # Damped Least Squares solver using MuJoCo's native Jacobian.  Numerically
        # stable near singularities (damping term), and rot_weight balances
        # position vs orientation tracking in a single solve — no wrist bypass
        # or FK compensation needed.  Inspired by the reference Pika→JAKA impl.
        self._gripper_max_distance_mm = gripper_max_distance_mm
        self._home_joints_deg = home_joints_deg
        self._ctrl_smoothing_alpha = ctrl_smoothing_alpha
        self._position_deadband_m = position_deadband_m
        self._rotation_deadband_rad = rotation_deadband_rad
        self._workspace_radius_m = workspace_radius_m
        # Adaptive rot_weight: for pure translations (rotation delta below
        # this threshold) the DLS uses translation_rot_weight instead of the
        # normal rot_weight.  The rotational Jacobian is scaled (or dropped
        # at 0) so folding the elbow is not treated as an angular-velocity
        # cost.  shoulder_pan can then rotate for lateral (Y) motion.
        self._translation_rot_weight = translation_rot_weight
        self._rotation_significant_rad = rotation_significant_rad
        self._snap_pos_err_m = snap_pos_err_m
        self._rest_gain = rest_gain
        self._home_capture_m = float(home_capture_m)
        self._home_rest_gain = float(home_rest_gain)
        self._home_seed_blend = float(home_seed_blend)
        # Public diagnostics — tests and 1 Hz logs read these after each solve.
        self.last_reach_err_m: float = 0.0
        self.last_manipulability: float = 0.0
        self.last_overshoot: bool = False
        self.last_snapped: bool = False  # alias of last_overshoot (no longer rewrites intent)
        self.last_home_err_m: float = 0.0
        self._escaped: bool = False  # #13: last solve was re-seeded from home
        self._last_achieved_pos: np.ndarray | None = None
        self._last_achieved_rot: np.ndarray | None = None
        self._last_solved_pos: np.ndarray | None = None
        self._last_solved_rot: np.ndarray | None = None
        if action_mode == "pose_delta":
            from .dls_ik import DLSIKSolver

            rest_qpos = np.radians(np.asarray(home_joints_deg, dtype=float))
            # Optional hard elbow floor.  Default is None — the home-side
            # configuration is preferred via the null-space rest task, but
            # the solver may cross 0° when that is what reaches the target
            # (horizontal arcs around the base).
            qpos_lo_override = None
            if elbow_floor_deg is not None:
                qpos_lo_override = np.array(
                    [self._model.jnt_range[i, 0] for i in range(len(BODY_JOINTS))],
                    dtype=float,
                )
                if len(qpos_lo_override) > 2:
                    qpos_lo_override[2] = max(
                        qpos_lo_override[2], math.radians(elbow_floor_deg)
                    )
            self._ik_solver = DLSIKSolver(
                self._model,
                site_name="gripperframe",
                body_dofs=list(range(len(BODY_JOINTS))),
                rot_weight=rot_weight,
                rest_qpos=rest_qpos,
                rest_gain=rest_gain,
                max_dq_rad=math.radians(max_dq_deg),
                qpos_lo_override=qpos_lo_override,
            )
            # Home posture in radians — reused as the escape re-seed (#13).
            self._home_rad = np.radians(np.asarray(home_joints_deg, dtype=float))

            # --- Workspace bubble (#13) -------------------------------------
            # The follower owns the URDF, so it derives the safe |ΔT| bound
            # itself.  Max reach from the base is sampled once at startup;
            # the bubble is then the *clearance around the current T_zero*
            # (reach − |T_zero|) × ratio — recomputed on every re-latch so an
            # extended arm gets a smaller bubble.  Prevents the leader (a
            # whole room of tracking volume) from commanding the arm far
            # beyond reach.
            self._workspace_bubble_ratio = float(workspace_bubble_ratio)
            self._workspace_escape = bool(workspace_escape)
            self._bubble_auto = workspace_bubble_m is None
            if self._bubble_auto:
                self._max_reach_m = self._sample_max_reach_m()
                self._workspace_bubble_m = (
                    self._workspace_bubble_ratio * self._max_reach_m
                )  # placeholder until T_zero locks
                logger.info(
                    "Workspace bubble auto from URDF: max_reach=%.0fmm "
                    "(bubble locks to T_zero clearance on Connect/SetReference)",
                    self._max_reach_m * 1000.0,
                )
            else:
                self._max_reach_m = None
                self._workspace_bubble_m = float(workspace_bubble_m)
            self._last_clamp_log_ts = 0.0
            self._last_escape_log_ts = 0.0
        else:
            self._workspace_bubble_m = None
            self._max_reach_m = None
            self._bubble_auto = False
            self._home_rad = np.radians(np.asarray(home_joints_deg, dtype=float))
            self._workspace_escape = False
            self._last_clamp_log_ts = 0.0
            self._last_escape_log_ts = 0.0

        # --- Optional viewer ------------------------------------------------
        self._viewer = None
        if render:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

        logger.info(
            "MuJoCoSO101Servicer ready: action_mode=%s xml=%s render=%s rot_weight=%s "
            "translation_rot_weight=%s overshoot=%.1fmm rest_gain=%s "
            "ctrl_smoothing_alpha=%s deadband=%.1fmm workspace_radius=%.0fmm "
            "home_capture=%.0fmm bubble=%s",
            action_mode, xml_path, render, rot_weight,
            translation_rot_weight, snap_pos_err_m * 1000, rest_gain,
            ctrl_smoothing_alpha, position_deadband_m * 1000, workspace_radius_m * 1000,
            home_capture_m * 1000,
            (
                "auto"
                if self._workspace_bubble_m is not None and workspace_bubble_m is None
                else f"{self._workspace_bubble_m * 1000:.0f}mm"
                if self._workspace_bubble_m
                else "off"
            ),
        )
        if action_mode == "pose_delta":
            logger.warning(
                "pose_delta ΔT is the current offset from SetReference, not a "
                "per-frame increment. Pair with a latch-once Pika Sense leader; "
                "an old incremental leader will accumulate and the arm will fly."
            )

    # ------------------------------------------------------------------
    # Unit conversion helpers
    # ------------------------------------------------------------------

    def _joint_action_to_ctrl(self, joint_action: dict[str, float]) -> np.ndarray:
        """Converts a lerobot joint-action dict to a MuJoCo ctrl array (radians)."""
        ctrl = np.zeros(len(JOINTS))
        for i, joint in enumerate(JOINTS):
            val = joint_action.get(f"{joint}.pos", 0.0)
            if joint == "gripper":
                ctrl[i] = (val / 100.0) * (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN) + GRIPPER_RAD_MIN
            else:
                ctrl[i] = math.radians(val)
        return ctrl

    def _qpos_to_observation(self) -> dict[str, float]:
        """Reads MuJoCo ``qpos`` and converts to lerobot-normalised values."""
        obs: dict[str, float] = {}
        for i, joint in enumerate(JOINTS):
            rad = float(self._data.qpos[i])
            if joint == "gripper":
                obs[f"{joint}.pos"] = (rad - GRIPPER_RAD_MIN) / (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN) * 100.0
            else:
                obs[f"{joint}.pos"] = math.degrees(rad)
        return obs

    def _gripper_distance_to_0_100(self, distance_mm: float) -> float:
        """Maps gripper finger distance (mm) to SO-101 gripper position (0-100)."""
        return float(np.clip(distance_mm / self._gripper_max_distance_mm * 100.0, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Pose-delta → joint pipeline
    # ------------------------------------------------------------------

    def _lock_t_zero_from_fk(self) -> None:
        """Set T_zero / intent / cmd from the current gripperframe FK."""
        self._mj.mj_forward(self._model, self._data)
        sid = self._ik_solver._site_id
        pose = np.eye(4)
        pose[:3, 3] = self._data.site_xpos[sid].copy()
        pose[:3, :3] = self._data.site_xmat[sid].reshape(3, 3).copy()
        self._t_zero = pose
        self._target_pose = pose.copy()
        self._cmd_pose = pose.copy()

        # Auto bubble tracks the clearance around T_zero (#13): an arm
        # re-latched far out gets a smaller |ΔT| bound.
        if self._bubble_auto and self._max_reach_m is not None:
            t_zero_dist = float(np.linalg.norm(self._t_zero[:3, 3]))
            clearance = self._max_reach_m - t_zero_dist
            self._workspace_bubble_m = max(
                self._workspace_bubble_ratio * clearance,
                0.020,  # floor — never shrink to zero
            )
            logger.info(
                "Workspace bubble: T_zero at %.0fmm → clearance %.0fmm × %.2f "
                "= %.0fmm",
                t_zero_dist * 1000.0,
                clearance * 1000.0,
                self._workspace_bubble_ratio,
                self._workspace_bubble_m * 1000.0,
            )

    def _limit_overshoot_q(
        self,
        q_new: np.ndarray,
        q_prev: np.ndarray,
        lock_elbow: bool = True,
    ) -> np.ndarray:
        """When the intent is unreachable, do not flip configuration in one frame.

        ``lock_elbow=False`` is used by the workspace escape (#13): a
        deliberate unfold back to the home side is allowed to cross the elbow
        sign (the per-joint 8°/frame clip still smooths the transition).
        """
        max_step = math.radians(8.0)
        dq = np.clip(q_new - q_prev, -max_step, max_step)
        limited = q_prev + dq
        # Keep the elbow on the same side of straight if it was clearly folded.
        elbow = 2
        if (
            lock_elbow
            and abs(q_prev[elbow]) > math.radians(15.0)
            and q_prev[elbow] * limited[elbow] < 0.0
        ):
            limited[elbow] = q_prev[elbow]
        return limited

    def _at_joint_limit(self, qpos: np.ndarray, tol_deg: float = 0.5) -> bool:
        """True when any body joint sits at (or within ``tol_deg`` of) its limit."""
        lo, hi = self._ik_solver.qpos_limits
        tol = math.radians(tol_deg)
        return bool(
            np.any(qpos <= lo + tol) or np.any(qpos >= hi - tol)
        )

    def _sample_max_reach_m(self, samples: int = 8000, seed: int = 13) -> float:
        """Max gripper-site distance from the arm base over the joint ranges.

        Uniform sampling inside the URDF joint limits (deterministic seed)
        plus all limit-corner configurations — the max extension lives on the
        boundary.  The arm base is at the world origin in ``scene.xml``, so
        the distance is ``|site_xpos|``.  Used to auto-derive the workspace
        bubble (#13).
        """
        lo, hi = self._ik_solver.qpos_limits
        n = len(lo)
        rng = np.random.default_rng(seed)
        uniform = rng.uniform(lo, hi, size=(samples, n))
        corners = np.array(
            np.meshgrid(*[(lo[i], hi[i]) for i in range(n)]),
        ).T.reshape(-1, n)
        q = np.vstack([uniform, corners])
        scratch = self._mj.MjData(self._model)
        sid = self._ik_solver.site_id
        best = 0.0
        for i in range(len(q)):
            scratch.qpos[:n] = q[i]
            self._mj.mj_forward(self._model, scratch)
            best = max(best, float(np.linalg.norm(scratch.site_xpos[sid])))
        return best

    def _slew_cmd_toward_intent(self) -> None:
        """Rate-limit T_cmd toward T_intent.  ``workspace_radius_m<=0`` = unlimited."""
        intent = self._target_pose
        if self._cmd_pose is None:
            self._cmd_pose = intent.copy()
            return
        cmd = self._cmd_pose.copy()
        dpos = intent[:3, 3] - cmd[:3, 3]
        if self._workspace_radius_m <= 0.0:
            cmd[:3, 3] = intent[:3, 3]
            cmd[:3, :3] = intent[:3, :3]
        else:
            cmd[:3, 3] = cmd[:3, 3] + _clamp_vector(dpos, self._workspace_radius_m)
            from lerobot.utils.rotation import Rotation as Rot

            r_err = cmd[:3, :3].T @ intent[:3, :3]
            rv = Rot.from_matrix(r_err).as_rotvec()
            # ~15 mm of arc at 0.3 m reach is ~3°; use 6° so rotation is not
            # the bottleneck next to the translation slew.
            max_drot = 0.105
            n = float(np.linalg.norm(rv))
            if n > max_drot:
                rv = rv * (max_drot / n)
            cmd[:3, :3] = cmd[:3, :3] @ Rot.from_rotvec(rv).as_matrix()
        self._cmd_pose = cmd

    def _pose_delta_to_joint_action(self, delta_action: dict[str, float]) -> dict[str, float]:
        """Converts a latch-once pose offset to a joint-space action via DLS IK.

        Pipeline (assign intent → slew cmd → optional IK skip → DLS):

        1. **Assign intent** — ``T_intent = T_zero ⊕ ΔT``.  The same offset
           sent twice holds the arm.  Offset identity returns to ``T_zero``.
        2. **Slew cmd** — ``T_cmd`` chases the intent at ``workspace_radius_m``.
           The intent is never discarded.
        3. **Deadband** — if ``T_cmd`` barely moved since the last solve,
           skip IK and reuse body joints (gripper still updates).
        4. **Adaptive rot_weight DLS IK** — pure translation drops the
           rotational Jacobian.  Near ``T_zero`` the rest task is strengthened
           so a flipped elbow folds back to home.
        5. **Overshoot log** — residual above ``snap_pos_err_m`` is flagged.
           The intent is **not** rewritten to FK.

        Smoothing happens in :meth:`GetObservation` (per-physics-loop EMA on
        ``data.ctrl``), NOT here.
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

        if self._t_zero is None or self._target_pose is None:
            self._lock_t_zero_from_fk()

        # --- Workspace bubble (#13) ------------------------------------------
        # The leader's tracking volume is far larger than the arm: clamp |ΔT|
        # to the bubble so the intent never leaves a conservative reachable
        # shell around T_zero.  No hysteresis — the arm returns as soon as the
        # hand comes back inside.
        if self._workspace_bubble_m and self._workspace_bubble_m > 0.0:
            n = float(np.linalg.norm(delta_pos))
            if n > self._workspace_bubble_m:
                delta_pos = delta_pos * (self._workspace_bubble_m / n)
                now = time.time()
                if now - self._last_clamp_log_ts > 1.0:
                    self._last_clamp_log_ts = now
                    logger.info(
                        "WORKSPACE clamp: ΔT %.0fmm → %.0fmm (bubble %.0fmm)",
                        n * 1000.0,
                        self._workspace_bubble_m * 1000.0,
                        self._workspace_bubble_m * 1000.0,
                    )

        # Assign intent from T_zero — do not accumulate.
        intent = np.eye(4)
        intent[:3, 3] = self._t_zero[:3, 3] + delta_pos
        intent[:3, :3] = self._t_zero[:3, :3] @ r_delta.as_matrix()
        self._target_pose = intent
        self.last_home_err_m = float(np.linalg.norm(intent[:3, 3] - self._t_zero[:3, 3]))
        self._slew_cmd_toward_intent()

        cmd_pos = self._cmd_pose[:3, 3]
        cmd_rot = self._cmd_pose[:3, :3]
        if (
            self._last_joint_action is not None
            and self._last_solved_pos is not None
            and float(np.linalg.norm(cmd_pos - self._last_solved_pos)) < self._position_deadband_m
        ):
            r_cmd_err = self._last_solved_rot.T @ cmd_rot
            cmd_rot_n = float(np.linalg.norm(Rot.from_matrix(r_cmd_err).as_rotvec()))
            if cmd_rot_n < self._rotation_deadband_rad:
                joint_action = dict(self._last_joint_action)
                joint_action["gripper.pos"] = self._gripper_distance_to_0_100(gripper_distance)
                return joint_action

        if rot_angle < self._rotation_significant_rad:
            effective_rw = self._translation_rot_weight
        else:
            effective_rw = None
        near_home = self.last_home_err_m < self._home_capture_m
        rest_gain = self._home_rest_gain if near_home else self._rest_gain

        seed = self._data.qpos.copy()
        if self._last_ik_rad is not None:
            seed[: len(self._last_ik_rad)] = self._last_ik_rad
        if near_home and self._home_seed_blend > 0.0:
            home_rad = np.radians(np.asarray(self._home_joints_deg, dtype=float))
            b = self._home_seed_blend
            seed[: len(home_rad)] = (1.0 - b) * seed[: len(home_rad)] + b * home_rad

        result = self._ik_solver.solve(
            cmd_pos,
            cmd_rot,
            seed,
            rot_weight=effective_rw,
            rest_gain=rest_gain,
        )
        raw_rad = result.qpos

        # --- Workspace escape (#13) -------------------------------------------
        # Warm-starting from the previous solution is a local search: a flipped
        # elbow reaches most bubble targets too, so once folded over it never
        # unfolds (bench: elbow pinned at -96.8° for 11 s with a reachable
        # intent).  When the warm result is flipped / limit-saturated, re-solve
        # from the home posture and take it when its accuracy is comparable —
        # the arm folds back to the home side as soon as the target allows.
        self._escaped = False
        if self._workspace_escape and len(self._home_rad) == len(BODY_JOINTS):
            elbow = 2
            flipped = float(raw_rad[elbow]) < -math.radians(15.0)
            if flipped or self._at_joint_limit(raw_rad):
                home_seed = np.zeros_like(self._data.qpos)
                home_seed[: len(self._home_rad)] = self._home_rad
                alt = self._ik_solver.solve(
                    cmd_pos,
                    cmd_rot,
                    home_seed,
                    rot_weight=effective_rw,
                    rest_gain=rest_gain,
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
                            "IK escape: home re-seed accepted "
                            "(pos_err %.1f→%.1fmm, elbow %.0f→%.0f°)",
                            warm_err_m * 1000.0,
                            result.pos_err * 1000.0,
                            warm_elbow_deg,
                            math.degrees(float(raw_rad[elbow])),
                        )

        self.last_reach_err_m = result.pos_err
        self.last_manipulability = result.manipulability
        if (
            np.isfinite(raw_rad).all()
            and result.pos_err > self._snap_pos_err_m
            and self._last_ik_rad is not None
        ):
            raw_rad = self._limit_overshoot_q(
                raw_rad,
                self._last_ik_rad,
                lock_elbow=not self._escaped,
            )

        if not np.isfinite(raw_rad).all():
            logger.warning("IK returned NaN — holding last joint action.")
            if self._last_joint_action is not None:
                joint_action = dict(self._last_joint_action)
                joint_action["gripper.pos"] = self._gripper_distance_to_0_100(gripper_distance)
                return joint_action
            raw_rad = np.zeros(len(BODY_JOINTS))
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
        if now - getattr(self, "_last_ik_debug_ts", 0.0) > 1.0:
            self._last_ik_debug_ts = now
            logger.info(
                "IK: reach=%.1fmm home=%.1fmm overshoot=%s manip=%.4f q_deg=[%s]",
                self.last_reach_err_m * 1000.0,
                self.last_home_err_m * 1000.0,
                self.last_overshoot,
                self.last_manipulability,
                ", ".join(f"{math.degrees(q):.1f}" for q in raw_rad),
            )

        joint_action: dict[str, float] = {}
        for i, joint in enumerate(BODY_JOINTS):
            joint_action[f"{joint}.pos"] = math.degrees(float(raw_rad[i]))
        joint_action["gripper.pos"] = self._gripper_distance_to_0_100(gripper_distance)
        self._last_joint_action = dict(joint_action)
        return joint_action

    # ------------------------------------------------------------------
    # FollowerServicer RPCs
    # ------------------------------------------------------------------

    def GetInfo(self, request, context):
        return device_pb2.GetInfoResponse(
            observation_features=list(self._obs_ft_info.values()),
            action_features=list(self._act_ft_info.values()),
            feedback_features=list(self._act_ft_info.values()),
        )

    def Connect(self, request, context):
        with self._lock:
            # Reset to a clean physics state on each connect so a reconnect after
            # a crashed/disconnected session doesn't inherit stale qpos/ctrl.
            self._mj.mj_resetData(self._model, self._data)
            # Override the all-zeros default with a non-singular home pose so
            # the arm starts inside the workspace (bent elbow) with room to
            # move in every direction without coupling.
            for i, deg in enumerate(self._home_joints_deg):
                rad = math.radians(deg)
                self._data.qpos[i] = rad
                self._data.ctrl[i] = rad
            self._connected = True
            self._t_zero = None
            self._target_pose = None
            self._cmd_pose = None
            self._last_ik_rad = None
            self._last_joint_action = None
            self._last_solved_pos = None
            self._last_solved_rot = None
            self._target_ctrl = self._data.ctrl.copy()  # home ctrl → no EMA drift
            self.last_reach_err_m = 0.0
            self.last_manipulability = 0.0
            self.last_overshoot = False
            self.last_snapped = False
            self.last_home_err_m = 0.0
            self._last_achieved_pos = None
            self._last_achieved_rot = None
            if self._action_mode == "pose_delta" and self._ik_solver is not None:
                self._lock_t_zero_from_fk()
        # Sim is pre-calibrated — no manual range-of-motion recording needed.
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        with self._lock:
            self._connected = False
            # Keep viewer alive — same pattern as the leader's hardware
            # persistence.  The viewer freezes (no sync calls) until a new
            # client connects and GetObservation resumes.
        return Empty()

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)

    def SetReference(self, request, context):
        """Re-lock ``T_zero`` (and intent/cmd) at the current gripperframe FK.

        Clutch re-engage contract (wayfinder #10): the client calls this on
        the engage edge *before* the leader's SetReference, so the next
        ``ΔT = 0`` action maps onto the arm's current pose instead of pulling
        it back to the Connect home.  Joint targets / IK seed / ctrl are
        deliberately untouched — the arm must not move at re-engage.

        In ``joint`` mode this is a no-op (no Cartesian reference exists).
        """
        with self._lock:
            if self._action_mode == "pose_delta" and self._ik_solver is not None:
                self._lock_t_zero_from_fk()
                logger.info(
                    "SetReference: T_zero re-locked at current FK pos=[%.4f %.4f %.4f]",
                    *self._t_zero[:3, 3],
                )
            else:
                logger.info("SetReference: no-op (action_mode=%s)", self._action_mode)
        return Empty()

    def GetObservation(self, request, context):
        """Persistent stream: step MuJoCo physics at ~50 Hz and stream joint angles.

        Two responsibilities beyond physics stepping:

        1. **Real-time substeps** — advances ``n_substeps`` physics steps per
           loop so the sim matches wall-clock time (not 10 % real-time).
        2. **Per-loop ctrl EMA** — blends ``data.ctrl`` toward ``_target_ctrl``
           by ``ctrl_smoothing_alpha`` each iteration.  This produces
           *continuous* motion between ``SendAction`` calls instead of
           "jolt-then-freeze" — critical at low action rates (e.g. the demo's
           4 Hz).  The stiff PD (kp≈1000) sees a gradual ramp, not a step.
        """
        n_substeps = max(1, int(round(_PHYSICS_PERIOD_S / self._model.opt.timestep)))
        while context.is_active():
            with self._lock:
                # Per-loop EMA: ramp ctrl toward target (50 Hz continuous motion)
                if self._target_ctrl is not None:
                    a = self._ctrl_smoothing_alpha
                    self._data.ctrl[:] = a * self._target_ctrl + (1.0 - a) * self._data.ctrl
                for _ in range(n_substeps):
                    self._mj.mj_step(self._model, self._data)
                if self._viewer:
                    self._viewer.sync()
                obs = self._qpos_to_observation()
            yield from encode_feature(self._obs_ft_info, obs)
            time.sleep(_PHYSICS_PERIOD_S)

    def SendAction(self, request, context):
        # Decode wire features into a flat dict
        action: dict[str, float] = {}
        for feat in request.features:
            load_feature(feat, self._act_ft_info, action, aux_behavior="ignore")

        with self._lock:
            if self._action_mode == "pose_delta":
                joint_action = self._pose_delta_to_joint_action(action)
            else:
                joint_action = action

            # Store desired ctrl — GetObservation's per-loop EMA ramps toward it.
            # Writing data.ctrl directly would create a step function that the
            # stiff PD (kp≈1000, ζ≈0.26) overshoots on.
            self._target_ctrl = self._joint_action_to_ctrl(joint_action)
            self._latest_action = action

        # Echo commanded action back (A-class semantics, same as MockFollowerServicer).
        return device_pb2.Action(features=list(encode_feature(self._act_ft_info, action)))

    def GetFeedback(self, request, context):
        with self._lock:
            return encode_feature(self._act_ft_info, dict(self._latest_action))
