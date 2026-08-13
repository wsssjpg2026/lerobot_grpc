"""MuJoCo-backed SO-101 follower servicer with joint and pose_delta action modes.

A standalone :class:`FollowerServicer` (following ``MockFollowerServicer``) that
wraps a MuJoCo simulation of the SO-101 arm.  In ``pose_delta`` mode it receives
end-effector pose deltas (8 FLOAT32 features from the shared pose_delta_schema),
converts them to joint targets via forward kinematics + inverse kinematics
(``RobotKinematics`` / placo), and drives the MuJoCo physics engine.  In ``joint``
mode it accepts joint-space actions directly — the same contract as the real
``SO101FollowerServicer`` minus the hardware.

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

    1. Reads the current body-joint angles from MuJoCo ``qpos``.
    2. Forward-kinematics → current end-effector pose (4×4).
    3. Composes the delta: ``pos_new = pos_current + delta_pos``;
       ``R_new = R_current @ R_delta`` (body-frame composition).
    4. Inverse-kinematics (placo, warm-started from current joints) → joint targets.
    5. Maps gripper distance (mm) → SO-101 gripper (0-100).
    6. Writes the 6 joint targets (converted to radians) into ``data.ctrl``.

    The MuJoCo physics engine (stepped inside :meth:`GetObservation`) drives the
    PD actuators toward those targets, producing realistic motion.

    Parameters
    ----------
    xml_path
        Path to the MuJoCo scene XML (e.g. ``scene.xml`` which includes
        ``so101_new_calib.xml``).
    urdf_path
        Path to the SO-101 URDF — required for *pose_delta* mode (FK/IK).
    action_mode
        ``"pose_delta"`` (default) or ``"joint"``.
    render
        If ``True``, launch a passive MuJoCo viewer window.
    orientation_weight
        IK orientation weight — ``0.01`` gives soft-orientation tracking suited
        to the 5-DOF under-actuated arm.
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
        # Accumulated target EE pose (4×4).  Deltas compose onto *this*, not onto
        # FK(actual qpos), so +Δ followed by −Δ always cancels regardless of PD
        # convergence lag.  Lazily synced from the current joints on first
        # SendAction (or after a reconnect that calls mj_resetData).
        self._target_pose: np.ndarray | None = None
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
        # normal rot_weight.  This lets shoulder_pan — the only Z-axis joint
        # — rotate freely to achieve lateral (Y) motion, accepting some yaw
        # drift (the 5-DOF kinematic coupling the user has accepted).
        self._translation_rot_weight = translation_rot_weight
        self._rotation_significant_rad = rotation_significant_rad
        if action_mode == "pose_delta":
            from .dls_ik import DLSIKSolver

            self._ik_solver = DLSIKSolver(
                self._model,
                site_name="gripperframe",
                body_dofs=list(range(len(BODY_JOINTS))),
                rot_weight=rot_weight,
            )

        # --- Optional viewer ------------------------------------------------
        self._viewer = None
        if render:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

        logger.info(
            "MuJoCoSO101Servicer ready: action_mode=%s xml=%s render=%s rot_weight=%s "
            "ctrl_smoothing_alpha=%s deadband=%.1fmm workspace_radius=%.0fmm",
            action_mode, xml_path, render, rot_weight,
            ctrl_smoothing_alpha, position_deadband_m * 1000, workspace_radius_m * 1000,
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

    def _pose_delta_to_joint_action(self, delta_action: dict[str, float]) -> dict[str, float]:
        """Converts a pose-delta action to a joint-space action via DLS IK.

        Pipeline (deadband → clamp → adaptive-weight DLS solve):

        1. **Deadband** — if both ``‖delta_pos‖`` and the rotation angle are
           below their thresholds, skip IK and reuse the last body-joint
           targets (gripper still updates).  Filters static hand noise.
        2. **Workspace clamp** — cap ``‖delta_pos‖`` per step.
        3. **Adaptive rot_weight DLS IK** — when the rotation delta is
           negligible (pure translation), a low ``translation_rot_weight``
           is used so shoulder_pan (the only Z-axis joint) can rotate freely
           for lateral (Y) motion, accepting yaw drift.  When rotation IS
           commanded, the normal ``rot_weight`` ensures yaw is tracked.

        Smoothing happens in :meth:`GetObservation` (per-physics-loop EMA on
        ``data.ctrl``), NOT here — this method returns the raw IK target and
        ``SendAction`` stores it as ``_target_ctrl``.
        """
        from lerobot.utils.rotation import Rotation as R

        # --- Decode pose delta ---
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

        # --- Magnitudes for deadband + workspace clamp ---
        pos_norm = float(np.linalg.norm(delta_pos))
        rot_angle = 2.0 * math.acos(min(abs(delta_quat[3]), 1.0))

        # --- Deadband: hold body joints, update only gripper ---
        if (
            self._last_joint_action is not None
            and pos_norm < self._position_deadband_m
            and rot_angle < self._rotation_deadband_rad
        ):
            joint_action = dict(self._last_joint_action)
            joint_action["gripper.pos"] = self._gripper_distance_to_0_100(gripper_distance)
            return joint_action

        # --- Workspace clamp: cap per-step displacement ---
        if pos_norm > self._workspace_radius_m:
            delta_pos = delta_pos * (self._workspace_radius_m / pos_norm)

        # --- Decode rotation delta (no yaw projection) ---
        # The SO-101 is a 5-DOF arm: shoulder_pan is the only yaw joint and
        # rotating it inevitably moves the EE laterally.  Yaw at constant XYZ
        # is kinematically impossible — but the user WANTS yaw response, so we
        # let the DLS handle it (rot_weight balances position vs rotation).
        # The resulting position drift is the correct 5-DOF behaviour.
        r_delta = R.from_quat(delta_quat)

        # --- Lazy-init target pose from current MuJoCo site state ---
        if self._target_pose is None:
            self._mj.mj_forward(self._model, self._data)
            sid = self._ik_solver._site_id
            self._target_pose = np.eye(4)
            self._target_pose[:3, 3] = self._data.site_xpos[sid].copy()
            self._target_pose[:3, :3] = self._data.site_xmat[sid].reshape(3, 3).copy()

        # --- Accumulate delta onto tracked target (body-frame rotation) ---
        self._target_pose = self._target_pose.copy()
        self._target_pose[:3, 3] += delta_pos
        self._target_pose[:3, :3] = self._target_pose[:3, :3] @ r_delta.as_matrix()

        # --- DLS IK: seed from previous solution, not PD-lagged qpos ---
        # Adaptive rot_weight: pure translations use a low weight so
        # shoulder_pan (the only Z-axis joint) can rotate freely for lateral
        # (Y) motion.  When rotation IS commanded, use the normal weight so
        # yaw is tracked.
        if rot_angle < self._rotation_significant_rad:
            effective_rw = self._translation_rot_weight
        else:
            effective_rw = None  # solver constructor default (rot_weight)
        seed = self._data.qpos.copy()
        if self._last_ik_rad is not None:
            seed[: len(self._last_ik_rad)] = self._last_ik_rad
        raw_rad = self._ik_solver.solve(
            self._target_pose[:3, 3],
            self._target_pose[:3, :3],
            seed,
            rot_weight=effective_rw,
        )
        self._last_ik_rad = raw_rad.copy()

        # --- Build joint action (radians → degrees + gripper from distance) ---
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
            self._target_pose = None  # re-sync from reset qpos on next SendAction
            self._last_ik_rad = None
            self._last_joint_action = None
            self._target_ctrl = self._data.ctrl.copy()  # home ctrl → no EMA drift
        # Sim is pre-calibrated — no manual range-of-motion recording needed.
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        with self._lock:
            self._connected = False
            if self._viewer:
                self._viewer.close()
                self._viewer = None
        return Empty()

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)

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
