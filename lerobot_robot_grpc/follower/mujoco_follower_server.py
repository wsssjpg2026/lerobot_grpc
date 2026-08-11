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
        orientation_weight: float = 0.01,
        gripper_max_distance_mm: float = DEFAULT_GRIPPER_MAX_DISTANCE_MM,
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
        # IK is seeded from current MuJoCo joint angles each frame — the arm is
        # already near the target (per-frame deltas are small), so this converges
        # fast.  Equivalent to InverseKinematicsEEToJoints(initial_guess_current_joints=True).
        self._kin = None
        self._orientation_weight = orientation_weight
        self._gripper_max_distance_mm = gripper_max_distance_mm
        if action_mode == "pose_delta":
            if urdf_path is None:
                raise ValueError("urdf_path is required for pose_delta action mode")
            from lerobot.model.kinematics import RobotKinematics

            self._kin = RobotKinematics(
                str(urdf_path),
                target_frame_name="gripper_frame_link",
                joint_names=list(BODY_JOINTS),
            )

        # --- Optional viewer ------------------------------------------------
        self._viewer = None
        if render:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

        logger.info(
            "MuJoCoSO101Servicer ready: action_mode=%s xml=%s urdf=%s render=%s",
            action_mode,
            xml_path,
            urdf_path,
            render,
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

    def _get_body_joint_degrees(self) -> np.ndarray:
        """Current body-joint angles in degrees (input to FK/IK)."""
        return np.array(
            [math.degrees(float(self._data.qpos[i])) for i in range(len(BODY_JOINTS))],
            dtype=float,
        )

    def _gripper_distance_to_0_100(self, distance_mm: float) -> float:
        """Maps gripper finger distance (mm) to SO-101 gripper position (0-100)."""
        return float(np.clip(distance_mm / self._gripper_max_distance_mm * 100.0, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Pose-delta → joint pipeline
    # ------------------------------------------------------------------

    def _pose_delta_to_joint_action(self, delta_action: dict[str, float]) -> dict[str, float]:
        """Converts a pose-delta action to a joint-space action via FK + IK.

        Pipeline: FK(current joints) → compose delta → IK → joint targets.
        Uses body-frame rotation composition (``R_current @ R_delta``), matching
        ``robot_kinematic_processor.EEReferenceAndDelta`` — confirmed correct by
        wayfinder #05 (coordinate frame alignment resolution).
        """
        from lerobot.utils.rotation import Rotation as R

        # 1. Decode pose delta
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

        # 2. FK: current body joints → current EE pose
        q_current = self._get_body_joint_degrees()
        t_current = self._kin.forward_kinematics(q_current)

        # 3. Compose delta onto current pose (body-frame)
        t_target = t_current.copy()
        t_target[:3, 3] = t_current[:3, 3] + delta_pos
        r_delta = R.from_quat(delta_quat).as_matrix()
        t_target[:3, :3] = t_current[:3, :3] @ r_delta

        # 4. IK: target EE pose → joint angles (warm-started from current)
        q_target = self._kin.inverse_kinematics(
            q_current,
            t_target,
            orientation_weight=self._orientation_weight,
        )

        # 5. Build joint action (body joints from IK + gripper from distance mapping)
        joint_action: dict[str, float] = {}
        for i, joint in enumerate(BODY_JOINTS):
            joint_action[f"{joint}.pos"] = float(q_target[i])
        joint_action["gripper.pos"] = self._gripper_distance_to_0_100(gripper_distance)
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
            self._connected = True
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
        """Persistent stream: step MuJoCo physics at ~50 Hz and stream joint angles."""
        while context.is_active():
            with self._lock:
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

            # Drive MuJoCo PD actuators
            self._data.ctrl[:] = self._joint_action_to_ctrl(joint_action)
            self._latest_action = action

        # Echo commanded action back (A-class semantics, same as MockFollowerServicer).
        return device_pb2.Action(features=list(encode_feature(self._act_ft_info, action)))

    def GetFeedback(self, request, context):
        with self._lock:
            return encode_feature(self._act_ft_info, dict(self._latest_action))
