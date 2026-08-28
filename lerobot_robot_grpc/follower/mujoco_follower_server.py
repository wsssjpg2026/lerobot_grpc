"""MuJoCo-backed SO-101 follower servicer with joint and pose_delta action modes.

A standalone FollowerServicer (following MockFollowerServicer) that wraps a
MuJoCo simulation of the SO-101 arm.  In pose_delta mode it receives
end-effector pose deltas (8 FLOAT32 features from the shared pose_delta_schema)
and delegates the compose + DLS + official-safety-check pipeline (PikaAnyArm
alignment, see pose_delta_law.py) to
the shared PoseDeltaLaw (the "one law" also driven by the real Feetech servicer),
then writes the resulting joint targets into the MuJoCo physics engine.  In
joint mode it accepts joint-space actions directly -- the same contract as the
real SO101FollowerServicer minus the hardware.

This servicer is the MuJoCo backend adapter over the shared law: it reads the
current joint vector from data.qpos and writes solved joints back to data.ctrl
(via a per-loop EMA in GetObservation).  Designed as the prototype validation
harness for delta-pose teleoperation (wayfinder ticket #08 / real map #03): no
real robot, no serial port, no calibration -- the servicer is the robot.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np
from google.protobuf.empty_pb2 import Empty

from .follower_server import FollowerServicer
from .pose_delta_law import PoseDeltaLaw
from .utils import encode_feature, load_feature
from lerobot_robot_grpc.pose_delta_schema import build_pose_delta_feature_info
from lerobot_robot_grpc.protos import device_pb2

logger = logging.getLogger(__name__)

# SO-101 joint names -- MuJoCo actuator names match lerobot motor names exactly.
JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# Body joints (excludes gripper) -- the DOF the IK solver controls.
BODY_JOINTS: tuple[str, ...] = JOINTS[:-1]

# MuJoCo gripper actuator range in radians (from so101_new_calib.xml).
# lerobot uses 0-100 (0 = closed, 100 = open); MuJoCo uses radians.
GRIPPER_RAD_MIN: float = -0.17453
GRIPPER_RAD_MAX: float = 1.74533

# Default full-open distance for the Pika Sense gripper sensor (millimetres).
DEFAULT_GRIPPER_MAX_DISTANCE_MM: float = 60.0


def norm_value_to_rad(joint: str, val: float) -> float:
    """One joint's lerobot-normalised value -> model radians (SO-101 unit
    convention: body joints in degrees, gripper 0-100 over GRIPPER_RAD range).
    Shared by both backend adapters -- the sim writes ctrl with it, the real
    servicer builds its FK/IK qpos seed with it."""
    if joint == "gripper":
        return (val / 100.0) * (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN) + GRIPPER_RAD_MIN
    return math.radians(val)


def rad_to_norm_value(joint: str, rad: float) -> float:
    """Inverse of norm_value_to_rad (model radians -> lerobot-normalised)."""
    if joint == "gripper":
        return (rad - GRIPPER_RAD_MIN) / (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN) * 100.0
    return math.degrees(rad)

# Physics step period (seconds) -- 50 Hz matches the real SO-101 observation rate.
_PHYSICS_PERIOD_S: float = 1.0 / 50.0

# Home joint angles (degrees) for the 5 body joints.  Bent elbow avoids the
# full-extension singularity; the arm starts inside the workspace with room to
# move in every direction.  Gripper stays at its MuJoCo default (near-closed).
HOME_JOINTS_DEG: tuple[float, ...] = (0.0, -20.0, 60.0, -40.0, 0.0)


def _scalar_feature_info(key: str) -> device_pb2.OneFeatureInfo:
    """Builds a CRITICAL FLOAT32 scalar feature info for key.

    Mirrors mock_follower.scalar_feature_info but adds WATCH_DOG_LEVEL_A to match
    the real servicer convention (SO101FollowerServicer).
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

    In pose_delta mode each SendAction receives 8 FLOAT32 features and drives the
    shared PoseDeltaLaw (the same law the real Feetech servicer will drive).
    This servicer is only the MuJoCo backend: it hands the law the current
    data.qpos as the FK/IK seed and writes the law's joint_action back to
    data.ctrl (smoothed by GetObservation's per-loop EMA).

    Parameters
    ----------
    xml_path
        Path to the MuJoCo scene XML (e.g. scene.xml which includes
        so101_new_calib.xml).
    action_mode
        "pose_delta" (default) or "joint".
    render
        If True, launch a passive MuJoCo viewer window.
    max_dq_deg / max_dq_frame_deg
        DLS per-iteration clip and the official per-frame published-step cap
        (forwarded to the law; identical defaults to the real servicer).
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
        max_dq_deg: float = 6.0,
        max_dq_frame_deg: float = 6.7,
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
        # Desired ctrl targets (radians, len(JOINTS)).  Set by SendAction;
        # GetObservation blends data.ctrl toward this at 50 Hz so motion is
        # continuous between actions (no jolt-then-freeze at low action rates).
        # None until first Connect.
        self._target_ctrl: np.ndarray | None = None
        self._home_joints_deg = home_joints_deg
        self._ctrl_smoothing_alpha = ctrl_smoothing_alpha

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

        # --- Shared pose_delta law (the one law, sim + real) ----------------
        # The law runs the PikaAnyArm official safety stack (see
        # pose_delta_law.py) — identical parameters to the real Feetech
        # servicer.  This servicer is only the MuJoCo backend: it feeds the
        # law the current qpos and writes the solved joints back.
        self._law: PoseDeltaLaw | None = None
        if action_mode == "pose_delta":
            self._law = PoseDeltaLaw(
                self._model,
                site_name="gripperframe",
                body_dofs=list(range(len(BODY_JOINTS))),
                body_joint_names=BODY_JOINTS,
                home_joints_deg=home_joints_deg,
                rot_weight=rot_weight,
                max_dq_deg=max_dq_deg,
                max_dq_frame_deg=max_dq_frame_deg,
                gripper_max_distance_mm=gripper_max_distance_mm,
            )

        # --- Optional viewer ------------------------------------------------
        self._viewer = None
        if render:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

        logger.info(
            "MuJoCoSO101Servicer ready: action_mode=%s xml=%s render=%s law=%s "
            "ctrl_smoothing_alpha=%s",
            action_mode, xml_path, render,
            "pose_delta" if self._law is not None else "none",
            ctrl_smoothing_alpha,
        )
        if action_mode == "pose_delta":
            logger.warning(
                "pose_delta dT is the current offset from SetReference, not a "
                "per-frame increment. Pair with a latch-once Pika Sense leader; "
                "an old incremental leader will accumulate and the arm will fly."
            )

    # ------------------------------------------------------------------
    # Unit conversion helpers (MuJoCo backend)
    # ------------------------------------------------------------------

    def _joint_action_to_ctrl(self, joint_action: dict[str, float]) -> np.ndarray:
        """Converts a lerobot joint-action dict to a MuJoCo ctrl array (radians)."""
        ctrl = np.zeros(len(JOINTS))
        for i, joint in enumerate(JOINTS):
            ctrl[i] = norm_value_to_rad(joint, joint_action.get(f"{joint}.pos", 0.0))
        return ctrl

    def _qpos_to_observation(self) -> dict[str, float]:
        """Reads MuJoCo qpos and converts to lerobot-normalised values."""
        obs: dict[str, float] = {}
        for i, joint in enumerate(JOINTS):
            obs[f"{joint}.pos"] = rad_to_norm_value(joint, float(self._data.qpos[i]))
        return obs

    # ------------------------------------------------------------------
    # Law delegation (the shared pose_delta pipeline lives in self._law)
    # ------------------------------------------------------------------

    def _lock_t_zero_from_fk(self) -> None:
        """Re-latch the law reference (T_arm_ref) at the current MuJoCo FK.

        Thin backend adapter: the law is pure in the qpos argument, so this
        servicer supplies its own data.qpos as the FK seed.  Pose_delta only --
        callers (Connect/SetReference) gate on self._law is not None.
        """
        assert self._law is not None
        self._law.lock_reference(self._data.qpos.copy())

    def _pose_delta_to_joint_action(self, delta_action: dict[str, float]) -> dict[str, float]:
        """Run the shared law on one latch-once delta and return a joint action.

        Backend adapter: seeds the law with the current data.qpos and returns the
        solved joint_action for the servicer to write to data.ctrl.  Pose_delta only.
        """
        assert self._law is not None
        return self._law.solve(delta_action, self._data.qpos.copy()).joint_action

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
            self._target_ctrl = self._data.ctrl.copy()  # home ctrl -> no EMA drift
            if self._law is not None:
                self._law.reset()
                self._lock_t_zero_from_fk()
        # Sim is pre-calibrated -- no manual range-of-motion recording needed.
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        with self._lock:
            self._connected = False
            # Keep viewer alive -- same pattern as the leader's hardware
            # persistence.  The viewer freezes (no sync calls) until a new
            # client connects and GetObservation resumes.
        return Empty()

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)

    def SetReference(self, request, context):
        """Re-lock T_arm_ref at the current gripperframe FK.

        Clutch re-engage contract (wayfinder #10): the client calls this on the
        engage transaction after the leader's SetReference and before action
        transport resumes, so the next dT=0 action maps onto the arm's current
        pose instead of pulling it back to the Connect home.  Joint targets /
        IK seed / ctrl are deliberately untouched -- the arm must not move at
        re-engage.

        In joint mode this is a no-op (no Cartesian reference exists).
        """
        with self._lock:
            if self._law is not None:
                self._lock_t_zero_from_fk()
                logger.info(
                    "SetReference: T_arm_ref re-locked at current FK pos=[%.4f %.4f %.4f]",
                    *self._law.arm_reference[:3, 3],
                )
            else:
                logger.info("SetReference: no-op (action_mode=%s)", self._action_mode)
        return Empty()

    def GetObservation(self, request, context):
        """Persistent stream: step MuJoCo physics at ~50 Hz and stream joint angles.

        Two responsibilities beyond physics stepping:

        1. Real-time substeps -- advances n_substeps physics steps per loop so
           the sim matches wall-clock time (not 10% real-time).
        2. Per-loop ctrl EMA -- blends data.ctrl toward _target_ctrl by
           ctrl_smoothing_alpha each iteration.  This produces continuous motion
           between SendAction calls instead of jolt-then-freeze -- critical at
           low action rates (e.g. the demo's 4 Hz).  The stiff PD (kp~1000) sees
           a gradual ramp, not a step.
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

            # Store desired ctrl -- GetObservation's per-loop EMA ramps toward it.
            # Writing data.ctrl directly would create a step function that the
            # stiff PD (kp~1000) overshoots on.
            self._target_ctrl = self._joint_action_to_ctrl(joint_action)
            self._latest_action = action

        # Echo commanded action back (A-class semantics, same as MockFollowerServicer).
        return device_pb2.ActionResult(
            features=list(encode_feature(self._act_ft_info, action))
        )

    def GetFeedback(self, request, context):
        with self._lock:
            return encode_feature(self._act_ft_info, dict(self._latest_action))
