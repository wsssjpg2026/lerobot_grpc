import logging
import math
import queue
import sys
import time
import threading
import traceback
from collections.abc import Sequence
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

from google.protobuf.empty_pb2 import Empty

from .follower_server import FollowerServicer
from .mujoco_follower_server import (
    BODY_JOINTS,
    DEFAULT_GRIPPER_MAX_DISTANCE_MM,
    HOME_JOINTS_DEG,
    JOINTS,
    norm_value_to_rad,
)
from .pose_delta_law import PoseDeltaLaw
from .utils import (
    H264FrameEncoder,
    H264_AVAILABLE,
    _now_produce_ts,
    _ts_from_wall,
    _wall_now,
    encode_feature,
    load_feature,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.motors import MotorCalibration
from lerobot.lerobot_types import RobotAction
from lerobot_robot_grpc.protos import device_pb2
from lerobot_robot_grpc.pose_delta_schema import build_pose_delta_feature_info
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)

type NameOrID = str | int
type Value = int | float


class CalibrationAbortedError(Exception):
    """Raised when an in-progress calibration is aborted (e.g. new Connect arrived)."""

_PROTO_BY_PYTHON: dict[type, device_pb2.DataType] = {
    int: device_pb2.DataType.INT32,
    float: device_pb2.DataType.FLOAT32,
    np.uint8: device_pb2.DataType.UINT8,
    np.uint16: device_pb2.DataType.UINT16,
    np.float32: device_pb2.DataType.FLOAT32,
    np.int32: device_pb2.DataType.INT32
}

# MuJoCo scene the real servicer loads as its FK/IK/max-reach oracle -- the
# SAME model the sim adapter uses (wayfinder #03: one kinematics engine, only
# the qpos source differs).  Only imported/loaded in pose_delta action mode.
_DEFAULT_KINEMATICS_XML = (
    Path(__file__).resolve().parents[2] / "assets" / "so101" / "scene.xml"
)

# Solver-bias rest posture for the REAL arm's pose_delta law (#07).
#
# This is NOT a pose the arm is ever commanded to -- Connect/SetReference
# write nothing and teleop starts from whatever unpowered droop pose the
# user parks the arm in.  It is where the DLS nullspace rest task and the
# limit-escape re-seed point, and it must be REACHABLE: the sim home
# (elbow +60 deg) sits past the real over-fold wall (+2 deg), so with it
# every solve pinned into the wall/singularity and the R1 bench froze with
# the body immovable while the gripper followed (residual 26.8 mm > 18 mm
# hold, manip 0.0133 at the droop).
#
# Chosen offline (lerobot-grpc-serve, /tmp/rest_scan.py, 2026-08-17) under the effective
# limits (model n follower.json calibration n elbow wall +2 deg), against
# the measured R1 droop seed (pan 6.3 / lift -2.4 / elbow +1.1 / wf +46.4 /
# wr -5.6): manip 0.0134, FK radius 463 mm, all six 15 mm axis intents
# tracked 14.8-15.2 mm at <= 0.3 mm residual with zero holds/escapes, and
# commanded elbows stay -2.0..+1.2 deg (off the +2 wall, no limit-saturation
# escape churn).  The old sim home under the same drive tracked nothing
# (held 29/30 frames, residual ~27 mm).
REAL_REST_POSTURE_DEG: tuple[float, ...] = (0.0, 30.0, -20.0, 0.0, 0.0)


def _protobuf_type_for(python_type: type) -> device_pb2.DataType:
    """Maps a Python type to the corresponding protobuf DataType."""
    if python_type in _PROTO_BY_PYTHON:
        return _PROTO_BY_PYTHON[python_type]
    else:
        raise ValueError(f"Unsupported Python type: {python_type}")

class SO101FollowerAdapted(SO101Follower):
    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        # The feetech SDK accounts packet timeouts with the wall clock (time.time());
        # a backward clock step (NTP/Windows time sync) mid-read makes isPacketTimeout()
        # never fire and rxPacket() spin forever, wedging the bus. Switch the port
        # handler to monotonic time so timeouts always elapse.
        self.bus.port_handler.getCurrentTime = lambda: time.monotonic() * 1000.0
        # rxPacket() consumes bytes while data arrives WITHOUT checking the packet timeout
        # on that path (protocol_packet_handler.rxPacket: the header-scan/wait_length
        # `continue` branches). A continuous garbage stream on the serial line (broken bus,
        # stuck transceiver) would therefore wedge the read forever, holding the servicer
        # bus lock. Choke every read through the packet deadline: once it elapsed, readPort
        # returns empty and any loop reaches the isPacketTimeout() branch and terminates.
        port = self.bus.port_handler
        _original_read_port = port.readPort

        def _bounded_read_port(length: int):
            if port.isPacketTimeout():
                return b""
            return _original_read_port(length)

        port.readPort = _bounded_read_port
        # Set by the client's CalibrateDone RPC to signal the end of manual range-of-motion recording.
        # Cleared in the servicer's Calibrate handler *before* the calibration thread starts, so a
        # CalibrateDone arriving at any point afterwards is never lost (unlike a plain bool flag).
        self._calibrate_done = threading.Event()
        self._calibrate_aborted = threading.Event()

        self.latest_action: RobotAction | None = None

    def record_ranges_of_motion_w_grpc(self, bus: FeetechMotorsBus,
        motors: NameOrID | Sequence[NameOrID] | None = None) -> tuple[dict[str, Value], dict[str, Value]]:

        motor_names = bus._get_motors_list(motors)

        start_positions = bus.sync_read("Present_Position", motor_names, normalize=False, num_retry=5)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        while not self._calibrate_done.is_set():
            if self._calibrate_aborted.is_set():
                raise CalibrationAbortedError()
            positions = bus.sync_read("Present_Position", motor_names, normalize=False, num_retry=5)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}
            q = getattr(self, "_calib_frame_queue", None)
            if q is not None:
                frame = device_pb2.CalibrationFrame()
                for motor in motor_names:
                    frame.readings.add(
                        name=motor,
                        position=int(positions[motor]),
                        range_min=int(mins[motor]),
                        range_max=int(maxes[motor]),
                    )
                q.put(frame)
            time.sleep(0.02)

        if self._calibrate_aborted.is_set():
            raise CalibrationAbortedError()

        same_min_max = [motor for motor in motor_names if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def calibrate(self, force: bool = False) -> None:
        # Only trust the cached file when the motors already match it (bus.is_calibrated).
        # Otherwise (file exists but motors disagree, e.g. after an interrupted calibration)
        # we MUST run the real procedure: skip this shortcut or the motors stay torque-locked
        # while the client shows the manual "move joints through full range" prompt.
        if self.calibration and not force and self.bus.is_calibrated:
            self.bus.write_calibration(self.calibration)
            return

        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        range_mins, range_maxes = self.record_ranges_of_motion_w_grpc(self.bus, unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        # Mirror stock lerobot (connect -> calibrate -> configure leaves torque on): the
        # manual phase above disabled it; without re-enabling, SendAction later cannot
        # move the arm within the same session.
        self.bus.enable_torque()

class SO101FollowerServicer(FollowerServicer):
    """gRPC servicer for the SO101 Follower robot."""

    def __init__(
        self,
        robot: SO101FollowerAdapted,
        camera_encoding: str = "h264",
        calibration_timeout_s: float = 300.0,
        bus_call_timeout_s: float = 5.0,
        action_mode: str = "joint",
        home_joints_deg: tuple[float, ...] = REAL_REST_POSTURE_DEG,
        stale_timeout_s: float = 1.0,
        max_dq_deg: float = 6.0,
        max_dq_frame_deg: float = 6.7,
        gripper_max_distance_mm: float = DEFAULT_GRIPPER_MAX_DISTANCE_MM,
        elbow_max_deg: float | None = 2.0,
    ):
        self.robot = robot
        if camera_encoding not in ("jpeg", "h264"):
            raise ValueError(f"Unsupported camera_encoding '{camera_encoding}'; expected 'jpeg' or 'h264'.")
        self.camera_encoding = camera_encoding
        if action_mode not in ("joint", "pose_delta"):
            raise ValueError(
                f"action_mode must be 'joint' or 'pose_delta', got {action_mode!r}"
            )
        self._action_mode = action_mode
        # Action schema is mode-dependent (joint space vs the shared pose-delta
        # schema); observation schema is mode-independent.
        self._act_ft_info: dict[str, device_pb2.OneFeatureInfo] = (
            build_pose_delta_feature_info()
            if action_mode == "pose_delta"
            else self._encode_feature_info(robot.action_features)
        )
        # --- Shared pose_delta law (the one law, sim + real) ----------------
        # Real-arm adapter over the same PoseDeltaLaw the sim drives: this
        # servicer only reads Present_Position -> rad as the FK/IK seed and
        # writes the solved joint_action via send_action.  The MuJoCo model is
        # loaded purely as a kinematics oracle (no sim state).  The law runs
        # the PikaAnyArm official safety stack (see pose_delta_law.py):
        # IK hard limits (model + measured calibration range + elbow wall),
        # 30° jump warm-start reset, FK consistency 0.3 m, per-frame step cap,
        # self-collision gate (auto-bypassed on SO-101's visual-only meshes).
        self._law: PoseDeltaLaw | None = None
        if action_mode == "pose_delta":
            import mujoco

            # mujoco ships no py.typed (same accepted category as the baseline).
            # The oracle is fixed to the shared SO-101 scene (#03 mandate).
            model = mujoco.MjModel.from_xml_path(  # pyright: ignore[reportAttributeAccessIssue]
                str(_DEFAULT_KINEMATICS_XML)
            )
            self._law = PoseDeltaLaw(
                model,
                site_name="gripperframe",
                body_dofs=list(range(len(BODY_JOINTS))),
                body_joint_names=BODY_JOINTS,
                home_joints_deg=home_joints_deg,
                max_dq_deg=max_dq_deg,
                max_dq_frame_deg=max_dq_frame_deg,
                gripper_max_distance_mm=gripper_max_distance_mm,
            )
        # Stale-hold (real-only): wall-clock of the last pose_delta SendAction;
        # a gap > stale_timeout_s makes the next solve hold the last joints.
        self._stale_timeout_s = float(stale_timeout_s)
        self._last_action_monotonic: float | None = None
        # Elbow over-fold hard wall (#05 bench): the physical arm binds at
        # ~+3-4 deg past the folded-rest calibration zero (positive = over-fold,
        # model qpos == normalised degrees), while the recorded calibration
        # range overstates that side.  None disables the cap.
        self._elbow_max_deg = None if elbow_max_deg is None else float(elbow_max_deg)
        # Held by the calibration thread for its whole duration; GetObservation/SendAction use a
        # non-blocking acquire to reject bus access while the robot is being manually moved.
        self._calibration_lock = threading.Lock()
        self._calibration_state_lock = threading.Lock()
        self._calibrating = False
        self._calibrate_error: str | None = None
        self._calib_frame_queue: queue.Queue | None = None
        self._force_recalibrate = False
        self._calibrate_thread: threading.Thread | None = None
        # Watchdog: a calibration whose client dies before sending CalibrateDone would otherwise
        # hold the bus lock forever (and block SendAction); abort after this many seconds.
        self.calibration_timeout_s = calibration_timeout_s
        # Bus-call watchdog: a stuck low-level bus call (e.g. a dead serial port, or a clock
        # regression in the feetech SDK timeouts) would otherwise hold the lock forever; after
        # bus_call_timeout_s the watchdog force-releases it and dumps the stuck thread's stack.
        self.bus_call_timeout_s = bus_call_timeout_s
        self._bus_held = False
        self._bus_owner: str | None = None
        self._bus_owner_start = 0.0
        self._bus_owner_thread: threading.Thread | None = None
        # Poisoned flag: once the watchdog detects a stuck bus call, the physical serial
        # port is in an unrecoverable state (the stuck owner is still blocked in ser.read()).
        # Force-releasing the lock would let SendAction write to the same handle concurrently
        # and corrupt the feetech protocol (unsafe robot motion). Instead, mark the bus
        # poisoned and refuse all further bus calls until a fresh Connect/Disconnect cycle.
        self._bus_poisoned = False
        threading.Thread(target=self._bus_watchdog, daemon=True, name="grpc-bus-watchdog").start()

    def _acquire_bus(self, what: str, timeout: float = 0.0) -> bool:
        """Bus lock acquire; records the owner so the stuck-call watchdog can dump its
        stack. Calibration is excluded: it legitimately holds the lock for the whole
        manual recording phase.

        Refuses immediately while the bus is poisoned (a prior bus call wedged and the
        serial port is unsafe to touch). `timeout > 0` blocks up to `timeout` seconds
        instead of failing immediately, for callers (SendAction) that can tolerate a
        short wait behind the observation stream — rejecting at teleop rate would starve
        the control loop.
        """
        if self._bus_poisoned:
            logger.error(f"Bus is poisoned (a prior call wedged); rejecting '{what}'.")
            return False
        if timeout > 0:
            if not self._calibration_lock.acquire(blocking=True, timeout=timeout):
                return False
        elif not self._calibration_lock.acquire(blocking=False):
            return False
        with self._calibration_state_lock:
            self._bus_held = True
            self._bus_owner = what
            self._bus_owner_start = time.monotonic()
            self._bus_owner_thread = threading.current_thread()
        return True

    def _release_bus(self) -> None:
        """Releases the bus lock.

        Only the recorded owner thread may release: a stale thread returning from a
        wedged call must not clear a newer owner's state nor release its lock. With the
        poisoned-flag design the watchdog never force-releases, so this guard is
        defense-in-depth against ownership confusion.
        """
        with self._calibration_state_lock:
            if not self._bus_held:
                return
            if threading.current_thread() is not self._bus_owner_thread:
                logger.warning(
                    "_release_bus called by a non-owner thread; ignoring to avoid "
                    "clobbering the current owner's lock state."
                )
                return
            self._bus_held = False
            self._bus_owner = None
            self._bus_owner_start = 0.0
            self._bus_owner_thread = None
        self._calibration_lock.release()

    def _reset_bus_lock_state(self) -> None:
        """Clears poisoned/ownership state and releases the lock so a fresh session can
        recover after a watchdog trip. Safe to call after the serial port has been closed
        (disconnect) or reopened (connect): no concurrent I/O is in flight on a fresh port.
        """
        with self._calibration_state_lock:
            was_held = self._bus_held
            self._bus_poisoned = False
            self._bus_held = False
            self._bus_owner = None
            self._bus_owner_start = 0.0
            self._bus_owner_thread = None
        if was_held:
            try:
                self._calibration_lock.release()
            except RuntimeError:
                pass

    def _bus_watchdog(self) -> None:
        """Marks the bus poisoned when a non-calibration bus call appears stuck, and
        dumps the stuck thread's stack for diagnosis (scservo_sdk reads can wedge forever
        if its wall-clock timeouts misbehave, e.g. after an NTP/Windows clock step).

        The lock is NOT force-released: the stuck owner is still blocked in the serial
        read, and releasing the lock would let another caller write to the same pyserial
        handle concurrently (protocol corruption, unsafe motion). Instead the bus is
        poisoned and all further bus calls are refused until a fresh Connect/Disconnect
        recovers. The monotonic-clock + bounded-read patches upstream prevent most wedges;
        this is the fail-safe that stops the robot rather than sending garbled commands.
        """
        while True:
            time.sleep(self.bus_call_timeout_s)
            with self._calibration_state_lock:
                held = self._bus_held
                owner = self._bus_owner
                start = self._bus_owner_start
                thread = self._bus_owner_thread
                poisoned = self._bus_poisoned
            if not held or owner == "calibration" or poisoned:
                continue
            if time.monotonic() - start < self.bus_call_timeout_s:
                continue
            logger.critical(
                f"Bus call '{owner}' (thread {thread.name!r}) stuck for "
                f">{self.bus_call_timeout_s:.0f}s; poisoning the bus (refusing further "
                f"bus calls until a fresh Connect/Disconnect)."
            )
            if thread is not None and thread.ident is not None:
                for tid, frame in sys._current_frames().items():
                    if tid == thread.ident:
                        logger.critical(
                            f"Stuck thread {thread.name} stack:\n{''.join(traceback.format_stack(frame))}"
                        )
                        break
            with self._calibration_state_lock:
                if not self._bus_held or self._bus_owner_start != start or self._bus_poisoned:
                    continue  # the call finished, a newer one started, or already poisoned
                self._bus_poisoned = True

    @staticmethod
    def _camera_encoding_for(key: str, shape: tuple[int, int, int], encoding: str) -> device_pb2.Encoding:
        """Picks the wire encoding for one camera feature.

        Depth maps stay RAW (uint16 mm precision; JPEG/H264 would truncate).
        H264 needs 3 channels + even dimensions + an available encoder;
        anything else falls back to JPEG.
        """
        if key.endswith("_depth") and shape[2] == 1:
            return device_pb2.Encoding.RAW
        if (
            encoding == "h264"
            and H264_AVAILABLE
            and shape[2] == 3
            and shape[0] % 2 == 0
            and shape[1] % 2 == 0
        ):
            return device_pb2.Encoding.H264
        return device_pb2.Encoding.JPEG

    def _encode_feature_info(self, feature_info: dict[str, Any]) -> dict[str, device_pb2.OneFeatureInfo]:
        # Returns a dict (not a generator): encode_feature/load_feature subscript it by key,
        # so a generator would raise KeyError. GetInfo fills the GetInfoResponse feature
        # fields from the result's ...values().
        result: dict[str, device_pb2.OneFeatureInfo] = {}
        for key, val in feature_info.items():
            if isinstance(val, tuple):
                shape = device_pb2.ImageShape(H=val[0], W=val[1], C=val[2])
                encoding = self._camera_encoding_for(key, val, self.camera_encoding)
                if encoding == device_pb2.Encoding.RAW:
                    # Depth maps (e.g. '<cam>_depth', uint16 mm) are sent RAW to preserve
                    # full precision; JPEG/H264 would silently truncate them to 8 bits.
                    val_type = device_pb2.DataType.UINT16
                else:
                    val_type = device_pb2.DataType.UINT8
            else:
                val_type = _protobuf_type_for(val)
                shape = device_pb2.ImageShape(H=1, W=1, C=1)
                encoding = device_pb2.Encoding.RAW
            result[key] = device_pb2.OneFeatureInfo(
                key=key,
                criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
                watchdog = device_pb2.WatchDogLevel.WATCH_DOG_LEVEL_A,
                type=val_type,
                shape=shape,
                encoding=encoding,
                img_quality=90,
            )
        return result

    # ------------------------------------------------------------------
    # Pose-delta backend adapter (the real-arm half of the shared law seam)
    # ------------------------------------------------------------------

    def _read_qpos_rad(self) -> np.ndarray:
        """Reads Feetech Present_Position and converts to the model qpos (rad).

        Backend-injected qpos source for the shared law: the bus returns
        lerobot-normalised values (body joints in degrees, gripper in 0-100,
        per the registered calibration); the MuJoCo oracle speaks radians with
        the sim's gripper mapping.  Caller must hold the bus lock.
        """
        pos = self.robot.bus.sync_read(
            "Present_Position", num_retry=self.robot.config.num_read_retries
        )
        qpos = np.zeros(len(JOINTS))
        for i, joint in enumerate(JOINTS):
            qpos[i] = norm_value_to_rad(joint, float(pos[joint]))
        return qpos

    def _apply_calibration_qpos_limits(self) -> None:
        """Tightens the law's IK joint limits to the measured calibration range.

        DEGREES-mode normalisation puts 0 deg at the mid-range of the recorded
        raw ticks, so the measured range maps to a symmetric +/- half-span in
        radians -- the same convention the MuJoCo oracle uses (model qpos rad
        == normalised degrees).  Joints without a valid recorded range keep the
        model limit.  Pose_delta only.
        """
        assert self._law is not None
        calibration = self.robot.calibration
        if not calibration:
            return
        lo = np.full(len(BODY_JOINTS), -np.inf)
        hi = np.full(len(BODY_JOINTS), np.inf)
        for i, joint in enumerate(BODY_JOINTS):
            mc = calibration.get(joint)
            if mc is None or mc.range_max <= mc.range_min:
                continue
            # sts3215 raw resolution 0..4095 = one full turn (lerobot table).
            half_span_deg = (mc.range_max - mc.range_min) / 2.0 * 360.0 / 4095.0
            lo[i] = -math.radians(half_span_deg)
            hi[i] = math.radians(half_span_deg)
        # The over-fold hard wall cuts the elbow ceiling no matter how wide the
        # recorded range is (the calibration only bounds the reachable half-span;
        # the wall is a one-sided mechanical stop #05 measured at +2~4 deg).
        if self._elbow_max_deg is not None:
            elbow = BODY_JOINTS.index("elbow_flex")
            hi[elbow] = min(hi[elbow], math.radians(self._elbow_max_deg))
        self._law.ik_solver.set_qpos_limits(lo, hi)
        logger.info(
            "IK joint limits tightened to measured calibration range (deg): [%s]",
            ", ".join(
                f"{j} [{math.degrees(lo[i]):.1f}, {math.degrees(hi[i]):.1f}]"
                for i, j in enumerate(BODY_JOINTS)
            ),
        )

    def _lock_t_zero_from_bus(self, what: str = "set_reference") -> None:
        """Re-latches the law reference (T_zero) at the FK of the CURRENT arm pose.

        Clutch re-engage contract (same as the sim adapter): reads the arm under
        the bus lock and re-anchors; nothing here moves the arm.  Pose_delta only.
        """
        assert self._law is not None
        if not self._acquire_bus(what, timeout=self.bus_call_timeout_s):
            raise DeviceNotConnectedError(
                f"Cannot {what}: bus lock busy (stuck call or calibration)."
            )
        try:
            qpos_rad = self._read_qpos_rad()
        finally:
            self._release_bus()
        self._law.lock_reference(qpos_rad)

    def GetInfo(self, request, context):
        obs = self._encode_feature_info(self.robot.observation_features)
        # Action/feedback schemas are mode-dependent and fixed at construction.
        return device_pb2.GetInfoResponse(
            observation_features=obs.values(),
            action_features=self._act_ft_info.values(),
            feedback_features=self._act_ft_info.values(),
        )

    def Connect(self, request, context):
        if not self.robot.is_connected:
            self.robot.connect(False)
        self._reset_bus_lock_state()
        self._calibrate_error = None
        self._abort_stuck_calibration()
        if self._law is not None:
            # Fresh session: clear latch/solve state and the stale-hold clock.
            self._last_action_monotonic = None
            self._law.reset()
            if self.robot.is_calibrated:
                # IK limits from the measured range BEFORE any solve; then
                # latch T_zero at the arm's current pose.  Uncalibrated arms
                # cannot normalise Present_Position; the first solve then
                # auto-latches once calibration exists.  Reset to the model
                # range first so a RE-Connect after a looser recalibration
                # widens back (set_qpos_limits alone only narrows).
                self._law.ik_solver.reset_qpos_limits()
                self._apply_calibration_qpos_limits()
                self._lock_t_zero_from_bus("connect")
        if self.robot.is_calibrated:
            return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)
        else:
            return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.NEED_TO_CALIBRATE)

    def _abort_stuck_calibration(self) -> None:
        """Abort any in-progress calibration thread (e.g. previous client died with Ctrl+C)."""
        with self._calibration_state_lock:
            if not self._calibrating:
                return
            self.robot._calibrate_aborted.set()
            self.robot._calibrate_done.set()
            thread = self._calibrate_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self._calibration_state_lock:
            self._calibrating = False

    def _calibrate_in_thread(self) -> None:
        if not self._acquire_bus("calibration"):
            # Never queue behind another lock holder (e.g. a previous calibration whose client
            # died): a waiting thread with _calibrating=True would block SendAction anyway.
            logger.warning(
                "Rejected calibration: the bus lock is held (another calibration in progress "
                "or a stuck session)."
            )
            with self._calibration_state_lock:
                self._calibrating = False
            return
        try:
            self.robot.calibrate(force=self._force_recalibrate)
        except CalibrationAbortedError:
            logger.info("Calibration aborted.")
        except Exception as e:
            self._calibrate_error = traceback.format_exc()
            logger.exception(f"Calibration failed: {e}")
        finally:
            self._release_bus()
            with self._calibration_state_lock:
                self._calibrating = False
            if self._calib_frame_queue is not None:
                self._calib_frame_queue.put(None)

    def _watchdog_calibration(self, timeout_s: float) -> None:
        """Aborts an orphaned calibration (client never sent CalibrateDone) so the bus lock
        cannot stay held forever."""
        if not self.robot._calibrate_done.wait(timeout_s):
            logger.error(
                f"Calibration timed out after {timeout_s:.0f}s without CalibrateDone; aborting it."
            )
            self.robot._calibrate_aborted.set()
            self.robot._calibrate_done.set()

    def Calibrate(self, request, context):
        if not self.robot.is_connected:
            self.robot.connect(False)
        # Cheap early-out BEFORE touching the bus: a running calibration holds
        # the bus lock for its whole recording, so any is_calibrated read here
        # would race it (and the observation stream) on the serial port.
        with self._calibration_state_lock:
            if self._calibrating:
                return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATING)
        # is_calibrated is a raw EEPROM read: it MUST run under the bus lock.
        # Reading it unlocked races the observation stream's sync_read on the
        # same serial port -> SDK "Port is in use!" (real-arm #05 smoke).
        if not self._acquire_bus("calibrate", timeout=self.bus_call_timeout_s):
            raise DeviceNotConnectedError(
                "Cannot calibrate: bus lock busy (stuck call or just released)."
            )
        try:
            # connect(False) 跳过了 bus.write_calibration()；若标定文件已存在，先同步到 bus，
            # 使 is_calibrated 反映真实状态（避免已有标定时仍误启动标定线程→白屏）。
            if self.robot.calibration and not self.robot.is_calibrated and not request.force:
                self.robot.bus.write_calibration(self.robot.calibration)
            is_calibrated = self.robot.is_calibrated
        finally:
            self._release_bus()
        logger.info(
            f"Calibrate RPC received (peer={context.peer()!r}, force={request.force}, "
            f"is_calibrated={is_calibrated})."
        )
        if is_calibrated and not request.force:
            return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

        with self._calibration_state_lock:
            if self._calibrating:
                return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATING)
            self._calibrating = True
            self._calibrate_error = None
            self._force_recalibrate = request.force
            self.robot._calibrate_aborted.clear()
            # Clear *before* starting the thread so a CalibrateDone arriving at any point during
            # calibration is never lost (it would otherwise hang the recording loop forever).
            self.robot._calibrate_done.clear()
            self._calib_frame_queue = queue.Queue()
            self.robot._calib_frame_queue = self._calib_frame_queue

        self._calibrate_thread = threading.Thread(target=self._calibrate_in_thread, daemon=True)
        self._calibrate_thread.start()
        threading.Thread(
            target=self._watchdog_calibration,
            args=(self.calibration_timeout_s,),
            daemon=True,
            name="grpc-calib-watchdog",
        ).start()
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.NEED_TO_CALIBRATE)
    
    def CalibrateDone(self, request, context):
        self.robot._calibrate_done.set()
        return Empty()

    def StreamCalibration(self, request, context):
        q = self._calib_frame_queue
        if q is None:
            return
        while context.is_active():
            try:
                frame = q.get(timeout=1.0)
            except queue.Empty:
                continue
            if frame is None:
                break
            yield frame

    def Disconnect(self, request, context):
        self._abort_stuck_calibration()
        if self.robot.is_connected:
            self.robot.disconnect()
        # Reset the bus-lock state so a fresh Connect can recover after a watchdog trip.
        # disconnect() closed the serial port (unblocking any wedged read), so releasing
        # the lock here is safe — no concurrent I/O is possible.
        self._reset_bus_lock_state()
        return Empty()

    def SetReference(self, request, context):
        """Re-locks T_zero at the current measured arm pose (clutch re-engage).

        Same contract as the sim adapter: the client calls this on the engage
        edge so the next dT=0 maps onto the arm's current pose instead of
        pulling back to the Connect home.  Reads Present_Position under the bus
        lock; nothing is written -- the arm must not move at re-engage.  In
        joint mode this is a no-op (no Cartesian reference exists).
        """
        if self._law is not None:
            self._lock_t_zero_from_bus("set_reference")
            logger.info(
                "SetReference: T_arm_ref re-locked at current FK pos=[%.4f %.4f %.4f]",
                *self._law.arm_reference[:3, 3],
            )
        else:
            logger.info("SetReference: no-op (action_mode=%s)", self._action_mode)
        return Empty()

    def GetObservation(self, request, context):
        """Persistent observation stream.

        One continuous server-streaming call per client, kept open for the whole
        session: non-image / JPEG / RAW features are re-sent every tick (clients
        keep last values for keys they don't see), while H264 cameras are encoded
        with a per-stream encoder — P-frames reference previous frames, so camera
        bandwidth collapses vs. per-frame JPEG. A fresh encoder per stream makes
        the first packet a keyframe, and keyint limits recovery time for clients
        that (re)connect mid-stream.
        """
        obs_feature_info = self._encode_feature_info(self.robot.observation_features)
        h264_keys = [k for k, info in obs_feature_info.items() if info.encoding == device_pb2.Encoding.H264]
        encoders: dict[str, H264FrameEncoder] = {}
        failed_h264: set[str] = set()

        fps_list = [c.fps for c in self.robot.config.cameras.values() if getattr(c, "fps", None)]
        min_fps = min(fps_list, default=30)
        period = 1.0 / min(max(min_fps, 1), 60)

        try:
            while context.is_active():
                if not self._acquire_bus("get_observation"):
                    # Calibration owns the bus: idle the stream, clients keep last values.
                    time.sleep(period)
                    continue
                # Sample-time stamping (P1.2-B): get_observation() reads motors first, then
                # peeks camera buffers. Capture wall-clock before/after to stamp each feature
                # at its approximate sample time (motor ~t_motor, camera ~t_cam) instead of a
                # single encode time — so downstream data collection/inference can tell motor
                # and camera samples apart. Approximate (no lock split); precise per-phase
                # stamping is deferred with the bus-lock split (P1.2-A, not in this PR).
                t_motor = _wall_now()
                try:
                    raw_obs = self.robot.get_observation()
                finally:
                    self._release_bus()
                t_cam = _wall_now()

                ts_for_key: dict[str, device_pb2.Timestamp] = {}
                for k, info in obs_feature_info.items():
                    if k not in raw_obs:
                        continue
                    is_scalar = (info.shape.H, info.shape.W, info.shape.C) == (1, 1, 1)
                    ts_for_key[k] = _ts_from_wall(t_motor if is_scalar else t_cam)

                # Non-H264 features: joints + JPEG/RAW cameras, same encode_feature path as before.
                non_h264 = {
                    k: v for k, v in raw_obs.items()
                    if obs_feature_info[k].encoding != device_pb2.Encoding.H264
                }
                yield from encode_feature(obs_feature_info, non_h264, ts_for_key=ts_for_key)

                # H264 cameras: one access unit per frame per camera.
                for key in h264_keys:
                    if key not in raw_obs or key in failed_h264:
                        continue
                    info = obs_feature_info[key]
                    camera_fps = int(getattr(self.robot.config.cameras.get(key), "fps", None) or min_fps)
                    encoder = encoders.get(key)
                    if encoder is None:
                        encoder = H264FrameEncoder(
                            key,
                            info.shape.H,
                            info.shape.W,
                            fps=camera_fps,
                            bitrate_kbps=max(500, 2000 * (info.shape.H * info.shape.W) // (640 * 480)),
                            keyint_frames=max(30, camera_fps * 2),
                        )
                        encoders[key] = encoder
                    try:
                        units = encoder.encode(raw_obs[key])
                    except Exception:
                        logger.exception(f"H264 encoding failed for camera '{key}'; dropping it from the stream.")
                        failed_h264.add(key)
                        continue
                    cam_ts = ts_for_key.get(key, _now_produce_ts())
                    for unit in units:
                        yield device_pb2.OneFeature(key=key, data=unit, produce_ts=cam_ts)

                time.sleep(period)
        finally:
            # Release native PyAV encoder contexts when the stream ends/cancels (P1.1).
            for enc in encoders.values():
                enc.close()

    def SendAction(self, request, context):
        act_dict: RobotAction = {}
        for act in request.features:
            load_feature(act, self._act_ft_info, act_dict, aux_behavior="ignore")
        # Hard-gate only on calibration, which owns the bus for its whole manual phase;
        # any other holder is transient (the observation stream's per-tick read), so wait
        # for it instead of rejecting at teleop rate.
        with self._calibration_state_lock:
            calibrating = self._calibrating
        if calibrating:
            alive = self._calibrate_thread is not None and self._calibrate_thread.is_alive()
            detail = f"calibration in progress (thread alive={alive})"
            logger.error(f"SendAction rejected: {detail}.")
            raise DeviceNotConnectedError("Cannot send action: robot is calibrating.")
        if not self._acquire_bus("send_action", timeout=self.bus_call_timeout_s):
            logger.error(
                "SendAction rejected: bus lock busy for >%.0fs (stuck call or just released).",
                self.bus_call_timeout_s,
            )
            raise DeviceNotConnectedError("Cannot send action: robot is calibrating.")
        try:
            if self._action_mode == "pose_delta":
                joint_action = self._solve_pose_delta(act_dict)
            else:
                joint_action = act_dict
            self.robot.send_action(joint_action)
            self.robot.latest_action = act_dict  # Reflect the last action sent, for GetFeedback.
        finally:
            self._release_bus()
        # Echo the commanded action back (A-class semantics, same as the sim adapter).
        return device_pb2.Action(features=list(encode_feature(self._act_ft_info, act_dict)))

    def _solve_pose_delta(self, delta_action: dict[str, float]) -> dict[str, float]:
        """Runs the shared law on one raw leader delta, seeding FK/IK from the
        arm's measured joints.  Caller must hold the bus lock: the
        Present_Position read and the send_action write ride the SAME hold, so
        the 30 Hz observation stream interleaves between actions, never
        mid-action (bus-contention answer for #04)."""
        assert self._law is not None
        qpos_rad = self._read_qpos_rad()
        now = time.monotonic()
        stale = (
            self._last_action_monotonic is not None
            and now - self._last_action_monotonic > self._stale_timeout_s
        )
        self._last_action_monotonic = now
        sol = self._law.solve(delta_action, qpos_rad, stale=stale)
        if sol.held:
            logger.info(
                "pose_delta hold: stale=%s rejected=%s collided=%s pos_err=%.1fmm "
                "-- keeping last joints.",
                sol.stale, sol.rejected, sol.collided, sol.pos_err_m * 1000.0,
            )
        return sol.joint_action

    def GetFeedback(self, request, context):
        raw_fb = self.robot.latest_action if self.robot.latest_action is not None else {}
        return encode_feature(self._act_ft_info, raw_fb)

    def GetStatus(self, request, context):
        if not self.robot.is_connected:
            return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.IDLE)
        if self._calibrating:
            return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.IDLE)
        if self._calibrate_error is not None:
            return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.FATAL)
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)
