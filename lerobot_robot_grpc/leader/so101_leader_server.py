import logging
import queue
import sys
import time
import threading
import traceback
from collections.abc import Sequence
from pprint import pformat
from typing import Any

from lerobot.utils import check_if_not_connected
import numpy as np

from google.protobuf.empty_pb2 import Empty

from .leader_server import LeaderServicer
from .utils import encode_feature, load_feature
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.motors import MotorCalibration
from lerobot.lerobot_types import RobotAction
from lerobot_robot_grpc.protos import device_pb2
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

def _protobuf_type_for(python_type: type) -> device_pb2.DataType:
    """Maps a Python type to the corresponding protobuf DataType."""
    if python_type in _PROTO_BY_PYTHON:
        return _PROTO_BY_PYTHON[python_type]
    else:
        raise ValueError(f"Unsupported Python type: {python_type}")

class SO101LeaderAdapted(SO101Leader):
    def __init__(self, config: SO101LeaderConfig):
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
        self.reference_action: RobotAction | None = None

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
        # Only trust the cached file when the motors already match it (bus.is_calibrated);
        # otherwise run the real procedure or the motors stay torque-locked while the
        # client shows the manual "move joints through full range" prompt.
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

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        action = {f"{motor}.pos": val for motor, val in action.items()}
        if self.reference_action is not None:
            action = {key: action[key] - self.reference_action.get(key, 0) for key in action}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

class SO101LeaderServicer(LeaderServicer):
    """gRPC servicer for the SO101 Leader robot."""

    def __init__(self, robot: SO101LeaderAdapted, calibration_timeout_s: float = 300.0,
                 bus_call_timeout_s: float = 5.0):
        self.robot = robot
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
        # hold the bus lock forever (and block GetAction/SendFeedback); abort after this many seconds.
        self.calibration_timeout_s = calibration_timeout_s
        # Bus-call watchdog: a stuck low-level bus call (e.g. a dead serial port, or a clock
        # regression in the feetech SDK timeouts) would otherwise hold the lock forever; after
        # bus_call_timeout_s the watchdog marks the bus poisoned and dumps the stuck thread's stack.
        self.bus_call_timeout_s = bus_call_timeout_s
        self._bus_held = False
        self._bus_owner: str | None = None
        self._bus_owner_start = 0.0
        self._bus_owner_thread: threading.Thread | None = None
        # Poisoned flag: once the watchdog detects a stuck bus call, the physical serial
        # port is in an unrecoverable state (the stuck owner is still blocked in ser.read()).
        # Force-releasing the lock would let GetAction/SendFeedback write to the same handle
        # concurrently and corrupt the feetech protocol (unsafe robot motion). Instead, mark
        # the bus poisoned and refuse all further bus calls until a fresh Connect/Disconnect.
        self._bus_poisoned = False
        threading.Thread(target=self._bus_watchdog, daemon=True, name="grpc-bus-watchdog").start()

    def _acquire_bus(self, what: str) -> bool:
        """Non-blocking bus lock acquire; records the owner so the stuck-call watchdog can
        dump its stack. Calibration is excluded: it legitimately holds the lock for the whole
        manual recording phase.

        Refuses immediately while the bus is poisoned (a prior bus call wedged and the
        serial port is unsafe to touch).
        """
        if self._bus_poisoned:
            logger.error(f"Bus is poisoned (a prior call wedged); rejecting '{what}'.")
            return False
        if not self._calibration_lock.acquire(blocking=False):
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
        wedged call must not clear a newer owner's state nor release its lock.
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
    def _encode_feature_info(feature_info: dict[str, Any]) -> dict[str, device_pb2.OneFeatureInfo]:
        # Returns a dict (not a generator): encode_feature/load_feature subscript it by key,
        # so a generator would raise KeyError. GetInfo fills the GetInfoResponse feature
        # fields from the result's ...values().
        result: dict[str, device_pb2.OneFeatureInfo] = {}
        for key, val in feature_info.items():
            if isinstance(val, tuple):
                shape = device_pb2.ImageShape(H=val[0], W=val[1], C=val[2])
                # Depth maps (e.g. '<cam>_depth', uint16 mm) are sent RAW to preserve full precision;
                # JPEG would silently truncate them to 8 bits.
                if key.endswith("_depth") and val[2] == 1:
                    val_type = device_pb2.DataType.UINT16
                    encoding = device_pb2.Encoding.RAW
                else:
                    val_type = device_pb2.DataType.UINT8
                    encoding = device_pb2.Encoding.JPEG
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

    def GetInfo(self, request, context):
        # leader 无 observation features;action 与 feedback 同源(action_features)。
        act = self._encode_feature_info(self.robot.action_features)
        return device_pb2.GetInfoResponse(
            observation_features=[],
            action_features=act.values(),
            feedback_features=act.values(),
        )

    def Connect(self, request, context):
        if not self.robot.is_connected:
            self.robot.connect(False)
        self._calibrate_error = None
        self._abort_stuck_calibration()
        self._reset_bus_lock_state()
        # Reset reference on new connection — previous session's SetReference
        # would otherwise make get_action() return deltas indefinitely.
        self.robot.reference_action = None
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
            # died): a waiting thread with _calibrating=True would block GetAction anyway.
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
        # connect(False) 跳过了 bus.write_calibration()；若标定文件已存在，先同步到 bus，
        # 使 is_calibrated 反映真实状态（避免已有标定时仍误启动标定线程→白屏）。
        if self.robot.calibration and not self.robot.is_calibrated and not request.force:
            self.robot.bus.write_calibration(self.robot.calibration)
        logger.info(
            f"Calibrate RPC received (peer={context.peer()!r}, force={request.force}, "
            f"is_calibrated={self.robot.is_calibrated})."
        )
        if self.robot.is_calibrated and not request.force:
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
        self._reset_bus_lock_state()
        return Empty()

    def GetObservation(self, request, context):
        return iter(())

    def GetAction(self, request, context):
        if not self._acquire_bus("get_action"):
            if self._calibrating:
                alive = self._calibrate_thread is not None and self._calibrate_thread.is_alive()
                detail = f"calibration in progress (thread alive={alive})"
            else:
                detail = "bus lock busy but no calibration flagged (stuck call or just released)"
            logger.error(f"GetAction rejected: {detail}.")
            raise DeviceNotConnectedError("Cannot get action: robot is calibrating.")
        try:
            raw_act = self.robot.get_action()
            self.robot.latest_action = raw_act  # Store the last action for GetFeedback.
        finally:
            self._release_bus()
        act_feature_info = self._encode_feature_info(self.robot.action_features)
        return encode_feature(act_feature_info, raw_act)

    def SendFeedback(self, request_iterator, context):
        fb_dict: RobotAction = {}
        fb_feature_info = self._encode_feature_info(self.robot.action_features)
        for fb in request_iterator:
            load_feature(fb, fb_feature_info, fb_dict, aux_behavior="ignore")
        if not self._acquire_bus("send_feedback"):
            raise DeviceNotConnectedError("Cannot send feedback: robot is calibrating.")
        try:
            self.robot.send_feedback(fb_dict)
        finally:
            self._release_bus()
        return Empty()  # Return an empty response

    def GetStatus(self, request, context):
        if not self.robot.is_connected:
            return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.IDLE)
        if self._calibrating:
            return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.IDLE)
        if self._calibrate_error is not None:
            return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.FATAL)
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)

    def SetReference(self, request, context):
        if not self._acquire_bus("set_reference"):
            raise DeviceNotConnectedError("Cannot set reference: robot is calibrating.")
        try:
            self.robot.reference_action = self.robot.get_action()
        finally:
            self._release_bus()
        return Empty()