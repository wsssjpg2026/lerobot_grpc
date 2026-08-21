from __future__ import annotations

import logging
import random
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _grpc_available, require_package
from lerobot.utils.utils import enter_pressed, move_cursor_up

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_grpc import GRPCLeaderConfig
from .utils import encode_feature, feature_meta_for, load_feature, python_scalar_type_for

if TYPE_CHECKING or _grpc_available:
    import grpc
    from google.protobuf.empty_pb2 import Empty
    from lerobot_robot_grpc.protos import device_pb2
    from lerobot_robot_grpc.protos import device_pb2_grpc
else:
    grpc = None
    Empty = None
    device_pb2 = None
    device_pb2_grpc = None

logger = logging.getLogger(__name__)


class ReferenceNotReadyError(DeviceNotConnectedError):
    """The leader is connected, but cannot safely capture a reference yet."""


# The server rejects GetAction while the bus is busy (real calibration, or a just-released
# stuck bus call). Retry briefly instead of crashing the teleop loop on a transient busy.
_BUSY_RETRIES = 5


def _busy_retry_delay(attempt: int) -> float:
    """Exponential backoff + jitter for transient busy retries: ~0.2s, 0.4, 0.8, capped at
    1.0s, plus up to 100ms jitter. The previous fixed 1.0s blocked the teleop loop for up to
    5s on a transient busy; this bounds it to ~2.5s and spreads retries to avoid thundering."""
    base = min(1.0, 0.2 * (2 ** attempt))
    return base + random.uniform(0.0, 0.1)


class GRPCLeader(Teleoperator):
    config_class = GRPCLeaderConfig
    name = "grpc_leader"

    def __init__(self, config: GRPCLeaderConfig):
        require_package("grpcio", extra="grpcio-dep", import_name="grpc")
        super().__init__(config)
        self.config = config
        self.address = config.address

        self.warmup_timeout_s = config.warmup_timeout_s
        self.connect_timeout_s = config.connect_timeout_s
        self.data_timeout_s = config.data_timeout_s
        self.reference_timeout_s = config.reference_timeout_s

        self._obs_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}
        self._act_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}
        self._fb_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}

        self._latest_obs_ft: RobotObservation = {}
        self._latest_act_ft: RobotAction = {}
        self._latest_fb_ft: dict[str, Any] = {}
        self._last_action_quality = device_pb2.FrameQuality.FRAME_QUALITY_GOOD
        self._last_tracking_state = (
            device_pb2.TrackingState.TRACKING_STATE_UNSPECIFIED
        )

        self.stub: device_pb2_grpc.TeleoperatorStub | None = None
        self.channel: grpc.Channel | None = None

        self._is_connected = False
        self._is_calibrated = False
        self.need_warmup = config.need_warmup
        self.force_recalibrate = config.force_recalibrate

    def _get_or_create_stub(self) -> device_pb2_grpc.TeleoperatorStub:
        """Lazily create the gRPC channel and stub, reused by connect()."""
        if self.stub is None:
            if self.config.use_ssl:
                with open(self.config.ssl_cert_path, "rb") as f:
                    creds = grpc.ssl_channel_credentials(f.read())
                self.channel = grpc.secure_channel(self.address, creds)
            else:
                self.channel = grpc.insecure_channel(self.address)
            self.stub = device_pb2_grpc.TeleoperatorStub(self.channel)
        return self.stub

    def _ensure_feature_info(self) -> None:
        """Populate feature info dicts via GetInfo RPC if not yet populated.

        This allows lerobot scripts to access action_features / feedback_features
        BEFORE connect(), since some scripts build schemas before calling connect().
        """
        if self._act_ft_info:
            return
        try:
            info = self._get_or_create_stub().GetInfo(
                device_pb2.GetInfoRequest(), timeout=self.connect_timeout_s
            )
            for fi in info.observation_features:
                self._obs_ft_info[fi.key] = fi
            for fi in info.action_features:
                self._act_ft_info[fi.key] = fi
            for fi in info.feedback_features:
                self._fb_ft_info[fi.key] = fi
        except grpc.RpcError as e:
            logger.warning(f"Pre-connect GetInfo failed for {self.address}: {e}")

    @property
    def feedback_features(self) -> dict[str, type | tuple]:
        self._ensure_feature_info()
        return {k: self._decode_feature_info(v)[1] for k, v in self._fb_ft_info.items()}

    @property
    def action_features(self) -> dict[str, type | tuple]:
        self._ensure_feature_info()
        return {k: self._decode_feature_info(v)[1] for k, v in self._act_ft_info.items()}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def last_action_quality(self) -> int:
        """Worst per-feature quality from the latest action snapshot."""
        return int(self._last_action_quality)

    @property
    def last_tracking_state(self) -> int:
        """Teleoperator lifecycle state from the latest action snapshot.

        Older servers do not populate this additive field; in that case the
        client derives a conservative state from ``FrameQuality``.
        """
        return int(self._last_tracking_state)

    @staticmethod
    def _verify_features(feature: RobotObservation | RobotAction, feature_dict: dict[str, type | tuple]) -> bool:
        """Verifies that the provided feature dictionary matches the expected features."""
        for key, expected_type in feature_dict.items():
            if key not in feature:
                raise ValueError(f"Missing expected feature '{key}' in the provided data.")
            if isinstance(expected_type, tuple):
                if np.shape(feature[key]) != expected_type:
                    return False
            elif not isinstance(feature[key], expected_type):
                return False
        return True

    @staticmethod
    def _decode_feature_info(feature_info: device_pb2.OneFeatureInfo) -> tuple[str, type | tuple]:
        """Decodes a OneFeatureInfo protobuf message into a key and its corresponding type."""
        key = feature_info.key
        data_type = feature_info.type

        if not (feature_info.shape.H == 1 and feature_info.shape.W == 1 and feature_info.shape.C == 1):
            # LeRobot convention: arrays (including images) are (height, width, channels).
            return key, (feature_info.shape.H, feature_info.shape.W, feature_info.shape.C)
        return key, python_scalar_type_for(data_type)

    def _init_feature(self, feature_info: device_pb2.OneFeatureInfo,
                      dest: RobotObservation | RobotAction | dict[str, Any]) -> None:
        """Initializes a feature based on its OneFeatureInfo protobuf message."""
        key, decoded_type = self._decode_feature_info(feature_info)
        if isinstance(decoded_type, tuple):
            # It's an image or multi-dimensional array
            dest[key] = np.zeros(decoded_type, dtype=feature_meta_for(feature_info.type)[1])
        else:
            # It's a scalar value
            dest[key] = decoded_type(0)

    def _cleanup(self):
        """Cleans up resources, such as closing the gRPC channel."""
        if self.stub:
            try:
                self.stub.Disconnect(Empty(), timeout=self.data_timeout_s)
            except grpc.RpcError:
                logger.debug(f"Failed to notify Disconnect to GRPCLeader at {self.address}.")
        if self.channel:
            self.channel.close()
            self.channel = None
        self.stub = None
        self._is_connected = False
        self._is_calibrated = False
        self._obs_ft_info.clear()
        self._act_ft_info.clear()
        self._fb_ft_info.clear()
        self._latest_obs_ft.clear()
        self._latest_act_ft.clear()
        self._latest_fb_ft.clear()
        self._last_action_quality = device_pb2.FrameQuality.FRAME_QUALITY_GOOD

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self.id} to GRPCLeader at {self.address}...")

        try:
            self._get_or_create_stub()

            calib_info = self.stub.Connect(Empty(), timeout=self.warmup_timeout_s)
            self._is_connected = True
            logger.info(f"Connected {self.id} to GRPCLeader at {self.address}.")
            if calib_info.status == device_pb2.CalibrationStatus.CALIBRATED:
                self._is_calibrated = True
                logger.info(f"GRPCLeader at {self.address} is already calibrated.")
            elif calib_info.status == device_pb2.CalibrationStatus.NEED_TO_CALIBRATE:
                self._is_calibrated = False
                if calibrate:
                    logger.warning(f"GRPCLeader at {self.address} needs calibration.")
                    self.calibrate()
            elif calib_info.status == device_pb2.CalibrationStatus.CALIBRATING:
                self._is_calibrated = False
                if calibrate:
                    self.calibrate()
                    logger.warning(f"GRPCLeader at {self.address} is currently calibrating.")
            else:
                raise DeviceNotConnectedError("Failed to retrieve calibration info from GRPCLeader.")

            self._ensure_feature_info()
            for fi in self._obs_ft_info.values():
                self._init_feature(fi, self._latest_obs_ft)
            for fi in self._act_ft_info.values():
                self._init_feature(fi, self._latest_act_ft)
            for fi in self._fb_ft_info.values():
                self._init_feature(fi, self._latest_fb_ft)

            if self.need_warmup and self._is_calibrated:
                temp_act = self.get_action()
                if not self._verify_features(temp_act, self.action_features):
                    raise DeviceNotConnectedError("Failed to warm up the GRPCLeader: action features mismatch.")
        except Exception as e:
            logger.error(f"Error connecting to GRPCLeader at {self.address}: {e}")
            self._cleanup()
            raise DeviceNotConnectedError(f"Failed to connect to GRPCLeader at {self.address}: {e}")

    def _wait_for_calibration(self, only_once: bool = False) -> None:
        """Polls the server until calibration completes, fails, or times out.

        Raises:
            DeviceNotConnectedError: if the server reports a fatal state (e.g. a failed calibration),
                or if calibration does not complete within `max_calibration_attempts`.
        """
        attempts = 0
        max_attempts = 1 if only_once else self.config.max_calibration_attempts
        while attempts < max_attempts:
            time.sleep(self.connect_timeout_s)
            if self.get_device_status() == device_pb2.DeviceStatus.FATAL:
                raise DeviceNotConnectedError(
                    f"Calibration of GRPCLeader at {self.address} failed on the server. Check the server logs."
                )
            calib_info = self.stub.Calibrate(
                device_pb2.CalibrateRequest(force=False), timeout=self.connect_timeout_s
            )
            if calib_info.status == device_pb2.CalibrationStatus.CALIBRATED:
                self._is_calibrated = True
                logger.info(f"GRPCLeader at {self.address} calibrated successfully.")
                return
            if calib_info.status == device_pb2.CalibrationStatus.CALIBRATING:
                logger.info(
                    f"Still calibrating... (attempt {attempts + 1}/{max_attempts})"
                )
                attempts += 1
                continue
            if calib_info.status == device_pb2.CalibrationStatus.NEED_TO_CALIBRATE:
                # The server rejected this calibration round; ask the user to run another one.
                self._calibrate_once(only_one_attempt=True)
                return
            raise DeviceNotConnectedError("Unknown calibration status.")
        raise DeviceNotConnectedError(
            f"Calibration of GRPCLeader at {self.address} stuck for too long. Check the server logs."
        )

    def _calibrate_once(self, only_one_attempt: bool = False) -> None:
        logger.warning(f"GRPCLeader at {self.address} needs calibration.")
        # Clear screen so each step starts fresh (multi-step calibration
        # recurses through _calibrate_once; leftover lines garble the
        # move_cursor_up refresh).
        print("\033[2J\033[H", end="", flush=True)
        print("Follow the calibration instructions below, then press Enter when ready.")
        # Drain any buffered Enter presses from the inter-step delay so
        # they don't auto-advance this step before the user is ready.
        while enter_pressed():
            pass

        latest: dict[str, Any] = {"frame": None}
        stop = threading.Event()

        def _recv_frames():
            try:
                for frame in self.stub.StreamCalibration(Empty()):
                    latest["frame"] = frame
                    if stop.is_set():
                        break
            except grpc.RpcError:
                pass

        recv_thread = threading.Thread(target=_recv_frames, daemon=True)
        recv_thread.start()

        n = 0
        while not enter_pressed():
            frame = latest["frame"]
            if frame is not None and len(frame.readings) > 0:
                n = len(frame.readings)
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for r in frame.readings:
                    print(f"{r.name:<15} | {r.range_min:>6} | {r.position:>6} | {r.range_max:>6}")
                move_cursor_up(n + 3)
            time.sleep(0.05)
        stop.set()

        self.stub.CalibrateDone(Empty(), timeout=self.connect_timeout_s)
        self._wait_for_calibration(only_once=only_one_attempt)

    def calibrate(self) -> None:
        logger.info(f"Calibrating GRPCLeader at {self.address}...")
        if not self.is_connected:
            raise DeviceNotConnectedError("Cannot calibrate: GRPCLeader is not connected.")
        if self.is_calibrated and not self.force_recalibrate:
            logger.info(f"GRPCLeader at {self.address} is already calibrated.")
            return
        if self.force_recalibrate:
            logger.info(f"Force recalibration requested for GRPCLeader at {self.address}.")
            self._is_calibrated = False

        try:
            calib_response = self.stub.Calibrate(
                device_pb2.CalibrateRequest(force=self.force_recalibrate),
                timeout=self.connect_timeout_s,
            )
            if calib_response.status == device_pb2.CalibrationStatus.CALIBRATED:
                self._is_calibrated = True
                logger.info(f"GRPCLeader at {self.address} calibrated successfully.")
            elif calib_response.status == device_pb2.CalibrationStatus.NEED_TO_CALIBRATE:
                self._calibrate_once(only_one_attempt=True)
                logger.info(f"GRPCLeader at {self.address} calibrated successfully after manual calibration.")
            elif calib_response.status == device_pb2.CalibrationStatus.CALIBRATING:
                logger.info(f"GRPCLeader at {self.address} is currently calibrating. Please wait...")
                self._wait_for_calibration(only_once=False)
            else:
                raise DeviceNotConnectedError("Unknown calibration status.")
        except grpc.RpcError as e:
            logger.error(f"gRPC error during calibration: {e}")
            raise DeviceNotConnectedError(f"Failed to calibrate GRPCLeader at {self.address}: {e}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Capture observations from the remote robot: force updates of CRITICAL
        features; use latest possible AUXILIARY features
        """
        received_keys: set[str] = set()
        try:
            for obs in self.stub.GetObservation(Empty(), timeout=self.data_timeout_s):
                load_feature(obs, self._obs_ft_info, self._latest_obs_ft)
                received_keys.add(obs.key)
        except grpc.RpcError as e:
            raise DeviceNotConnectedError(
                f"Failed to receive observation from GRPCLeader at {self.address}: {e}"
            ) from e
        if not received_keys:
            raise DeviceNotConnectedError("No observation received from GRPCLeader.")
        missing_critical = {
            key
            for key, info in self._obs_ft_info.items()
            if info.criticality == device_pb2.Criticality.CRITICALITY_CRITICAL and key not in received_keys
        }
        if missing_critical:
            raise DeviceNotConnectedError(
                f"Missing critical feature(s) {sorted(missing_critical)} in the received observation."
            )

        return self._latest_obs_ft.copy()

    def configure(self):
        """No local configuration needed: all hardware configuration is done on the remote gRPC server."""
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        attempts = _BUSY_RETRIES
        last_error: grpc.RpcError | None = None
        while attempts > 0:
            attempts -= 1
            received_keys: set[str] = set()
            qualities: list[int] = []
            tracking_states: list[int] = []
            try:
                for act in self.stub.GetAction(Empty(), timeout=self.data_timeout_s):
                    load_feature(act, self._act_ft_info, self._latest_act_ft)
                    received_keys.add(act.key)
                    quality = int(act.quality)
                    qualities.append(
                        quality
                        if quality != device_pb2.FrameQuality.FRAME_QUALITY_UNSPECIFIED
                        else device_pb2.FrameQuality.FRAME_QUALITY_GOOD
                    )
                    tracking_state = int(
                        getattr(
                            act,
                            "tracking_state",
                            device_pb2.TrackingState.TRACKING_STATE_UNSPECIFIED,
                        )
                    )
                    if (
                        tracking_state
                        != device_pb2.TrackingState.TRACKING_STATE_UNSPECIFIED
                    ):
                        tracking_states.append(tracking_state)
                last_error = None
                break
            except grpc.RpcError as e:
                last_error = e
                if "calibrating" not in str(e):
                    break  # not a transient busy rejection; surface immediately
                if attempts > 0:
                    logger.warning(
                        f"GetAction rejected while the robot is busy ({e}); retrying "
                        f"({_BUSY_RETRIES - attempts}/{_BUSY_RETRIES})..."
                    )
                    time.sleep(_busy_retry_delay(_BUSY_RETRIES - attempts - 1))
        if last_error is not None:
            raise DeviceNotConnectedError(
                f"Failed to receive action from GRPCLeader at {self.address}: {last_error}"
            ) from last_error
        if not received_keys:
            raise DeviceNotConnectedError("No action received from GRPCLeader.")
        missing_critical = {
            key
            for key, info in self._act_ft_info.items()
            if info.criticality == device_pb2.Criticality.CRITICALITY_CRITICAL and key not in received_keys
        }
        if missing_critical:
            raise DeviceNotConnectedError(
                f"Missing critical feature(s) {sorted(missing_critical)} in the received action."
            )

        self._last_action_quality = max(
            qualities,
            default=device_pb2.FrameQuality.FRAME_QUALITY_GOOD,
        )
        unique_tracking_states = set(tracking_states)
        if len(unique_tracking_states) == 1:
            self._last_tracking_state = unique_tracking_states.pop()
        elif len(unique_tracking_states) > 1:
            # A snapshot carrying conflicting lifecycle states is not safe to
            # interpret as ready.  Quality still independently gates motion.
            self._last_tracking_state = (
                device_pb2.TrackingState.TRACKING_STATE_UNSPECIFIED
            )
        elif (
            self._last_action_quality
            == device_pb2.FrameQuality.FRAME_QUALITY_GOOD
        ):
            self._last_tracking_state = (
                device_pb2.TrackingState.TRACKING_STATE_READY
            )
        elif (
            self._last_action_quality
            == device_pb2.FrameQuality.FRAME_QUALITY_STALE
        ):
            self._last_tracking_state = (
                device_pb2.TrackingState.TRACKING_STATE_LOST
            )
        else:
            self._last_tracking_state = (
                device_pb2.TrackingState.TRACKING_STATE_HELD
            )

        return self._latest_act_ft.copy()

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        for key in feedback.keys():
            self._latest_fb_ft[key] = feedback[key]
        try:
            self.stub.SendFeedback(
                encode_feature(self._fb_ft_info, self._latest_fb_ft), timeout=self.data_timeout_s
            )
        except grpc.RpcError as e:
            raise DeviceNotConnectedError(
                f"Failed to send feedback to GRPCLeader at {self.address}: {e}"
            ) from e

    @check_if_not_connected
    def disconnect(self):
        """Cleans gRPC comms"""

        self._cleanup()

    @check_if_not_connected
    def get_device_status(self) -> device_pb2.DeviceStatus:
        """Returns the device status from the remote robot."""
        try:
            return self.stub.GetStatus(Empty(), timeout=self.data_timeout_s).status
        except grpc.RpcError as e:
            raise DeviceNotConnectedError(
                f"Failed to get device status from GRPCLeader at {self.address}: {e}"
            ) from e

    @check_if_not_connected
    def set_reference(self) -> None:
        """Sets the reference for the remote robot. On receiving this command, the remote robot will set its current state as the reference state."""
        try:
            self.stub.SetReference(Empty(), timeout=self.reference_timeout_s)
        except grpc.RpcError as e:
            if e.code() in (
                grpc.StatusCode.FAILED_PRECONDITION,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            ):
                raise ReferenceNotReadyError(
                    f"Reference is not ready on GRPCLeader at {self.address}: {e}"
                ) from e
            raise DeviceNotConnectedError(
                f"Failed to set reference on GRPCLeader at {self.address}: {e}"
            ) from e
