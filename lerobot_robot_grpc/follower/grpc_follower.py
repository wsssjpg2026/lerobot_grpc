from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _grpc_available, require_package
from lerobot.utils.utils import enter_pressed, move_cursor_up

from lerobot.robots.robot import Robot
from .config_grpc import GRPCFollowerConfig
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


class GRPCFollower(Robot):
    config_class = GRPCFollowerConfig
    name = "grpc_follower"

    def __init__(self, config: GRPCFollowerConfig):
        require_package("grpcio", extra="grpcio-dep", import_name="grpc")
        super().__init__(config)
        self.config = config
        self.address = config.address

        self.warmup_timeout_s = config.warmup_timeout_s
        self.connect_timeout_s = config.connect_timeout_s
        self.data_timeout_s = config.data_timeout_s

        self._obs_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}
        self._act_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}
        self._fb_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}

        self._latest_obs_ft: RobotObservation = {}
        self._latest_act_ft: RobotAction = {}
        self._latest_fb_ft: dict[str, Any] = {}

        self.stub: device_pb2_grpc.RobotStub | None = None
        self.channel: grpc.Channel | None = None

        self._is_connected = False
        self._is_calibrated = False
        self.need_warmup = config.need_warmup

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {k: self._decode_feature_info(v)[1] for k, v in self._obs_ft_info.items()}

    @property
    def action_features(self) -> dict[str, type | tuple]:
        return {k: self._decode_feature_info(v)[1] for k, v in self._act_ft_info.items()}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

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
                logger.debug(f"Failed to notify Disconnect to GRPCFollower at {self.address}.")
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

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self.id} to GRPCFollower at {self.address}...")

        try:
            if self.config.use_ssl:
                with open(self.config.ssl_cert_path, "rb") as f:
                    creds = grpc.ssl_channel_credentials(f.read())
                self.channel = grpc.secure_channel(self.address, creds)
            else:
                self.channel = grpc.insecure_channel(self.address)
            self.stub = device_pb2_grpc.RobotStub(self.channel)

            calib_info = self.stub.Connect(Empty(), timeout=self.warmup_timeout_s)
            self._is_connected = True
            logger.info(f"Connected {self.id} to GRPCFollower at {self.address}.")
            if calib_info.status == device_pb2.CalibrationStatus.CALIBRATED:
                self._is_calibrated = True
                logger.info(f"GRPCFollower at {self.address} is already calibrated.")
            elif calib_info.status == device_pb2.CalibrationStatus.NEED_TO_CALIBRATE:
                self._is_calibrated = False
                if calibrate:
                    logger.warning(f"GRPCFollower at {self.address} needs calibration.")
                    self.calibrate()
            elif calib_info.status == device_pb2.CalibrationStatus.CALIBRATING:
                self._is_calibrated = False
                if calibrate:
                    self.calibrate()
                    logger.warning(f"GRPCFollower at {self.address} is currently calibrating.")
            else:
                raise DeviceNotConnectedError("Failed to retrieve calibration info from GRPCFollower.")

            for feature_info in self.stub.GetObservationFeatureInfo(Empty(), timeout=self.warmup_timeout_s):
                self._obs_ft_info[feature_info.key] = feature_info
                self._init_feature(feature_info, self._latest_obs_ft)
            for feature_info in self.stub.GetActionFeatureInfo(Empty(), timeout=self.warmup_timeout_s):
                self._act_ft_info[feature_info.key] = feature_info
                self._init_feature(feature_info, self._latest_act_ft)
            for feature_info in self.stub.GetFeedbackFeatureInfo(Empty(), timeout=self.warmup_timeout_s):
                self._fb_ft_info[feature_info.key] = feature_info
                self._init_feature(feature_info, self._latest_fb_ft)

            if self.need_warmup:
                if self.get_device_status() == device_pb2.DeviceStatus.FATAL:
                    raise DeviceNotConnectedError(
                        f"GRPCFollower at {self.address} reported a fatal state (e.g. failed calibration); "
                        "cannot warm up. Check the server logs."
                    )
                temp_obs = self.get_observation()
                if not self._verify_features(temp_obs, self.observation_features):
                    raise DeviceNotConnectedError("Failed to warm up the GRPCFollower: observation features mismatch.")
        except Exception as e:
            logger.error(f"Error connecting to GRPCFollower at {self.address}: {e}")
            self._cleanup()
            raise DeviceNotConnectedError(f"Failed to connect to GRPCFollower at {self.address}: {e}")

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
                    f"Calibration of GRPCFollower at {self.address} failed on the server. Check the server logs."
                )
            calib_info = self.stub.Calibrate(Empty(), timeout=self.connect_timeout_s)
            if calib_info.status == device_pb2.CalibrationStatus.CALIBRATED:
                self._is_calibrated = True
                logger.info(f"GRPCFollower at {self.address} calibrated successfully.")
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
            f"Calibration of GRPCFollower at {self.address} stuck for too long. Check the server logs."
        )

    def _calibrate_once(self, only_one_attempt: bool = False) -> None:
        logger.warning(f"GRPCFollower at {self.address} needs calibration.")
        print("Starting calibration process... (move joints through full range, then press Enter)")

        latest: dict[str, Any] = {"frame": None}
        stop = threading.Event()

        def _recv_frames():
            try:
                for frame in self.stub.StreamCalibration(Empty(), timeout=self.connect_timeout_s):
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
        logger.info(f"Calibrating GRPCFollower at {self.address}...")
        if not self.is_connected:
            raise DeviceNotConnectedError("Cannot calibrate: GRPCFollower is not connected.")
        if self.is_calibrated:
            logger.info(f"GRPCFollower at {self.address} is already calibrated.")
            return

        try:
            calib_response = self.stub.Calibrate(Empty(), timeout=self.connect_timeout_s)
            if calib_response.status == device_pb2.CalibrationStatus.CALIBRATED:
                self._is_calibrated = True
                logger.info(f"GRPCFollower at {self.address} calibrated successfully.")
            elif calib_response.status == device_pb2.CalibrationStatus.NEED_TO_CALIBRATE:
                self._calibrate_once(only_one_attempt=True)
                logger.info(f"GRPCFollower at {self.address} calibrated successfully after manual calibration.")
            elif calib_response.status == device_pb2.CalibrationStatus.CALIBRATING:
                logger.info(f"GRPCFollower at {self.address} is currently calibrating. Please wait...")
                self._wait_for_calibration(only_once=False)
            else:
                raise DeviceNotConnectedError("Unknown calibration status.")
        except grpc.RpcError as e:
            logger.error(f"gRPC error during calibration: {e}")
            raise DeviceNotConnectedError(f"Failed to calibrate GRPCFollower at {self.address}: {e}")

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
                f"Failed to receive observation from GRPCFollower at {self.address}: {e}"
            ) from e
        if not received_keys:
            raise DeviceNotConnectedError("No observation received from GRPCFollower.")
        missing_critical = {
            key
            for key, info in self._obs_ft_info.items()
            if info.criticality == device_pb2.Criticality.CRITICAL and key not in received_keys
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
    def send_action(self, action: RobotAction) -> RobotAction:

        for key in action.keys():
            self._latest_act_ft[key] = action[key]
        try:
            self.stub.SendAction(
                encode_feature(self._act_ft_info, self._latest_act_ft), timeout=self.data_timeout_s
            )
        except grpc.RpcError as e:
            raise DeviceNotConnectedError(
                f"Failed to send action to GRPCFollower at {self.address}: {e}"
            ) from e

        return self._latest_act_ft.copy()

    @check_if_not_connected
    def get_feedback(self) -> dict[str, Any]:
        received_keys: set[str] = set()
        try:
            for fb in self.stub.GetFeedback(Empty(), timeout=self.data_timeout_s):
                load_feature(fb, self._fb_ft_info, self._latest_fb_ft)
                received_keys.add(fb.key)
        except grpc.RpcError as e:
            raise DeviceNotConnectedError(
                f"Failed to receive feedback from GRPCFollower at {self.address}: {e}"
            ) from e
        if not received_keys:
            raise DeviceNotConnectedError("No feedback received from GRPCFollower.")
        missing_critical = {
            key
            for key, info in self._fb_ft_info.items()
            if info.criticality == device_pb2.Criticality.CRITICAL and key not in received_keys
        }
        if missing_critical:
            raise DeviceNotConnectedError(
                f"Missing critical feature(s) {sorted(missing_critical)} in the received feedback."
            )

        return self._latest_fb_ft.copy()

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
                f"Failed to get device status from GRPCFollower at {self.address}: {e}"
            ) from e