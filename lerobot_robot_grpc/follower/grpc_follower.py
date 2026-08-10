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

from lerobot.robots.robot import Robot
from .config_grpc import GRPCFollowerConfig
from .utils import (
    H264FrameDecoder,
    TeleopMonitor,
    TeleopStats,
    _wall_now,
    encode_feature,
    feature_meta_for,
    load_feature,
    python_scalar_type_for,
)

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

# The server rejects SendAction while the bus is busy (real calibration, or a just-released
# stuck bus call). Retry briefly instead of crashing the teleop loop on a transient busy.
_BUSY_RETRIES = 5


def _busy_retry_delay(attempt: int) -> float:
    """Exponential backoff + jitter for transient busy retries: ~0.2s, 0.4, 0.8, capped at
    1.0s, plus up to 100ms jitter. The previous fixed 1.0s blocked the teleop loop for up to
    5s on a transient busy; this bounds it to ~2.5s and spreads retries to avoid thundering."""
    base = min(1.0, 0.2 * (2 ** attempt))
    return base + random.uniform(0.0, 0.1)


# Latency outside [0, this] ms implies server/client wall-clock skew (cross-host, unsynced).
# Fall back to relative jitter then; absolute latency is only valid same-host / NTP-synced.
_LATENCY_SANE_MAX_MS = 10_000.0


class GRPCFollower(Robot):
    config_class = GRPCFollowerConfig
    name = "grpc_follower"

    def __init__(self, config: GRPCFollowerConfig, stats: TeleopStats | None = None):
        require_package("grpcio", extra="grpcio-dep", import_name="grpc")
        super().__init__(config)
        self.config = config
        self.address = config.address

        self.warmup_timeout_s = config.warmup_timeout_s
        self.connect_timeout_s = config.connect_timeout_s
        self.data_timeout_s = config.data_timeout_s

        # Optional telemetry hook (see utils.py): records bytes + end-to-end
        # latency per feature and SendAction RTT. `produce_ts` is the server's wall
        # clock stamped at sample time; latency is computed as an absolute wall-clock
        # delta (accurate same-host / NTP-synced). `_stats_offset` is only used by the
        # cross-host skew fallback (relative-to-first-frame jitter).
        if stats is not None:
            self._stats = stats
        elif config.teleop_stats:
            self._stats = TeleopStats()
        else:
            self._stats = None
        self._stats_offset: float | None = None
        self._stats_clock_skewed = False
        # Auto-lifecycle monitor (config flag): started on connect(), prints the
        # session summary on disconnect(). If the caller passed `stats=` they own
        # the monitor (see examples/teleop_monitor_demo.py).
        self._monitor: TeleopMonitor | None = None
        if stats is None and config.teleop_stats:
            self._monitor = TeleopMonitor(self._stats, interval=config.stats_interval_s)

        self._obs_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}
        self._act_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}
        self._fb_ft_info: dict[str, device_pb2.OneFeatureInfo] = {}

        self._latest_obs_ft: RobotObservation = {}
        self._latest_act_ft: RobotAction = {}
        self._latest_fb_ft: dict[str, Any] = {}

        # Persistent observation stream: a background thread holds the GetObservation RPC
        # open and continuously refreshes `_latest_obs_ft` (H264 cameras are decoded here).
        self._obs_stop = threading.Event()
        self._obs_thread: threading.Thread | None = None
        self._latest_obs_time = 0.0

        self.stub: device_pb2_grpc.RobotStub | None = None
        self.channel: grpc.Channel | None = None

        self._is_connected = False
        self._is_calibrated = False
        self.need_warmup = config.need_warmup
        self.force_recalibrate = config.force_recalibrate

    def _get_or_create_stub(self) -> device_pb2_grpc.RobotStub:
        """Lazily create the gRPC channel and stub, reused by connect()."""
        if self.stub is None:
            if self.config.use_ssl:
                with open(self.config.ssl_cert_path, "rb") as f:
                    creds = grpc.ssl_channel_credentials(f.read())
                self.channel = grpc.secure_channel(self.address, creds)
            else:
                self.channel = grpc.insecure_channel(self.address)
            self.stub = device_pb2_grpc.RobotStub(self.channel)
        return self.stub

    def _ensure_feature_info(self) -> None:
        """Populate feature info dicts via GetInfo RPC if not yet populated.

        This allows lerobot-record to access observation_features / action_features
        BEFORE connect(), since the record script builds the dataset schema before
        calling connect().
        """
        if self._obs_ft_info:
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
    def observation_features(self) -> dict[str, type | tuple]:
        self._ensure_feature_info()
        return {k: self._decode_feature_info(v)[1] for k, v in self._obs_ft_info.items()}

    @property
    def action_features(self) -> dict[str, type | tuple]:
        self._ensure_feature_info()
        return {k: self._decode_feature_info(v)[1] for k, v in self._act_ft_info.items()}

    @property
    def cameras(self) -> dict[str, None]:
        self._ensure_feature_info()
        return {k: None for k, v in self.observation_features.items() if isinstance(v, tuple)}

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
        self._obs_stop.set()
        if self._obs_thread is not None:
            self._obs_thread.join(timeout=2.0)
            self._obs_thread = None
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
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
        self._latest_obs_time = 0.0

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self.id} to GRPCFollower at {self.address}...")

        try:
            self._get_or_create_stub()

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

            self._ensure_feature_info()
            for fi in self._obs_ft_info.values():
                self._init_feature(fi, self._latest_obs_ft)
            for fi in self._act_ft_info.values():
                self._init_feature(fi, self._latest_act_ft)
            for fi in self._fb_ft_info.values():
                self._init_feature(fi, self._latest_fb_ft)

            # Start the persistent observation stream before warmup: get_observation()
            # now returns the latest frame consumed by this thread.
            self._obs_stop.clear()
            self._obs_thread = threading.Thread(
                target=self._obs_stream_loop,
                daemon=True,
                name=f"grpc-follower-obs-{self.id}",
            )
            self._obs_thread.start()
            if self._monitor is not None:
                self._monitor.start()

            if self.need_warmup and self._is_calibrated:
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
            calib_info = self.stub.Calibrate(
                device_pb2.CalibrateRequest(force=False), timeout=self.connect_timeout_s
            )
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
        logger.info(f"Calibrating GRPCFollower at {self.address}...")
        if not self.is_connected:
            raise DeviceNotConnectedError("Cannot calibrate: GRPCFollower is not connected.")
        if self.is_calibrated and not self.force_recalibrate:
            logger.info(
                f"GRPCFollower at {self.address} is already calibrated. "
                "Use --robot.force_recalibrate=true to run calibration again."
            )
            return
        if self.force_recalibrate:
            logger.info(f"Force recalibration requested for GRPCFollower at {self.address}.")
            self._is_calibrated = False

        try:
            calib_response = self.stub.Calibrate(
                device_pb2.CalibrateRequest(force=self.force_recalibrate),
                timeout=self.connect_timeout_s,
            )
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

    def _obs_stream_loop(self) -> None:
        """Background consumer for the persistent GetObservation stream.

        Holds the RPC open and refreshes `_latest_obs_ft` continuously; reconnects
        with a 1s backoff when the stream dies. H264 decoder state is reset per
        stream because the server always opens with a keyframe.
        """
        decoders: dict[str, H264FrameDecoder] = {}
        try:
            while not self._obs_stop.is_set():
                try:
                    stream = self.stub.GetObservation(Empty(), timeout=None)
                    # Close native decoder contexts before dropping them (P1.1): the server
                    # always opens with a keyframe, so a fresh decoder per stream is safe.
                    for dec in decoders.values():
                        dec.close()
                    decoders.clear()
                    for feat in stream:
                        if self._obs_stop.is_set():
                            break
                        self._consume_obs_feature(feat, decoders)
                except ImportError:
                    logger.error("H.264 observation stream requires PyAV; install `av` (e.g. via lerobot[dataset]).")
                    break
                except grpc.RpcError as e:
                    if self._obs_stop.is_set():
                        break
                    logger.warning(
                        f"Observation stream from GRPCFollower at {self.address} interrupted ({e}); reconnecting..."
                    )
                except Exception:
                    if self._obs_stop.is_set():
                        break
                    logger.exception(
                        f"Observation stream from GRPCFollower at {self.address} crashed; reconnecting..."
                    )
                if not self._obs_stop.is_set():
                    time.sleep(1.0)
        finally:
            for dec in decoders.values():
                dec.close()

    def _consume_obs_feature(
        self, feat: device_pb2.OneFeature, decoders: dict[str, H264FrameDecoder]
    ) -> None:
        """Decodes one streamed feature into `_latest_obs_ft` (H264 via per-camera decoder state)."""
        info = self._obs_ft_info.get(feat.key)
        if info is None:
            return
        if info.encoding == device_pb2.Encoding.H264:
            decoder = decoders.get(feat.key)
            if decoder is None:
                decoder = H264FrameDecoder(feat.key, info.shape.H, info.shape.W)
                decoders[feat.key] = decoder
            image = decoder.decode(feat.data)
            if image is not None:
                self._latest_obs_ft[feat.key] = image
        else:
            load_feature(feat, self._obs_ft_info, self._latest_obs_ft)
        self._latest_obs_time = time.monotonic()
        if self._stats is not None:
            stamp = feat.produce_ts.seconds + feat.produce_ts.nanos / 1e9
            latency_ms = None
            if stamp > 0:
                # Absolute end-to-end latency: server wall-clock sample time -> local now.
                # Accurate when client and server share a clock (same host or NTP-synced).
                abs_ms = (_wall_now() - stamp) * 1000.0
                if 0.0 <= abs_ms <= _LATENCY_SANE_MAX_MS:
                    latency_ms = abs_ms
                    self._stats_clock_skewed = False
                else:
                    # Cross-host clock skew: fall back to relative-to-first-frame jitter
                    # (immune to skew) so the monitor still shows a meaningful trend.
                    # TODO(P1.4c): replace with RTT-based clock-offset estimation
                    # (server echoes its wall time in SendAction; client EWMA-smooths the
                    # offset) so cross-host deployments get true absolute latency too.
                    if self._stats_offset is None:
                        self._stats_offset = time.perf_counter() - stamp
                    else:
                        latency_ms = (time.perf_counter() - (stamp + self._stats_offset)) * 1000.0
                    if not self._stats_clock_skewed:
                        self._stats_clock_skewed = True
                        logger.warning(
                            "teleop_stats: server/client clock skew detected (produce_ts -> "
                            "local wall clock out of range); falling back to relative jitter. "
                            "Absolute latency is accurate only on same-host/NTP-synced setups."
                        )
            self._stats.record_feature(feat.key, len(feat.data), latency_ms)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Returns the latest observation from the remote robot.

        The background stream thread refreshes `_latest_obs_ft` continuously;
        this call blocks until the first frame arrives (warmup), then serves
        the latest snapshot, raising if the stream has gone stale.
        """
        deadline = time.monotonic() + self.warmup_timeout_s
        while self._latest_obs_time == 0.0 and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._latest_obs_time == 0.0:
            raise DeviceNotConnectedError("No observation received from GRPCFollower.")
        if time.monotonic() - self._latest_obs_time > self.data_timeout_s:
            raise DeviceNotConnectedError(
                f"Observation stream from GRPCFollower at {self.address} is stale "
                f"(no data for {self.data_timeout_s:.1f}s)."
            )
        return self._latest_obs_ft.copy()

    def configure(self):
        """No local configuration needed: all hardware configuration is done on the remote gRPC server."""
        pass

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:

        for key in action.keys():
            self._latest_act_ft[key] = action[key]
        t0 = time.perf_counter() if self._stats is not None else None
        attempts = _BUSY_RETRIES
        last_error: grpc.RpcError | None = None
        while attempts > 0:
            attempts -= 1
            try:
                req = device_pb2.Action(
                    features=list(encode_feature(self._act_ft_info, self._latest_act_ft))
                )
                resp = self.stub.SendAction(req, timeout=self.data_timeout_s)
                last_error = None
                break
            except grpc.RpcError as e:
                last_error = e
                if "calibrating" not in str(e):
                    break  # not a transient busy rejection; surface immediately
                if attempts > 0:
                    logger.warning(
                        f"SendAction rejected while the robot is busy ({e}); retrying "
                        f"({_BUSY_RETRIES - attempts}/{_BUSY_RETRIES})..."
                    )
                    time.sleep(_busy_retry_delay(_BUSY_RETRIES - attempts - 1))
        if last_error is not None:
            raise DeviceNotConnectedError(
                f"Failed to send action to GRPCFollower at {self.address}: {last_error}"
            ) from last_error
        if t0 is not None:
            self._stats.record_action((time.perf_counter() - t0) * 1000.0)

        # 解码服务端返回的 executed(so101 A 类 ≈ commanded;B 类为服务端 IK 后关节角)。
        executed: RobotAction = {}
        for feat in resp.features:
            load_feature(feat, self._act_ft_info, executed)
        self._latest_act_ft.update(executed)
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
            if info.criticality == device_pb2.Criticality.CRITICALITY_CRITICAL and key not in received_keys
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