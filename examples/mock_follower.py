"""Shared no-hardware mock follower for example scripts (loopback test, demos).

`MockFollowerServicer` is a drop-in `FollowerServicer`: joints climb with wall
clock, one synthetic H.264 camera (moving white square on green) is streamed,
and actions are echoed back as feedback — the same shape as the real
SO101 follower server, minus the hardware.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from google.protobuf.empty_pb2 import Empty

from lerobot_robot_grpc.follower.follower_server import FollowerServicer
from lerobot_robot_grpc.follower.utils import (
    H264FrameEncoder,
    _now_produce_ts,
    encode_feature,
    load_feature,
)
from lerobot_robot_grpc.protos import device_pb2

# Two dummy scalar joints — the "hardware" this mock pretends to be.
ACTION_KEYS = ("joint_0.pos", "joint_1.pos")
CAM_HEIGHT, CAM_WIDTH = 240, 320


def scalar_feature_info(key: str) -> device_pb2.OneFeatureInfo:
    return device_pb2.OneFeatureInfo(
        key=key,
        criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
        type=device_pb2.DataType.FLOAT32,
        shape=device_pb2.ImageShape(H=1, W=1, C=1),
        encoding=device_pb2.Encoding.RAW,
        img_quality=100,
    )


def camera_feature_info(key: str, height: int, width: int) -> device_pb2.OneFeatureInfo:
    """A CRITICAL H.264-encoded RGB camera feature."""
    return device_pb2.OneFeatureInfo(
        key=key,
        criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
        type=device_pb2.DataType.UINT8,
        shape=device_pb2.ImageShape(H=height, W=width, C=3),
        encoding=device_pb2.Encoding.H264,
        img_quality=90,
    )


class MockFollowerServicer(FollowerServicer):
    """A no-hardware follower: observations stream continuously, feedback echoes actions."""

    def __init__(self):
        self._obs_ft_info = {k: scalar_feature_info(k) for k in ACTION_KEYS}
        self._obs_ft_info["cam"] = camera_feature_info("cam", CAM_HEIGHT, CAM_WIDTH)
        self._act_ft_info = {k: scalar_feature_info(k) for k in ACTION_KEYS}
        self._lock = threading.Lock()
        # Joint values climb 1 unit per second (wall-clock based, so assertions are
        # immune to the stream rate); the camera shows a moving white square on green.
        self._t0 = time.monotonic()
        self._phase = 0
        self._last_action: dict[str, float] = {k: 0.0 for k in ACTION_KEYS}

    def _camera_frame(self) -> np.ndarray:
        img = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        img[..., 0] = 40
        img[..., 1] = 180
        img[..., 2] = 40
        x = (self._phase * 8) % (CAM_WIDTH - 40)
        img[100:140, x : x + 40] = 255
        self._phase += 1
        return img

    # --- feature introspection: single GetInfo returns all three schemas ---
    def GetInfo(self, request, context):
        return device_pb2.GetInfoResponse(
            observation_features=list(self._obs_ft_info.values()),
            action_features=list(self._act_ft_info.values()),
            feedback_features=list(self._act_ft_info.values()),
        )

    # --- lifecycle -----------------------------------------------------------
    def Connect(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        return Empty()

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)

    # --- data flow -----------------------------------------------------------
    def GetObservation(self, request, context):
        # Persistent stream: joints at ~50Hz + one H.264 access unit per camera frame.
        # Fresh encoder per stream → first camera packet is always a keyframe.
        encoders: dict[str, H264FrameEncoder] = {}
        while context.is_active():
            with self._lock:
                tick = time.monotonic() - self._t0
            obs = {
                "joint_0.pos": float(tick),
                "joint_1.pos": float(tick) * 2.0,
                "cam": self._camera_frame(),
            }
            yield from encode_feature(self._obs_ft_info, {k: v for k, v in obs.items() if k != "cam"})
            info = self._obs_ft_info["cam"]
            encoder = encoders.get("cam")
            if encoder is None:
                encoder = H264FrameEncoder("cam", info.shape.H, info.shape.W, fps=30)
                encoders["cam"] = encoder
            for unit in encoder.encode(obs["cam"]):
                yield device_pb2.OneFeature(key="cam", data=unit, produce_ts=_now_produce_ts())
            time.sleep(0.02)

    def SendAction(self, request, context):
        action: dict[str, float] = {}
        for feat in request.features:
            load_feature(feat, self._act_ft_info, action)
        with self._lock:
            self._last_action = action
        # 返回 executed(mock echo commanded,等价于 so101 A 类语义)。
        return device_pb2.ActionResult(
            features=list(encode_feature(self._act_ft_info, action))
        )

    def GetFeedback(self, request, context):
        with self._lock:
            return encode_feature(self._act_ft_info, dict(self._last_action))
