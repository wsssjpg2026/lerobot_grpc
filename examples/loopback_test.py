"""Loopback test — exercise the gRPC follower path with NO hardware.

Runs a mock `FollowerServicer` in-process (it produces dummy scalar observations and
echoes received actions as feedback), then connects the real `GRPCFollower` client to
it and round-trips observation / action / feedback. This validates the whole stack the
unit tests don't: proto encode/decode, feature-schema negotiation over Get*FeatureInfo,
streaming RPCs, and the client class end-to-end.

Run from the repo root (with the env that has lerobot[grpcio-dep] + this package):

    python examples/loopback_test.py
"""

from __future__ import annotations

import logging
import threading
import time

from google.protobuf.empty_pb2 import Empty

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.follower_server import (
    FollowerServer,
    FollowerServerConfig,
    FollowerServicer,
)
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.follower.utils import encode_feature, load_feature
from lerobot_robot_grpc.protos import device_pb2

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("loopback")

ADDRESS = "127.0.0.1:50051"

# Two dummy scalar joints — the "hardware" this mock pretends to be.
ACTION_KEYS = ("joint_0.pos", "joint_1.pos")


def _scalar_feature_info(key: str) -> device_pb2.OneFeatureInfo:
    """A CRITICAL float32 scalar feature (the shape SO-101 uses per joint)."""
    return device_pb2.OneFeatureInfo(
        key=key,
        criticality=device_pb2.Criticality.CRITICAL,
        type=device_pb2.DataType.FLOAT32,
        shape=device_pb2.ImageShape(H=1, W=1, C=1),
        encoding=device_pb2.Encoding.RAW,
    )


class MockFollowerServicer(FollowerServicer):
    """A no-hardware follower: observations are a live counter, feedback echoes actions."""

    def __init__(self):
        self._ft_info = {k: _scalar_feature_info(k) for k in ACTION_KEYS}
        self._calls = 0
        self._last_action: dict[str, float] = {k: 0.0 for k in ACTION_KEYS}
        self._lock = threading.Lock()

    # --- feature introspection: return the dict's values as the gRPC stream ---
    def GetObservationFeatureInfo(self, request, context):
        return iter(self._ft_info.values())

    def GetActionFeatureInfo(self, request, context):
        return iter(self._ft_info.values())

    def GetFeedbackFeatureInfo(self, request, context):
        return iter(self._ft_info.values())

    # --- lifecycle -----------------------------------------------------------
    def Connect(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        return Empty()

    # --- data flow -----------------------------------------------------------
    def GetObservation(self, request, context):
        with self._lock:
            self._calls += 1
            obs = {
                "joint_0.pos": float(self._calls),
                "joint_1.pos": float(self._calls) * 2.0,
            }
        # NOTE: encode_feature needs a dict[str, OneFeatureInfo], NOT a generator.
        return encode_feature(self._ft_info, obs)

    def SendAction(self, request_iterator, context):
        action: dict[str, float] = {}
        for feat in request_iterator:
            load_feature(feat, self._ft_info, action)
        with self._lock:
            self._last_action = action
        return Empty()

    def GetFeedback(self, request, context):
        with self._lock:
            return encode_feature(self._ft_info, dict(self._last_action))

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)


def main() -> None:
    server = FollowerServer(FollowerServerConfig(address=ADDRESS), MockFollowerServicer())
    server.start()
    log.warning("mock follower server listening on %s", ADDRESS)
    try:
        client = GRPCFollower(GRPCFollowerConfig(address=ADDRESS, need_warmup=False))
        client.connect(calibrate=False)

        print("\n=== feature schemas negotiated from the server ===")
        print("observation_features:", client.observation_features)
        print("action_features:     ", client.action_features)

        print("\n=== get_observation() x2 (values should climb: 1,2 then 2,4) ===")
        obs1 = client.get_observation()
        obs2 = client.get_observation()
        print("obs #1:", obs1)
        print("obs #2:", obs2)
        assert obs1["joint_0.pos"] == 1.0 and obs1["joint_1.pos"] == 2.0, obs1
        assert obs2["joint_0.pos"] == 2.0 and obs2["joint_1.pos"] == 4.0, obs2

        print("\n=== send_action() then get_feedback() (server echoes action) ===")
        sent = client.send_action({"joint_0.pos": 1.5, "joint_1.pos": -2.5})
        fb = client.get_feedback()
        print("sent      :", sent)
        print("feedback  :", fb)
        assert abs(fb["joint_0.pos"] - 1.5) < 1e-5, fb
        assert abs(fb["joint_1.pos"] - (-2.5)) < 1e-5, fb

        client.disconnect()
        print("\nLOOPBACK_PASS  — gRPC plumbing (proto/negotiation/streaming) OK")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
