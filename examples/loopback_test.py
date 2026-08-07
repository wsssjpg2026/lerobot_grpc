"""Loopback test — exercise the gRPC follower path with NO hardware.

Runs a mock `FollowerServicer` in-process (it produces dummy scalar observations,
a synthetic H.264-encoded camera stream, and echoes received actions as
feedback), then connects the real `GRPCFollower` client to it and round-trips
observation / action / feedback. This validates the whole stack the unit tests
don't: proto encode/decode, feature-schema negotiation over GetInfo, streaming
RPCs (including the persistent observation stream + H.264 inter-frame codec
round-trip), and the client class end-to-end.

Run from the repo root (with the env that has lerobot[grpcio-dep] + this package):

    python examples/loopback_test.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.mock_follower import ACTION_KEYS, CAM_HEIGHT, CAM_WIDTH, MockFollowerServicer

from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.follower_server import FollowerServer, FollowerServerConfig

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("loopback")

ADDRESS = "127.0.0.1:50051"


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
        assert client.observation_features["cam"] == (CAM_HEIGHT, CAM_WIDTH, 3)
        assert "cam" in client.cameras

        print("\n=== get_observation() — persistent stream + H.264 camera ===")
        obs1 = client.get_observation()
        print("obs #1 joints:", {k: round(obs1[k], 3) for k in ACTION_KEYS})
        cam1 = obs1["cam"]
        print("cam #1:", cam1.shape, cam1.dtype)
        assert cam1.shape == (CAM_HEIGHT, CAM_WIDTH, 3) and cam1.dtype == np.uint8
        assert cam1.max() > 200, "expected the white square to be visible"
        assert abs(int(cam1[10, 300, 1]) - 180) < 25, "expected green background"

        obs2 = client.get_observation()  # may be the same snapshot — stream is async
        assert obs2["joint_0.pos"] >= obs1["joint_0.pos"]

        print("\n=== stream liveness: joint values must climb with wall clock ===")
        time.sleep(1.2)
        obs3 = client.get_observation()
        print("obs #3 joints:", {k: round(obs3[k], 3) for k in ACTION_KEYS})
        assert obs3["joint_0.pos"] >= obs1["joint_0.pos"] + 1.0, "observation stream did not advance"
        assert obs3["joint_1.pos"] == 2.0 * obs3["joint_0.pos"]

        print("\n=== send_action() then get_feedback() (server echoes action) ===")
        sent = client.send_action({"joint_0.pos": 1.5, "joint_1.pos": -2.5})
        fb = client.get_feedback()
        print("sent      :", sent)
        print("feedback  :", fb)
        assert abs(fb["joint_0.pos"] - 1.5) < 1e-5, fb
        assert abs(fb["joint_1.pos"] - (-2.5)) < 1e-5, fb

        client.disconnect()
        print("\nLOOPBACK_PASS  — gRPC plumbing (proto/negotiation/streaming + H.264) OK")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
