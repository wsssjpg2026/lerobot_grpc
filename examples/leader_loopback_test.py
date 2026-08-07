"""Loopback test — exercise the gRPC leader path with NO hardware.

Runs a mock ``LeaderServicer`` in-process (it produces dummy scalar actions,
records SetReference calls), then connects the real ``GRPCLeader`` client to it
and round-trips action / set_reference. This validates the leader side of the
gRPC stack end-to-end.

Run from the repo root:

    python examples/leader_loopback_test.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.mock_leader import ACTION_KEYS, MockLeaderServicer

from lerobot_robot_grpc.leader.grpc_leader import GRPCLeader
from lerobot_robot_grpc.leader.config_grpc import GRPCLeaderConfig
from lerobot_robot_grpc.leader.leader_server import LeaderServer, LeaderServerConfig

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("leader_loopback")

ADDRESS = "127.0.0.1:50052"


def main() -> None:
    servicer = MockLeaderServicer()
    server = LeaderServer(LeaderServerConfig(address=ADDRESS), servicer)
    server.start()
    log.warning("mock leader server listening on %s", ADDRESS)
    try:
        client = GRPCLeader(GRPCLeaderConfig(address=ADDRESS, need_warmup=False))
        client.connect(calibrate=False)

        print("\n=== feature schemas negotiated from the server ===")
        print("action_features:", client.action_features)
        assert "joint_0.pos" in client.action_features

        print("\n=== get_action() — persistent stream ===")
        act1 = client.get_action()
        print("act #1:", {k: round(act1[k], 3) for k in ACTION_KEYS})

        time.sleep(0.5)
        act2 = client.get_action()
        print("act #2:", {k: round(act2[k], 3) for k in ACTION_KEYS})
        assert act2["joint_0.pos"] >= act1["joint_0.pos"], "action stream did not advance"

        print("\n=== set_reference() — lock current pose as delta origin ===")
        client.set_reference()
        assert servicer.set_reference_called, "SetReference was not recorded by the mock"

        time.sleep(0.3)
        act3 = client.get_action()
        print("act #3 (post-reference):", {k: round(act3[k], 3) for k in ACTION_KEYS})
        # After SetReference, action should be relative to the reference timestamp,
        # so joint_0.pos should be small (time elapsed since reference, not since t0).
        assert act3["joint_0.pos"] < act2["joint_0.pos"], "action should reset after SetReference"

        client.disconnect()
        print("\nLEADER_LOOPBACK_PASS  — gRPC leader plumbing (proto/negotiation/streaming + SetReference) OK")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
