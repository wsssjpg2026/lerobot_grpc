"""Teleop stats demo — live bandwidth/latency monitor, no hardware.

Starts the shared mock follower (examples/mock_follower.py: two dummy
joints + a synthetic H.264 camera stream), connects the real `GRPCFollower`
with a `TeleopStats` hooked in, and runs a fake "teleop session"
(get_observation + send_action at ~50 Hz) for `--seconds`, showing:

  1. a live multi-line block repainted in place every `--interval` seconds —
     total and per-feature bandwidth, fps, latency avg/p95/max, action RTT;
  2. a final summary table when the session ends.

Copy the `stats`/`monitor` wiring in `main()` into a real leader-follower
teleop loop — nothing else in the client API changes.

Run from the repo root (env with av + this package installed editable):

    python examples/teleop_monitor_demo.py --seconds 10 --interval 1.0
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.mock_follower import ACTION_KEYS, MockFollowerServicer

from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.follower_server import FollowerServer, FollowerServerConfig
from lerobot_robot_grpc.follower.utils import TeleopMonitor, TeleopStats

ADDRESS = "127.0.0.1:50059"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0, help="session length (s)")
    parser.add_argument("--interval", type=float, default=1.0, help="live refresh interval (s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    server = FollowerServer(FollowerServerConfig(address=ADDRESS, server_grace_period_s=1.0), MockFollowerServicer())
    server.start()

    # The two pieces to copy into a real teleop loop:
    stats = TeleopStats()
    monitor = TeleopMonitor(stats, interval=args.interval)
    client = GRPCFollower(GRPCFollowerConfig(address=ADDRESS, need_warmup=False), stats=stats)
    try:
        client.connect(calibrate=True)
        monitor.start()
        t0 = time.monotonic()
        print("teleop session running (joints + H.264 camera stream)...")
        while time.monotonic() - t0 < args.seconds:
            obs = client.get_observation()
            client.send_action({k: float(obs[k]) for k in ACTION_KEYS})
            time.sleep(0.02)
    finally:
        monitor.stop()  # clears the live line and prints the session summary
        client.disconnect()
        server.stop()


if __name__ == "__main__":
    main()
