"""One-shot SO-101 teleoperation: leader (COM7) drives follower (COM5).

Starts both gRPC servers in-process (follower server wrapping the follower arm
on `--follower-port`, leader server wrapping the leader arm on `--leader-port`),
connects the GRPCFollower / GRPCLeader clients, then streams leader joint
actions to the follower at `--rate` Hz. Ctrl+C to stop and clean up.

Windows / PowerShell:

    python examples/teleop_so101.py
    python examples/teleop_so101.py --follower-port=COM5 --leader-port=COM7 `
        --cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
        --teleop-stats --time 60

Bare joint-only teleop (no camera):

    python examples/teleop_so101.py --follower-port=COM5 --leader-port=COM7
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import draccus

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 -- register for draccus
try:
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401 -- register for draccus
except ImportError:
    # pyrealsense2 not installed — RealSense cameras unavailable, but OpenCV cameras still work.
    RealSenseCameraConfig = None
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.follower_server import FollowerServer, FollowerServerConfig
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.follower.so101_follower_server import (
    SO101FollowerAdapted,
    SO101FollowerServicer,
)
from lerobot_robot_grpc.leader.config_grpc import GRPCLeaderConfig
from lerobot_robot_grpc.leader.grpc_leader import GRPCLeader
from lerobot_robot_grpc.leader.leader_server import LeaderServer, LeaderServerConfig
from lerobot_robot_grpc.leader.so101_leader_server import (
    SO101LeaderAdapted,
    SO101LeaderServicer,
)


def parse_cameras(cameras_yaml: str | None) -> dict[str, CameraConfig]:
    if not cameras_yaml:
        return {}
    cameras = draccus.loads(dict[str, CameraConfig], cameras_yaml)
    if not cameras:
        raise ValueError("--cameras parsed to an empty dict; provide at least one camera.")
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-port", default="COM5", help="serial port of the follower arm")
    parser.add_argument("--leader-port", default="COM7", help="serial port of the leader arm")
    parser.add_argument(
        "--follower-address", default="0.0.0.0:5555", help="follower gRPC server address"
    )
    parser.add_argument(
        "--leader-address", default="0.0.0.0:5556", help="leader gRPC server address"
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="inline YAML dict of follower cameras, e.g. "
        "'{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}'",
    )
    parser.add_argument("--rate", type=float, default=50.0, help="control loop rate (Hz)")
    parser.add_argument("--time", type=float, default=None, help="teleop duration (s); omit for unlimited")
    parser.add_argument(
        "--calibrate", action=argparse.BooleanOptionalAction, default=True,
        help="run calibration if the arms need it on connect",
    )
    parser.add_argument(
        "--teleop-stats", action="store_true",
        help="live bandwidth/latency monitor + end-of-session summary (follower)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger("teleop_so101")

    follower_robot = SO101FollowerAdapted(
        SOFollowerRobotConfig(port=args.follower_port, id="follower", cameras=parse_cameras(args.cameras))
    )
    leader_robot = SO101LeaderAdapted(SO101LeaderConfig(port=args.leader_port, id="leader"))
    follower_server = FollowerServer(
        FollowerServerConfig(address=args.follower_address),
        SO101FollowerServicer(follower_robot, camera_encoding="h264"),
    )
    leader_server = LeaderServer(
        LeaderServerConfig(address=args.leader_address),
        SO101LeaderServicer(leader_robot),
    )
    follower_server.start()
    leader_server.start()
    logger.info(
        "Servers ready: follower=%s (COM %s) leader=%s (COM %s)",
        args.follower_address, args.follower_port, args.leader_address, args.leader_port,
    )

    follower = GRPCFollower(
        GRPCFollowerConfig(
            address="127.0.0.1:" + args.follower_address.rsplit(":", 1)[-1],
            id="follower",
            teleop_stats=args.teleop_stats,
        )
    )
    leader = GRPCLeader(
        GRPCLeaderConfig(address="127.0.0.1:" + args.leader_address.rsplit(":", 1)[-1], id="leader")
    )

    stop = False
    period = 1.0 / args.rate
    last = time.monotonic()
    try:
        logger.info("Connecting follower (may calibrate; move joints through full range when asked)...")
        follower.connect(calibrate=args.calibrate)
        logger.info("Connecting leader (may calibrate; move joints through full range when asked)...")
        leader.connect(calibrate=args.calibrate)

        action_keys = sorted(leader.action_features)
        logger.info(
            "Teleoperating: leader -> follower @ %.1f Hz, joints=%s. Ctrl+C to stop.",
            args.rate, action_keys,
        )
        t_start = time.monotonic()
        n = 0
        while not stop:
            action = leader.get_action()
            follower.send_action(action)
            n += 1
            if n % args.rate == 0:
                pos = ", ".join(f"{k}={action[k]:.1f}" for k in action_keys)
                logger.info("[%.0fs] %s", time.monotonic() - t_start, pos)
            if args.time is not None and time.monotonic() - t_start >= args.time:
                break
            next_t = last + period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            last = time.monotonic()
    except KeyboardInterrupt:
        logger.info("Stopping teleoperation...")
    finally:
        try:
            leader.disconnect()
        except Exception:
            pass
        try:
            follower.disconnect()
        except Exception:
            pass
        follower_server.stop()
        leader_server.stop()
        if follower_robot.is_connected:
            follower_robot.disconnect()
        if leader_robot.is_connected:
            leader_robot.disconnect()


if __name__ == "__main__":
    main()
