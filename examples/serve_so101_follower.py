"""Launch a gRPC follower server wrapping a real SO-101 follower arm.

Run this on the machine that has the follower arm plugged in. A remote (or
local) recording/teleop machine then drives it with:

    lerobot-teleoperate \
        --robot.type=grpc_follower --robot.address=<this_host>:5555 \
        --teleop.type=so101_leader --teleop.port=<leader_port>

The arm MUST be calibrated first (same id as used at calibration time):

    lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=follower

See docs/extending.md §9 for the general launch pattern.
"""
import argparse
import logging
import time

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot_robot_grpc.follower.follower_server import FollowerServer, FollowerServerConfig
from lerobot_robot_grpc.follower.so101_follower_server import (
    SO101FollowerAdapted,
    SO101FollowerServicer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="COM6", help="Serial port of the follower arm (e.g. COM6, /dev/ttyACM0)")
    parser.add_argument("--id", default="follower", help="Robot id — must match the id used for calibration")
    parser.add_argument("--address", default="0.0.0.0:5555", help="gRPC listen address")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    robot = SO101FollowerAdapted(SOFollowerRobotConfig(port=args.port, id=args.id))
    servicer = SO101FollowerServicer(robot)
    server = FollowerServer(FollowerServerConfig(address=args.address), servicer)
    server.start()
    logging.info(
        "SO-101 follower server ready: address=%s serial=%s id=%s "
        "(arm connects on first client Connect RPC; calibration via Calibrate/CalibrateDone RPCs)",
        args.address, args.port, args.id,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping follower server...")
        server.stop()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
