"""Launch a gRPC leader server wrapping a real SO-101 leader arm.

Run this on the machine that has the leader (teleop) arm plugged in. A remote
(or local) recording/teleop machine then reads operator input from it with:

    lerobot-teleoperate \
        --robot.type=grpc_follower --robot.address=<follower_host>:5555 \
        --teleop.type=grpc_leader --teleop.address=<this_host>:5556

The arm MUST be calibrated first (same id as used at calibration time):

    lerobot-calibrate --teleop.type=so101_leader --teleop.port=<port> --teleop.id=leader

See docs/extending.md §9 for the general launch pattern.
"""
import argparse
import logging
import time

from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot_robot_grpc.leader.leader_server import LeaderServer, LeaderServerConfig
from lerobot_robot_grpc.leader.so101_leader_server import (
    SO101LeaderAdapted,
    SO101LeaderServicer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="COM3", help="Serial port of the leader arm (e.g. COM3, /dev/ttyACM1)")
    parser.add_argument("--id", default="leader", help="Teleop id — must match the id used for calibration")
    parser.add_argument("--address", default="0.0.0.0:5556", help="gRPC listen address")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    robot = SO101LeaderAdapted(SO101LeaderConfig(port=args.port, id=args.id))
    servicer = SO101LeaderServicer(robot)
    server = LeaderServer(LeaderServerConfig(address=args.address), servicer)
    server.start()
    logging.info(
        "SO-101 leader server ready: address=%s serial=%s id=%s "
        "(arm connects on first client Connect RPC; calibration via Calibrate/CalibrateDone RPCs)",
        args.address, args.port, args.id,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping leader server...")
        server.stop()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
