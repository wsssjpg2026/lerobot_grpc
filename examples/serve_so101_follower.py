"""Launch a gRPC follower server wrapping a real SO-101 follower arm.

Uses standard lerobot config syntax (including --robot.cameras.*):

    python serve_so101_follower.py \\
        --robot.port=COM6 --robot.id=follower \\
        --robot.cameras.top.type=opencv \\
        --robot.cameras.top.index_or_path=0 \\
        --robot.cameras.top.width=640 --robot.cameras.top.height=480 \\
        --robot.cameras.top.fps=30 \\
        --address=localhost:5555

The arm connects on first client Connect RPC. Calibration via
Calibrate/CalibrateDone RPCs (use lerobot-calibrate --robot.type=grpc_follower ...).
"""
import logging
import time
from dataclasses import dataclass

import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 -- register for draccus
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot_robot_grpc.follower.follower_server import FollowerServer, FollowerServerConfig
from lerobot_robot_grpc.follower.so101_follower_server import (
    SO101FollowerAdapted,
    SO101FollowerServicer,
)


@dataclass
class ServeFollowerConfig:
    robot: SOFollowerRobotConfig
    address: str = "0.0.0.0:5555"


@draccus.wrap()
def main(cfg: ServeFollowerConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    robot = SO101FollowerAdapted(cfg.robot)
    servicer = SO101FollowerServicer(robot)
    server = FollowerServer(FollowerServerConfig(address=cfg.address), servicer)
    server.start()
    cam_names = list(cfg.robot.cameras.keys())
    logging.info(
        "SO-101 follower server ready: address=%s serial=%s id=%s cameras=%s",
        cfg.address, cfg.robot.port, cfg.robot.id, cam_names,
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
