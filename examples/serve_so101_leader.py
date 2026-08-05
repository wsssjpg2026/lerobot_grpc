"""Launch a gRPC leader server wrapping a real SO-101 leader arm.

Uses standard lerobot config syntax:

    python serve_so101_leader.py \\
        --robot.port=COM4 --robot.id=leader \\
        --address=localhost:5556

The arm connects on first client Connect RPC. Calibration via
Calibrate/CalibrateDone RPCs (use lerobot-calibrate --teleop.type=grpc_leader ...).
"""
import logging
import time
from dataclasses import dataclass

import draccus

from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot_robot_grpc.leader.leader_server import LeaderServer, LeaderServerConfig
from lerobot_robot_grpc.leader.so101_leader_server import (
    SO101LeaderAdapted,
    SO101LeaderServicer,
)


@dataclass
class ServeLeaderConfig:
    robot: SO101LeaderConfig
    address: str = "0.0.0.0:5556"


@draccus.wrap()
def main(cfg: ServeLeaderConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    robot = SO101LeaderAdapted(cfg.robot)
    servicer = SO101LeaderServicer(robot)
    server = LeaderServer(LeaderServerConfig(address=cfg.address), servicer)
    server.start()
    logging.info(
        "SO-101 leader server ready: address=%s serial=%s id=%s",
        cfg.address, cfg.robot.port, cfg.robot.id,
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
