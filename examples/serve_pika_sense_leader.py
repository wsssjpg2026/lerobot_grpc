"""Launch a Pika Sense leader server for delta-pose teleoperation.

Connects to the Pika Sense hardware (Vive Tracker + gripper sensor), wraps it
in a :class:`PikaSenseServicer`, and starts a gRPC ``LeaderServer`` that
streams 8-FLOAT32 pose-delta actions.

Calibration (lighthouse→base axis alignment) is done via the servicer's
built-in 8-step StreamCalibration protocol — no ``--calibrate`` flag needed.

Usage::

    # Basic — loads calibration from file, or runs 8-step calibration on Connect
    python serve_pika_sense_leader.py --port /dev/ttyUSB0 --address 0.0.0.0:5556

The produced actions are consumed by a ``MuJoCoSO101Servicer`` (or real
``SO101FollowerServicer``) in ``pose_delta`` action mode.
"""

import argparse
import logging
import time

from lerobot_robot_grpc.leader.leader_server import (
    LeaderServer,
    LeaderServerConfig,
)
from lerobot_robot_grpc.leader.pika_sense_leader_server import PikaSenseServicer


def main():
    parser = argparse.ArgumentParser(
        description="Pika Sense leader server (delta-pose teleoperation)"
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port path for the Pika Sense device",
    )
    parser.add_argument(
        "--address",
        default="0.0.0.0:5556",
        help="gRPC bind address",
    )
    parser.add_argument(
        "--dead-zone-mm",
        type=float,
        default=2.0,
        help="Position dead-zone for published-offset updates when far from origin (default 2.0)",
    )
    parser.add_argument(
        "--dead-zone-deg",
        type=float,
        default=0.5,
        help="Rotation dead-zone for published-offset updates when far from origin (default 0.5)",
    )
    parser.add_argument(
        "--pos-gain",
        type=float,
        default=0.45,
        help="Hand-to-arm offset scale (default 0.45; 1.0 = 1:1)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.15,
        help="Pose EMA alpha when the tracker is still (default 0.15; 1.0 = off)",
    )
    parser.add_argument(
        "--ema-alpha-fast",
        type=float,
        default=0.70,
        help="Pose EMA alpha when the tracker is moving quickly (default 0.70)",
    )
    parser.add_argument(
        "--max-delta-mm",
        type=float,
        default=20.0,
        help="Max change of the published offset per frame in mm (default 20)",
    )
    parser.add_argument(
        "--home-capture-mm",
        type=float,
        default=40.0,
        help="Inside this radius of T_begin the offset dead-zone is disabled (default 40)",
    )
    parser.add_argument(
        "--jump-mm",
        type=float,
        default=50.0,
        help="Ignore a tracker sample that jumps farther than this in one frame (default 50)",
    )
    parser.add_argument(
        "--gripper-rate-mm-s",
        type=float,
        default=250.0,
        help="Max gripper distance change in mm/s after EMA (default 250)",
    )
    parser.add_argument(
        "--device-id",
        default="pika_sense",
        help="Device ID for the calibration file name (default: pika_sense)",
    )
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Directory for calibration files (default: ~/.cache/huggingface/lerobot/"
        "calibration/teleoperators/pika_sense/)",
    )
    parser.add_argument(
        "--auto-reference",
        action="store_true",
        help="Latch T_begin and engage on Connect, for clients without an alignment "
        "step (e.g. lerobot-teleoperate). NOTE: the clutch gesture still freezes "
        "the publish in this mode and nobody re-latches — resume by reconnecting.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    # --- Start server ---
    servicer = PikaSenseServicer(
        port=args.port,
        dead_zone_mm=args.dead_zone_mm,
        dead_zone_deg=args.dead_zone_deg,
        pos_gain=args.pos_gain,
        ema_alpha=args.ema_alpha,
        ema_alpha_fast=args.ema_alpha_fast,
        max_delta_mm=args.max_delta_mm,
        home_capture_mm=args.home_capture_mm,
        jump_mm=args.jump_mm,
        gripper_rate_mm_s=args.gripper_rate_mm_s,
        calibration_dir=args.calibration_dir,
        device_id=args.device_id,
        auto_reference=args.auto_reference,
    )
    server = LeaderServer(LeaderServerConfig(address=args.address), servicer)
    server.start()
    logging.info("Pika Sense leader server ready: address=%s", args.address)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping Pika Sense leader server...")
        server.stop()


if __name__ == "__main__":
    main()
