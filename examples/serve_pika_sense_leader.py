"""Launch a Pika Sense leader server for delta-pose teleoperation.

Connects to the Pika Sense hardware (Vive Tracker + gripper sensor), wraps it
in a :class:`PikaSenseServicer`, and starts a gRPC ``LeaderServer`` that
streams 8-FLOAT32 pose-delta actions.

Calibration persists the gripper travel range (plus the legacy
lighthouse→base rotation, unused by the teleop path) via the servicer's
built-in 8-step StreamCalibration protocol — no ``--calibrate`` flag needed.
Published deltas are raw 1:1 relative transforms (PikaAnyArm official
semantics; follower-side safety stack handles the rest).

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
        "step (e.g. lerobot-teleoperate).",
    )
    parser.add_argument(
        "--arm-prefix",
        choices=("left", "right"),
        default=None,
        help="Namespace pose actions for an S1 arm; omit for SO101 compatibility",
    )
    parser.add_argument(
        "--cumulative-clutch",
        action="store_true",
        help="Resume from the frozen target after clutch repositioning without "
        "requiring client SetReference calls (recommended for lerobot-teleoperate)",
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
        calibration_dir=args.calibration_dir,
        device_id=args.device_id,
        auto_reference=args.auto_reference,
        arm_prefix=args.arm_prefix,
        cumulative_clutch=args.cumulative_clutch,
    )
    server = LeaderServer(LeaderServerConfig(address=args.address), servicer)
    server.start()
    logging.info(
        "Pika Sense leader server ready: address=%s arm_prefix=%s cumulative_clutch=%s",
        args.address,
        args.arm_prefix,
        args.cumulative_clutch,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping Pika Sense leader server...")
        server.stop()


if __name__ == "__main__":
    main()
