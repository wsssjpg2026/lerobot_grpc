"""Launch the bundled Galaxy General Galbot S1 MuJoCo follower.

The selected arm consumes a namespaced 6D pose delta plus Pika gripper
distance.  Every other S1 actuator is held at its configured home target.
The MuJoCo viewer is enabled by default; pass ``--headless`` in CI or on a
machine without a display.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from lerobot_robot_grpc.follower.follower_server import (
    FollowerServer,
    FollowerServerConfig,
)
from lerobot_robot_grpc.follower.mujoco_s1_follower_server import (
    MuJoCoS1Servicer,
)

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_XML = _PKG_ROOT / "assets" / "s1" / "mjcf" / "galbot_s1.xml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo Galbot S1 pose-delta follower server"
    )
    parser.add_argument(
        "--xml-path",
        default=str(_DEFAULT_XML),
        help=f"S1 MJCF path (default: {_DEFAULT_XML})",
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right"),
        default="left",
        help="Arm controlled by the namespaced pose action (default: left)",
    )
    parser.add_argument(
        "--address", default="0.0.0.0:5555", help="gRPC bind address"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the MuJoCo viewer (viewer is on by default)",
    )
    parser.add_argument(
        "--reset-on-connect",
        action="store_true",
        help="Reset to the safe home on every client Connect; default preserves pose",
    )
    parser.add_argument(
        "--rot-weight",
        type=float,
        default=0.3,
        help="DLS orientation weight (default: 0.3)",
    )
    parser.add_argument(
        "--ctrl-smoothing-alpha",
        type=float,
        default=0.20,
        help="Per-physics-loop ctrl EMA alpha (default: 0.20)",
    )
    parser.add_argument(
        "--max-dq-deg",
        type=float,
        default=6.0,
        help="DLS iteration joint-step limit in degrees (default: 6.0)",
    )
    parser.add_argument(
        "--max-dq-frame-deg",
        type=float,
        default=2.291831,
        help="30Hz joint-step cap: 1.2rad/s = 2.292deg/frame (default)",
    )
    parser.add_argument(
        "--collision-margin-mm",
        type=float,
        default=5.0,
        help="Independent collision candidate clearance in mm (default: 5)",
    )
    parser.add_argument(
        "--stale-timeout-s",
        type=float,
        default=0.5,
        help="Atomically hold the selected arm and gripper after this command gap (default: 0.5)",
    )
    parser.add_argument(
        "--torso-home-m",
        type=float,
        default=0.6,
        help="Initial fixed torso lift in metres, within [0, 0.735] (default: 0.6)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    servicer = MuJoCoS1Servicer(
        xml_path=args.xml_path,
        arm=args.arm,
        render=not args.headless,
        reset_on_connect=args.reset_on_connect,
        rot_weight=args.rot_weight,
        ctrl_smoothing_alpha=args.ctrl_smoothing_alpha,
        max_dq_deg=args.max_dq_deg,
        max_dq_frame_deg=args.max_dq_frame_deg,
        collision_margin_m=args.collision_margin_mm / 1000.0,
        stale_timeout_s=args.stale_timeout_s,
        torso_home_m=args.torso_home_m,
    )
    server = FollowerServer(FollowerServerConfig(address=args.address), servicer)
    server.start()
    logging.info(
        "MuJoCo S1 follower ready: address=%s arm=%s viewer=%s",
        args.address,
        args.arm,
        not args.headless,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping MuJoCo S1 follower...")
        server.stop()


if __name__ == "__main__":
    main()
