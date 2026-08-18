"""Launch a MuJoCo-backed SO-101 follower server.

Supports two action modes:

``pose_delta`` (default) — receives end-effector pose deltas (8 FLOAT32)
  from a delta-pose leader (e.g. Pika Sense), converts them to joint targets
  via DLS IK (MuJoCo Jacobian), and drives the MuJoCo simulation.

``joint`` — accepts joint-space actions directly (same contract as the real
  ``SO101FollowerServicer``, minus the hardware).

Usage::

    # pose_delta mode (full delta-pose → DLS IK pipeline)
    python serve_mujoco_follower.py \\
        --xml-path assets/so101/scene.xml \\
        --action-mode pose_delta \\
        --render

    # joint mode (simplest — just drives MuJoCo with joint angles)
    python serve_mujoco_follower.py \\
        --xml-path assets/so101/scene.xml \\
        --action-mode joint \\
        --render

The server is transparent to gRPC clients: ``teleop_so101.py``,
``loopback_test.py``, and data-collection scripts connect without modification.
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot_robot_grpc.follower.follower_server import (
    FollowerServer,
    FollowerServerConfig,
)
from lerobot_robot_grpc.follower.mujoco_follower_server import (
    MuJoCoSO101Servicer,
)

# Resolve the default asset paths relative to the package root so the script
# works from any CWD when the repo is on sys.path.
_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_XML = _PKG_ROOT / "assets" / "so101" / "scene.xml"


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo SO-101 follower server (sim-backed, no hardware)"
    )
    parser.add_argument(
        "--xml-path",
        default=str(_DEFAULT_XML),
        help=f"Path to MuJoCo scene XML (default: {_DEFAULT_XML})",
    )
    parser.add_argument(
        "--action-mode",
        choices=["joint", "pose_delta"],
        default="pose_delta",
        help="Action mode: 'pose_delta' (DLS IK) or 'joint' (direct joint control)",
    )
    parser.add_argument("--address", default="0.0.0.0:5555", help="gRPC bind address")
    parser.add_argument("--render", action="store_true", help="Launch MuJoCo viewer window")
    parser.add_argument(
        "--rot-weight",
        type=float,
        default=0.3,
        help="DLS rotation weight — lower prioritises position (default: 0.3 ≈ 3:1 pos:rot; "
        "yaw on 5-DOF arm needs ≥0.2 for visible shoulder_pan response)",
    )
    parser.add_argument(
        "--gripper-max-distance-mm",
        type=float,
        default=60.0,
        help="Gripper full-open distance in mm (for distance→0-100 mapping)",
    )
    parser.add_argument(
        "--home-joints",
        default="0,-20,60,-40,0",
        help="Comma-separated home joint angles in degrees for the 5 body joints "
        "(shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll). "
        "Bent elbow avoids the full-extension singularity (default: 0,-20,60,-40,0)",
    )
    parser.add_argument(
        "--ctrl-smoothing-alpha",
        type=float,
        default=0.20,
        help="Per-physics-loop ctrl EMA alpha — lower = smoother but laggier "
        "(default: 0.20; 1.0 = no smoothing, step-function ctrl)",
    )
    parser.add_argument(
        "--max-dq-deg",
        type=float,
        default=6.0,
        help="DLS per-iteration per-joint step clip in degrees (default 6)",
    )
    parser.add_argument(
        "--max-dq-frame-deg",
        type=float,
        default=6.7,
        help="Per-joint per-frame cap on the published step — the official "
        ">30° 200Hz-interpolation equivalent (default 6.7 = 200°/s at 30 Hz)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    home_joints_deg = tuple(float(x) for x in args.home_joints.split(","))

    servicer = MuJoCoSO101Servicer(
        xml_path=args.xml_path,
        action_mode=args.action_mode,
        render=args.render,
        rot_weight=args.rot_weight,
        gripper_max_distance_mm=args.gripper_max_distance_mm,
        home_joints_deg=home_joints_deg,
        ctrl_smoothing_alpha=args.ctrl_smoothing_alpha,
        max_dq_deg=args.max_dq_deg,
        max_dq_frame_deg=args.max_dq_frame_deg,
    )
    server = FollowerServer(FollowerServerConfig(address=args.address), servicer)
    server.start()
    logging.info(
        "MuJoCo SO-101 follower server ready: address=%s action_mode=%s render=%s",
        args.address,
        args.action_mode,
        args.render,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping MuJoCo follower server...")
        server.stop()


if __name__ == "__main__":
    main()
