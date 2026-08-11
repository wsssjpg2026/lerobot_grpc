"""Launch a MuJoCo-backed SO-101 follower server.

Supports two action modes:

``pose_delta`` (default) — receives end-effector pose deltas (8 FLOAT32)
  from a delta-pose leader (e.g. Pika Sense), converts them to joint targets
  via FK + IK, and drives the MuJoCo simulation.  Requires a URDF for the IK
  solver.

``joint`` — accepts joint-space actions directly (same contract as the real
  ``SO101FollowerServicer``, minus the hardware).

Usage::

    # pose_delta mode (full delta-pose → IK pipeline)
    python serve_mujoco_follower.py \\
        --xml-path assets/so101/scene.xml \\
        --urdf-path assets/so101/so101_new_calib.urdf \\
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
_DEFAULT_URDF = _PKG_ROOT / "assets" / "so101" / "so101_new_calib.urdf"


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
        "--urdf-path",
        default=str(_DEFAULT_URDF),
        help=f"Path to URDF for IK solver (default: {_DEFAULT_URDF})",
    )
    parser.add_argument(
        "--action-mode",
        choices=["joint", "pose_delta"],
        default="pose_delta",
        help="Action mode: 'pose_delta' (FK+IK) or 'joint' (direct joint control)",
    )
    parser.add_argument("--address", default="0.0.0.0:5555", help="gRPC bind address")
    parser.add_argument("--render", action="store_true", help="Launch MuJoCo viewer window")
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=0.01,
        help="IK orientation weight (0.01 = soft for 5-DOF under-actuation)",
    )
    parser.add_argument(
        "--gripper-max-distance-mm",
        type=float,
        default=60.0,
        help="Gripper full-open distance in mm (for distance→0-100 mapping)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    servicer = MuJoCoSO101Servicer(
        xml_path=args.xml_path,
        urdf_path=args.urdf_path,
        action_mode=args.action_mode,
        render=args.render,
        orientation_weight=args.orientation_weight,
        gripper_max_distance_mm=args.gripper_max_distance_mm,
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
