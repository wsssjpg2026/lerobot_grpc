"""Launch a Pika Sense leader server for delta-pose teleoperation.

Connects to the Pika Sense hardware (Vive Tracker + gripper sensor), wraps it
in a :class:`PikaSenseServicer`, and starts a gRPC ``LeaderServer`` that
streams 8-FLOAT32 pose-delta actions.

Usage::

    # Basic — identity axis mapping, latch mode
    python serve_pika_sense_leader.py --port /dev/ttyUSB0 --address 0.0.0.0:5556

    # With interactive direction calibration (determines lighthouse→base axes)
    python serve_pika_sense_leader.py --port /dev/ttyUSB0 --calibrate

    # Continuous (velocity) mode
    python serve_pika_sense_leader.py --port /dev/ttyUSB0 --reference-mode continuous

The produced actions are consumed by a ``MuJoCoSO101Servicer`` (or real
``SO101FollowerServicer``) in ``pose_delta`` action mode.
"""

import argparse
import logging
import time

import numpy as np

from lerobot_robot_grpc.leader.leader_server import (
    LeaderServer,
    LeaderServerConfig,
)
from lerobot_robot_grpc.leader.pika_sense_leader_server import PikaSenseServicer


def calibrate_direction(device, tracker_device: str) -> np.ndarray:
    """Interactive direction calibration to determine ``R_lh2base``.

    Prompts the operator to move the tracker in two known directions
    (forward, right) relative to the robot.  From the observed lighthouse-
    frame displacement vectors, constructs the 3×3 rotation matrix that maps
    lighthouse axes to robot-base axes (X=forward, Y=right, Z=up).

    Returns ``np.eye(3)`` if the operator skips calibration.
    """
    def _capture_delta(prompt_ready: str, prompt_move: str) -> np.ndarray:
        input(prompt_ready)
        pose0 = device.get_pose(tracker_device)
        if pose0 is None:
            raise RuntimeError("No tracker data — ensure the tracker is tracked.")
        input(prompt_move)
        pose1 = device.get_pose(tracker_device)
        if pose1 is None:
            raise RuntimeError("Tracker lost during calibration.")
        delta = np.array(pose1.position) - np.array(pose0.position)
        norm = np.linalg.norm(delta)
        if norm < 0.005:  # 5 mm minimum movement
            raise RuntimeError(
                f"Movement too small ({norm*1000:.1f} mm). Move at least ~10 cm."
            )
        return delta / norm

    print("\n=== Direction Calibration ===")
    print("This determines how lighthouse axes map to robot-base axes.\n")

    forward_lh = _capture_delta(
        "Hold the tracker steady at a starting position. Press Enter when ready.",
        "Now move the tracker FORWARD (~10 cm, away from you). Press Enter when done.",
    )
    right_lh = _capture_delta(
        "Return to a neutral position. Press Enter when ready.",
        "Now move the tracker RIGHT (~10 cm). Press Enter when done.",
    )

    # Robot base frame: X=forward, Y=right, Z=up
    # R_lh2base maps lighthouse column vectors to base column vectors.
    # Columns of R_lh2base are the lighthouse-frame directions of the base axes.
    # forward_lh is where base-X points in lighthouse frame → first column.
    # right_lh   is where base-Y points in lighthouse frame → second column.
    # up         = forward × right (right-handed) → third column.
    up_lh = np.cross(forward_lh, right_lh)
    up_lh /= np.linalg.norm(up_lh)

    # Re-orthogonalise right against the computed up.
    right_lh = np.cross(up_lh, forward_lh)

    R_lh2base = np.column_stack([forward_lh, right_lh, up_lh])

    # Verify it's a valid rotation (det ≈ +1).
    det = np.linalg.det(R_lh2base)
    if abs(det - 1.0) > 0.01:
        raise RuntimeError(f"Calibration produced invalid rotation (det={det:.4f}).")

    print(f"\nCalibration complete. R_lh2base =\n{np.round(R_lh2base, 4)}\n")
    return R_lh2base


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
        "--reference-mode",
        choices=["latch", "continuous"],
        default="latch",
        help="Delta reference: 'latch' (position control, default) or 'continuous' (velocity control)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor (0-1, default 0.25)",
    )
    parser.add_argument(
        "--dead-zone-mm",
        type=float,
        default=2.0,
        help="Position dead-zone threshold in mm (default 2.0)",
    )
    parser.add_argument(
        "--pos-gain",
        type=float,
        default=1.0,
        help="Position delta gain factor (default 1.0 = 1:1; tune for SO-101 workspace)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run interactive direction calibration before starting the server",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    # --- Optional direction calibration (#05) ---
    R_lh2base = np.eye(3)
    if args.calibrate:
        from pika.sense import Sense

        cal_device = Sense(args.port)
        cal_device.connect()
        cal_device.get_vive_tracker()
        time.sleep(2)
        devices = cal_device.get_tracker_devices()
        if not devices:
            raise RuntimeError("No Vive Tracker devices found for calibration.")
        R_lh2base = calibrate_direction(cal_device, devices[0])
        cal_device.disconnect()

    # --- Start server ---
    servicer = PikaSenseServicer(
        port=args.port,
        reference_mode=args.reference_mode,
        ema_alpha=args.ema_alpha,
        dead_zone_mm=args.dead_zone_mm,
        pos_gain=args.pos_gain,
        R_lh2base=R_lh2base,
    )
    server = LeaderServer(LeaderServerConfig(address=args.address), servicer)
    server.start()
    logging.info(
        "Pika Sense leader server ready: address=%s mode=%s calibrate=%s",
        args.address,
        args.reference_mode,
        args.calibrate,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping Pika Sense leader server...")
        server.stop()


if __name__ == "__main__":
    main()
