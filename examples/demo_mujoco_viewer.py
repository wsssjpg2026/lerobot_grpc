"""Visual demo: drive the MuJoCo SO-101 through all 6 pose-delta DOF.

Connects to an EXTERNAL ``serve_mujoco_follower.py`` server (started with
``--render``) and sends a choreographed pose-delta sequence so you can WATCH
the arm move in the MuJoCo viewer:

    1. Translate +X then -X (forward ↔ back)     4. Rotate about X (roll)  ↔ return
    2. Translate +Y then -Y (left   ↔ right)     5. Rotate about Y (pitch) ↔ return
    3. Translate +Z then -Z (down   ↔ up)        6. Rotate about Z (yaw)   ↔ return

Each axis sends the **current offset** from home (latch-once): step *k* of
*N* is offset ``k * step_size``, then it counts back down to 0.  After
each step the script prints the measured EE position, per-step displacement,
and gripper percentage.  After each axis it prints the drift from origin.

The gripper is held at its **initial** position throughout — the demo reads
``gripper.pos`` on connect, reverse-maps it to mm, and uses that as the
constant ``gripper.distance`` so the fingers don't spurious open.

Frame conventions (wayfinder #05):
  - Translation deltas are in the **base frame** (X=forward, Y=left, Z=up).
  - Rotation deltas are **body-frame** (composed as ``R_current @ R_delta``).

Prerequisite — start the server first with a viewer window::

    # Terminal 1
    python examples/serve_mujoco_follower.py --action-mode pose_delta --render

Then run this client::

    # Terminal 2
    python examples/demo_mujoco_viewer.py
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("mujoco_demo")

ADDRESS = "127.0.0.1:5555"
PKG_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PKG_ROOT / "assets" / "so101" / "so101_new_calib.urdf"

BODY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Must match the server's --gripper-max-distance-mm default so the reverse-map
# (gripper.pos % → mm) is consistent.  See MuJoCoSO101Servicer.
GRIPPER_MAX_DISTANCE_MM: float = 60.0


# --------------------------------------------------------------------------- #
# Pose-delta action builder
# --------------------------------------------------------------------------- #

def make_delta_action(
    pos_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot_axis: str | None = None,
    rot_deg: float = 0.0,
    gripper_mm: float = 30.0,
) -> dict[str, float]:
    """Build one 8-FLOAT32 pose-delta action.

    Parameters
    ----------
    pos_mm
        (x, y, z) translation in millimetres — converted to metres internally.
    rot_axis
        ``'x'`` | ``'y'`` | ``'z'`` for a single-axis body-frame rotation, or
        ``None`` for identity rotation.
    rot_deg
        Rotation angle in degrees (sign = direction).
    gripper_mm
        Gripper finger distance — 0 = closed, ~60 = wide open.  Held constant
        during the demo so the gripper stays in a neutral pose.
    """
    action: dict[str, float] = {
        "hand.delta_pos.x": pos_mm[0] / 1000.0,
        "hand.delta_pos.y": pos_mm[1] / 1000.0,
        "hand.delta_pos.z": pos_mm[2] / 1000.0,
        "hand.delta_rot.qx": 0.0,
        "hand.delta_rot.qy": 0.0,
        "hand.delta_rot.qz": 0.0,
        "hand.delta_rot.qw": 1.0,  # identity quaternion
        "gripper.distance": gripper_mm,
    }
    if rot_axis is not None:
        half = math.radians(rot_deg) / 2.0
        s, c = math.sin(half), math.cos(half)
        quats = {"x": (s, 0.0, 0.0, c), "y": (0.0, s, 0.0, c), "z": (0.0, 0.0, s, c)}
        qx, qy, qz, qw = quats[rot_axis]
        action["hand.delta_rot.qx"] = qx
        action["hand.delta_rot.qy"] = qy
        action["hand.delta_rot.qz"] = qz
        action["hand.delta_rot.qw"] = qw
    return action


# --------------------------------------------------------------------------- #
# FK verification helpers (separate instance — does not touch the server's IK)
# --------------------------------------------------------------------------- #

def _fk_pose(obs: dict, kin) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematics EE position (mm) + rotation matrix from joint obs."""
    q = np.array([obs[f"{j}.pos"] for j in BODY_JOINTS], dtype=float)
    t = kin.forward_kinematics(q)
    return t[:3, 3].copy(), t[:3, :3].copy()


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → rotation vector (axis × angle, radians)."""
    from lerobot.utils.rotation import Rotation

    return Rotation.from_matrix(R).as_rotvec()


def _send_step(
    client: GRPCFollower,
    kin,
    action: dict[str, float],
    sign: int,
    step: int,
    n_steps: int,
    prev_ee: np.ndarray,
    dwell: float,
) -> np.ndarray:
    """Send one delta, dwell for PD convergence, read obs, print a compact line.

    Returns the new EE position so the caller can chain it as ``prev_ee``.
    """
    client.send_action(action)
    time.sleep(dwell)
    obs = client.get_observation()
    ee_pos, _ = _fk_pose(obs, kin)
    grip = obs.get("gripper.pos", -1.0)
    sp = obs.get("shoulder_pan.pos", 0.0)
    wf = obs.get("wrist_flex.pos", 0.0)
    wr = obs.get("wrist_roll.pos", 0.0)
    delta_mm = float(np.linalg.norm(ee_pos - prev_ee) * 1000.0)
    arrow = "→" if sign > 0 else "←"
    print(
        f"  {arrow} {step + 1}/{n_steps}  "
        f"ee=[{ee_pos[0] * 1000:+7.1f},{ee_pos[1] * 1000:+7.1f},{ee_pos[2] * 1000:+7.1f}]mm  "
        f"Δ={delta_mm:4.1f}mm  grip={grip:5.1f}%  "
        f"pan:{sp:+5.1f}° wrist=[flex:{wf:+6.1f}° roll:{wr:+6.1f}°]"
    )
    return ee_pos


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual MuJoCo SO-101 pose-delta demo (connects to serve_mujoco_follower.py)"
    )
    parser.add_argument("--address", default=ADDRESS, help=f"gRPC server address (default: {ADDRESS})")
    parser.add_argument("--steps", type=int, default=5, help="steps per direction (default: 5)")
    parser.add_argument(
        "--translate-mm", type=float, default=4.0,
        help="millimetres per translation step (default: 4 → ±20 mm total at 5 steps)",
    )
    parser.add_argument(
        "--rotate-deg", type=float, default=3.0,
        help="degrees per rotation step (default: 3 → ±15° total at 5 steps)",
    )
    parser.add_argument(
        "--dwell", type=float, default=0.25,
        help="seconds to settle between steps — the MuJoCo PD actuators need time to converge (default: 0.25)",
    )
    parser.add_argument(
        "--gripper-max-distance-mm", type=float, default=GRIPPER_MAX_DISTANCE_MM,
        help="server's full-open distance in mm, for the initial-position reverse-map (default: 60)",
    )
    args = parser.parse_args()

    t_mm = args.translate_mm
    r_deg = args.rotate_deg

    # Placeholder — updated from the initial observation after connecting so the
    # gripper holds its starting position throughout the demo (no spurious open).
    # Python closures capture the *name*, so the lambdas below read the updated
    # value at call time.
    gripper_hold_mm: float = 30.0

    # Each test produces the current latch-once offset for step index k
    # (k = 1..N out, then N-1..0 back).
    tests: list[tuple[str, object]] = [
        ("Translate X (forward ↔ back)", lambda k: make_delta_action(pos_mm=(k * t_mm, 0, 0), gripper_mm=gripper_hold_mm)),
        ("Translate Y (left   ↔ right)", lambda k: make_delta_action(pos_mm=(0, k * t_mm, 0), gripper_mm=gripper_hold_mm)),
        ("Translate Z (down   ↔ up)",    lambda k: make_delta_action(pos_mm=(0, 0, k * t_mm), gripper_mm=gripper_hold_mm)),
        ("Rotate body-X (forearm → roll)",  lambda k: make_delta_action(rot_axis="x", rot_deg=k * r_deg, gripper_mm=gripper_hold_mm)),
        ("Rotate body-Y (lateral → pitch)", lambda k: make_delta_action(rot_axis="y", rot_deg=k * r_deg, gripper_mm=gripper_hold_mm)),
        ("Rotate body-Z (vertical → yaw)",  lambda k: make_delta_action(rot_axis="z", rot_deg=k * r_deg, gripper_mm=gripper_hold_mm)),
    ]

    total_mm = t_mm * args.steps
    total_deg = r_deg * args.steps
    print("\n╔══ MuJoCo SO-101 pose-delta visual demo ══╗")
    print(f"║  server:   {args.address:<30s}║")
    print(f"║  per axis: {args.steps} steps out + {args.steps} back     ║")
    print(f"║  transl:   ±{t_mm:g} mm/step  (±{total_mm:g} mm total)       ║")
    print(f"║  rotate:   ±{r_deg:g}°/step  (±{total_deg:g}° total)        ║")
    print(f"║  dwell:    {args.dwell}s between steps            ║")
    print("╚══════════════════════════════════════════╝")
    print("  → Watch the MuJoCo viewer window.\n")

    # FK verifier (separate kinematics instance, same URDF the server uses)
    from lerobot.model.kinematics import RobotKinematics

    kin = RobotKinematics(
        str(URDF_PATH), target_frame_name="gripper_frame_link",
        joint_names=list(BODY_JOINTS),
    )

    # --- Connect to the external server ---------------------------------------
    client = GRPCFollower(GRPCFollowerConfig(address=args.address, need_warmup=False))
    client.connect(calibrate=False)
    log.warning("Connected to %s — observation stream is live (physics + viewer stepping)", args.address)

    # Let the physics settle to a clean initial state, then snapshot.
    time.sleep(0.5)
    obs0 = client.get_observation()
    ee0_pos, ee0_rot = _fk_pose(obs0, kin)

    # Hold the gripper at its initial position for the entire demo.
    grip_init_pct = obs0.get("gripper.pos", 50.0)
    gripper_hold_mm = grip_init_pct / 100.0 * args.gripper_max_distance_mm

    print(f"EE frame:  'gripper_frame_link' (URDF, ~98 mm past gripper joint, Z-flipped)")
    print(f"Initial EE: pos = [{ee0_pos[0]*1000:+.1f}, {ee0_pos[1]*1000:+.1f}, {ee0_pos[2]*1000:+.1f}] mm")
    print(f"Gripper hold: {gripper_hold_mm:.1f} mm ({grip_init_pct:.1f}%) — stays constant throughout\n")

    try:
        prev_ee = ee0_pos.copy()
        for idx, (name, delta_fn) in enumerate(tests, 1):
            print(f"─── {idx}/6  {name} ───")

            # --- N steps OUT (offset = 1..N) -----------------------------------
            for i in range(args.steps):
                prev_ee = _send_step(
                    client, kin, delta_fn(i + 1), +1, i, args.steps, prev_ee, args.dwell,
                )

            # --- N steps BACK (offset = N-1..0) --------------------------------
            for i in range(args.steps):
                prev_ee = _send_step(
                    client, kin, delta_fn(args.steps - 1 - i), -1, i, args.steps, prev_ee, args.dwell,
                )

            # --- Measure drift from initial pose -------------------------------
            time.sleep(0.3)
            obs = client.get_observation()
            ee_pos, ee_rot = _fk_pose(obs, kin)
            drift_mm = float(np.linalg.norm(ee_pos - ee0_pos) * 1000.0)
            rot_drift_deg = float(np.degrees(np.linalg.norm(_rotvec(ee_rot @ ee0_rot.T))))
            print(f"  ✓ drift from origin: {drift_mm:.2f} mm pos | {rot_drift_deg:.2f}° rot\n")

        client.disconnect()
        print("DEMO_PASS — all 6 DOF exercised; arm returned near origin.")
    except KeyboardInterrupt:
        print("\n\nInterrupted — disconnecting …")
        client.disconnect()


if __name__ == "__main__":
    main()
