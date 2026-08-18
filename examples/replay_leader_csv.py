#!/usr/bin/env python
"""Offline replay of a recorded leader-delta CSV through the follower law.

Acceptance harness for the PikaAnyArm official alignment
(decision 1): feed a
bench-recorded leader stream (teleop_hitl_bench.py CSV, act_d* columns)
frame-by-frame through the shared PoseDeltaLaw in a MuJoCo oracle loop —
no hardware, the solved joints are written straight back as the next
qpos (perfect tracking) — and report how the official safety stack
(30° jump reset, FK consistency 0.3 m, per-frame step cap, self-collision
gate) handled the stream.

The primary input is the r4 real-bench CSV (the uncontrollable round):
decision 5's validation question is whether the official mechanisms
suppress the tracker z-garbage without any leader-side gate.

Two flags adapt the r4 recording to the new semantics:

--legacy-base-frame   r4 was recorded under the OLD composition
                      (intent = T_zero + R_lh2base @ Δ_world, pos_gain
                      0.45, EMA, jump-rebase).  This flag treats the
                      recorded delta_pos as a BASE-frame offset and folds
                      it back through R_ref^T so the new law reproduces
                      the exact old intent positions.
--pos-gain            Scales the recorded position deltas (default 1.0).
                      r4 was recorded at 0.45 gain; --pos-gain 2.22
                      emulates the 1:1 raw publish of the new leader.

Clutch windows are mirrored: on an engaged 0→1 edge the law reference is
re-locked at the current oracle pose (the bench client's relatch
sequence); rows with sent=0 are skipped; a >1.2 s row gap makes the next
solve stale (the real servicer's stale hold).

Usage::

    conda run -n lerobot-grpc-serve python examples/replay_leader_csv.py \
        --csv /tmp/hitl_r4.csv --out /tmp/replay_r4.csv \
        --legacy-base-frame --pos-gain 1.0
"""

import argparse
import csv as csv_mod
import logging
import math
from pathlib import Path

import numpy as np

from lerobot_robot_grpc.follower.hitl_bench import load_rows
from lerobot_robot_grpc.follower.mujoco_follower_server import (
    BODY_JOINTS,
    DEFAULT_GRIPPER_MAX_DISTANCE_MM,
    JOINTS,
    norm_value_to_rad,
)
from lerobot_robot_grpc.follower.pose_delta_law import PoseDeltaLaw

logger = logging.getLogger(__name__)

_DEFAULT_XML = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"

_OUT_HEADER = (
    "frame", "t_s", "engaged",
    "tgt_x_m", "tgt_y_m", "tgt_z_m",
    "fk_x_m", "fk_y_m", "fk_z_m",
    "pos_err_mm", "held", "rejected", "collided", "jumped",
    "q_pan_deg", "q_lift_deg", "q_elbow_deg", "q_wf_deg", "q_wr_deg",
    "gripper",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, help="bench CSV recorded by teleop_hitl_bench.py")
    parser.add_argument("--out", required=True, help="per-frame replay trajectory CSV to write")
    parser.add_argument("--xml-path", default=str(_DEFAULT_XML))
    parser.add_argument(
        "--legacy-base-frame", action="store_true",
        help="recorded delta_pos is a BASE-frame offset (old composition); "
             "fold it back through R_ref^T so the new law reproduces the "
             "old intent positions exactly",
    )
    parser.add_argument(
        "--pos-gain", type=float, default=1.0,
        help="scale on recorded position deltas (r4 was recorded at 0.45; "
             "2.22 emulates the 1:1 raw leader)",
    )
    parser.add_argument(
        "--home-joints", default="0,30,-20,0,0",
        help="law rest posture in degrees (default 0,30,-20,0,0 = the real "
             "servicer's REAL_REST_POSTURE_DEG)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    import mujoco

    rows = load_rows(args.csv)
    rows = [r for r in rows if r["t_s"] is not None]
    logger.info("Loaded %d rows from %s", len(rows), args.csv)

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    home = tuple(float(x) for x in args.home_joints.split(","))

    law = PoseDeltaLaw(
        model,
        site_name="gripperframe",
        body_dofs=list(range(len(BODY_JOINTS))),
        body_joint_names=BODY_JOINTS,
        home_joints_deg=home,
        max_dq_deg=6.0,
        max_dq_frame_deg=6.7,
        gripper_max_distance_mm=DEFAULT_GRIPPER_MAX_DISTANCE_MM,
    )

    # Oracle joints from the first row's observation columns.
    def qpos_from_obs(row) -> np.ndarray:
        obs_deg = [row[k] for k in (
            "obs_pan_deg", "obs_lift_deg", "obs_elbow_deg", "obs_wf_deg", "obs_wr_deg",
        )]
        return np.array(
            [math.radians(v) for v in obs_deg]
            + [norm_value_to_rad("gripper", row["obs_gripper"])],
            dtype=float,
        )

    qpos = qpos_from_obs(rows[0])
    law.lock_reference(qpos)
    logger.info(
        "Reference locked at FK pos=[%.3f %.3f %.3f]",
        *law.arm_reference[:3, 3],
    )

    out_rows: list[list] = []
    stats = {
        "frames": 0, "held": 0, "rejected": 0, "collided": 0, "jumped": 0,
        "stale": 0, "relatch": 0,
    }
    prev_t = rows[0]["t_s"]
    prev_engaged = 1
    max_step_deg = 0.0
    prev_q_deg = [math.degrees(v) for v in qpos[:5]]
    fk_z_min, fk_z_max = float("inf"), float("-inf")
    fk_r_max = 0.0

    for frame, row in enumerate(rows):
        t = row["t_s"]
        gap_s = t - prev_t
        # Mirror the servicer's stale hold: a >1 s command gap makes the
        # next action stale-flagged.
        stale = gap_s > 1.2
        if stale:
            stats["stale"] += 1

        # Mirror the bench client's relatch on a clutch re-engage edge.
        engaged = int(row["engaged"] or 0)
        if engaged == 1 and prev_engaged == 0:
            law.lock_reference(qpos)
            stats["relatch"] += 1
        prev_engaged = engaged

        if not row.get("sent") or row.get("act_dx_m") is None:
            prev_t = t
            continue  # hold window / leader-down: no command flowed

        delta_pos = np.array(
            [row["act_dx_m"], row["act_dy_m"], row["act_dz_m"]], dtype=float
        ) * args.pos_gain
        if args.legacy_base_frame:
            # Old composition: intent = p_ref + delta_base.  New law wants a
            # body-frame delta; R_ref @ (R_ref^T @ delta_base) = delta_base.
            delta_pos = law.arm_reference[:3, :3].T @ delta_pos

        action = {
            "hand.delta_pos.x": float(delta_pos[0]),
            "hand.delta_pos.y": float(delta_pos[1]),
            "hand.delta_pos.z": float(delta_pos[2]),
            "hand.delta_rot.qx": row["act_qx"],
            "hand.delta_rot.qy": row["act_qy"],
            "hand.delta_rot.qz": row["act_qz"],
            "hand.delta_rot.qw": row["act_qw"],
            "gripper.distance": row["act_grip_mm"] or 0.0,
        }

        # Log target position for the trajectory CSV (same composition as
        # the law: p_ref + R_ref @ delta_pos).
        tgt_pos = law.arm_reference[:3, 3] + law.arm_reference[:3, :3] @ delta_pos

        sol = law.solve(action, qpos, stale=stale)
        stats["frames"] += 1
        if sol.held:
            stats["held"] += 1
        if sol.rejected:
            stats["rejected"] += 1
        if sol.collided:
            stats["collided"] += 1
        if sol.jumped:
            stats["jumped"] += 1

        # Closed-loop oracle: commanded joints become the next state.
        for i, joint in enumerate(JOINTS):
            qpos[i] = norm_value_to_rad(joint, sol.joint_action[f"{joint}.pos"])
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        fk = data.site_xpos[site].copy()
        fk_z_min, fk_z_max = min(fk_z_min, fk[2]), max(fk_z_max, fk[2])
        fk_r_max = max(fk_r_max, float(np.linalg.norm(fk)))

        q_deg = [sol.joint_action[f"{j}.pos"] for j in BODY_JOINTS]
        max_step_deg = max(
            max_step_deg, max(abs(a - b) for a, b in zip(q_deg, prev_q_deg))
        )
        prev_q_deg = q_deg

        out_rows.append([
            frame, f"{t:.4f}", engaged,
            f"{tgt_pos[0]:.4f}", f"{tgt_pos[1]:.4f}", f"{tgt_pos[2]:.4f}",
            f"{fk[0]:.4f}", f"{fk[1]:.4f}", f"{fk[2]:.4f}",
            f"{sol.pos_err_m * 1000:.2f}",
            int(sol.held), int(sol.rejected), int(sol.collided), int(sol.jumped),
            *(f"{v:.2f}" for v in q_deg),
            f"{sol.joint_action['gripper.pos']:.1f}",
        ])
        prev_t = t

    with open(args.out, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(_OUT_HEADER)
        writer.writerows(out_rows)

    print("=" * 62)
    print(f"Replay: {stats['frames']} solved frames from {len(rows)} rows")
    print(f"  held={stats['held']}  rejected={stats['rejected']}  "
          f"collided={stats['collided']}  jumped={stats['jumped']}")
    print(f"  stale-fed={stats['stale']}  relatch={stats['relatch']}")
    print(f"  max published joint step: {max_step_deg:.2f} deg/frame (cap 6.7)")
    print(f"  oracle FK z range: [{fk_z_min * 1000:.0f}, {fk_z_max * 1000:.0f}] mm")
    print(f"  oracle FK max radius: {fk_r_max * 1000:.0f} mm")
    print(f"  trajectory CSV -> {args.out}")
    print("=" * 62)


if __name__ == "__main__":
    main()
