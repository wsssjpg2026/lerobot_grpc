#!/usr/bin/env python
"""Real-arm pose_delta HITL bench driver + offline report (wayfinder #07).

Run mode: teleop exactly like examples/teleop_pika_mujoco.py (same clutch
helpers, same relatch sequence, same processors) against a REAL
``serve_so101_follower.py --action_mode=pose_delta`` server, and record one
CSV row per loop tick: clutch state, what was sent, and the follower's
observed joints/gripper.  Clutch semantics per #06: a single natural
squeeze-release toggles follow/hold (no double-click).

Pre-pose phase (pika-sense-real #05, default on): after connecting, before
the Enter alignment, slowly walk the arm to the law's rest posture
(REAL_REST_POSTURE_DEG) via small pose-delta ramps — a torque-free arm that
sagged after calibration picks itself up — then re-lock T_zero there, so the
human-aligned Enter start happens from a manipulable configuration with
~zero offset.  ``--no-pre-pose`` restores the old behavior.

Driver-only tolerances (the bench needs the loop to survive them):

- Leader down (``kill`` the leader server mid-run): status/action RPCs raise;
  the driver marks ``leader_ok=0``, sends nothing, keeps logging the frozen
  follower -- the leader-down freeze is then judged offline.
- The client itself being SIGSTOPped shows up as a ``t_s`` gap; the servicer's
  stale-hold on the first post-gap action is judged offline.

Report mode: ``--report <csv>`` prints every machine-checked judgement
(hold freeze, relock jitter, stale gaps, leader-down, sphere, desktop
evidence) via lerobot_robot_grpc.follower.hitl_bench.

Usage::

    # Terminal 1: real follower (full command in pika-sense-real map Notes)
    conda run -n lrg python examples/serve_so101_follower.py \\
        --action_mode=pose_delta --robot.port=/dev/ttyACM0 --robot.id=follower \\
        --robot.position_p_coefficient=32 --address=0.0.0.0:5555 ...

    # Terminal 2: Pika Sense leader (ALSO /dev/ttyUSB0 -- never confuse them)
    conda run -n lrg python examples/serve_pika_sense_leader.py --port /dev/ttyUSB0

    # Terminal 3: this driver
    conda run -n lrg python examples/teleop_hitl_bench.py --csv /tmp/hitl_r1.csv

    # After the round: the offline judgement
    conda run -n lrg python examples/teleop_hitl_bench.py --report /tmp/hitl_r1.csv
"""

import argparse
import csv as csv_mod
import logging
import math
import select
import sys
import time
from pathlib import Path

from google.protobuf.empty_pb2 import Empty
from lerobot.processor import RobotAction, make_default_processors
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.follower.hitl_bench import CSV_HEADER, build_report, load_rows
from lerobot_robot_grpc.leader.config_grpc import GRPCLeaderConfig
from lerobot_robot_grpc.leader.grpc_leader import GRPCLeader
from lerobot_robot_grpc.protos import device_pb2
from lerobot_robot_grpc.teleop_clutch import auto_clutch_step, keyboard_clutch_step

logger = logging.getLogger(__name__)

_XML = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"

OBS_TO_CSV = {
    "shoulder_pan.pos": "obs_pan_deg",
    "shoulder_lift.pos": "obs_lift_deg",
    "elbow_flex.pos": "obs_elbow_deg",
    "wrist_flex.pos": "obs_wf_deg",
    "wrist_roll.pos": "obs_wr_deg",
    "gripper.pos": "obs_gripper",
}
ACT_TO_CSV = {
    "hand.delta_pos.x": "act_dx_m",
    "hand.delta_pos.y": "act_dy_m",
    "hand.delta_pos.z": "act_dz_m",
    "hand.delta_rot.qx": "act_qx",
    "hand.delta_rot.qy": "act_qy",
    "hand.delta_rot.qz": "act_qz",
    "hand.delta_rot.qw": "act_qw",
    "gripper.distance": "act_grip_mm",
}


def relatch(robot: GRPCFollower, teleop: GRPCLeader) -> None:
    """Same re-engage sequence as the sim client: follower T_zero first, then
    leader T_begin (order is the zero-motion contract, #10/#12)."""
    import grpc

    try:
        robot.stub.SetReference(Empty(), timeout=robot.data_timeout_s)
    except grpc.RpcError as e:
        logger.warning("follower SetReference failed (%s) — continuing anyway.", e)
    teleop.set_reference()
    logger.info("Clutch re-engaged: T_zero and T_begin re-latched.")


# Observation keys in BODY_JOINTS order (pan, lift, elbow, wrist_flex,
# wrist_roll) — the qpos assembly mirrors hitl_bench.ee_positions.
_OBS_JOINT_KEYS = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos",
)

# Mirror of SO101FollowerServicer's law home, REAL_REST_POSTURE_DEG
# (so101_follower_server.py) — keep in sync with the server.
_REST_POSTURE_DEG = (0.0, 30.0, -20.0, 0.0, 0.0)


def pre_pose(robot: GRPCFollower, args) -> None:
    """Slowly walk the arm to the law's rest posture before the Enter gate.

    The law drives intents as T_zero + delta, so: re-lock T_zero at the
    current pose, read the joints and FK them (same oracle as the offline
    report), then ramp the delta toward FK(rest posture) in ``--pre-pose-step-mm``
    bites at 30 Hz.  The first action engages the servos, so a torque-free
    arm that sagged after calibration carries itself along the ramp.  Stall
    detection backs out of unreachable approaches; either way T_zero is
    re-locked at the pose actually reached, so the human alignment at Enter
    starts with ~zero offset from a manipulable configuration.
    """
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot

    from lerobot_robot_grpc.follower.hitl_bench import (
        _GRIPPER_RAD_MAX,
        _GRIPPER_RAD_MIN,
    )

    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    def fk(joint_deg, gripper_pct):
        qpos = [math.radians(v) for v in joint_deg]
        qpos.append(
            (gripper_pct / 100.0) * (_GRIPPER_RAD_MAX - _GRIPPER_RAD_MIN)
            + _GRIPPER_RAD_MIN
        )
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        return (
            data.site_xpos[site].copy(),
            data.site_xmat[site].reshape(3, 3).copy(),
        )

    obs = robot.get_observation()
    grip_pct = float(obs["gripper.pos"])
    cur = [obs[k] for k in _OBS_JOINT_KEYS]
    t0_pos, t0_rot = fk(cur, grip_pct)
    tgt_pos, tgt_rot = fk(_REST_POSTURE_DEG, grip_pct)
    need = tgt_pos - t0_pos
    dist0_mm = float(np.linalg.norm(need)) * 1000.0
    if dist0_mm < args.pre_pose_tol_mm:
        logger.info("Pre-pose: already at the rest posture (%.0fmm) — skipping.", dist0_mm)
        return

    print("\n" + "=" * 55)
    print("  预摆位：把机械臂缓慢移到中位工作区（末端移动约 %.0fcm，约 %.0f 秒）"
          % (dist0_mm / 10.0, dist0_mm / args.pre_pose_step_mm / 30.0))
    print("  确认臂的活动范围内无人、无障碍物；夹爪会半张开，注意防夹手")
    print("  按 Enter 开始预摆位，Ctrl+C 取消")
    print("=" * 55)
    input()

    # T_zero := current pose, so the ramp below is relative to here.
    robot.stub.SetReference(Empty(), timeout=robot.data_timeout_s)
    need_rotvec = Rot.from_matrix(t0_rot.T @ tgt_rot).as_rotvec()

    print("\n🤖 预摆位进行中（可随时 Ctrl+C 急停）...")
    sent_mm = 0.0
    best_mm = dist0_mm
    best_t = time.monotonic()
    stalled = False
    frame = 0
    deadline = time.monotonic() + 60.0
    while True:
        if dist0_mm - sent_mm <= args.pre_pose_step_mm:
            frac = 1.0
        else:
            sent_mm += args.pre_pose_step_mm
            frac = sent_mm / dist0_mm
        delta = need * frac
        quat = Rot.from_rotvec(need_rotvec * frac).as_quat()  # [x, y, z, w]
        robot.send_action({
            "hand.delta_pos.x": float(delta[0]),
            "hand.delta_pos.y": float(delta[1]),
            "hand.delta_pos.z": float(delta[2]),
            "hand.delta_rot.qx": float(quat[0]),
            "hand.delta_rot.qy": float(quat[1]),
            "hand.delta_rot.qz": float(quat[2]),
            "hand.delta_rot.qw": float(quat[3]),
            "gripper.distance": 30.0,  # half-open; fingers clear
        })
        frame += 1
        if frame % 15 == 0:  # progress check at ~2 Hz (obs rides the bus)
            obs = robot.get_observation()
            cur_pos, _ = fk([obs[k] for k in _OBS_JOINT_KEYS], obs["gripper.pos"])
            d_mm = float(np.linalg.norm(cur_pos - tgt_pos)) * 1000.0
            if d_mm < args.pre_pose_tol_mm:
                break
            if d_mm < best_mm - 0.5:
                best_mm, best_t = d_mm, time.monotonic()
            elif time.monotonic() - best_t > args.pre_pose_stall_s:
                stalled = True
                break
        if time.monotonic() > deadline:
            stalled = True
            break
        precise_sleep(1.0 / 30.0)

    # T_zero := the pose actually reached: the Enter alignment starts here.
    robot.stub.SetReference(Empty(), timeout=robot.data_timeout_s)
    obs = robot.get_observation()
    cur_pos, _ = fk([obs[k] for k in _OBS_JOINT_KEYS], obs["gripper.pos"])
    d_mm = float(np.linalg.norm(cur_pos - tgt_pos)) * 1000.0
    if stalled:
        logger.warning(
            "Pre-pose stalled %.0fmm short of the rest posture (residual-hold or "
            "unreachable approach); continuing from the reached pose.", d_mm)
    else:
        logger.info("Pre-pose done: %.0fmm -> %.0fmm from the rest posture.",
                    dist0_mm, d_mm)


def key_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


def run_bench(args) -> None:
    teleop = GRPCLeader(GRPCLeaderConfig(
        address=args.leader_address, id=args.leader_id, warmup_timeout_s=120.0,
    ))
    robot = GRPCFollower(GRPCFollowerConfig(
        address=args.follower_address, id=args.follower_id, warmup_timeout_s=30.0,
    ))

    logger.info("Connecting to Pika Sense leader at %s ...", args.leader_address)
    teleop.connect()
    logger.info("Connecting to follower at %s ...", args.follower_address)
    robot.connect()

    if args.pre_pose:
        try:
            pre_pose(robot, args)
        except KeyboardInterrupt:
            print("\n预摆位被中断（Ctrl+C）——退出。")
            teleop.disconnect()
            robot.disconnect()
            return

    print("\n" + "=" * 55)
    print("  Tracker 就绪后拿起 Pika Sense，夹爪朝前，与 SO-101 末端姿态对应")
    print("  按 Enter 对齐并开始跟随")
    print("  离合 = 单次自然捏合-松开（#06：单击即 toggle，勿刻意慢按）")
    print("  停 = 臂冻结原位、夹爪仍跟手；再捏合-松开 = 当前手=当前臂续跟")
    print("  Ctrl+C 结束（CSV 会完整落盘）")
    print("=" * 55)
    input()

    teleop.set_reference()
    logger.info("Reference set — delta origin locked to current tracker pose.")

    teleop_action_processor, robot_action_processor, _ = make_default_processors()
    csv_file = open(args.csv, "w", newline="")
    writer = csv_mod.writer(csv_file)
    writer.writerow(CSV_HEADER)

    period = 1.0 / args.fps
    t0 = time.monotonic()
    engaged = True
    if args.clutch_source == "auto":
        try:
            engaged = teleop.get_device_status() == device_pb2.DeviceStatus.COLLECTION
        except Exception:
            engaged = True  # status poll hiccup at start; the loop re-syncs
    last_action: RobotAction = {}
    last_report_t = 0.0

    print(f"\n🤖 HITL 台架录制中 -> {args.csv}\n")

    try:
        while True:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            leader_ok = True
            raw_action: RobotAction = {}
            try:
                raw_action = teleop.get_action()
                if args.clutch_source == "keyboard":
                    toggled = key_pressed()
                    if toggled:
                        sys.stdin.readline()
                    engaged, raw_action, should_send = keyboard_clutch_step(
                        engaged=engaged, key_toggled=toggled, raw_action=raw_action,
                        fetch_action=teleop.get_action,
                        relatch=lambda: relatch(robot, teleop),
                    )
                else:
                    status = teleop.get_device_status()
                    prev = engaged
                    engaged, raw_action, should_send = auto_clutch_step(
                        status=status, engaged=engaged, raw_action=raw_action,
                        fetch_action=teleop.get_action,
                        relatch=lambda: relatch(robot, teleop),
                    )
                    if prev != engaged:
                        logger.info("CLUTCH: toggle -> %s at t=%.2fs",
                                    "FOLLOW" if engaged else "HOLD",
                                    time.monotonic() - t0)
            except Exception as e:  # leader down: keep logging the frozen arm
                leader_ok = False
                should_send = False
                logger.warning("leader RPC failed (%s) — holding, still logging", e)

            sent = 0
            if should_send and leader_ok:
                teleop_action = teleop_action_processor((raw_action, obs))
                robot_action = robot_action_processor((teleop_action, obs))
                robot.send_action(robot_action)
                last_action = dict(robot_action)
                sent = 1

            row: dict = {k: "" for k in CSV_HEADER}
            row["t_s"] = f"{time.monotonic():.4f}"
            row["engaged"] = "1" if engaged else "0"
            row["sent"] = "1" if sent else "0"
            row["leader_ok"] = "1" if leader_ok else "0"
            if sent:
                for key, col in ACT_TO_CSV.items():
                    row[col] = f"{last_action.get(key, 0.0):.5f}"
            for key, col in OBS_TO_CSV.items():
                row[col] = f"{obs.get(key, 0.0):.3f}"
            writer.writerow([row[k] for k in CSV_HEADER])

            now = time.monotonic()
            if now - last_report_t >= 1.0:
                csv_file.flush()
                last_report_t = now
                state = ("FOLLOW" if engaged else "HOLD") if leader_ok else "LEADER-DOWN"
                print(f"[{now - t0:7.1f}s] {state:<12} elbow={obs['elbow_flex.pos']:7.2f} "
                      f"grip={obs['gripper.pos']:5.1f} rows/sec={args.fps}")

            dt = time.perf_counter() - loop_start
            precise_sleep(max(period - dt, 0.0))
    except KeyboardInterrupt:
        print("\nStopping bench...")
    finally:
        csv_file.close()
        teleop.disconnect()
        robot.disconnect()
        logger.info("CSV closed: %s", args.csv)


def report_bench(args) -> None:
    rows = load_rows(args.report)
    report = build_report(
        rows,
        xml_path=str(_XML),
        sphere_radius_m=args.sphere_mm / 1000.0,
    )
    print(report.render())


def main():
    parser = argparse.ArgumentParser(description="Real-arm pose_delta HITL bench (#07)")
    parser.add_argument("--leader-address", default="127.0.0.1:5556")
    parser.add_argument("--follower-address", default="127.0.0.1:5555")
    parser.add_argument("--leader-id", default="pika_sense")
    parser.add_argument("--follower-id", default="follower")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clutch-source", choices=("auto", "keyboard"), default="auto")
    parser.add_argument("--csv", default=None, help="run mode: CSV path to record")
    parser.add_argument("--report", default=None, help="report mode: judge a recorded CSV")
    parser.add_argument(
        "--pre-pose", action=argparse.BooleanOptionalAction, default=True,
        help="walk the arm to the law rest posture before the Enter alignment "
             "(default: on; --no-pre-pose restores the old behavior)",
    )
    parser.add_argument("--pre-pose-step-mm", type=float, default=2.0,
                        help="per-frame ramp bite toward the rest posture (default 2mm)")
    parser.add_argument("--pre-pose-tol-mm", type=float, default=10.0,
                        help="pre-pose success distance from the rest posture (default 10mm)")
    parser.add_argument("--pre-pose-stall-s", type=float, default=2.5,
                        help="pre-pose stall: no progress for this long -> stop ramp (default 2.5s)")
    parser.add_argument("--sphere-mm", type=float, default=0.72 * 543,
                        help="base safety sphere radius for the report (default 391)")
    args = parser.parse_args()

    init_logging()
    if args.report:
        report_bench(args)
        return
    if not args.csv:
        parser.error("either --csv <path> (record) or --report <csv> (judge) is required")
    run_bench(args)


if __name__ == "__main__":
    main()
