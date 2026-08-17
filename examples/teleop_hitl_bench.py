#!/usr/bin/env python
"""Real-arm pose_delta HITL bench driver + offline report (wayfinder #07).

Run mode: teleop exactly like examples/teleop_pika_mujoco.py (same clutch
helpers, same relatch sequence, same processors) against a REAL
``serve_so101_follower.py --action_mode=pose_delta`` server, and record one
CSV row per loop tick: clutch state, what was sent, and the follower's
observed joints/gripper.  Clutch semantics per #06: a single natural
squeeze-release toggles follow/hold (no double-click).

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
