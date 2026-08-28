#!/usr/bin/env python
"""Pika Sense leader → MuJoCo pose-delta follower teleop with clutch.

Replaces ``lerobot-teleoperate`` for the Pika Sense delta-pose workflow.
Adds a mandatory alignment step, then a double-click clutch with official
UR/Piper semantics:

- **Enter** — first alignment: ``SetReference`` on the leader, teleop engages.
- **Pika double-click** (command-state change) — toggles follow / hold.
  Hold = the arm stops at its current pose; re-engage = both bases re-latch
  (leader ``T_begin`` first, then follower FK ``T_zero``), so the current
  hand pose maps onto the current arm pose — no crawl back to Connect home.
- **Gripper** (official PikaAnyArm semantics): the clutch gates the arm only.
  In ``auto`` mode the client keeps transporting actions while holding — the
  leader freezes the arm offset itself and the gripper stays live.  The
  ``keyboard`` fallback owns the clutch locally and stops sending entirely on
  hold (arm AND gripper freeze — no button for the leader to read).

Flow::

    1. Connect leader (returns after hardware setup)
    2. Connect follower
    3. Poll Tracker readiness until Lighthouse convergence is READY
    4. Prompt: "Hold tracker forward, press Enter to align"
    5. teleop.set_reference()  →  locks current pose as delta origin, engages
    6. Teleop loop (get_action → send_action at --fps), gated by the clutch

Clutch edge sources (``--clutch-source``):

- ``auto`` (default): poll the leader's ``GetStatus`` (COLLECTION = follow,
  IDLE = clutch off).  This is the leader→client channel for the Pika button.
- ``keyboard``: press Enter in the terminal to toggle.  For bench/tests when
  the button is unavailable; the client owns the clutch and still sequences
  leader.SetReference → follower.SetReference on every engage.

Usage::

    # Terminal 1: MuJoCo follower
    conda run -n lerobot-grpc-serve python examples/serve_mujoco_follower.py --action-mode pose_delta --render

    # Terminal 2: Pika Sense leader
    conda run -n lerobot-grpc-serve python examples/serve_pika_sense_leader.py --port /dev/ttyUSB0

    # Terminal 3: This script
    conda run -n collection-serve python examples/teleop_pika_mujoco.py [--fps=30]
"""

import argparse
import logging
import select
import sys
import time

from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.leader.config_grpc import GRPCLeaderConfig
from lerobot_robot_grpc.leader.grpc_leader import (
    GRPCLeader,
    ReferenceNotReadyError,
)
from lerobot_robot_grpc.protos import device_pb2
from lerobot_robot_grpc.teleop_clutch import auto_clutch_step, keyboard_clutch_step

logger = logging.getLogger(__name__)


def relatch(robot: GRPCFollower, teleop: GRPCLeader) -> bool:
    """Commit a failure-safe leader-then-follower reference transaction.

    The leader readiness check is the fallible validation step, so it runs
    before mutating the follower reference.  No action is sent until both
    calls succeed.  If the follower call then fails, the next loop observes
    the leader's COLLECTION state and retries the complete transaction while
    transport remains held.  The IK stays in the adapters; this client only
    triggers the transaction (#10: 采集进程只搬运).
    """
    try:
        teleop.set_reference()
    except ReferenceNotReadyError as exc:
        logger.warning("Leader reference is not ready; remaining held: %s", exc)
        return False
    except DeviceNotConnectedError as exc:
        logger.error("Leader SetReference failed; remaining held: %s", exc)
        return False
    try:
        robot.set_reference()
    except DeviceNotConnectedError as exc:
        logger.error("Follower SetReference failed; remaining held: %s", exc)
        return False
    logger.info("Clutch re-engaged: T_begin and T_zero re-latched.")
    return True


def key_pressed() -> bool:
    """Non-blocking stdin check for the ``--clutch-source keyboard`` toggle."""
    return bool(select.select([sys.stdin], [], [], 0)[0])


def wait_for_tracking_readiness(
    teleop: GRPCLeader,
    *,
    timeout_s: float = 120.0,
) -> str | None:
    """Wait for a current reference token before prompting the operator."""
    deadline = time.monotonic() + timeout_s
    last_summary: tuple[int, str] | None = None
    last_log = 0.0
    while True:
        readiness = teleop.get_tracking_readiness()
        state = int(readiness.state)
        if state == (
            device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_READY
        ):
            return str(readiness.token or "")
        if state == (
            device_pb2.TrackingReadinessState.
            TRACKING_READINESS_STATE_NOT_APPLICABLE
        ):
            return None
        if state == (
            device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_ERROR
        ):
            raise RuntimeError(f"Tracker readiness failed: {readiness.reason}")
        now = time.monotonic()
        summary = (state, str(readiness.reason))
        if summary != last_summary or now - last_log >= 10.0:
            state_name = device_pb2.TrackingReadinessState.Name(state).removeprefix(
                "TRACKING_READINESS_STATE_"
            )
            logger.info("Tracker readiness %s: %s", state_name, readiness.reason)
            last_summary = summary
            last_log = now
        if now >= deadline:
            raise TimeoutError(
                f"Tracker did not become ready within {timeout_s:.0f}s: "
                f"{readiness.reason}"
            )
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pika Sense → MuJoCo pose-delta follower teleop with "
            "alignment + double-click clutch"
        )
    )
    parser.add_argument(
        "--leader-address",
        default="127.0.0.1:5556",
        help="Pika Sense leader gRPC address (default: 127.0.0.1:5556)",
    )
    parser.add_argument(
        "--follower-address",
        default="127.0.0.1:5555",
        help="MuJoCo follower gRPC address (default: 127.0.0.1:5555)",
    )
    parser.add_argument("--leader-id", default="pika_sense", help="Leader device ID")
    parser.add_argument("--follower-id", default="so101_sim", help="Follower device ID")
    parser.add_argument("--fps", type=int, default=30, help="Teleop loop frequency (Hz)")
    parser.add_argument(
        "--clutch-source",
        choices=("auto", "keyboard"),
        default="auto",
        help="Clutch edge source: 'auto' = leader GetStatus (Pika button), "
        "'keyboard' = Enter key toggles in this terminal",
    )
    args = parser.parse_args()

    init_logging()

    # --- Create devices ---
    # Connect establishes hardware; readiness is polled explicitly below.
    teleop = GRPCLeader(
        GRPCLeaderConfig(
            address=args.leader_address,
            id=args.leader_id,
            warmup_timeout_s=120.0,
        )
    )
    robot = GRPCFollower(
        GRPCFollowerConfig(
            address=args.follower_address,
            id=args.follower_id,
            warmup_timeout_s=30.0,
        )
    )

    # --- Connect ---
    logger.info("Connecting to Pika Sense leader at %s ...", args.leader_address)
    teleop.connect()
    logger.info("Pika Sense leader connected.")

    logger.info("Connecting to MuJoCo follower at %s ...", args.follower_address)
    robot.connect()
    logger.info("MuJoCo follower connected.")

    # --- Readiness + alignment step ---
    while True:
        readiness_token = wait_for_tracking_readiness(teleop)
        print("\n" + "=" * 55)
        print("  Tracker 已收敛，现在可以调整 Pika 参考姿态")
        print("  拿起 Pika Sense（和 Vive Tracker 一体），夹爪朝前")
        print("  与当前机械臂末端姿态对应后按 Enter 对齐并开始跟随")
        print("  之后 Pika 双击：停 / 跟。停止时臂停在当前末端")
        print("  再双击重锁：当前手 = 当前臂，从这里继续跟")
        print("  两端须用同一套代码（发的是相对对齐姿态的当前偏移）")
        print("=" * 55)
        input()
        try:
            teleop.set_reference(readiness_token=readiness_token)
        except ReferenceNotReadyError as exc:
            logger.warning(
                "Tracker readiness changed before alignment; retrying: %s", exc
            )
            continue
        break
    logger.info("Reference set — delta origin locked to current tracker pose.")

    # --- Teleop loop ---
    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    if args.clutch_source == "keyboard":
        print(
            "\n🤖 遥操已开始！键盘模式：按 Enter 切换 跟/停。Ctrl+C 退出。\n"
            "回退模式：停止时臂与夹爪一起冻结（官方夹爪语义只在按钮模式有效）。\n"
        )
    else:
        print(
            "\n🤖 遥操已开始！移动 Pika Sense 控制机械臂。"
            "Pika 双击切换 跟/停。Ctrl+C 退出。\n"
            "停止时臂冻结、夹爪仍跟手（官方语义）。\n"
        )

    display_len = max(len(key) for key in robot.action_features)
    period = 1.0 / args.fps

    # Following state.  Auto mode is initialised from the leader status after
    # the Enter alignment (engaged); keyboard mode owns it locally.
    engaged = True
    if args.clutch_source == "auto":
        engaged = (
            teleop.get_device_status() == device_pb2.DeviceStatus.COLLECTION
        )
    last_action: RobotAction = {}

    try:
        while True:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            raw_action: RobotAction = teleop.get_action()

            # --- Clutch edge detection (#10/#12) ---
            # Always pull an action: the leader reads the Pika button inside
            # GetAction, so the stream must keep flowing even while holding.
            # Auto mode keeps sending on IDLE too — the leader freezes the
            # arm offset itself and the gripper stays live (official grip
            # semantics).  Keyboard mode owns the clutch locally and stops
            # sending entirely on hold (arm + gripper freeze — fallback).
            if args.clutch_source == "keyboard":
                toggled = key_pressed()
                if toggled:
                    sys.stdin.readline()
                engaged, raw_action, should_send = keyboard_clutch_step(
                    engaged=engaged,
                    key_toggled=toggled,
                    raw_action=raw_action,
                    fetch_action=teleop.get_action,
                    relatch=lambda: relatch(robot, teleop),
                )
            else:
                status = teleop.get_device_status()
                engaged, raw_action, should_send = auto_clutch_step(
                    status=status,
                    engaged=engaged,
                    reference_pending=(
                        teleop.last_tracking_state
                        == device_pb2.TrackingState.
                        TRACKING_STATE_REFERENCE_PENDING
                    ),
                    raw_action=raw_action,
                    fetch_action=teleop.get_action,
                    relatch=lambda: relatch(robot, teleop),
                )

            if should_send:
                teleop_action = teleop_action_processor((raw_action, obs))
                robot_action = robot_action_processor((teleop_action, obs))
                robot.send_action(robot_action)
                last_action = dict(robot_action)

            # --- Live display ---
            dt = time.perf_counter() - loop_start
            precise_sleep(max(period - dt, 0.0))
            loop_ms = (time.perf_counter() - loop_start) * 1000
            state = "FOLLOW" if engaged else "HOLD"
            print(f"\nCLUTCH: {state:<6} | source={args.clutch_source:<8} | Loop: {loop_ms:.1f}ms")
            print("-" * (display_len + 12))
            print(f"{'NAME':<{display_len}} | {'VALUE':>8}")
            for key, value in last_action.items():
                print(f"{key:<{display_len}} | {value:>8.4f}")
            move_cursor_up(len(last_action) + 4)

    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        teleop.disconnect()
        robot.disconnect()
        logger.info("Disconnected.")


if __name__ == "__main__":
    main()
