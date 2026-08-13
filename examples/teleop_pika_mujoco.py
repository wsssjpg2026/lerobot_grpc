#!/usr/bin/env python
"""Pika Sense leader → MuJoCo SO-101 follower teleop with alignment step.

Replaces ``lerobot-teleoperate`` for the Pika Sense delta-pose workflow.
Adds a mandatory alignment step: the operator picks up the tracker, holds
it in a natural forward-facing pose, and presses Enter to set the delta
reference *before* teleop begins.

Flow::

    1. Connect leader (blocks until tracker solver converges, up to 60s)
    2. Connect follower
    3. Prompt: "Hold tracker forward, press Enter to align"
    4. teleop.set_reference()  →  locks current pose as delta origin
    5. Teleop loop (get_action → send_action at --fps)

Usage::

    # Terminal 1: MuJoCo follower
    conda run -n lrg python examples/serve_mujoco_follower.py --action-mode pose_delta --render

    # Terminal 2: Pika Sense leader
    conda run -n lerobot-grpc-test python examples/serve_pika_sense_leader.py --port /dev/ttyUSB0

    # Terminal 3: This script
    conda run -n lrg python examples/teleop_pika_mujoco.py [--fps=30]
"""

import argparse
import logging
import time

from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.leader.config_grpc import GRPCLeaderConfig
from lerobot_robot_grpc.leader.grpc_leader import GRPCLeader

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pika Sense → MuJoCo SO-101 teleop with alignment step"
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
    args = parser.parse_args()

    init_logging()

    # --- Create devices ---
    # warmup_timeout_s=120 covers the libsurvive solver convergence wait
    # (the Connect RPC blocks until get_pose() returns valid data, up to 60s).
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
    logger.info(
        "(Connect blocks until tracker solver converges — "
        "may take 30-60s on first connect)"
    )
    teleop.connect()
    logger.info("Pika Sense leader connected.")

    logger.info("Connecting to MuJoCo follower at %s ...", args.follower_address)
    robot.connect()
    logger.info("MuJoCo follower connected.")

    # --- Alignment step ---
    print("\n" + "=" * 55)
    print("  ✅ Tracker 就绪！")
    print("  拿起 Pika Sense，保持自然朝前握持姿势")
    print("  按 Enter 完成对齐，开始遥操")
    print("=" * 55)
    input()

    teleop.set_reference()
    logger.info("Reference set — delta origin locked to current tracker pose.")

    # --- Teleop loop ---
    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    print("\n🤖 遥操已开始！移动 Pika Sense 控制 SO-101。Ctrl+C 退出。\n")

    display_len = max(len(key) for key in robot.action_features)
    period = 1.0 / args.fps

    try:
        while True:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            raw_action: RobotAction = teleop.get_action()
            teleop_action = teleop_action_processor((raw_action, obs))
            robot_action = robot_action_processor((teleop_action, obs))
            robot.send_action(robot_action)

            # --- Live display ---
            print("\n" + "-" * (display_len + 12))
            print(f"{'NAME':<{display_len}} | {'VALUE':>8}")
            for key, value in robot_action.items():
                print(f"{key:<{display_len}} | {value:>8.4f}")
            move_cursor_up(len(robot_action) + 3)

            dt = time.perf_counter() - loop_start
            precise_sleep(max(period - dt, 0.0))
            loop_ms = (time.perf_counter() - loop_start) * 1000
            print(f"Loop: {loop_ms:.1f}ms ({1000.0 / max(loop_ms, 0.1):.0f}Hz)")
            move_cursor_up(1)

    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        teleop.disconnect()
        robot.disconnect()
        logger.info("Disconnected.")


if __name__ == "__main__":
    main()
