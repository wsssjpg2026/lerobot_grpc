#!/usr/bin/env python
"""Pika Sense pose-quality probe — leader only, no follower (wayfinder #05/#07).

The follower is out of the loop on purpose: this probe measures what the
LEADER publishes, so tracker noise / jump glitches / the R_lh2base direction
mapping are judged on leader data alone.  Four guided phases, one CSV row per
tick (30 Hz):

- ``still``   hold the device still in the teleop grip -> noise floor (the
  base number the #05 static-follow criterion is written against);
- ``forward`` / ``left`` / ``up``  slow ~10cm moves along each room axis ->
  direction + gain check for the #07 calibration question (published delta
  should be ``pos_gain * 10cm`` along the matching base axis).

Each phase re-latches the origin (``set_reference``), so every phase starts
from ~zero offset.  The follower server may be running or not — nothing here
touches it.

Usage::

    conda run --no-capture-output -n lerobot-grpc-serve \\
        python examples/probe_pika_pose.py --csv /tmp/pika_probe.csv \\
        2>&1 | tee /tmp/probe_client.log
"""

import argparse
import csv as csv_mod
import logging
import time

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot_robot_grpc.leader.config_grpc import GRPCLeaderConfig
from lerobot_robot_grpc.leader.grpc_leader import GRPCLeader

logger = logging.getLogger(__name__)

_PHASES = [
    ("still", "把 pika sense 拿在手里，保持遥操握姿，完全不动（30 秒）", 30.0),
    ("forward", "缓慢向前（远离自己）推约 10cm → 停 2 秒 → 缓慢回到原位", 14.0),
    ("left", "缓慢向左移约 10cm → 停 2 秒 → 缓慢回到原位", 14.0),
    ("up", "缓慢向上抬约 10cm → 停 2 秒 → 缓慢回到原位", 14.0),
]

_COLS = ["t_s", "phase", "dx_m", "dy_m", "dz_m",
         "qx", "qy", "qz", "qw", "grip_mm"]


def main():
    parser = argparse.ArgumentParser(description="Pika Sense pose-quality probe (#05/#07)")
    parser.add_argument("--address", default="127.0.0.1:5556")
    parser.add_argument("--leader-id", default="pika_sense")
    parser.add_argument("--csv", default="/tmp/pika_probe.csv")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    init_logging()

    teleop = GRPCLeader(GRPCLeaderConfig(
        address=args.address, id=args.leader_id, warmup_timeout_s=120.0,
    ))
    logger.info("Connecting to Pika Sense leader at %s ...", args.address)
    teleop.connect()

    with open(args.csv, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(_COLS)
        t0 = time.monotonic()
        for name, desc, dur in _PHASES:
            print("\n" + "=" * 55)
            print("  阶段 [%s]：%s" % (name, desc))
            print("  按 Enter 开始（阶段开始时重锁原点）")
            print("=" * 55)
            input()
            teleop.set_reference()
            start = time.monotonic()
            print("[%s] 录制中 %.0f 秒 -> %s" % (name, dur, args.csv))
            last_flush = 0.0
            while time.monotonic() - start < dur:
                a = teleop.get_action()
                writer.writerow([
                    f"{time.monotonic() - t0:.3f}", name,
                    f"{a['hand.delta_pos.x']:.5f}", f"{a['hand.delta_pos.y']:.5f}",
                    f"{a['hand.delta_pos.z']:.5f}",
                    f"{a['hand.delta_rot.qx']:.5f}", f"{a['hand.delta_rot.qy']:.5f}",
                    f"{a['hand.delta_rot.qz']:.5f}", f"{a['hand.delta_rot.qw']:.5f}",
                    f"{a['gripper.distance']:.2f}",
                ])
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    f.flush()
                    last_flush = now
                    print("  [%s] 剩 %4.1fs  pub=[%6.1f %6.1f %6.1f]mm" % (
                        name, start + dur - now,
                        a["hand.delta_pos.x"] * 1000.0,
                        a["hand.delta_pos.y"] * 1000.0,
                        a["hand.delta_pos.z"] * 1000.0))
                precise_sleep(1.0 / args.fps)

    teleop.disconnect()
    logger.info("Probe done: %s", args.csv)


if __name__ == "__main__":
    main()
