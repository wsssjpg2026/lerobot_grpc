"""Joint-smoke gate client -- HITL, drives a REAL SO-101 over gRPC (#05).

Protocol locked in .scratch/pika-sense-real/issues/05-joint-smoke-gate.md:
server joint mode; base pose = power-on pose (never HOME); per joint
shoulder_pan..wrist_roll +/-5 then +/-10 with 5deg-margin pre-check (skip,
never clip); gripper 0->100->original; stream-cut mid-scan 3s + kill -9.
Pass: steady <=2deg, no oscillation, overshoot <=2deg, drift <=0.5deg.

Run (from lerobot_grpc/):
    python examples/joint_smoke_client.py --mode scan
    kill -9 <printed pid>
    python examples/joint_smoke_client.py --mode observe
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.follower.joint_smoke import (
    BODY_SCAN_ORDER,
    ScanStep,
    SkipStep,
    build_scan_steps,
    evaluate_dwell,
    evaluate_freeze,
    limits_from_raw_ranges,
)

DEFAULT_CALIBRATION = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"
)
SAMPLE_PERIOD_S = 0.1

def _load_limits(path: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(path.read_text())
    ranges = {}
    for joint, entry in raw.items():
        if joint in BODY_SCAN_ORDER:
            ranges[joint] = (int(entry["range_min"]), int(entry["range_max"]))
    return limits_from_raw_ranges(ranges)


def _obs_joints(client: GRPCFollower) -> dict[str, float]:
    obs = client.get_observation()
    return {j: float(obs[j + ".pos"]) for j in (*BODY_SCAN_ORDER, "gripper")}


def _dwell(client: GRPCFollower, seconds: float) -> list[dict[str, float]]:
    """Sample all joints at 10 Hz for seconds; one row per sample."""
    rows = []
    for _ in range(max(1, round(seconds / SAMPLE_PERIOD_S))):
        rows.append({"ts": time.time(), **_obs_joints(client)})
        time.sleep(SAMPLE_PERIOD_S)
    return rows


def run_scan(args) -> int:
    limits = _load_limits(Path(args.calibration))
    print("calibration limits (deg):",
          {j: (round(lo, 1), round(hi, 1)) for j, (lo, hi) in limits.items()})

    client = GRPCFollower(GRPCFollowerConfig(address=args.address, need_warmup=False))
    client.connect(calibrate=False)

    base = _obs_joints(client)
    print("base pose (deg):", {k: round(v, 1) for k, v in base.items()})
    tiers = tuple(float(t) for t in args.tiers.split(","))
    steps = build_scan_steps(base, limits, tiers=tiers, margin_deg=args.margin_deg)
    for s in steps:
        print("  plan:", s)
    input(">>> ARM AREA CLEAR, hand on power. Press ENTER to start the scan <<<")

    out = open(args.out_csv, "a", newline="")
    writer = None

    def log(rows, phase, step=""):
        nonlocal writer
        for r in rows:
            row = {"phase": phase, "step": step, **r}
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
        out.flush()

    failures: list[str] = []
    silence_at = round(len(steps) * args.silence_fraction)
    baseline_rows: list[dict[str, float]] = []

    for idx, step in enumerate(steps):
        if idx == silence_at:
            print("=== STREAM-CUT #1: 3 s silence (client alive, no commands) ===")
            silence = _dwell(client, 3.0)
            log(silence, "streamcut1")
            for j in BODY_SCAN_ORDER:
                res = evaluate_freeze([r[j] for r in baseline_rows][-3:],
                                      [r[j] for r in silence], tol_deg=0.5)
                print("  freeze " + j + ": drift=" + format(res.max_drift_deg, ".2f")
                      + (" PASS" if res.passed else " FAIL"))
                if not res.passed:
                    failures.append("streamcut1-" + j)
        if isinstance(step, SkipStep):
            print("SKIP " + step.joint + " " + format(step.direction * step.tier_deg, "+.0f")
                  + " deg (would be " + format(step.would_be_target_deg, ".1f")
                  + ", outside margin)")
            continue
        assert isinstance(step, ScanStep)
        key = step.joint + ".pos"
        label = step.joint + format(step.direction * step.tier_deg, "+.0f")
        print("step " + str(idx) + ": " + label + " -> " + format(step.target_deg, ".1f"))
        client.send_action({key: step.target_deg})
        go = _dwell(client, args.dwell_s)
        log(go, "scan", label)
        res = evaluate_dwell([r[step.joint] for r in go], step.target_deg)
        print("  go: steady=" + format(res.steady_error_deg, ".2f")
              + " over=" + format(res.overshoot_deg, ".2f")
              + " osc=" + format(res.end_oscillation_deg, ".2f")
              + (" PASS" if res.passed else " FAIL"))
        if not res.passed:
            failures.append(label + "-go")
        client.send_action({key: base[step.joint]})
        back = _dwell(client, args.dwell_s)
        log(back, "scan", label + "-return")
        res = evaluate_dwell([r[step.joint] for r in back], base[step.joint])
        print("  back: steady=" + format(res.steady_error_deg, ".2f")
              + " over=" + format(res.overshoot_deg, ".2f")
              + " osc=" + format(res.end_oscillation_deg, ".2f")
              + (" PASS" if res.passed else " FAIL"))
        if not res.passed:
            failures.append(label + "-back")
        baseline_rows = back

    print("=== gripper cycle 0 -> 100 -> original ===")
    client.send_action({"gripper.pos": 0.0})
    log(_dwell(client, args.dwell_s), "gripper", "close")
    client.send_action({"gripper.pos": 100.0})
    log(_dwell(client, args.dwell_s), "gripper", "open")
    client.send_action({"gripper.pos": base["gripper"]})
    log(_dwell(client, args.dwell_s), "gripper", "return")

    print("================ SUMMARY ================")
    if failures:
        print("FAILURES: " + ", ".join(failures))
    else:
        print("ALL STEPS PASS")
    print("CSV: " + args.out_csv)
    print("For stream-cut #2: kill -9 " + str(os.getpid())
          + " , then run --mode observe.")
    print("Client idle-holding (sending nothing). Ctrl+C to exit without the kill test.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        return 1 if failures else 0

def run_observe(args) -> int:
    """Post-kill observation: 3 s of joint samples; drift vs first sample."""
    client = GRPCFollower(GRPCFollowerConfig(address=args.address, need_warmup=False))
    client.connect(calibrate=False)
    rows = _dwell(client, 3.0)
    out = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=["phase", "step", *rows[0].keys()])
    for r in rows:
        writer.writerow({"phase": "streamcut2", "step": "", **r})
    out.flush()
    for j in (*BODY_SCAN_ORDER, "gripper"):
        vals = [r[j] for r in rows]
        drift = max(abs(v - vals[0]) for v in vals)
        print(j + ": first=" + format(vals[0], ".2f")
              + " max_drift=" + format(drift, ".2f") + " deg")
    print("CSV: " + args.out_csv)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["scan", "observe"], default="scan")
    p.add_argument("--address", default="127.0.0.1:5555")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    p.add_argument("--out-csv", default="/tmp/joint_smoke_" + str(int(time.time())) + ".csv")
    p.add_argument("--tiers", default="5,10")
    p.add_argument("--margin-deg", type=float, default=5.0)
    p.add_argument("--dwell-s", type=float, default=2.0)
    p.add_argument("--silence-fraction", type=float, default=0.5)
    args = p.parse_args()
    if args.mode == "observe":
        return run_observe(args)
    return run_scan(args)


if __name__ == "__main__":
    sys.exit(main())
