"""Joint-smoke gate client -- HITL, drives a REAL SO-101 over gRPC (#05).

Protocol locked in .scratch/pika-sense-real/issues/05-joint-smoke-gate.md:
server joint mode; base pose = power-on pose (never HOME); per joint
shoulder_pan..wrist_roll +/-5 then +/-10 with 5deg-margin pre-check (skip,
never clip); gripper 0->100->original; stream-cut mid-scan 3s + kill -9.
Pass: steady <=2deg, no oscillation, overshoot <=2deg, drift <=0.5deg.

Round-3 amendment (criteria unchanged, send waveform only): every action is
a FULL dict of all 6 joints + gripper, and ramp mode (default) streams the
moving joint's goal in <=ramp-step-deg steps at ramp-hz -- the same waveform
as official lerobot-teleoperate.  Full dicts are mandatory because
GRPCFollower zero-initialises its action cache at every connect and merges
partial dicts into it: a single-joint send would command the other joints
to 0.0 (the calibration mid = folded rest pose), yanking them after every
reconnect.

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
    check_base_headroom,
    evaluate_dwell,
    evaluate_freeze,
    limits_from_raw_ranges,
    ramp_goal,
    retrying_rpc,
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


def _reconnect(client: GRPCFollower) -> None:
    try:
        client.disconnect()
    except Exception:
        pass  # server may already see us gone; channel cleanup is best-effort
    client.connect(calibrate=False)


def _obs_joints(client: GRPCFollower) -> dict[str, float]:
    def _note(e: BaseException) -> None:
        print("  [reconnect] observation failed (" + repr(e)[:120] + "), retrying")
    obs = retrying_rpc(lambda: client.get_observation(),
                       reconnect=lambda: _reconnect(client), on_retry=_note)
    return {j: float(obs[j + ".pos"]) for j in (*BODY_SCAN_ORDER, "gripper")}


def _send(client: GRPCFollower, action: dict[str, float]) -> None:
    def _note(e: BaseException) -> None:
        print("  [reconnect] send failed (" + repr(e)[:120] + "), retrying")
    retrying_rpc(lambda: client.send_action(action),
                 reconnect=lambda: _reconnect(client), on_retry=_note)


def _send_joint(client: GRPCFollower, command_state: dict[str, float],
                key: str, target: float, joint: str, args) -> None:
    """Drive one joint to target with FULL-dict actions; ramp unless told not to.

    command_state holds the last commanded value of every feature and is sent
    in full each time, so the server-side zero-initialised action cache can
    never leak a 0.0 onto joints we did not mean to move.  ramp mode streams
    the goal in small steps at a fixed rate (teleoperate waveform) and stops
    pushing early if the joint stops making progress -- the dwell then judges.
    """
    def send_value(value: float) -> None:
        _send(client, {**command_state, key: value})
        command_state[key] = value

    if args.send_mode == "step":
        send_value(target)
        return

    max_step = args.ramp_step_gripper if joint == "gripper" else args.ramp_step_deg
    start = command_state[key]
    direction = 1.0 if target >= start else -1.0
    check_t = time.perf_counter()
    check_pos: float | None = None
    next_t = time.perf_counter()
    for goal in ramp_goal(start, target, max_step):
        send_value(goal)
        now = time.perf_counter()
        if now - check_t >= 1.0:
            pos = _obs_joints(client)[joint]
            if check_pos is not None and (pos - check_pos) * direction < 0.3:
                print("  [ramp] " + joint + " stalled at "
                      + format(pos, ".1f") + " -- holding here, dwell will judge")
                break
            check_pos, check_t = pos, now
        next_t += 1.0 / args.ramp_hz
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)


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
    command_state = {j + ".pos": base[j] for j in (*BODY_SCAN_ORDER, "gripper")}
    if args.repose_json:
        # Motor-driven repositioning from the (torque-on) connect pose to a
        # mid-travel base -- hands-free for the human: the servos hold every
        # intermediate pose.  Order lifts the arm before folding the elbow,
        # both directions proven smooth on the real arm.
        repose = json.loads(args.repose_json)
        order = [j for j in ("shoulder_lift", "shoulder_pan", "elbow_flex",
                             "wrist_flex", "wrist_roll") if j in repose]
        print("repose to:", {j: repose[j] for j in order})
        for joint in order:
            _send_joint(client, command_state, joint + ".pos",
                        float(repose[joint]), joint, args)
            time.sleep(1.0)
        base = _obs_joints(client)
        print("reposed base pose (deg):", {k: round(v, 1) for k, v in base.items()})
        command_state = {j + ".pos": base[j] for j in (*BODY_SCAN_ORDER, "gripper")}
        measured = json.loads(args.measured_travel) if args.measured_travel else None
        pre = check_base_headroom(base, limits, measured,
                                  required_deg=args.required_headroom_deg)
        for r in pre:
            print("  preflight " + r.joint.ljust(13)
                  + " min " + format(r.min_headroom_deg, "+7.1f")
                  + "  " + ("PASS" if r.passed else "FAIL"))
        if not all(r.passed for r in pre):
            print("PREFLIGHT FAIL after repose -- aborting before any scan step")
            return 1
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
        _send_joint(client, command_state, key, step.target_deg, step.joint, args)
        go = _dwell(client, args.dwell_s)
        log(go, "scan", label)
        res = evaluate_dwell([r[step.joint] for r in go], step.target_deg)
        print("  go: steady=" + format(res.steady_error_deg, ".2f")
              + " over=" + format(res.overshoot_deg, ".2f")
              + " osc=" + format(res.end_oscillation_deg, ".2f")
              + (" PASS" if res.passed else " FAIL"))
        if not res.passed:
            failures.append(label + "-go")
        _send_joint(client, command_state, key, base[step.joint], step.joint, args)
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
    _send_joint(client, command_state, "gripper.pos", 0.0, "gripper", args)
    log(_dwell(client, args.dwell_s), "gripper", "close")
    _send_joint(client, command_state, "gripper.pos", 100.0, "gripper", args)
    log(_dwell(client, args.dwell_s), "gripper", "open")
    _send_joint(client, command_state, "gripper.pos", base["gripper"], "gripper", args)
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

def run_preflight(args) -> int:
    """Base-pose gate before any scan: >= required headroom both ways per joint.

    Calibrated limits proved insufficient on the real arm (round 1: elbow
    parked on a mechanical wall inside a +/-180 deg calibrated range), so
    --measured-travel takes hand-measured free travel from the CURRENT pose
    as JSON of room magnitudes in deg, e.g. '{"elbow_flex": [40, 30]}' =
    40 deg available toward minus, 30 toward plus.
    """
    limits = _load_limits(Path(args.calibration))
    measured = json.loads(args.measured_travel) if args.measured_travel else None
    client = GRPCFollower(GRPCFollowerConfig(address=args.address, need_warmup=False))
    client.connect(calibrate=False)
    base = _obs_joints(client)
    print("base pose (deg):", {k: round(v, 1) for k, v in base.items()})
    results = check_base_headroom(base, limits, measured,
                                  required_deg=args.required_headroom_deg)
    all_ok = True
    for r in results:
        travel = "        n/a" if r.travel_deg is None \
            else format(r.travel_deg[0], "+7.1f") + format(r.travel_deg[1], "+7.1f")
        print("  " + r.joint.ljust(13)
              + " cal " + format(r.headroom_deg[0], "+7.1f") + format(r.headroom_deg[1], "+7.1f")
              + "  hand" + travel
              + "  min " + format(r.min_headroom_deg, "+7.1f")
              + "  " + ("PASS" if r.passed else "FAIL"))
        if r.travel_deg is None:
            print("    note: no hand-measured travel for " + r.joint
                  + " -- calibrated limits only (mechanical walls invisible)")
        all_ok = all_ok and r.passed
    print("PREFLIGHT " + ("PASS (>= " + format(args.required_headroom_deg, ".0f")
                          + " deg both ways everywhere)" if all_ok else "FAIL -- reposition the arm"))
    return 0 if all_ok else 1


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
    p.add_argument("--mode", choices=["preflight", "scan", "observe"], default="scan")
    p.add_argument("--address", default="127.0.0.1:5555")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    p.add_argument("--out-csv", default="/tmp/joint_smoke_" + str(int(time.time())) + ".csv")
    p.add_argument("--tiers", default="5,10")
    p.add_argument("--margin-deg", type=float, default=5.0)
    p.add_argument("--dwell-s", type=float, default=2.0)
    p.add_argument("--silence-fraction", type=float, default=0.5)
    p.add_argument("--repose-json", default="",
                   help='JSON {joint: deg} motor-driven repositioning to a mid-travel'
                        ' base after connect, e.g. \'{"shoulder_lift": 40, "elbow_flex": -30}\''
                        " (scan mode; arm holds itself, hands-free for the human)")
    p.add_argument("--measured-travel", default="",
                   help="JSON {joint: [minus_room_deg, plus_room_deg]} hand-measured"
                        " free travel from the current pose (preflight and the"
                        " post-repose check in scan mode)")
    p.add_argument("--required-headroom-deg", type=float, default=15.0)
    p.add_argument("--send-mode", choices=["ramp", "step"], default="ramp",
                   help="ramp: stream <=ramp-step goals at ramp-hz (lerobot-teleoperate"
                        " waveform); step: round-1 one-shot behaviour for A/B")
    p.add_argument("--ramp-hz", type=float, default=60.0)
    p.add_argument("--ramp-step-deg", type=float, default=0.25,
                   help="max per-tick goal change in deg for body joints (ramp mode)")
    p.add_argument("--ramp-step-gripper", type=float, default=10.0,
                   help="max per-tick goal change in 0-100 units for the gripper")
    args = p.parse_args()
    if args.mode == "observe":
        return run_observe(args)
    if args.mode == "preflight":
        return run_preflight(args)
    return run_scan(args)


if __name__ == "__main__":
    sys.exit(main())
