"""Pika button edge bench -- HITL, records the REAL button's serial frames (#06).

Protocol locked in issue #06:
one recording window; the human presses 5 slow (~0.5-1 s hold, >=2 s apart),
5 quick taps, 2 double-clicks (~300 ms pairs) and one 5 s hold.  The script
logs EVERY serial frame's Command value with monotonic timestamps -- the
same Sense class the leader server consumes, so the measurement is of the
production path -- flashes the tracker LED on each observed edge, then
reports frame-level edges, 30/60 Hz poller replays and missed edges.
Rounds R3/R4 replaced the instructed protocol with free/natural gestures
plus raw-frame recording (the trigger turned out to be gesture-profile
dependent; see the ticket's Answer).

Verdict criteria (fixed before the bench): exactly 1 frame-level edge per
press and no <50 ms burst -> latched toggle, no debounce; paired edges
hold-length apart -> level semantics; <50 ms bursts -> bounce.

!! Pika controller only (/dev/ttyUSB0).  The SO-101 arm is /dev/ttyACM0 --
never open it here.

Run (from lerobot_grpc/, env lerobot-grpc-serve):
    python examples/button_probe.py --duration 90 --csv /tmp/button_probe_r1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_grpc.leader.button_probe import (
    cluster_edges,
    frame_edges,
    gripper_mm,
    missed_edges,
    replay_samples,
    squeeze_events,
    transition_count,
)
from pika.sense import Sense  # pyright: ignore[reportMissingImports]  # editable install

DEFAULT_PORT = "/dev/ttyUSB0"
REPLAY_RATES = (30.0, 60.0)  # leader GetAction rates: teleop default 30, smoke 60
# Verdict criteria: bursts of edges closer than this = bounce.
BOUNCE_WINDOW_S = 0.05
EDGE_LIGHTS = (3, 4)  # blue / yellow, alternating per edge (official driver flashes blue)


class ProbeSense(Sense):
    """Sense that logs (monotonic t, Command) per frame plus every raw frame.

    Final #06 reading (R1-R4): Command is a rare LATCHED toggle that flips at
    the release of a qualifying closure; most closures (all instructed
    variants) leave no trace in it.  So every frame is kept raw (jsonl) for
    offline re-analysis -- the R1/R2 "long holds pass through as a level
    pair" reading was a misread of that latching.
    """

    def __init__(self, port: str, feedback: bool = True) -> None:
        super().__init__(port=port)
        self.frame_log: list[tuple[float, int]] = []
        self.pose_log: list[tuple[float, float, float]] = []
        self.raw_log: list[tuple[float, dict]] = []
        self.feedback = feedback
        self._flash = 0

    def _data_callback(self, data: dict) -> None:
        super()._data_callback(data)
        t = time.monotonic()
        self.raw_log.append((t, data))
        encoder = data.get("AS5047")
        if encoder is not None:
            self.pose_log.append((t, float(encoder.get("angle", 0.0)),
                                  float(encoder.get("rad", 0.0))))
        if "Command" not in data:
            return
        value = int(data["Command"])
        if self.frame_log and value != self.frame_log[-1][1] and self.feedback:
            try:  # LED flash is cosmetic; it must never kill the read callback
                self.light_ctrl(EDGE_LIGHTS[self._flash % len(EDGE_LIGHTS)])
                self._flash += 1
            except Exception:
                pass
        self.frame_log.append((t, value))


def _pulse(sense: ProbeSense, light_id: int) -> None:
    """Start/end cue: flash a light colour and buzz briefly (not continuously)."""
    try:
        sense.light_ctrl(light_id)
        sense.vibrate_ctrl(1)
        time.sleep(0.3)
        sense.vibrate_ctrl(0)
    except Exception:
        pass  # cues are cosmetic


def record(args: argparse.Namespace) -> list[tuple[float, dict]]:
    sense = ProbeSense(args.port, feedback=not args.no_feedback)
    if not sense.connect():
        print("CONNECT FAIL on " + args.port
              + " -- wrong port, device off, or a leader server is holding it")
        raise SystemExit(1)
    print("connected; cached Command value: " + str(sense.get_command_state()))
    if not args.no_feedback:
        _pulse(sense, 2)  # green + buzz: recording starts now
    deadline = time.monotonic() + args.duration
    print("RECORDING " + str(args.duration) + " s -- press per protocol; Ctrl+C ends early")
    shown = 0
    grip_t = time.monotonic()
    try:
        while time.monotonic() < deadline:
            edges = frame_edges(sense.frame_log)
            while shown < len(edges):
                e = edges[shown]
                origin = sense.frame_log[0][0] if sense.frame_log else 0.0
                print("EDGE  +%07.3fs  %d->%d" % (e.t - origin, e.prev, e.new), flush=True)
                shown += 1
            if sense.pose_log and time.monotonic() - grip_t >= 2.0:
                grip_t = time.monotonic()
                _, angle_deg, angle_rad = sense.pose_log[-1]
                print("GRIP   %6.1f mm (angle %5.1f deg)"
                      % (gripper_mm(angle_rad), angle_deg), flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("interrupted -- analysing what was recorded")
    if not args.no_feedback:
        _pulse(sense, 4)  # yellow + buzz: recording over
    sense.disconnect()
    frames = sense.frame_log
    origin = frames[0][0] if frames else 0.0
    with open(args.csv, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["t_s", "command", "angle_deg", "gripper_mm"])
        poses = sense.pose_log
        pose_times = [p[0] for p in poses]
        for t, value in frames:
            j = bisect_right(pose_times, t) - 1  # last pose at or before this frame
            if j >= 0:
                writer.writerow([format(t - origin, ".6f"), value,
                                 poses[j][1], gripper_mm(poses[j][2])])
            else:  # Command frame before any AS5047 frame
                writer.writerow([format(t - origin, ".6f"), value, "", ""])
    raw_path = args.csv + ".raw.jsonl"
    with open(raw_path, "w") as out:
        for t, data in sense.raw_log:
            out.write(json.dumps({"t_s": round(t - origin, 6), **data},
                                 separators=(",", ":")) + "\n")
    print("CSV: " + args.csv + "  (" + str(len(frames)) + " frames)")
    print("RAW: " + raw_path + "  (" + str(len(sense.raw_log)) + " frames)")
    return sense.raw_log


def report(raw_rows: list[tuple[float, dict]]) -> int:
    frames = [(t, int(d["Command"])) for t, d in raw_rows if "Command" in d]
    poses = [(t, float(d["AS5047"]["angle"]), float(d["AS5047"]["rad"]))
             for t, d in raw_rows if "AS5047" in d]
    if not frames:
        print("NO FRAMES recorded -- device not streaming?")
        return 1
    origin = frames[0][0]
    duration = frames[-1][0] - frames[0][0]
    idle_mm = gripper_mm(poses[0][2]) if poses else float("nan")
    print("================ SUMMARY ================")
    print("frames: %d  duration: %.1fs  rate: %.1f fps  initial Command: %d  idle grip: %.1f mm"
          % (len(frames), duration, len(frames) / duration if duration > 0 else 0.0,
             frames[0][1], idle_mm))
    edges = frame_edges(frames)
    print("frame-level edges: %d" % len(edges))
    prev_t: float | None = None
    for i, e in enumerate(edges):
        gap = "" if prev_t is None else "  (gap %+.3fs)" % (e.t - prev_t)
        print("  #%02d  +%07.3fs  %d->%d%s" % (i + 1, e.t - origin, e.prev, e.new, gap))
        prev_t = e.t
    clusters = cluster_edges(edges, BOUNCE_WINDOW_S)
    bursts = [c for c in clusters if len(c) > 1]
    min_gap_ms = min(((b.t - a.t) * 1000.0 for a, b in zip(edges, edges[1:])),
                     default=None)
    print("bounce check (<%.0f ms bursts): %s  (min inter-edge gap %s)"
          % (BOUNCE_WINDOW_S * 1000, "%d BURSTS" % len(bursts) if bursts else "clean",
             "n/a" if min_gap_ms is None else "%.0f ms" % min_gap_ms))
    if poses:
        # Command of the last Command-bearing frame at or before each pose
        # frame -- timestamps are exact floats from the same callback, but a
        # frame missing either field must not desynchronise the two streams.
        frame_times = [t for t, _ in frames]
        commands = [frames[max(bisect_right(frame_times, p[0]) - 1, 0)][1]
                    for p in poses]
        events = squeeze_events(poses, commands)
        print("squeeze events (glove closures): %d" % len(events))
        for i, ev in enumerate(events):
            print("  #%02d  +%07.3fs  closed %.2fs  min %5.1f deg / %4.1f mm  %s"
                  % (i + 1, ev.t0 - origin, ev.t1 - ev.t0, ev.min_angle_deg,
                     ev.min_mm, "TRIGGER" if ev.triggered else ""))
    for rate_hz in REPLAY_RATES:
        samples = replay_samples(frames, rate_hz)
        missed = missed_edges(frames, rate_hz)
        line = "replay @%02.0fHz: %d transitions" % (rate_hz, transition_count(samples))
        if missed:
            line += ("  MISSED %d: " % len(missed)
                     + ", ".join("+%.3fs %d->%d" % (m.t - origin, m.prev, m.new)
                                 for m in missed))
        print(line)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--duration", type=float, default=90.0)
    p.add_argument("--csv", default="/tmp/button_probe.csv")
    p.add_argument("--no-feedback", action="store_true",
                   help="no LED flash / vibration cues")
    args = p.parse_args()
    raw_rows = record(args)
    return report(raw_rows)


if __name__ == "__main__":
    sys.exit(main())
