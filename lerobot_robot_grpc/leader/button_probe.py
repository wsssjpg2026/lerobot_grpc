"""Pure logic behind examples/button_probe.py (wayfinder pika-sense-real #06).

The bench answers: does one physical press of the real Pika button produce
exactly one 0/1 transition in the SDK-reported Command state (latched-toggle
semantics -- the working hypothesis, since sim #11 saw 4 presses -> 4 toggles
at 30 Hz through the full chain), or does the device report raw button level
or bounce that the leader's dual-edge command_state_edge() would misread?
The example client records every serial frame's Command value; this module
turns that recording into frame-level edges, fixed-rate poller replays and
missed-edge reports -- all hardware-free and unit-tested.

Timestamps are time.monotonic() seconds taken per frame; values are the
device's Command field, 0/1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# (monotonic seconds, Command value) as reported for one serial frame.
Frame = tuple[float, int]

# (monotonic seconds, AS5047 angle deg, AS5047 rad) as reported per frame.
Pose = tuple[float, float, float]

# Ticks landing within this of a frame timestamp count as "at" it (float math).
_EPS = 1e-9

# Squeeze segmentation (deg): below ENTER = squeezing, above EXIT = released.
SQUEEZE_ENTER_DEG = 50.0
SQUEEZE_EXIT_DEG = 90.0
# "Closed to the bottom" in gripper opening (mm).
CLOSED_MM = 5.0


@dataclass(frozen=True)
class Edge:
    """One 0<->1 transition in the frame timeline."""

    t: float
    prev: int
    new: int


def frame_edges(frames: Sequence[Frame]) -> list[Edge]:
    """All transitions in the recording; consecutive equal values are ignored."""
    return [
        Edge(t=t, prev=v_prev, new=v)
        for (t_prev, v_prev), (t, v) in zip(frames, frames[1:])
        if v != v_prev
    ]


def replay_samples(frames: Sequence[Frame], rate_hz: float) -> list[Frame]:
    """Simulate a fixed-rate poller of get_command_state() over the recording.

    Ticks start at the first frame's timestamp with period 1/rate_hz and span
    the recording; the value at a tick is the last frame at or before it --
    exactly what the SDK's cached command_state would return.  A state change
    that reverts before the next tick (sub-period pulse) therefore vanishes.
    """
    if not frames or rate_hz <= 0:
        return []
    period = 1.0 / rate_hz
    t_first, t_last = frames[0][0], frames[-1][0]
    samples: list[Frame] = []
    i = 0  # frames[i-1] is the newest frame at or before the current tick
    k = 0
    while True:
        t = t_first + k * period
        if t > t_last + _EPS:
            return samples
        while i < len(frames) and frames[i][0] <= t + _EPS:
            i += 1
        samples.append((t, frames[i - 1][1]))
        k += 1


def transition_count(samples: Sequence[Frame]) -> int:
    """Changes between consecutive poller samples (production CLUTCH toggles)."""
    return sum(1 for (_, a), (_, b) in zip(samples, samples[1:]) if a != b)


def missed_edges(frames: Sequence[Frame], rate_hz: float) -> list[Edge]:
    """Frame-level edges a fixed-rate poller would never observe.

    The poller sees only its sampled sequence: an observed transition is
    caused by the LAST frame edge before the tick whose value changed.  An
    edge is observed iff it plays exactly that role; a sub-period pulse
    (0->1->0 between two ticks) cancels itself and both its edges are
    missed -- the poller never sees any transition at all.
    """
    edges = frame_edges(frames)
    if not edges:
        return []
    samples = replay_samples(frames, rate_hz)
    observed: set[int] = set()
    last_edge_idx = -1
    prev_value = samples[0][1] if samples else None
    for t, value in samples[1:]:
        for ei in range(last_edge_idx + 1, len(edges)):
            if edges[ei].t <= t + _EPS:
                last_edge_idx = ei
            else:
                break
        if last_edge_idx >= 0 and value != prev_value:
            observed.add(last_edge_idx)
        prev_value = value
    return [edge for i, edge in enumerate(edges) if i not in observed]


def cluster_edges(edges: Sequence[Edge], min_gap_s: float) -> list[list[Edge]]:
    """Group consecutive edges into bursts; a gap >= min_gap_s starts a new one."""
    clusters: list[list[Edge]] = []
    for edge in edges:
        if clusters and edge.t - clusters[-1][-1].t < min_gap_s:
            clusters[-1].append(edge)
        else:
            clusters.append([edge])
    return clusters


def gripper_mm(angle_rad: float) -> float:
    """AS5047 rad -> gripper opening in mm (replica of pika Sense.get_distance,
    pika/sense.py:154-171: (get_distance(a) - get_distance(0)) * 2)."""
    def width(angle: float) -> float:
        angle = (180.0 - 43.99) / 180.0 * math.pi - angle
        height = 0.0325 * math.sin(angle)
        width_d = 0.0325 * math.cos(angle)
        return math.sqrt(0.058**2 - (height - 0.01456)**2) + width_d
    return (width(angle_rad) - width(0.0)) * 2.0 * 1000.0


@dataclass(frozen=True)
class Squeeze:
    """One squeeze: gripper angle dropped below ENTER_DEG then re-opened."""

    t0: float
    t1: float
    min_angle_deg: float
    min_mm: float
    bottom_dwell_s: float  # time gripper stayed below CLOSED_MM
    triggered: bool  # Command flipped inside this squeeze


def squeeze_events(poses: Sequence[Pose], commands: Sequence[int]) -> list[Squeeze]:
    """Segment the recording into squeezes and their trigger outcome.

    A squeeze starts when the angle drops below SQUEEZE_ENTER_DEG and ends
    when it re-rises above SQUEEZE_EXIT_DEG; ``triggered`` is True when the
    Command value changed during the squeeze (the firmware flips it near the
    release of a qualifying closure).
    """
    events: list[Squeeze] = []
    inside = False
    start = bottom0 = bottom1 = 0
    for i, (t, angle_deg, angle_rad) in enumerate(poses):
        if not inside and angle_deg < SQUEEZE_ENTER_DEG:
            inside, start, bottom0, bottom1 = True, i, -1, -1
        elif inside and angle_deg > SQUEEZE_EXIT_DEG:
            inside = False
            seg_poses = poses[start:i + 1]
            seg_cmds = commands[start:i + 1]
            dists = [gripper_mm(rad) for _, _, rad in seg_poses]
            for j, d in enumerate(dists):
                if d < CLOSED_MM:
                    if bottom0 < 0:
                        bottom0 = j
                    bottom1 = j
            dwell = (seg_poses[bottom1][0] - seg_poses[bottom0][0]) if bottom0 >= 0 else 0.0
            events.append(Squeeze(
                t0=seg_poses[0][0],
                t1=seg_poses[-1][0],
                min_angle_deg=min(p[1] for p in seg_poses),
                min_mm=min(dists),
                bottom_dwell_s=dwell,
                triggered=any(c != seg_cmds[0] for c in seg_cmds),
            ))
    return events
