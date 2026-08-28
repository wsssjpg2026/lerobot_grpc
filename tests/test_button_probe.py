"""Pure-logic tests for the #06 button-edge bench.

The HITL bench itself lives in examples/button_probe.py (it opens the real
Pika on /dev/ttyUSB0 -- humans pressing the button).  Everything decidable
without hardware lives here: frame timeline -> edges, fixed-rate poller
replay (what a 30/60 Hz consumer of get_command_state() would see), and
missed-edge detection (sub-period pulses that vanish at the poll rate).

Verdict criteria locked before the bench (issue #06): exactly 1 frame-level
edge per press and no <50 ms burst -> latched toggle, no debounce; paired
edges hold-length apart -> level semantics; <50 ms bursts -> bounce.
"""

from __future__ import annotations

import math

import pytest

from lerobot_robot_grpc.leader.button_probe import (
    Edge,
    cluster_edges,
    frame_edges,
    gripper_mm,
    missed_edges,
    replay_samples,
    squeeze_events,
    transition_count,
)

P30 = 1.0 / 30.0


def test_frame_edges_empty_and_constant():
    assert frame_edges([]) == []
    assert frame_edges([(0.0, 0), (0.01, 0), (0.02, 0)]) == []


def test_frame_edges_single_flip_then_back():
    frames = [(0.0, 0), (0.5, 1), (2.9, 1), (3.0, 0)]
    assert frame_edges(frames) == [
        Edge(t=0.5, prev=0, new=1),
        Edge(t=3.0, prev=1, new=0),
    ]


def test_frame_edges_alternating():
    frames = [(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 1)]
    assert frame_edges(frames) == [
        Edge(t=1.0, prev=0, new=1),
        Edge(t=2.0, prev=1, new=0),
        Edge(t=3.0, prev=0, new=1),
    ]


def test_replay_samples_sees_last_frame_before_each_tick():
    # Frames flip at t=0.5; every 30 Hz tick from 0.5333 on must see 1.
    frames = [(0.0, 0), (0.5, 1), (1.0, 1)]
    samples = replay_samples(frames, 30.0)
    assert samples[0] == (0.0, 0)
    assert samples[14] == (14 * P30, 0)   # t=0.4667 still before the flip
    assert samples[15] == (15 * P30, 1)   # t=0.5 == flip time: that frame counts
    assert samples[-1] == (30 * P30, 1)   # ticks span the full recording


def test_replay_samples_subperiod_pulse_vanishes():
    # 0->1 at 0.010 and back at 0.015: no 30 Hz tick lands in [0.010, 0.015).
    frames = [(0.0, 0), (0.010, 1), (0.015, 0), (1.0, 0)]
    assert transition_count(replay_samples(frames, 30.0)) == 0


def test_replay_samples_hold_level_press_is_two_transitions():
    # Level semantics: press 0.1-0.2 s (100 ms) -> poller sees 0,1,0.
    frames = [(0.0, 0), (0.1, 1), (0.2, 0), (1.0, 0)]
    assert transition_count(replay_samples(frames, 30.0)) == 2


def test_replay_samples_latched_toggle_is_one_transition():
    # Toggle semantics: state flips at 0.1 and stays -> exactly one transition.
    frames = [(0.0, 0), (0.1, 1), (1.0, 1)]
    assert transition_count(replay_samples(frames, 30.0)) == 1


def test_missed_edges_short_pulse_both_missed():
    frames = [(0.0, 0), (0.010, 1), (0.015, 0), (1.0, 0)]
    missed = missed_edges(frames, 30.0)
    assert [(e.prev, e.new) for e in missed] == [(0, 1), (1, 0)]


def test_missed_edges_persistent_flip_not_missed():
    frames = [(0.0, 0), (0.1, 1), (1.0, 1)]
    assert missed_edges(frames, 30.0) == []


def test_missed_edges_100ms_level_press_observed():
    frames = [(0.0, 0), (0.1, 1), (0.2, 0), (1.0, 0)]
    assert missed_edges(frames, 30.0) == []


def test_transition_count():
    assert transition_count([]) == 0
    assert transition_count([(0.0, 0)]) == 0
    assert transition_count([(0.0, 0), (P30, 1), (2 * P30, 1), (3 * P30, 0)]) == 2


def test_cluster_edges_splits_on_min_gap():
    edges = [
        Edge(t=1.0, prev=0, new=1),
        Edge(t=1.01, prev=1, new=0),   # 10 ms later: same bounce burst
        Edge(t=2.0, prev=0, new=1),    # 990 ms later: separate interaction
    ]
    clusters = cluster_edges(edges, min_gap_s=0.05)
    assert [len(c) for c in clusters] == [2, 1]


def test_cluster_edges_exact_boundary_starts_new_cluster():
    # Gap == min_gap is NOT one burst: [t_i, t_{i+1}) groupings use strict <.
    edges = [Edge(t=1.0, prev=0, new=1), Edge(t=1.05, prev=1, new=0)]
    assert [len(c) for c in cluster_edges(edges, min_gap_s=0.05)] == [1, 1]


def test_gripper_mm_matches_sdk_formula():
    # Same literals as pika Sense.get_distance (the formula get_gripper_distance
    # applies to the live angle): 0 rad -> 0 mm closed,
    # the R3 idle angle 97.63 deg -> ~98.1 mm open.
    assert gripper_mm(0.0) == pytest.approx(0.0, abs=1e-9)
    assert gripper_mm(math.radians(97.63)) == pytest.approx(98.1, abs=0.5)


def _squeeze_rows(t0: float, closed_s: float, command_flip: bool):
    """Synthesize poses+commands for one closure: open -> closed -> open."""
    poses: list[tuple[float, float, float]] = []
    cmds: list[int] = []
    t = t0
    for _ in range(2):  # open approach (100 deg)
        poses.append((t, 100.0, math.radians(100.0))); cmds.append(1); t += 0.04
    n_closed = max(2, int(closed_s / 0.04))
    for _ in range(n_closed):
        poses.append((t, 0.0, 0.0)); cmds.append(1); t += 0.04
    for _ in range(2):  # re-open
        poses.append((t, 100.0, math.radians(100.0))); cmds.append(1); t += 0.04
    if command_flip:  # firmware flips near the release
        cmds[-2] = 0
    return poses, cmds


def test_squeeze_events_dwell_and_trigger():
    poses, cmds = [], []
    for t0, closed, flip in ((1.0, 0.32, True),      # in the trigger band
                             (5.0, 0.12, False),     # too short
                             (9.0, 1.20, False)):    # too long
        p, c = _squeeze_rows(t0, closed, flip)
        poses += p; cmds += c
    events = squeeze_events(poses, cmds)
    assert len(events) == 3
    assert [e.triggered for e in events] == [True, False, False]
    assert events[0].bottom_dwell_s == pytest.approx(0.32, abs=0.05)
    assert events[1].bottom_dwell_s == pytest.approx(0.12, abs=0.05)
    assert events[2].bottom_dwell_s == pytest.approx(1.20, abs=0.05)
    assert all(e.min_mm < 5.0 for e in events)


def test_squeeze_events_partial_closure_not_counted_closed():
    # Dip to 21.8 deg (17.2 mm): a squeeze event, but bottom_dwell 0 (never <5mm).
    poses: list[tuple[float, float, float]] = []
    cmds: list[int] = []
    t = 1.0
    for angle_deg in (100.0, 100.0, 21.8, 21.8, 21.8, 21.8, 100.0, 100.0):
        poses.append((t, angle_deg, math.radians(angle_deg)))
        cmds.append(1)
        t += 0.1
    events = squeeze_events(poses, cmds)
    assert len(events) == 1
    assert events[0].bottom_dwell_s == 0.0
    assert events[0].min_mm > 15.0
