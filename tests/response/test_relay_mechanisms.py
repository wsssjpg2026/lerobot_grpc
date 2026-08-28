"""Layer b — relay mechanisms (the clutch/teleop plumbing around the law):

- startup handshake: GetInfo schema negotiation; no action -> no motion;
- gating: silence freezes the follower (deep assertions live in layer c);
- engage/disengage switching: SetReference re-locks the base at the stop
  pose — Δ=0 afterwards maps onto the stop pose, never back to home;
- smooth return: dropping the offset walks the arm home without a swing.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_robot_grpc.pose_delta_schema import ACTION_KEYS  # noqa: E402
from lerobot_robot_grpc.protos import device_pb2  # noqa: E402

from .injectors import Sequence, identity, with_offset  # noqa: E402
from .metrics import BODY_JOINTS  # noqa: E402


def test_startup_handshake_negotiates_pose_delta_schema(runner, backend):
    """GetInfo must present exactly the shared 8-feature schema, CRITICAL
    FLOAT32 — the contract the leader negotiates before sending anything."""
    info = backend.client.stub.GetInfo(device_pb2.GetInfoRequest())
    keys = {f.key: f for f in info.action_features}
    assert set(keys) == set(ACTION_KEYS)
    for fi in keys.values():
        assert fi.type == device_pb2.DataType.FLOAT32
        assert fi.criticality == device_pb2.Criticality.CRITICALITY_CRITICAL


def test_no_motion_before_first_action(runner, backend):
    """After Connect the follower may not move: the arm only follows once
    the first action lands (the leader's alignment gate)."""
    runner.reset_session()
    sol_count = len(backend.solutions) if backend.law_spy_available else None

    t0 = time.monotonic()
    quiet: list[dict] = []
    while time.monotonic() - t0 < 1.0:
        quiet.append(backend.client.get_observation())
        time.sleep(0.05)

    home = quiet[0]
    drift = max(
        max(abs(obs[f"{j}.pos"] - home[f"{j}.pos"]) for j in BODY_JOINTS)
        for obs in quiet
    )
    assert drift < 0.3, f"arm drifted {drift:.3f} deg with no action sent"
    if sol_count is not None:
        assert len(backend.solutions) == sol_count, (
            "law was solved without any SendAction"
        )


def test_setreference_relock_maps_zero_onto_stop_pose(runner):
    """Clutch re-engage: walk +60 mm, SetReference at the stop pose, then
    Δ=0 must HOLD the stop pose (no crawl home), and a fresh +20 mm offset
    must still move the arm from there."""
    hold_s = 0.6
    walk_s = 1.4
    lock_s = 1.2
    offset = with_offset((0.06, 0.0, 0.0))

    def frame(t: float):
        if t < hold_s:
            return identity()
        if t < hold_s + walk_s:
            return dict(offset)
        return identity()

    seq = Sequence(
        name="relay_setreference_relock",
        category="relay",
        duration_s=hold_s + walk_s + lock_s,
        frame=frame,
        meta={"lead_s": hold_s, "offset_m": 0.06,
              "walk_until_s": hold_s + walk_s},
    )
    result = runner.run(
        seq,
        events={hold_s + walk_s: runner.set_reference},
        note="tracking error is vs the PRE-relock reference — after "
        "SetReference the law targets the stop pose, so post-lock rows "
        "carry the relock offset by construction",
    )
    assert result.events_fired, "SetReference event never fired"

    samples = result.samples
    walk_end = [s for s in samples if s.t <= hold_s + walk_s]
    after_lock = [s for s in samples if s.t >= hold_s + walk_s + 0.4]
    stop_ee = np.mean([s.ee for s in walk_end[-5:]], axis=0)
    end_ee = np.mean([s.ee for s in after_lock[-5:]], axis=0)
    home_ee = np.mean([s.ee for s in samples if s.t < hold_s], axis=0)

    stay = float(np.linalg.norm(end_ee - stop_ee)) * 1000.0
    assert stay < 10.0, f"post-relock Δ=0 drifted {stay:.1f} mm off the stop pose"
    assert float(np.linalg.norm(end_ee - home_ee)) * 1000.0 > 30.0, (
        "post-relock Δ=0 walked back toward the Connect home"
    )


def test_fresh_offset_after_relock_moves_from_stop_pose(runner):
    """Re-engage continuation: after SetReference a NEW offset is applied
    on top of the stop pose (composition base = relatched pose)."""
    hold_s = 0.5
    walk_s = 1.3

    def frame(t: float):
        if t < hold_s:
            return identity()
        if t < hold_s + walk_s:
            return dict(with_offset((0.05, 0.0, 0.0)))
        return dict(with_offset((0.07, 0.0, 0.0)))  # +20 mm fresh on top

    seq = Sequence(
        name="relay_postlock_follow",
        category="relay",
        duration_s=hold_s + walk_s + 1.2,
        frame=frame,
        meta={"lead_s": hold_s},
    )
    result = runner.run(
        seq,
        events={hold_s + walk_s: runner.set_reference},
        note="tracking error is vs the PRE-relock reference (relock mid-run)",
    )
    samples = result.samples
    before = [s for s in samples if s.t < hold_s + walk_s]
    stop_ee = np.mean([s.ee for s in before[-5:]], axis=0)
    end_ee = np.mean([s.ee for s in samples[-5:]], axis=0)
    moved = float(np.linalg.norm(end_ee - stop_ee)) * 1000.0
    assert moved > 8.0, f"fresh +20 mm offset moved only {moved:.1f} mm"


def test_smooth_return_to_reference(runner):
    """Dropping the offset walks the EE home: converges, no big overshoot,
    bounded jerk (numbers recorded for the baseline)."""
    hold_s = 0.6
    return_s = 1.8

    def frame(t: float):
        if t < hold_s:
            return dict(with_offset((0.08, 0.0, 0.0)))
        return identity()

    seq = Sequence(
        name="relay_smooth_return",
        category="relay",
        duration_s=hold_s + return_s,
        frame=frame,
        meta={"lead_s": 0.0, "hold_s": hold_s},
    )
    # lead_s=0 keeps every sample in the tracking set; the walk-out happens
    # inside the lead of the runner's reset instead.
    result = runner.run(seq)
    m = result.metrics
    assert m["final_err_mm"] < 12.0, (
        f"return-to-home final error {m['final_err_mm']:.1f} mm"
    )
    ee = np.array([s.ee for s in result.samples])
    ref_pos = result.ref[:3, 3]
    dists = np.linalg.norm(ee - ref_pos, axis=1) * 1000.0
    # No swing past the outbound excursion during the return.
    assert dists.max() < 80.0 + 15.0, "return path overshot the excursion"
    assert np.isfinite(m["jerk_rms_m_s3"])
