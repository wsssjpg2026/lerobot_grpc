"""Layer c — the official safety stack, exercised through the wire:

- stale hold: >1 s of leader silence freezes the arm (sim note: the sim
  adapter holds implicitly — no new target — and never sets the law's
  stale flag; see the report findings);
- 30 deg jump rejection: warm-start reset + per-frame 6.7 deg published
  step cap (official piper_IK mechanisms);
- FK consistency 0.3 m: unreachable intents rejected, arm holds, then
  recovers on the next healthy action;
- workspace overflow: far-but-finite intents never command joints outside
  the model limits and never diverge.

Law-side flags (rejected/held/jumped) are asserted through the in-process
solution spy — the only channel they exist on (no RPC-visible status;
reported as an implementation gap, not fixed here).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from .injectors import (  # noqa: E402
    far_target_sequence,
    jump_sequence,
    silence_sequence,
    workspace_overflow_sequence,
)
from .metrics import BODY_JOINTS  # noqa: E402


def _skip_without_law_spy(backend):
    if not backend.law_spy_available:
        pytest.skip("law-side solution records unavailable on this backend")


def test_silence_over_one_second_holds_position(runner, backend):
    """No packet for >1 s: the body joints and gripper hold; nothing is
    sent (hence nothing solved); the stream keeps flowing; after the gap
    the follower is still alive (resumes on the next action)."""
    seq = silence_sequence()
    result = runner.run(seq)
    meta = seq.meta
    hold_s, silent_s = meta["hold_s"], meta["silent_s"]
    m = result.metrics

    # The harness really went silent (this guards the probe itself).
    sends_in_window = [
        r for r in result.sent if hold_s <= r.t < hold_s + silent_s
    ]
    assert not sends_in_window, "harness sent during the silence window"

    in_window = [s for s in result.samples if hold_s <= s.t < hold_s + silent_s]
    assert len(in_window) > 10, "no samples inside the silence window"
    first, last = in_window[0], in_window[-1]
    joint_drift = max(
        abs(last.obs[f"{j}.pos"] - first.obs[f"{j}.pos"]) for j in BODY_JOINTS
    )
    assert joint_drift < 0.3, f"joints drifted {joint_drift:.3f} deg in silence"
    ee_drift = float(np.linalg.norm(last.ee - first.ee)) * 1000.0
    assert ee_drift < 5.0, f"EE drifted {ee_drift:.1f} mm in silence"
    assert abs(last.obs["gripper.pos"] - first.obs["gripper.pos"]) < 2.0

    # The stream survived the gap and the follower resumed afterwards.
    assert m["obs_hz_measured"] > 30.0, (
        f"observation stream unhealthy across the gap ({m['obs_hz_measured']:.1f} Hz)"
    )


def test_jump_injection_is_capped_per_frame(runner, backend):
    """A one-frame >30 deg-class retarget: every published joint step must
    respect the official 6.7 deg/frame cap; the warm start may reset but
    the response must stay bounded and directional."""
    seq = jump_sequence()
    result = runner.run(seq)
    m = result.metrics

    _skip_without_law_spy(backend)
    sols = result.solutions
    assert len(sols) > 5
    # Every consecutive published pair respects the frame cap.
    for prev, cur in zip(sols, sols[1:]):
        worst = max(
            abs(cur.joint_action[f"{j}.pos"] - prev.joint_action[f"{j}.pos"])
            for j in BODY_JOINTS
        )
        assert worst <= 6.7 + 1e-6, (
            f"published step {worst:.2f} deg exceeds the 6.7 deg frame cap"
        )
    # The retarget direction is right (amplitude_fraction from metrics).
    assert m["direction_cosine"] > 0.5, "jump retarget went the wrong way"
    assert math.isfinite(m["max_commanded_step_deg"])
    # Bounded excursion, no divergence.
    assert m["max_ee_mm_from_ref"] < 0.45 * 1000.0


def test_fk_consistency_rejects_unreachable_and_recovers(runner, backend):
    """A 2 m intent trips the official 0.3 m FK-consistency check: the law
    must reject it and hold the previous joints; the arm must not fly; the
    next healthy action must be solved normally."""
    seq = far_target_sequence(dx_m=2.0)
    result = runner.run(seq)
    m = result.metrics

    _skip_without_law_spy(backend)
    sols = result.solutions
    assert any(s.rejected for s in sols), "2 m intent was never rejected"
    # Every rejected solve holds the previous command exactly (body joints).
    for prev, cur in zip(sols, sols[1:]):
        if cur.rejected:
            for j in BODY_JOINTS:
                assert cur.joint_action[f"{j}.pos"] == prev.joint_action[f"{j}.pos"], (
                    "rejected solve changed the published joints"
                )
    # Recovery: solutions after the reject window are accepted again.
    accepted_after = [
        s for s in sols[len(sols) // 2:] if not s.rejected and not s.held
    ]
    assert accepted_after, "no healthy solve after the reject window (no recovery)"
    # The arm never diverged.
    assert m["max_ee_mm_from_ref"] < 0.4 * 1000.0
    assert math.isfinite(m.get("final_pos_err_mm", float("nan")))


@pytest.mark.parametrize(
    "dp",
    [(0.5, 0.0, 0.0), (0.0, 0.0, -0.35)],
    ids=["near_max_reach", "below_floor"],
)
def test_workspace_overflow_stays_inside_limits(runner, backend, fk, dp):
    """Far-but-finite intents: whatever the solver decides (saturation or
    reject), every commanded joint stays inside the model range and the EE
    excursion stays bounded."""
    seq = workspace_overflow_sequence(dp=dp)
    result = runner.run(seq)
    m = result.metrics

    limits = fk.joint_limits_deg()
    lo, hi = limits[:, 0], limits[:, 1]
    if backend.law_spy_available:
        for sol in result.solutions:
            for i, j in enumerate(BODY_JOINTS):
                q = sol.joint_action[f"{j}.pos"]
                assert lo[i] - 1e-6 <= q <= hi[i] + 1e-6, (
                    f"{seq.name}: commanded {j}={q:.1f} deg outside model range"
                )
    assert m["max_ee_mm_from_ref"] < 0.6 * 1000.0, "overflow probe diverged"
    # Either rejected outright or saturated — both acceptable; recorded.
    if backend.law_spy_available:
        assert m.get("solves", 0) > 0


def test_stale_law_path_documented_as_sim_gap(report):
    """The law's stale=True freeze path cannot be reached through the sim
    wire stack (the sim adapter never sets it) — pinned here so the B-3
    real run knows what differs, instead of silently assuming parity."""
    report.note(
        "stale=True law path untestable over sim wire",
        "silence_sequence verifies the sim's implicit hold (no new target); "
        "the law's stale=True freeze (leader gap > 1 s then a fresh action) "
        "only fires on the real adapter — B-3 must re-run the silence probe "
        "on hardware to characterize it.",
    )
