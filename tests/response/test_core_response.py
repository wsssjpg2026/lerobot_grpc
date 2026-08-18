"""Layer a — core response: does the follower answer each command class
sanely (no divergence, right direction, converges), with metrics recorded?

Everything runs against the live gRPC stack (see conftest: sim by default).
Assertions are coarse sanity bounds; the precise numbers land in the
baseline report for B-2 human review.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from .injectors import (  # noqa: E402
    circle_sequence,
    gripper_ramp_sequence,
    gripper_step_sequence,
    rot_step_sequences,
    sine_sequence,
    step_sequences,
)

# --- sanity bounds (coarse by design; thresholds.md backfill is B-3) -----
# Final tracking error allowance per step amplitude class (mm).  Body-y is
# the weak axis of this stack (DLS + rot_weight leaves a standing lateral
# residual — solver-side, the arm executes what the IK commands; recorded
# as a report finding), so it carries its own relaxed band.
_STEP_FINAL_ERR_MM = {0.01: 8.0, 0.05: 25.0, 0.12: 70.0}
_STEP_FINAL_ERR_MM_Y = {0.01: 8.0, 0.05: 30.0, 0.12: 75.0}
# Minimum fraction of the commanded amplitude actually reached (saturated
# directions may fall short; wrong directions must not pass).
_STEP_MIN_AMPLITUDE_FRACTION = {0.01: 0.4, 0.05: 0.3, 0.12: 0.15}
_STEP_MIN_AMPLITUDE_FRACTION_Y = 0.4
# Continuous-tracking RMSE ceiling (mm) — generous, catches divergence only.
_TRACK_RMSE_MM = 60.0

# One-shot flag so the solver-side explanation lands in the report once.
_lateral_cause_noted = False


def _assert_no_divergence(result) -> None:
    m = result.metrics
    assert m.get("samples", 0) > 10, "observation sampler recorded nothing"
    for key in ("rmse_mm", "max_mm", "jerk_rms_m_s3", "obs_hz_measured"):
        value = m.get(key)
        if value is not None:
            assert math.isfinite(value), f"{key} not finite: {value}"
    for s in result.samples:
        assert np.isfinite(s.ee).all(), "non-finite EE sample"


@pytest.mark.parametrize("seq", step_sequences(), ids=lambda s: s.name)
def test_step_response_direction_and_convergence(runner, seq):
    result = runner.run(seq)
    _assert_no_divergence(result)
    m = result.metrics
    amp = seq.meta["amplitude_m"]
    is_lateral = seq.meta["axis"] == "y"
    frac_floor = (
        _STEP_MIN_AMPLITUDE_FRACTION_Y if is_lateral
        else _STEP_MIN_AMPLITUDE_FRACTION[amp]
    )
    final_cap = (
        _STEP_FINAL_ERR_MM_Y[amp] if is_lateral else _STEP_FINAL_ERR_MM[amp]
    )
    frac = m["amplitude_fraction"]
    assert m["direction_cosine"] > 0.7, (
        f"{seq.name}: wrong direction (cos={m['direction_cosine']:.2f})"
    )
    assert frac >= frac_floor, (
        f"{seq.name}: reached only {frac:.0%} of the commanded amplitude"
    )
    assert m["final_err_mm"] <= final_cap, (
        f"{seq.name}: final error {m['final_err_mm']:.1f} mm"
    )
    # The EE never wandered far past what was asked.
    assert m["max_ee_mm_from_ref"] <= (amp + 0.15) * 1000.0
    # Systematic lateral under-response is a headline baseline finding, not
    # a failure: flag it in the report whenever it shows up.
    if is_lateral and frac < 0.6 and runner.collector is not None:
        global _lateral_cause_noted
        if not _lateral_cause_noted:
            _lateral_cause_noted = True
            runner.collector.note(
                "systematic lateral (body-y) under-response: solver-side",
                "Across ±y steps the law's own IK residual equals the "
                "EE-side final error — the arm executes exactly what the "
                "IK commands; the DLS solve stops short of lateral targets "
                "under rot_weight=0.3 on the 5-DOF arm. Candidate "
                "follow-ups: more DLS iterations, lower rot_weight for "
                "pure-translation intents, or a larger damping schedule. "
                "Per-sequence evidence under the sibling finding.",
            )
        runner.collector.note(
            "systematic lateral (body-y) under-response: evidence",
            f"{seq.name}: reached {frac:.0%} of the commanded amplitude",
        )


@pytest.mark.parametrize("seq", rot_step_sequences(), ids=lambda s: s.name)
def test_rotation_step_response(runner, seq):
    result = runner.run(seq)
    _assert_no_divergence(result)
    m = result.metrics
    # The 5-DOF arm cannot fully track orientation (rot_weight 0.3): demand
    # partial response, record the rest. Some rotation must show up.
    assert m["rot_rmse_deg"] < 90.0, f"{seq.name}: rotation tracking diverged"
    assert m["rmse_mm"] < 80.0, f"{seq.name}: position blew up under rotation"


@pytest.mark.parametrize("freq", [0.1, 0.5, 1.0])
def test_sine_tracking(runner, freq):
    seq = sine_sequence(freq)
    result = runner.run(seq)
    _assert_no_divergence(result)
    m = result.metrics
    assert m["rmse_mm"] < _TRACK_RMSE_MM, (
        f"sine {freq} Hz rmse {m['rmse_mm']:.1f} mm"
    )
    assert m.get("lag_ms") is not None, "lag estimation returned nothing"
    assert abs(m["lag_ms"]) < 1000.0
    assert m.get("corr_peak", 0.0) > 0.5, (
        f"sine {freq} Hz correlation too weak ({m.get('corr_peak')})"
    )


def test_circle_tracking(runner):
    seq = circle_sequence()
    result = runner.run(seq)
    _assert_no_divergence(result)
    m = result.metrics
    assert m["rmse_mm"] < _TRACK_RMSE_MM
    assert m.get("corr_peak", 0.0) > 0.3, "circle tracking uncorrelated"


def test_gripper_step_response(runner):
    seq = gripper_step_sequence()
    result = runner.run(seq)
    _assert_no_divergence(result)
    m = result.metrics
    assert m["gripper_final_norm"] >= 0.9 * m["gripper_target_norm"], (
        f"gripper reached {m['gripper_final_norm']:.1f}/"
        f"{m['gripper_target_norm']:.1f}"
    )
    assert m["gripper_t63_ms"] is not None and m["gripper_t63_ms"] < 2000.0
    # Hand pose must not drift while only the gripper commands change.
    assert m["final_err_mm"] < 15.0, "hand moved during gripper-only sequence"


def test_gripper_ramp_follows_without_oscillation(runner):
    seq = gripper_ramp_sequence()
    result = runner.run(seq)
    _assert_no_divergence(result)
    m = result.metrics
    grips = [s.obs["gripper.pos"] for s in result.samples]
    assert all(0.0 <= g <= 100.0 for g in grips), "gripper out of 0-100 range"
    # The ramp is 0 -> high -> 0: the aperture must open meaningfully and
    # come back, without sign flips per sample (EMA/physics smoothing).
    assert max(grips) - min(grips) > 40.0, "aperture barely changed on ramp"
    diffs = np.diff(grips)
    sign_flips = int(np.sum(np.diff(np.sign(diffs[np.abs(diffs) > 0.2])) != 0))
    assert sign_flips <= 4, f"gripper oscillated ({sign_flips} direction flips)"


def test_identity_holds_reference_within_handshake_offset(runner):
    """Zero delta = hold: the identity window measures the handshake offset
    between the test-side T_ref and what the arm actually does — the floor
    every other tracking number sits on."""
    from .injectors import identity_sequence

    result = runner.run(identity_sequence())
    _assert_no_divergence(result)
    m = result.metrics
    lead = m.get("lead_offset_mm")
    assert lead is not None, "no lead-window samples recorded"
    assert lead < 15.0, f"identity hold drifted {lead:.1f} mm from reference"
