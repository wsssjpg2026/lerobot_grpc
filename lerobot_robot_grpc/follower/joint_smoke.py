"""Pure logic behind examples/joint_smoke_client.py (wayfinder pika-sense-real #05).

The joint-smoke gate proves the real arm's physical send_action path is safe
-- arrival, no overshoot/jerk, freeze on stream-cut -- before pose_delta
teleop may be enabled (map Destination hard gate).  Everything in this module
is hardware-free and unit-tested; the example client is a thin HITL driver.

Conventions: joint-mode actions are lerobot-normalised values (body joints in
degrees, gripper 0-100).  Degree-mode normalisation puts 0 deg at the
mid-range of the recorded raw ticks, so a calibration range maps to a
symmetric +/- half-span in degrees -- the same convention
SO101FollowerServicer._apply_calibration_qpos_limits uses for the IK limits.
"""

from __future__ import annotations

from collections.abc import Mapping

# Feetech STS3215 raw resolution: 0..4095 ticks = one full turn (lerobot table).
_RAW_TICKS_PER_TURN = 4095.0


def limits_from_raw_ranges(
    raw_ranges: Mapping[str, tuple[int, int]],
) -> dict[str, tuple[float, float]]:
    """Recorded raw tick ranges -> symmetric degree limits (lo, hi) per joint.

    Joints with a degenerate range (range_max <= range_min) are omitted: a
    limit the pre-check cannot trust must never silently pass.
    """
    limits: dict[str, tuple[float, float]] = {}
    for joint, (range_min, range_max) in raw_ranges.items():
        if range_max <= range_min:
            continue
        half_span_deg = (range_max - range_min) / 2.0 * 360.0 / _RAW_TICKS_PER_TURN
        limits[joint] = (-half_span_deg, half_span_deg)
    return limits

def check_target(
    joint: str,
    base_deg: float,
    delta_deg: float,
    limits: Mapping[str, tuple[float, float]],
    margin_deg: float = 5.0,
) -> float | None:
    """Pre-check one step against the joint's calibration limits.

    Returns the absolute target angle (deg) when it sits inside
    [lo + margin, hi - margin]; returns None to SKIP the step -- never clip:
    clipping would silently change what the smoke set out to prove.  A joint
    with no recorded limit is never moved.
    """
    lim = limits.get(joint)
    if lim is None:
        return None
    lo, hi = lim
    target = base_deg + delta_deg
    if lo + margin_deg <= target <= hi - margin_deg:
        return target
    return None

from dataclasses import dataclass  # noqa: E402

# Scan order locked by #05 grilling: proximal to distal, gripper excluded
# (gripper gets its own full 0->100->0 cycle at the end of the run).
BODY_SCAN_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


@dataclass(frozen=True)
class ScanStep:
    """One smoke motion: drive joint from base to target_deg, dwell, return."""

    joint: str
    tier_deg: float
    direction: int  # +1 or -1
    target_deg: float


@dataclass(frozen=True)
class SkipStep:
    """A step the limit pre-check rejected -- recorded, never clipped."""

    joint: str
    tier_deg: float
    direction: int
    would_be_target_deg: float


def build_scan_steps(
    base_deg: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
    tiers: tuple[float, ...] = (5.0, 10.0),
    margin_deg: float = 5.0,
) -> list[ScanStep | SkipStep]:
    """The full scan plan: per tier, per joint (scan order), + then -.

    Each step is pre-checked against the calibration limits; rejected steps
    become SkipStep records so the report shows exactly what was not tried.
    """
    steps: list[ScanStep | SkipStep] = []
    for tier in tiers:
        for joint in BODY_SCAN_ORDER:
            for direction in (+1, -1):
                delta = tier * direction
                target = check_target(joint, base_deg[joint], delta, limits, margin_deg)
                if target is None:
                    steps.append(SkipStep(joint, tier, direction, base_deg[joint] + delta))
                else:
                    steps.append(ScanStep(joint, tier, direction, target))
    return steps

@dataclass(frozen=True)
class DwellResult:
    """Machine-checked outcome of one 2 s dwell (pass criteria from #05)."""

    passed: bool
    steady_error_deg: float   # |mean(steady window) - target|
    overshoot_deg: float      # furthest excursion PAST the target
    end_oscillation_deg: float  # peak-to-peak inside the steady window


def evaluate_dwell(
    samples_deg: list[float],
    target_deg: float,
    tol_deg: float = 2.0,
    steady_window: int = 5,
    oscillation_tol_deg: float = 1.0,
    overshoot_tol_deg: float = 2.0,
) -> DwellResult:
    """Judge one dwell: steady arrival within tol, no oscillation, no big overshoot.

    The steady window is the last steady_window samples (0.5 s at the
    protocol's 100 ms cadence).  Overshoot is measured relative to the
    direction of travel; a small overshoot that settles is reported but
    passes, a swing beyond overshoot_tol_deg fails.
    """
    if len(samples_deg) < steady_window:
        raise ValueError(f"need >= {steady_window} samples, got {len(samples_deg)}")
    window = samples_deg[-steady_window:]
    steady_error = abs(sum(window) / len(window) - target_deg)
    oscillation = max(window) - min(window)
    direction = 1.0 if target_deg >= samples_deg[0] else -1.0
    overshoot = max(0.0, max(direction * (s - target_deg) for s in samples_deg))
    passed = (
        steady_error <= tol_deg
        and oscillation <= oscillation_tol_deg
        and overshoot <= overshoot_tol_deg
    )
    return DwellResult(passed, steady_error, overshoot, oscillation)

@dataclass(frozen=True)
class FreezeResult:
    """Stream-cut outcome: while the client was silent the arm must not move."""

    passed: bool
    max_drift_deg: float


def evaluate_freeze(
    baseline_deg: list[float],
    silence_deg: list[float],
    tol_deg: float = 0.5,
) -> FreezeResult:
    """Drift check for the stream-cut steps (mid-scan silence and kill -9).

    Reference is the mean of the samples taken just before the cut; every
    sample during the silence must stay within tol_deg of it.  Joint mode has
    no stale-hold code -- this verifies the servos' physical position-hold.
    """
    if not baseline_deg or not silence_deg:
        raise ValueError("need baseline and silence samples")
    ref = sum(baseline_deg) / len(baseline_deg)
    drift = max(abs(s - ref) for s in silence_deg)
    return FreezeResult(drift <= tol_deg, drift)
