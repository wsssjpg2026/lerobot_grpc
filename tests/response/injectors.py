"""Synthetic pose-delta command sequences — pure functions, data-driven.

Every :class:`Sequence` is a named, bounded-duration generator over the
shared 8-feature schema: ``frame(t)`` returns the **complete** action dict
for time ``t`` (latch-once semantics — the current offset from the
reference, NOT a per-frame increment), or ``None`` during a silence window
(no packet is sent; the follower must hold).

The same sequences drive the sim backend (this round) and the real arm
(B-3, human-triggered) through the same gRPC client, so the baseline
numbers are comparable across backends.  Amplitudes are bounded and safe:

- translations stay within +/-0.12 m of the latch pose (the arm reaches
  ~0.35 m at home, so +/-0.12 m exercises saturation without escaping);
- rotations stay within 25 deg;
- the far-target probes (2.0 m) exist to trip the FK-consistency reject,
  never to be reached.

Unit conventions follow ``pose_delta_schema`` / ``pose_delta_law``:
metres, xyzw quaternion, gripper distance in millimetres (0-60 mm maps to
lerobot 0-100).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from lerobot.utils.rotation import Rotation


ActionDict = dict[str, float]
FrameFn = Callable[[float], "ActionDict | None"]

# Nominal gripper aperture held during motion sequences (mm) — mid-range,
# matching the existing pose-delta tests.
MOTION_GRIPPER_MM = 30.0

# Translation step amplitudes (m): small / medium / large.
STEP_AMPLITUDES_M = (0.01, 0.05, 0.12)
# Rotation step amplitudes (deg).
ROT_STEP_AMPLITUDES_DEG = (10.0, 25.0)

# Amplitude of the continuous-tracking probes.
SINE_AMPLITUDE_M = 0.04
CIRCLE_RADIUS_M = 0.04


def identity(grip_mm: float = MOTION_GRIPPER_MM) -> ActionDict:
    """A zero-offset action (hold the latch pose)."""
    return {
        "hand.delta_pos.x": 0.0,
        "hand.delta_pos.y": 0.0,
        "hand.delta_pos.z": 0.0,
        "hand.delta_rot.qx": 0.0,
        "hand.delta_rot.qy": 0.0,
        "hand.delta_rot.qz": 0.0,
        "hand.delta_rot.qw": 1.0,
        "gripper.distance": grip_mm,
    }


def with_offset(
    dp: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotvec_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    grip_mm: float | None = None,
) -> ActionDict:
    """identity() + a translational and/or rotational offset (rotvec, deg)."""
    action = identity(grip_mm if grip_mm is not None else MOTION_GRIPPER_MM)
    action["hand.delta_pos.x"], action["hand.delta_pos.y"], action["hand.delta_pos.z"] = dp
    if any(rotvec_deg):
        quat = _quat_from_rotvec_deg(rotvec_deg)
        action["hand.delta_rot.qx"], action["hand.delta_rot.qy"], \
            action["hand.delta_rot.qz"], action["hand.delta_rot.qw"] = quat
    return action


def _quat_from_rotvec_deg(rotvec_deg) -> tuple[float, float, float, float]:
    """xyzw quaternion for a rotation-vector spec in degrees."""
    quat = Rotation.from_rotvec(
        [math.radians(v) for v in rotvec_deg]
    ).as_quat()
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


@dataclass(frozen=True)
class Sequence:
    """One named command sequence.

    frame(t) with t in [0, duration_s] returns the full 8-feature action or
    None (silence).  meta carries the generator parameters so the report can
    group and parameterise rows without parsing names.
    """

    name: str
    category: str
    duration_s: float
    frame: FrameFn
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Identity hold (handshake-offset baseline)
# ---------------------------------------------------------------------------


def identity_sequence(duration_s: float = 1.2) -> Sequence:
    """Pure zero-delta hold — measures the handshake offset + hold drift."""
    return Sequence(
        name="identity_hold",
        category="identity",
        duration_s=duration_s,
        frame=lambda t: identity(),
        meta={"lead_s": duration_s},
    )


# ---------------------------------------------------------------------------
# Single-axis translation steps (6 directions x 3 amplitudes)
# ---------------------------------------------------------------------------


def step_sequences(
    *,
    amplitudes_m=STEP_AMPLITUDES_M,
    axes=("x", "y", "z"),
    lead_s: float = 0.5,
    hold_s: float = 1.5,
) -> list[Sequence]:
    out = []
    for axis in axes:
        for sign in (1.0, -1.0):
            for amp in amplitudes_m:
                out.append(
                    step_sequence(axis, sign, amp, lead_s=lead_s, hold_s=hold_s)
                )
    return out


def step_sequence(
    axis: str,
    sign: float,
    amp_m: float,
    *,
    lead_s: float = 0.5,
    hold_s: float = 1.5,
) -> Sequence:
    """Identity lead, then a constant translational offset on one axis."""
    target = identity()
    target[f"hand.delta_pos.{axis}"] = sign * amp_m

    def frame(t: float) -> ActionDict:
        return identity() if t < lead_s else dict(target)

    return Sequence(
        name=f"step_{axis}{'p' if sign > 0 else 'm'}_{int(amp_m * 1000)}mm",
        category="step",
        duration_s=lead_s + hold_s,
        frame=frame,
        meta={"axis": axis, "sign": sign, "amplitude_m": amp_m,
              "lead_s": lead_s, "hold_s": hold_s},
    )


# ---------------------------------------------------------------------------
# Single-axis rotation steps
# ---------------------------------------------------------------------------


def rot_step_sequences(
    *,
    amplitudes_deg=ROT_STEP_AMPLITUDES_DEG,
    axes=("z", "y"),
    lead_s: float = 0.5,
    hold_s: float = 1.5,
) -> list[Sequence]:
    out = []
    for axis in axes:  # z = yaw, y = pitch
        for amp in amplitudes_deg:
            out.append(rot_step_sequence(axis, amp, lead_s=lead_s, hold_s=hold_s))
    return out


def rot_step_sequence(
    axis: str,
    angle_deg: float,
    *,
    lead_s: float = 0.5,
    hold_s: float = 1.5,
) -> Sequence:
    """Identity lead, then a constant rotation offset about one body axis."""
    dp = (0.0, 0.0, 0.0)
    rot = {a: (angle_deg if a == axis else 0.0) for a in "xyz"}
    target = with_offset(dp, (rot["x"], rot["y"], rot["z"]))

    def frame(t: float) -> ActionDict:
        return identity() if t < lead_s else dict(target)

    return Sequence(
        name=f"rotstep_{axis}_{int(angle_deg)}deg",
        category="rot_step",
        duration_s=lead_s + hold_s,
        frame=frame,
        meta={"axis": axis, "amplitude_deg": angle_deg,
              "lead_s": lead_s, "hold_s": hold_s},
    )


# ---------------------------------------------------------------------------
# Sine tracking (multi-frequency)
# ---------------------------------------------------------------------------


def sine_sequence(
    freq_hz: float,
    *,
    axis: str = "x",
    amp_m: float = SINE_AMPLITUDE_M,
    periods: int | None = None,
    lead_s: float = 0.3,
    ramp_s: float = 0.3,
) -> Sequence:
    """Sinusoidal offset on one body axis with a smooth amplitude ramp-in.

    periods=None picks a duration budget that keeps slow sines tractable:
    0.1 Hz -> 1 period, >=0.5 Hz -> 3-4 periods.
    """
    if periods is None:
        periods = 1 if freq_hz < 0.3 else (3 if freq_hz < 0.75 else 4)
    active_s = periods / freq_hz

    def frame(t: float) -> ActionDict:
        tau = t - lead_s
        if tau <= 0.0:
            return identity()
        ramp = min(1.0, tau / ramp_s) if ramp_s > 0 else 1.0
        action = identity()
        action[f"hand.delta_pos.{axis}"] = (
            amp_m * ramp * math.sin(2.0 * math.pi * freq_hz * tau)
        )
        return action

    return Sequence(
        name=f"sine_{axis}_{freq_hz:g}hz",
        category="sine",
        duration_s=lead_s + active_s,
        frame=frame,
        meta={"axis": axis, "freq_hz": freq_hz, "amplitude_m": amp_m,
              "periods": periods, "lead_s": lead_s},
    )


# ---------------------------------------------------------------------------
# XY circle
# ---------------------------------------------------------------------------


def circle_sequence(
    *,
    radius_m: float = CIRCLE_RADIUS_M,
    freq_hz: float = 0.2,
    loops: int = 1,
    lead_s: float = 0.3,
    ramp_s: float = 0.5,
) -> Sequence:
    """One or more loops of a circle in the tracker's XY body plane."""
    active_s = loops / freq_hz

    def frame(t: float) -> ActionDict:
        tau = t - lead_s
        if tau <= 0.0:
            return identity()
        ramp = min(1.0, tau / ramp_s) if ramp_s > 0 else 1.0
        phase = 2.0 * math.pi * freq_hz * tau
        action = identity()
        action["hand.delta_pos.x"] = radius_m * ramp * math.cos(phase)
        action["hand.delta_pos.y"] = radius_m * ramp * math.sin(phase)
        return action

    return Sequence(
        name=f"circle_xy_r{int(radius_m * 1000)}mm_{freq_hz:g}hz",
        category="circle",
        duration_s=lead_s + active_s,
        frame=frame,
        meta={"radius_m": radius_m, "freq_hz": freq_hz, "loops": loops,
              "lead_s": lead_s},
    )


# ---------------------------------------------------------------------------
# Gripper step / ramp
# ---------------------------------------------------------------------------


def gripper_step_sequence(
    *,
    from_mm: float = 5.0,
    to_mm: float = 55.0,
    lead_s: float = 0.3,
    hold_s: float = 1.7,
) -> Sequence:
    """Gripper-distance step with the hand pose held (identity delta)."""
    def frame(t: float) -> ActionDict:
        return identity(from_mm if t < lead_s else to_mm)

    return Sequence(
        name=f"gripstep_{int(from_mm)}to{int(to_mm)}mm",
        category="gripper",
        duration_s=lead_s + hold_s,
        frame=frame,
        meta={"from_mm": from_mm, "to_mm": to_mm, "lead_s": lead_s},
    )


def gripper_ramp_sequence(
    *,
    low_mm: float = 5.0,
    high_mm: float = 55.0,
    lead_s: float = 0.3,
    ramp_s: float = 2.4,
) -> Sequence:
    """Open-close-open linear ramp (distance sweep, hand pose held)."""
    def frame(t: float) -> ActionDict:
        tau = t - lead_s
        if tau <= 0.0:
            return identity(low_mm)
        if tau < ramp_s:
            u = tau / ramp_s
        else:
            u = 1.0
        # triangle: 0 -> 1 -> 0 over ramp_s, then hold low.
        tri = u if u < 0.5 else 2.0 - 2.0 * u
        return identity(low_mm + (high_mm - low_mm) * tri)

    return Sequence(
        name=f"gripramp_{int(low_mm)}-{int(high_mm)}-{int(low_mm)}mm",
        category="gripper",
        duration_s=lead_s + ramp_s,
        frame=frame,
        meta={"low_mm": low_mm, "high_mm": high_mm, "ramp_s": ramp_s},
    )


# ---------------------------------------------------------------------------
# Silence (stale-hold / gating probe)
# ---------------------------------------------------------------------------


def silence_sequence(
    *,
    offset_m: float = 0.05,
    hold_s: float = 0.8,
    silent_s: float = 1.6,
    resume_s: float = 1.0,
) -> Sequence:
    """Hold an offset, stop sending entirely (None frames) > 1 s, resume."""
    moved = identity()
    moved["hand.delta_pos.x"] = offset_m

    def frame(t: float) -> ActionDict | None:
        if t < hold_s:
            return identity()
        if t < hold_s + silent_s:
            return None  # silence: no packet leaves the client
        return dict(moved) if t < hold_s + silent_s + 0.3 else identity()

    # Resume at the SAME offset for 0.3 s then return home, so the probe also
    # shows the arm is still alive after the gap (real adapter would hold the
    # first post-gap action; sim has no stale wiring — reported as a gap).
    return Sequence(
        name=f"silence_{int(silent_s * 1000)}ms",
        category="silence",
        duration_s=hold_s + silent_s + resume_s,
        frame=frame,
        meta={"offset_m": offset_m, "hold_s": hold_s, "silent_s": silent_s,
              "resume_s": resume_s},
    )


# ---------------------------------------------------------------------------
# Jump / reject / workspace-overflow probes
# ---------------------------------------------------------------------------


def jump_sequence(
    *,
    dp: tuple[float, float, float] = (0.0, 0.10, -0.15),
    lead_s: float = 0.5,
    hold_s: float = 1.6,
) -> Sequence:
    """One-frame large retarget: >30 deg of joint motion is expected, so the
    law must reset its warm start and the per-frame cap must bound the
    published step (official 30 deg / 6.7 deg rules)."""
    target = with_offset(dp)

    def frame(t: float) -> ActionDict:
        return identity() if t < lead_s else dict(target)

    return Sequence(
        name="jump_oneframe",
        category="jump",
        duration_s=lead_s + hold_s,
        frame=frame,
        meta={"dp_m": list(dp), "lead_s": lead_s, "hold_s": hold_s},
    )


def far_target_sequence(
    *,
    dx_m: float = 2.0,
    lead_s: float = 0.3,
    bad_s: float = 0.9,
    recover_s: float = 1.2,
) -> Sequence:
    """An unreachable intent (FK-consistency >0.3 m reject) followed by a
    healthy one — the arm must hold through the reject and recover."""
    def frame(t: float) -> ActionDict:
        if t < lead_s:
            return identity()
        if t < lead_s + bad_s:
            return with_offset((dx_m, 0.0, 0.0))
        return identity()

    return Sequence(
        name=f"far_dx{dx_m:g}m_reject",
        category="fk_reject",
        duration_s=lead_s + bad_s + recover_s,
        frame=frame,
        meta={"dx_m": dx_m, "lead_s": lead_s, "bad_s": bad_s},
    )


def workspace_overflow_sequence(
    *,
    dp: tuple[float, float, float],
    hold_s: float = 1.4,
    lead_s: float = 0.3,
) -> Sequence:
    """A far-but-finite intent at/over the workspace edge: the solve either
    saturates inside the IK joint limits or is rejected — it must never
    command a joint outside the model range or diverge."""
    target = with_offset(dp)

    def frame(t: float) -> ActionDict:
        return identity() if t < lead_s else dict(target)

    sign = "".join("p" if v > 0 else "m" if v < 0 else "0" for v in dp)
    return Sequence(
        name=f"overflow_{sign}_{int(max(abs(v) for v in dp) * 100)}cm",
        category="workspace",
        duration_s=lead_s + hold_s,
        frame=frame,
        meta={"dp_m": list(dp), "lead_s": lead_s, "hold_s": hold_s},
    )


# ---------------------------------------------------------------------------
# Registries (what the test layers and the baseline report run)
# ---------------------------------------------------------------------------


def core_sequences() -> list[Sequence]:
    """Layer-a set: every step/rot/sine/circle/gripper probe."""
    return (
        step_sequences()
        + rot_step_sequences()
        + [sine_sequence(f) for f in (0.1, 0.5, 1.0)]
        + [circle_sequence()]
        + [gripper_step_sequence(), gripper_ramp_sequence()]
    )


def safety_sequences() -> list[Sequence]:
    """Layer-c set: silence, jump, FK reject, workspace overflow."""
    return [
        silence_sequence(),
        jump_sequence(),
        far_target_sequence(),
        workspace_overflow_sequence((0.5, 0.0, 0.0)),
        workspace_overflow_sequence((0.0, 0.0, -0.35)),
    ]
