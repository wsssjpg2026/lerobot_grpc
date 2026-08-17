"""Offline judge for the real-arm pose_delta HITL bench (wayfinder #07).

examples/teleop_hitl_bench.py is a thin recording driver: it teleops exactly
like examples/teleop_pika_mujoco.py (same clutch helpers, same relatch
sequence) and writes one CSV row per loop tick.  This module is everything
that can be judged offline from that CSV, machine-checked against the bench
criteria (values from #05 freeze evidence, sim #11/#12/#13 methodology):

- HOLD windows (clutch off): body joints frozen <= 0.5 deg while the hand
  flails; gripper stays LIVE (official PikaAnyArm semantics).
- RELOCK (clutch back on): first 300 ms cumulative joint motion <= 2 deg --
  the current-hand-equals-current-arm contract, no crawl to Connect home.
- STALE gaps (client paused > stale_timeout): the first frame after the gap
  must not jump to the hand's new pose (servicer stale-hold, body only).
- LEADER-DOWN windows (leader server killed): no commands flow, the servos
  physically hold -- drift <= 2 deg, never a return to home.
- SPHERE: FK end-effector radius stays <= base-safety-sphere + slew slack.
- DESKTOP: minimum FK end-effector height -- evidence for the table-collision
  scoping decision (no pass/fail).

Conventions: observation columns are lerobot-normalised (body joints in
degrees, gripper 0-100); the FK path mirrors mujoco_follower_server.
norm_value_to_rad (kept local to avoid importing the whole servicer stack).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# SO-101 body joints in model-DOF order (mirror of BODY_JOINTS in
# mujoco_follower_server -- the FK path indexes model DOFs by position).
CSV_HEADER: tuple[str, ...] = (
    "t_s", "engaged", "sent", "leader_ok",
    "act_dx_m", "act_dy_m", "act_dz_m",
    "act_qx", "act_qy", "act_qz", "act_qw", "act_grip_mm",
    "obs_pan_deg", "obs_lift_deg", "obs_elbow_deg", "obs_wf_deg", "obs_wr_deg",
    "obs_gripper",
)

BODY_JOINT_KEYS: tuple[str, ...] = (
    "obs_pan_deg", "obs_lift_deg", "obs_elbow_deg", "obs_wf_deg", "obs_wr_deg",
)
_JOINT_NAMES = {
    "obs_pan_deg": "shoulder_pan",
    "obs_lift_deg": "shoulder_lift",
    "obs_elbow_deg": "elbow_flex",
    "obs_wf_deg": "wrist_flex",
    "obs_wr_deg": "wrist_roll",
}

# MuJoCo gripper actuator range in radians (so101_new_calib.xml, same source
# as mujoco_follower_server.GRIPPER_RAD_MIN/MAX).
_GRIPPER_RAD_MIN = -0.17453
_GRIPPER_RAD_MAX = 1.74533

Row = dict  # one CSV row: float values, None for empty action cells


def load_rows(path: str | Path) -> list[Row]:
    """Loads the driver's CSV; empty cells become None, numbers become float."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows: list[Row] = []
        for raw in reader:
            row: Row = {}
            for key in CSV_HEADER:
                cell = (raw.get(key) or "").strip()
                row[key] = None if cell == "" else float(cell)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Window finding
# ---------------------------------------------------------------------------


def find_hold_windows(rows: list[Row], min_len: int = 3) -> list[tuple[int, int]]:
    """Inclusive index spans of consecutive engaged==0 (clutch off).

    A trailing unterminated window (bench ended while holding) is kept -- the
    freeze evidence in it is still valid.
    """
    windows: list[tuple[int, int]] = []
    start = None
    for i, r in enumerate(rows):
        if r["engaged"] == 0 and start is None:
            start = i
        elif r["engaged"] != 0 and start is not None:
            if i - start >= min_len:
                windows.append((start, i - 1))
            start = None
    if start is not None and len(rows) - start >= min_len:
        windows.append((start, len(rows) - 1))
    return windows


def find_leader_down_windows(rows: list[Row], min_len: int = 3) -> list[tuple[int, int]]:
    """Inclusive spans of consecutive leader_ok==0 (status poll failing)."""
    windows: list[tuple[int, int]] = []
    start = None
    for i, r in enumerate(rows):
        if r["leader_ok"] == 0 and start is None:
            start = i
        elif r["leader_ok"] != 0 and start is not None:
            if i - start >= min_len:
                windows.append((start, i - 1))
            start = None
    if start is not None and len(rows) - start >= min_len:
        windows.append((start, len(rows) - 1))
    return windows


@dataclass(frozen=True)
class Gap:
    """A timestamp hole in the client loop (paused client / machine stall)."""

    before_idx: int
    after_idx: int
    gap_s: float


def find_gaps(rows: list[Row], min_gap_s: float = 1.2) -> list[Gap]:
    """Timestamp gaps > min_gap_s (default just past the servicer's 1 s stale
    timeout, so every found gap means the next action was stale-flagged)."""
    gaps: list[Gap] = []
    for i in range(1, len(rows)):
        dt = rows[i]["t_s"] - rows[i - 1]["t_s"]
        if dt > min_gap_s:
            gaps.append(Gap(i - 1, i, dt))
    return gaps


# ---------------------------------------------------------------------------
# Judgements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldResult:
    passed: bool
    max_drift_deg: float
    worst_joint: str
    gripper_span: float
    gripper_live: bool
    start_t: float
    duration_s: float


def evaluate_hold(
    rows: list[Row],
    window: tuple[int, int],
    tol_deg: float = 0.5,
    baseline_n: int = 3,
    gripper_live_span: float = 5.0,
) -> HoldResult:
    """Freeze judgement for one hold window.

    Baseline per joint = the samples just BEFORE the disengage edge; every
    in-window sample must stay within tol_deg of it (same contract as the
    #05 stream-cut check).  Gripper liveness is a positive check: the official
    clutch gates the arm only, so the operator's gripper wiggles must show up.
    """
    i0, i1 = window
    start = max(0, i0 - baseline_n)
    worst, worst_key = 0.0, BODY_JOINT_KEYS[0]
    for key in BODY_JOINT_KEYS:
        baseline = [r[key] for r in rows[start:i0]] or [rows[i0][key]]
        ref = sum(baseline) / len(baseline)
        drift = max(abs(r[key] - ref) for r in rows[i0 : i1 + 1])
        if drift > worst:
            worst, worst_key = drift, key
    grips = [r["obs_gripper"] for r in rows[i0 : i1 + 1]]
    span = max(grips) - min(grips)
    return HoldResult(
        passed=worst <= tol_deg,
        max_drift_deg=worst,
        worst_joint=_JOINT_NAMES[worst_key],
        gripper_span=span,
        gripper_live=span >= gripper_live_span,
        start_t=rows[i0]["t_s"],
        duration_s=rows[i1]["t_s"] - rows[i0]["t_s"],
    )


@dataclass(frozen=True)
class RelockResult:
    passed: bool
    cumulative_deg: float
    max_step_deg: float
    worst_joint: str


def evaluate_relock(
    rows: list[Row],
    hold_end_idx: int,
    horizon_s: float = 0.3,
    tol_deg: float = 2.0,
) -> RelockResult:
    """Re-engage judgement: the 300 ms after clutch-on must barely move.

    The relatch made current-hand = current-arm, so the first intents are
    ~zero; cumulative motion past tol_deg means the arm yanked toward the old
    frozen offset or crawled back toward Connect home (#12/#11 contracts).
    """
    engage = hold_end_idx + 1
    if engage >= len(rows):
        return RelockResult(True, 0.0, 0.0, "-")
    t0 = rows[engage]["t_s"]
    horizon = [r for r in rows[engage:] if r["t_s"] <= t0 + horizon_s]
    worst, worst_key, max_step = 0.0, BODY_JOINT_KEYS[0], 0.0
    for key in BODY_JOINT_KEYS:
        base = rows[engage][key]
        cum = max(abs(r[key] - base) for r in horizon)
        if cum > worst:
            worst, worst_key = cum, key
        for prev, cur in zip(horizon, horizon[1:]):
            max_step = max(max_step, abs(cur[key] - prev[key]))
    return RelockResult(worst <= tol_deg, worst, max_step, _JOINT_NAMES[worst_key])


@dataclass(frozen=True)
class GapHoldResult:
    passed: bool
    max_jump_deg: float
    worst_joint: str
    gap_s: float


def evaluate_gap_hold(rows: list[Row], gap: Gap, tol_deg: float = 2.0) -> GapHoldResult:
    """Stale-hold judgement: across a > stale_timeout gap the arm must not
    jump -- the first post-gap action holds the last body joints (servicer
    stale-hold), whatever the hand did while the client was frozen."""
    worst, worst_key = 0.0, BODY_JOINT_KEYS[0]
    for key in BODY_JOINT_KEYS:
        jump = abs(rows[gap.after_idx][key] - rows[gap.before_idx][key])
        if jump > worst:
            worst, worst_key = jump, key
    return GapHoldResult(worst <= tol_deg, worst, _JOINT_NAMES[worst_key], gap.gap_s)


@dataclass(frozen=True)
class LeaderDownResult:
    passed: bool
    max_drift_deg: float
    worst_joint: str
    start_t: float
    duration_s: float


def evaluate_leader_down(
    rows: list[Row], window: tuple[int, int], tol_deg: float = 2.0
) -> LeaderDownResult:
    """Leader-death judgement: nothing may flow, so the servos hold physically
    -- drift within tol_deg, and any motion toward Connect home fails loudly."""
    i0, i1 = window
    worst, worst_key = 0.0, BODY_JOINT_KEYS[0]
    for key in BODY_JOINT_KEYS:
        ref = rows[i0][key]
        drift = max(abs(r[key] - ref) for r in rows[i0 : i1 + 1])
        if drift > worst:
            worst, worst_key = drift, key
    return LeaderDownResult(
        worst <= tol_deg,
        worst,
        _JOINT_NAMES[worst_key],
        rows[i0]["t_s"],
        rows[i1]["t_s"] - rows[i0]["t_s"],
    )


# ---------------------------------------------------------------------------
# FK (sphere + desktop evidence)
# ---------------------------------------------------------------------------


def ee_positions(rows: list[Row], xml_path: str) -> "np.ndarray":
    """Gripperframe site positions (N, 3), base frame, one per row.

    The same kinematics oracle the servicers load; joints are read from the
    observation columns so the judgement sees what the arm actually did, not
    what was commanded.
    """
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    out = np.zeros((len(rows), 3))
    for i, r in enumerate(rows):
        qpos = [math.radians(r[key]) for key in BODY_JOINT_KEYS]
        qpos.append(
            (r["obs_gripper"] / 100.0) * (_GRIPPER_RAD_MAX - _GRIPPER_RAD_MIN)
            + _GRIPPER_RAD_MIN
        )
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        out[i] = data.site_xpos[site]
    return out


@dataclass(frozen=True)
class SphereResult:
    passed: bool
    max_radius_m: float
    radius_m: float
    slack_m: float


def evaluate_sphere(pos, radius_m: float, slack_m: float = 0.020) -> SphereResult:
    """Base-safety-sphere judgement over the run's FK positions.

    slack covers the arm finishing an in-flight slew step after the intent was
    clamped (~5 mm/frame) plus IK residual, not a systematic overshoot.
    """
    import numpy as np

    max_r = float(np.linalg.norm(pos, axis=1).max())
    return SphereResult(max_r <= radius_m + slack_m, max_r, radius_m, slack_m)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Report:
    n_rows: int = 0
    duration_s: float = 0.0
    holds: list[HoldResult] = field(default_factory=list)
    relocks: list[RelockResult] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    gap_holds: list[GapHoldResult] = field(default_factory=list)
    leader_downs: list[LeaderDownResult] = field(default_factory=list)
    sphere: SphereResult | None = None
    desktop_min_z_m: float | None = None
    desktop_min_z_t: float | None = None

    @property
    def all_pass(self) -> bool:
        checks = [h.passed for h in self.holds]
        checks += [r.passed for r in self.relocks]
        checks += [g.passed for g in self.gap_holds]
        checks += [l.passed for l in self.leader_downs]
        if self.sphere is not None:
            checks.append(self.sphere.passed)
        return all(checks) if checks else False

    def render(self) -> str:
        lines = [f"HITL bench: {self.n_rows} rows, {self.duration_s:.1f}s"]
        lines.append(f"HOLD windows: {len(self.holds)}")
        for i, h in enumerate(self.holds, 1):
            lines.append(
                f"  #{i} t={h.start_t:.1f}s {h.duration_s:.1f}s "
                f"{'PASS' if h.passed else 'FAIL'} drift={h.max_drift_deg:.2f}deg "
                f"({h.worst_joint}) gripper span={h.gripper_span:.1f} "
                f"{'LIVE' if h.gripper_live else 'DEAD?'}"
            )
        lines.append(f"RELOCK: {len(self.relocks)}")
        for i, r in enumerate(self.relocks, 1):
            lines.append(
                f"  #{i} {'PASS' if r.passed else 'FAIL'} "
                f"cumulative={r.cumulative_deg:.2f}deg/300ms "
                f"(worst {r.worst_joint}, max step {r.max_step_deg:.2f}deg)"
            )
        lines.append(f"STALE gaps: {len(self.gaps)}")
        for i, g in enumerate(self.gap_holds, 1):
            lines.append(
                f"  #{i} gap={g.gap_s:.2f}s {'PASS' if g.passed else 'FAIL'} "
                f"jump={g.max_jump_deg:.2f}deg ({g.worst_joint})"
            )
        lines.append(f"LEADER-DOWN: {len(self.leader_downs)}")
        for i, l in enumerate(self.leader_downs, 1):
            lines.append(
                f"  #{i} t={l.start_t:.1f}s {l.duration_s:.1f}s "
                f"{'PASS' if l.passed else 'FAIL'} drift={l.max_drift_deg:.2f}deg "
                f"({l.worst_joint})"
            )
        if self.sphere is not None:
            s = self.sphere
            lines.append(
                f"SPHERE: max radius {s.max_radius_m * 1000:.0f}mm vs "
                f"{s.radius_m * 1000:.0f}+{s.slack_m * 1000:.0f}mm "
                f"{'PASS' if s.passed else 'FAIL'}"
            )
        if self.desktop_min_z_m is not None:
            lines.append(
                f"DESKTOP: min EE z {self.desktop_min_z_m * 1000:.0f}mm "
                f"at t={self.desktop_min_z_t:.1f}s (evidence, no criterion)"
            )
        lines.append(f"OVERALL: {'PASS' if self.all_pass else 'FAIL'}")
        return "\n".join(lines)


def build_report(
    rows: list[Row],
    xml_path: str | None = None,
    sphere_radius_m: float = 0.72 * 0.543,
    sphere_slack_m: float = 0.020,
) -> Report:
    """Every offline judgement over one bench CSV run."""
    report = Report()
    if not rows:
        return report
    report.n_rows = len(rows)
    report.duration_s = rows[-1]["t_s"] - rows[0]["t_s"]
    for window in find_hold_windows(rows):
        report.holds.append(evaluate_hold(rows, window))
        report.relocks.append(evaluate_relock(rows, window[1]))
    for gap in find_gaps(rows):
        report.gaps.append(gap)
        report.gap_holds.append(evaluate_gap_hold(rows, gap))
    for window in find_leader_down_windows(rows):
        report.leader_downs.append(evaluate_leader_down(rows, window))
    if xml_path is not None:
        import numpy as np

        pos = ee_positions(rows, xml_path)
        report.sphere = evaluate_sphere(pos, sphere_radius_m, sphere_slack_m)
        argmin = int(np.argmin(pos[:, 2]))
        report.desktop_min_z_m = float(pos[argmin, 2])
        report.desktop_min_z_t = float(rows[argmin]["t_s"])
    return report
