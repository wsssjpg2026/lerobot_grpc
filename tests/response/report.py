"""Baseline report writer — JSON + CSV + Markdown under outputs/.

Everything lands under ``outputs/response_baseline/<UTC-timestamp>_<backend>/``
(gitignored): ``report.json`` (full metrics), ``metrics.csv`` (one row per
sequence), ``timelines/<sequence>.csv`` (per-sample target/actual tracks)
and ``report.md`` (human review copy for stage B-2).

Threshold fields are PLACEHOLDERS (``null``) — the baseline is data for
human review and B-3 threshold backfill, never a pass/fail gate.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backends import REAL_LAUNCH_CMD
from .harness import RunResult
from .metrics import BODY_JOINTS, expected_target

PKG_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PKG_ROOT / "outputs" / "response_baseline"

# Metric columns of the flat CSV / Markdown table (order matters).  The
# CSV additionally prefixes name/duration/category columns.
_TABLE_COLUMNS = (
    ("rmse_mm", "rmse_mm"),
    ("p95_mm", "p95_mm"),
    ("max_mm", "max_mm"),
    ("final_err_mm", "final_err_mm"),
    ("rot_rmse_deg", "rot_rmse_deg"),
    ("lag_ms", "lag_ms"),
    ("jerk_rms_m_s3", "jerk_m_s3"),
    ("obs_hz_measured", "obs_hz"),
    ("drop_rate_vs_nominal", "drop_rate"),
    ("action_rtt_ms_p95", "rtt_p95_ms"),
    ("max_commanded_step_deg", "max_cmd_step_deg"),
    ("rejected", "rejected"),
    ("jumped", "jumped"),
)

# Implementation-gap notes gathered while building the suite (record-only —
# B-1 does not fix implementations).  Each entry: (title, detail).
KNOWN_FINDINGS = (
    (
        "sim adapter never sets stale=True",
        "MuJoCoSO101Servicer calls law.solve(delta, qpos) without the stale "
        "flag; the >1 s stale-hold clock (_last_action_monotonic) exists only "
        "in SO101FollowerServicer. In sim the arm holds a silence only "
        "implicitly (no new target arrives); the first post-gap action is "
        "solved immediately, while the real adapter would freeze it. "
        "Cross-backend baselines must not compare post-gap behaviour.",
    ),
    (
        "safety-stack state has no RPC-visible channel",
        "JointSolution flags (rejected/held/stale/collided/jumped) are "
        "discarded by both adapters: SendAction echoes the wire action and "
        "GetFeedback echoes the last wire action. Over-limit / hold state is "
        "observable only in-process (law object) or via throttled WARN logs. "
        "The suite asserts law-side flags through the in-process spy (sim).",
    ),
    (
        "each live GetObservation stream steps the physics",
        "Every GetObservation generator instance runs its own mj_step loop "
        "under the lock, so N concurrent streams advance the sim N x real "
        "time. The harness therefore keeps exactly one stream (the "
        "GRPCFollower background thread). Worth a follow-up if multi-client "
        "recording sessions ever share one sim server.",
    ),
    (
        "GRPCFollower has no set_reference() wrapper",
        "The leader client wraps SetReference (grpc_leader.py:444) but the "
        "follower client does not; clutch re-engage callers must use "
        "client.stub.SetReference(Empty()) directly (examples/"
        "teleop_pika_mujoco.py:82 pattern).",
    ),
    (
        "observation stream carries joints only",
        "No EE-pose feature exists in GetObservation, so end-effector ground "
        "truth is test-side FK on assets/so101/scene.xml. If the schema ever "
        "grows EE features the metrics can switch to server-reported truth.",
    ),
)


class ReportCollector:
    """Accumulates RunResults + findings; write() emits the three formats."""

    def __init__(self, backend_name: str, out_root: Path = OUT_ROOT):
        self.backend_name = backend_name
        self.out_root = out_root
        self.runs: list[RunResult] = []
        self.findings: list[tuple[str, str]] = list(KNOWN_FINDINGS)
        self.meta: dict = {"action_hz": 30.0, "obs_nominal_hz": 50.0}

    def add(self, run: RunResult) -> None:
        self.runs.append(run)

    def note(self, title: str, detail: str) -> None:
        """Add (or extend) a finding — same title merges its evidence."""
        for i, (t, d) in enumerate(self.findings):
            if t == title:
                if detail not in d:
                    self.findings[i] = (t, f"{d} | {detail}")
                return
        self.findings.append((title, detail))

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def write(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = self.out_root / f"{stamp}_{self.backend_name}"
        (out_dir / "timelines").mkdir(parents=True, exist_ok=True)

        self._write_json(out_dir)
        self._write_csv(out_dir)
        self._write_timelines(out_dir)
        self._write_markdown(out_dir)
        return out_dir

    # -- JSON ------------------------------------------------------------

    def _write_json(self, out_dir: Path) -> None:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "backend": self.backend_name,
            "git_rev": _git_rev(),
            "law_spy_available": self.backend_name == "sim",
            "real_launch_cmd": REAL_LAUNCH_CMD,
            "meta": self.meta,
            "thresholds": "PLACEHOLDER — backfill after B-3 real validation",
            "sequences": [
                {
                    "name": run.sequence.name,
                    "category": run.sequence.category,
                    "duration_s": run.sequence.duration_s,
                    "meta": run.sequence.meta,
                    "ref_offset_vs_law_mm": _num(run.metrics.get("ref_law_offset_mm")),
                    "metrics": _jsonify(run.metrics),
                    "gate_events": _jsonify(run.gate_events),
                }
                for run in self.runs
            ],
            "findings": [
                {"title": t, "detail": d} for t, d in self.findings
            ],
        }
        (out_dir / "report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- CSV -------------------------------------------------------------

    def _write_csv(self, out_dir: Path) -> None:
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["name", "category", "duration_s"]
                + [flat for flat, _ in _TABLE_COLUMNS]
                + [f"threshold_{flat}" for flat, _ in _TABLE_COLUMNS]
            )
            for run in self.runs:
                row = [
                    run.sequence.name,
                    run.sequence.category,
                    f"{run.sequence.duration_s:.2f}",
                ]
                row += [_fmt(run.metrics.get(flat)) for flat, _ in _TABLE_COLUMNS]
                row += [""] * len(_TABLE_COLUMNS)  # threshold placeholders
                writer.writerow(row)

    def _write_timelines(self, out_dir: Path) -> None:
        for run in self.runs:
            path = out_dir / "timelines" / f"{run.sequence.name}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["t_s"]
                    + [f"target_{a}_m" for a in "xyz"]
                    + [f"ee_{a}_m" for a in "xyz"]
                    + [f"err_{a}_mm" for a in "xyz"]
                    + ["err_norm_mm"]
                    + [f"{j}_deg" for j in BODY_JOINTS]
                    + ["gripper_norm"]
                )
                for s in run.samples:
                    action = run.sequence.frame(s.t)
                    if action is None:
                        tgt = [""] * 3
                        err = [""] * 4
                    else:
                        tp, _ = expected_target(run.ref, action)
                        e = (s.ee - tp) * 1000.0
                        tgt = [f"{v:.5f}" for v in tp]
                        err = [f"{v:.2f}" for v in e] + [
                            f"{float(np.linalg.norm(e)):.2f}"
                        ]
                    writer.writerow(
                        [f"{s.t:.4f}"]
                        + tgt
                        + [f"{v:.5f}" for v in s.ee]
                        + err
                        + [f"{np.degrees(s.qpos[i]):.2f}" for i in range(5)]
                        + [f"{s.obs['gripper.pos']:.2f}"]
                    )

    # -- Markdown ----------------------------------------------------------

    def _write_markdown(self, out_dir: Path) -> None:
        lines: list[str] = []
        lines.append("# Follower response baseline — %s" % self.backend_name)
        lines.append("")
        lines.append("- generated (UTC): %s" % datetime.now(timezone.utc).isoformat())
        lines.append("- git rev: `%s`" % _git_rev())
        lines.append("- action rate: %.0f Hz, nominal obs rate: %.0f Hz"
                     % (self.meta["action_hz"], self.meta["obs_nominal_hz"]))
        lines.append("- sequences: %d" % len(self.runs))
        lines.append(
            "- thresholds: **placeholders** — backfill after B-3 real "
            "validation; nothing here gates pass/fail."
        )
        if self.backend_name == "sim":
            lines.append(
                "- real-backend launch (B-3, human): `%s`"
                % REAL_LAUNCH_CMD
            )
        lines.append("")

        for category in sorted({r.sequence.category for r in self.runs}):
            runs = [r for r in self.runs if r.sequence.category == category]
            lines.append("## %s (%d)" % (category, len(runs)))
            lines.append("")
            header = "| sequence | " + " | ".join(
                disp for _, disp in _TABLE_COLUMNS
            ) + " |"
            lines.append(header)
            lines.append("|" + "---|" * (len(_TABLE_COLUMNS) + 1))
            for run in runs:
                cells = [_fmt(run.metrics.get(flat)) for flat, _ in _TABLE_COLUMNS]
                lines.append("| %s | %s |" % (run.sequence.name, " | ".join(cells)))
            lines.append("")

        lines.append("## Gate events (server-log channel, Gap-2 tooling)")
        lines.append("")
        lines.append(
            "In-process capture of the follower loggers during each run — the "
            "throttled reject/jump/hold/relatch lines the safety stack emits "
            "(the only server-side status channel besides in-process state)."
        )
        lines.append("")
        lines.append("| sequence | reject | jump | hold | relatch |")
        lines.append("|---|---|---|---|---|")
        for run in self.runs:
            g = run.gate_events
            if not g:
                continue
            lines.append(
                "| %s | %s | %s | %s | %s |" % (
                    run.sequence.name,
                    g.get("reject_count", 0), g.get("jump_count", 0),
                    g.get("hold_count", 0), g.get("relatch_count", 0),
                )
            )
        lines.append("")

        lines.append("## Handshake offset (test-side T_ref vs law T_arm_ref)")
        lines.append("")
        lines.append("| sequence | offset_mm |")
        lines.append("|---|---|")
        for run in self.runs:
            v = run.metrics.get("ref_law_offset_mm")
            if v is not None:
                lines.append("| %s | %.2f |" % (run.sequence.name, v))
        lines.append("")

        lines.append("## Findings / implementation gaps (record only)")
        lines.append("")
        for i, (title, detail) in enumerate(self.findings, 1):
            lines.append("%d. **%s** — %s" % (i, title, detail))
        lines.append("")
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.2f}"
    return str(v)


def _num(v):
    return None if v is None else float(v)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PKG_ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"
