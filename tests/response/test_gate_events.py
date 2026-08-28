"""Gate-event tooling tests (implementation-gap #2): the parser grammar,
the in-process capture, and the tee'd-file CLI merge.

The parser must stay in lockstep with the law/adapter log lines — these
tests pin the exact strings pose_delta_law.py / so101_follower_server.py
emit today, so a log-format change fails loudly here instead of silently
producing empty gate-event sections in every baseline report.
"""

from __future__ import annotations

import json
import logging

import pytest

from .gate_events import (
    GateLogCapture,
    merge_into_report,
    parse_gate_line,
    parse_log_file,
    slice_events,
    summarize,
)

# Verbatim lines from pose_delta_law.py (_hold / jump reset) — the exact
# formats the throttled warnings use.
LAW_REJECT = (
    "SOLVE rejected (fk-consistency): pos_err=2000mm -- holding last action."
)
LAW_REJECT_COLLISION = (
    "SOLVE rejected (collision): pos_err=12mm -- holding last action."
)
LAW_JUMP = (
    "IK jump: solution 120.0° from the previous — warm start reset "
    "(official 30° rule)."
)
# Verbatim from so101_follower_server.py (real adapter, INFO level).
REAL_HOLD = (
    "pose_delta hold: stale=True rejected=False collided=False pos_err=13.5mm "
    "-- keeping last joints."
)
# Verbatim from mujoco_follower_server.py SetReference.
SIM_RELATCH = "SetReference: T_arm_ref re-locked at current FK pos=[0.3230 -0.0000 0.1413]"


class TestParseGateLine:
    def test_reject_line_parses_reason_and_error(self):
        e = parse_gate_line(LAW_REJECT)
        assert e is not None and e.kind == "reject"
        assert e.detail == "fk-consistency"
        assert e.pos_err_mm == pytest.approx(2000.0)

    def test_collision_reject_reason(self):
        e = parse_gate_line(LAW_REJECT_COLLISION)
        assert e.detail == "collision"

    def test_jump_line(self):
        e = parse_gate_line(LAW_JUMP)
        assert e.kind == "jump" and e.detail == "120.0deg"

    def test_real_hold_line_flags(self):
        e = parse_gate_line(REAL_HOLD)
        assert e.kind == "hold"
        assert "stale=True" in e.detail
        assert "rejected" not in e.detail  # rejected=False omitted by design
        assert e.pos_err_mm == pytest.approx(13.5)

    def test_relatch_line(self):
        assert parse_gate_line(SIM_RELATCH).kind == "relatch"

    def test_noise_lines_return_none(self):
        for noise in (
            "PoseDeltaLaw ready: site='gripperframe' dofs=(0, 1, 2, 3, 4)",
            "IK: pos_err=0.1mm manip=0.0300 q_deg=[...]",
            "MuJoCoSO101Servicer ready: action_mode=pose_delta",
            "Completely unrelated text",
            "",
        ):
            assert parse_gate_line(noise) is None

    def test_rejected_true_hold_keeps_the_flag(self):
        e = parse_gate_line(
            "pose_delta hold: stale=False rejected=True collided=False "
            "pos_err=1.0mm -- keeping last joints."
        )
        assert "rejected=True" in e.detail


class TestGateLogCapture:
    def test_capture_collects_from_the_law_logger_namespace(self):
        capture = GateLogCapture()
        law_logger = logging.getLogger(
            "lerobot_robot_grpc.follower.pose_delta_law"
        )
        capture.attach()
        try:
            law_logger.warning(LAW_REJECT)
            law_logger.warning(LAW_JUMP)
            law_logger.info("IK: pos_err=0.1mm manip=0.03 q_deg=[x]")  # noise
        finally:
            capture.detach()
        kinds = [e.kind for e in capture.events]
        assert kinds == ["reject", "jump"]

    def test_info_level_hold_line_survives_default_warning_root(self):
        """The real adapter's hold line is INFO — capture must lift the
        subtree level to see it, and restore the level afterwards."""
        logger = logging.getLogger("lerobot_robot_grpc.follower")
        prev = logger.level
        capture = GateLogCapture()
        capture.attach()
        try:
            logging.getLogger(
                "lerobot_robot_grpc.follower.so101_follower_server"
            ).info(REAL_HOLD)
        finally:
            capture.detach()
        assert [e.kind for e in capture.events] == ["hold"]
        assert logging.getLogger("lerobot_robot_grpc.follower").level == prev

    def test_slice_by_wall_time(self):
        capture = GateLogCapture()
        capture.attach()
        try:
            logging.getLogger("lerobot_robot_grpc.follower.pose_delta_law").warning(
                LAW_REJECT
            )
        finally:
            capture.detach()
        e = capture.events[0]
        assert slice_events(capture.events, e.t_wall - 1, e.t_wall + 1) == [e]
        assert slice_events(capture.events, e.t_wall + 1, e.t_wall + 2) == []


class TestSummarizeAndMerge:
    def test_summarize_counts_and_samples(self):
        events = [
            parse_gate_line(LAW_REJECT, t_wall=1.0),
            parse_gate_line(LAW_JUMP, t_wall=2.0),
            parse_gate_line(LAW_REJECT, t_wall=3.0),
        ]
        s = summarize(events)
        assert s["total"] == 3 and s["reject_count"] == 2 and s["jump_count"] == 1
        assert len(s["reject_samples"]) == 2

    def test_merge_cli_round_trip(self, tmp_path):
        log = tmp_path / "follower.log"
        log.write_text(
            "\n".join(
                [
                    "2026-08-18 11:00:00,123 WARNING "
                    "lerobot_robot_grpc.follower.pose_delta_law: "
                    + LAW_REJECT,
                    "2026-08-18 11:00:01,456 INFO "
                    "lerobot_robot_grpc.follower.so101_follower_server: "
                    + REAL_HOLD,
                    "2026-08-18 11:00:02,000 INFO "
                    "lerobot_robot_grpc.follower.mujoco_follower_server: "
                    + SIM_RELATCH,
                    "2026-08-18 11:00:03,000 INFO someone.else: noise",
                ]
            ),
            encoding="utf-8",
        )
        parsed = parse_log_file(log)
        assert [e.kind for e in parsed] == ["reject", "hold", "relatch"]
        assert parsed[0].t_text == "2026-08-18 11:00:00,123"

        report = tmp_path / "report.json"
        report.write_text(json.dumps({"sequences": []}), encoding="utf-8")
        merged = merge_into_report(log, report)
        assert merged["reject_count"] == 1 and merged["hold_count"] == 1
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert "gate_events_from_log" in payload
