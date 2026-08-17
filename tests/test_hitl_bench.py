"""Pure-logic tests for the real-arm pose_delta HITL bench (wayfinder #07).

lerobot_robot_grpc.follower.hitl_bench is the offline judge over the CSV the
thin driver (examples/teleop_hitl_bench.py) records: hold windows, freeze
drift, re-lock first-frame jitter, stream gaps (stale hold), leader-down
freeze, and the FK end-effector radius for the base-safety-sphere check.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from lerobot_robot_grpc.follower.hitl_bench import (  # noqa: E402
    BODY_JOINT_KEYS,
    CSV_HEADER,
    build_report,
    ee_positions,
    evaluate_gap_hold,
    evaluate_hold,
    evaluate_leader_down,
    evaluate_relock,
    find_gaps,
    find_hold_windows,
    find_leader_down_windows,
    load_rows,
)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"


def _row(t, engaged=1, sent=1, leader_ok=1, obs=None, act=(0.0, 0.0, 0.0, 30.0)):
    """One CSV row; obs overrides body joints (pan, lift, elbow, wf, wr, grip)."""
    o = {
        "obs_pan_deg": 0.0, "obs_lift_deg": -20.0, "obs_elbow_deg": 60.0,
        "obs_wf_deg": -40.0, "obs_wr_deg": 0.0, "obs_gripper": 10.0,
    }
    if obs:
        o.update(obs)
    return {
        "t_s": t, "engaged": engaged, "sent": sent, "leader_ok": leader_ok,
        "act_dx_m": act[0], "act_dy_m": act[1], "act_dz_m": act[2],
        "act_qx": 0.0, "act_qy": 0.0, "act_qz": 0.0, "act_qw": 1.0,
        "act_grip_mm": act[3], **o,
    }


def _write_csv(path, rows):
    lines = [",".join(CSV_HEADER)]
    for r in rows:
        lines.append(",".join(
            "" if r.get(k) is None else f"{r[k]}" for k in CSV_HEADER
        ))
    path.write_text("\n".join(lines) + "\n")


class TestLoadRows:
    def test_round_trip_types(self, tmp_path):
        csv = tmp_path / "r.csv"
        rows = [_row(0.0), _row(0.033, obs={"obs_elbow_deg": -30.5})]
        _write_csv(csv, rows)
        loaded = load_rows(csv)
        assert len(loaded) == 2
        assert loaded[0]["obs_elbow_deg"] == pytest.approx(60.0)
        assert loaded[1]["obs_elbow_deg"] == pytest.approx(-30.5)
        assert loaded[1]["engaged"] == 1

    def test_unsent_action_cells_load_as_none(self, tmp_path):
        csv = tmp_path / "r.csv"
        rows = [_row(0.0, sent=0)]
        rows[0]["act_dx_m"] = None  # driver leaves action cells empty when unsent
        _write_csv(csv, rows)
        loaded = load_rows(csv)
        assert loaded[0]["act_dx_m"] is None
        assert loaded[0]["obs_pan_deg"] == pytest.approx(0.0)


class TestHoldWindows:
    def test_windows_are_the_disengaged_spans(self):
        rows = [
            _row(0.00), _row(0.033),                       # engaged
            _row(0.066, engaged=0), _row(0.099, engaged=0), _row(0.132, engaged=0),
            _row(0.165),                                    # re-engage
            _row(0.198), _row(0.231), _row(0.264),
            _row(0.297, engaged=0), _row(0.330, engaged=0), _row(0.363, engaged=0),
            _row(0.396),
        ]
        windows = find_hold_windows(rows)
        assert windows == [(2, 4), (9, 11)]

    def test_min_len_drops_short_blips(self):
        rows = [_row(0.0), _row(0.033, engaged=0), _row(0.066)]
        assert find_hold_windows(rows, min_len=3) == []

    def test_trailing_unterminated_window_is_kept(self):
        rows = [_row(0.0), _row(0.033, engaged=0), _row(0.066, engaged=0)]
        assert find_hold_windows(rows, min_len=2) == [(1, 2)]


class TestEvaluateHold:
    def _hold_rows(self, drift_deg=0.0, gripper_span=0.0):
        rows = [
            _row(0.00), _row(0.033), _row(0.066),          # baseline (engaged)
        ]
        for i in range(10):                                 # the hold
            rows.append(_row(
                0.099 + i * 0.033, engaged=0,
                obs={"obs_elbow_deg": 60.0 + drift_deg,
                     "obs_gripper": 10.0 + gripper_span * i / 9.0},
            ))
        rows.append(_row(0.45, engaged=1))                  # re-engage
        return rows

    def test_frozen_hold_passes_and_reports_gripper_alive(self):
        res = evaluate_hold(self._hold_rows(drift_deg=0.2, gripper_span=20.0), (3, 12))
        assert res.passed
        assert res.max_drift_deg == pytest.approx(0.2, abs=0.05)
        assert res.gripper_live          # >= 5 units of gripper motion = live

    def test_drift_beyond_tol_fails(self):
        res = evaluate_hold(self._hold_rows(drift_deg=1.0), (3, 12))
        assert not res.passed
        assert res.worst_joint == "elbow_flex"

    def test_dead_gripper_flags_not_live(self):
        res = evaluate_hold(self._hold_rows(drift_deg=0.0, gripper_span=0.0), (3, 12))
        assert res.passed                # freeze criterion still fine
        assert not res.gripper_live      # but official semantics need gripper live


class TestEvaluateRelock:
    def test_small_post_engage_motion_passes(self):
        rows = [_row(0.0, engaged=0)] + [
            _row(0.033 + i * 0.033, obs={"obs_lift_deg": -20.0 + 0.05 * i})
            for i in range(10)                            # 300 ms of follow
        ]
        res = evaluate_relock(rows, hold_end_idx=0)
        assert res.passed
        assert res.cumulative_deg == pytest.approx(0.45, abs=0.1)

    def test_yank_on_relock_fails(self):
        rows = [_row(0.0, engaged=0)] + [
            _row(0.033 + i * 0.033, obs={"obs_lift_deg": -20.0 + 1.0 * i})
            for i in range(10)                            # ~9 deg in 300 ms
        ]
        res = evaluate_relock(rows, hold_end_idx=0)
        assert not res.passed
        assert res.worst_joint == "shoulder_lift"


class TestGaps:
    def test_gap_detection(self):
        rows = [_row(0.0), _row(0.033), _row(2.0), _row(2.033)]
        gaps = find_gaps(rows, min_gap_s=1.2)
        assert len(gaps) == 1
        assert gaps[0].gap_s == pytest.approx(1.967, abs=0.01)

    def test_normal_cadence_produces_no_gaps(self):
        rows = [_row(i * 0.033) for i in range(50)]
        assert find_gaps(rows) == []

    def test_gap_jump_within_tol_passes_stale_hold(self):
        rows = [_row(0.0), _row(0.033), _row(1.8, obs={"obs_elbow_deg": 60.6})]
        gaps = find_gaps(rows)
        res = evaluate_gap_hold(rows, gaps[0])
        assert res.passed

    def test_gap_jump_beyond_tol_fails(self):
        rows = [_row(0.0), _row(0.033), _row(1.8, obs={"obs_elbow_deg": 64.0})]
        gaps = find_gaps(rows)
        res = evaluate_gap_hold(rows, gaps[0])
        assert not res.passed


class TestLeaderDown:
    def test_leader_down_window_and_freeze(self):
        rows = (
            [_row(0.033 * i) for i in range(5)]
            + [_row(0.165 + 0.033 * i, leader_ok=0, sent=0) for i in range(10)]
            + [_row(0.5 + 0.033 * i) for i in range(5)]
        )
        windows = find_leader_down_windows(rows)
        assert windows == [(5, 14)]
        res = evaluate_leader_down(rows, windows[0])
        assert res.passed          # no commands -> physical hold, drift ~0
        assert res.max_drift_deg == pytest.approx(0.0)

    def test_motion_during_leader_down_fails(self):
        rows = (
            [_row(0.033 * i) for i in range(5)]
            + [
                _row(0.165 + 0.033 * i, leader_ok=0, sent=0,
                     obs={"obs_pan_deg": 3.0 * (i + 1)})
                for i in range(10)
            ]
        )
        windows = find_leader_down_windows(rows)
        res = evaluate_leader_down(rows, windows[0])
        assert not res.passed


class TestEndEffectorPositions:
    def test_fk_matches_an_independent_model(self):
        rows = [_row(0.0, obs={"obs_pan_deg": 5.0, "obs_lift_deg": -30.0})]
        pos = ee_positions(rows, str(XML_PATH))
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        data = mujoco.MjData(model)
        qpos = [math.radians(v) for v in (5.0, -30.0, 60.0, -40.0, 0.0)]
        qpos.append((10.0 / 100.0) * (1.74533 - (-0.17453)) + (-0.17453))
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        assert pos.shape == (1, 3)
        assert pos[0] == pytest.approx(data.site_xpos[sid], abs=1e-9)


class TestBuildReport:
    def test_report_covers_all_sections(self, tmp_path):
        # One hold with drift, one relock, one gap -- every section populated.
        rows = [_row(0.033 * i) for i in range(5)]
        rows += [_row(0.165 + 0.033 * i, engaged=0, obs={"obs_gripper": 10.0 + i})
                 for i in range(10)]
        rows += [_row(0.5 + 0.033 * i, obs={"obs_lift_deg": -20.0 + 0.01 * i})
                 for i in range(5)]
        rows.append(_row(3.0))  # a 1.6 s gap -> stale check
        csv = tmp_path / "r.csv"
        _write_csv(csv, rows)
        report = build_report(load_rows(csv), xml_path=str(XML_PATH))
        assert len(report.holds) >= 1
        assert report.holds[0].passed
        assert report.holds[0].gripper_live
        assert len(report.relocks) >= 1
        assert report.relocks[0].passed
        assert len(report.gaps) >= 1
        assert report.leader_downs == []       # leader_ok stayed 1 throughout
        assert report.sphere is not None
        assert report.sphere.max_radius_m > 0.0
        text = report.render()
        assert "HOLD" in text and "RELOCK" in text and "SPHERE" in text
        assert "STALE" in text

    def test_sphere_violation_fails_the_report(self, tmp_path):
        # A pose far outside the 391 mm sphere cannot come from FK of a sane
        # joint set, so drive the radius check directly via a stub positions
        # array through the report's sphere evaluation helper instead.
        from lerobot_robot_grpc.follower.hitl_bench import evaluate_sphere
        import numpy as np
        pos = np.array([[0.0, 0.0, 0.30], [0.40, 0.0, 0.10]])
        ok = evaluate_sphere(pos, radius_m=0.72 * 0.543)
        assert not ok.passed
        assert ok.max_radius_m == pytest.approx(math.hypot(0.40, 0.10))
        inside = evaluate_sphere(np.array([[0.30, 0.0, 0.10]]), radius_m=0.72 * 0.543)
        assert inside.passed


def test_body_joint_keys_match_the_model_order():
    # The FK path indexes model DOFs by position -- the key order must be the
    # SO-101 body-joint order (pan, lift, elbow, wf, wr).
    assert BODY_JOINT_KEYS == (
        "obs_pan_deg", "obs_lift_deg", "obs_elbow_deg",
        "obs_wf_deg", "obs_wr_deg",
    )
