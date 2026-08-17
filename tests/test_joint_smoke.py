"""Pure-logic tests for the #05 joint-smoke gate client.

The HITL client itself lives in examples/joint_smoke_client.py (it drives a
real arm over gRPC -- humans in the loop).  Everything decidable without
hardware lives here: calibration-range -> degree limits, per-step limit
pre-check, scan-plan construction, dwell/stream-cut pass criteria.

Protocol locked by grilling (#05): symmetric +/-half-span degree limits from
the recorded raw tick range (same convention as
SO101FollowerServicer._apply_calibration_qpos_limits); target must sit inside
[min+5deg, max-5deg] or the step is SKIPPED (never clipped); tiers +/-5 then
+/-10; pass = steady |obs-cmd| <= 2 deg + no oscillation; stream-cut drift
<= 0.5 deg.
"""

from __future__ import annotations

import pytest

from lerobot_robot_grpc.follower.joint_smoke import limits_from_raw_ranges


def test_limits_from_raw_ranges_symmetric_half_span():
    # Known literal: raw span 1000..3095 -> half-span = (3095-1000)/2 * 360/4095.
    limits = limits_from_raw_ranges({"shoulder_pan": (1000, 3095)})
    expected = 2095 / 2.0 * 360.0 / 4095.0
    assert limits["shoulder_pan"] == pytest.approx((-expected, expected))


def test_limits_from_raw_ranges_full_turn():
    limits = limits_from_raw_ranges({"wrist_roll": (0, 4095)})
    assert limits["wrist_roll"] == pytest.approx((-180.0, 180.0))


def test_limits_from_raw_ranges_rejects_inverted_range():
    # A degenerate calibration (range_max <= range_min) must not produce a
    # limit the pre-check could trust -- the joint is simply absent.
    assert limits_from_raw_ranges({"elbow_flex": (2000, 2000)}) == {}

from lerobot_robot_grpc.follower.joint_smoke import check_target  # noqa: E402


class TestCheckTarget:
    LIMITS = {"shoulder_pan": (-90.0, 90.0)}

    def test_inside_margin_returns_target(self):
        assert check_target("shoulder_pan", base_deg=10.0, delta_deg=5.0,
                            limits=self.LIMITS) == 15.0

    def test_negative_delta_inside(self):
        assert check_target("shoulder_pan", base_deg=10.0, delta_deg=-10.0,
                            limits=self.LIMITS) == 0.0

    def test_outside_margin_skips_not_clips(self):
        # base 80 + 10 = 90, but hi-5 margin = 85 -> skip (None), never clip.
        assert check_target("shoulder_pan", base_deg=80.0, delta_deg=10.0,
                            limits=self.LIMITS) is None

    def test_exactly_at_margin_boundary_passes(self):
        # target == hi - margin is still inside [lo+m, hi-m].
        assert check_target("shoulder_pan", base_deg=80.0, delta_deg=5.0,
                            limits=self.LIMITS) == 85.0

    def test_unknown_joint_skips(self):
        # No trustworthy limit recorded -> do not move that joint at all.
        assert check_target("elbow_flex", base_deg=0.0, delta_deg=5.0,
                            limits=self.LIMITS) is None

    def test_custom_margin(self):
        assert check_target("shoulder_pan", base_deg=80.0, delta_deg=8.0,
                            limits=self.LIMITS, margin_deg=2.0) == 88.0

from lerobot_robot_grpc.follower.joint_smoke import (  # noqa: E402
    BODY_SCAN_ORDER,
    ScanStep,
    build_scan_steps,
)


class TestBuildScanSteps:
    BASE = {
        "shoulder_pan": 0.0, "shoulder_lift": -20.0, "elbow_flex": 60.0,
        "wrist_flex": -40.0, "wrist_roll": 0.0,
    }
    LIMITS = {
        "shoulder_pan": (-90.0, 90.0), "shoulder_lift": (-90.0, 90.0),
        "elbow_flex": (-90.0, 90.0), "wrist_flex": (-90.0, 90.0),
        "wrist_roll": (-90.0, 90.0),
    }

    def test_order_and_tiers(self):
        steps = build_scan_steps(self.BASE, self.LIMITS, tiers=(5.0, 10.0))
        sends = [s for s in steps if isinstance(s, ScanStep)]
        # tier 5 fully precedes tier 10; joints in scan order; + before -.
        tier5 = [s for s in sends if s.tier_deg == 5.0]
        tier10 = [s for s in sends if s.tier_deg == 10.0]
        assert len(tier5) == 10 and len(tier10) == 10
        assert [s.joint for s in tier5] == [j for j in BODY_SCAN_ORDER for _ in (1, -1)]
        assert [s.direction for s in tier5[:2]] == [+1, -1]
        assert sends.index(tier5[-1]) < sends.index(tier10[0])

    def test_targets_absolute_from_base(self):
        steps = build_scan_steps(self.BASE, self.LIMITS, tiers=(5.0,))
        sends = [s for s in steps if isinstance(s, ScanStep)]
        assert sends[0].target_deg == 5.0            # shoulder_pan 0 + 5
        assert sends[1].target_deg == -5.0           # shoulder_pan 0 - 5
        assert sends[2].target_deg == -15.0          # shoulder_lift -20 + 5

    def test_skip_recorded_not_clipped(self):
        base = dict(self.BASE, wrist_roll=83.0)
        steps = build_scan_steps(base, self.LIMITS, tiers=(5.0, 10.0))
        wr = [s for s in steps if getattr(s, "joint", None) == "wrist_roll"]
        # 83+5=88 > 85 and 83+10=93 > 85 -> both + steps skipped; - steps fine.
        skipped = [s for s in wr if not isinstance(s, ScanStep)]
        sends = [s for s in wr if isinstance(s, ScanStep)]
        assert len(skipped) == 2 and all(s.direction == +1 for s in skipped)
        assert [s.target_deg for s in sends] == [78.0, 73.0]

    def test_gripper_excluded(self):
        steps = build_scan_steps(self.BASE, self.LIMITS, tiers=(5.0,))
        assert all(getattr(s, "joint", None) != "gripper" for s in steps)

from lerobot_robot_grpc.follower.joint_smoke import evaluate_dwell  # noqa: E402


class TestEvaluateDwell:
    # 20 samples @100ms = one 2s dwell.  Arm transits 10 -> 15 deg.
    GOOD = [10.0, 12.5, 14.6, 15.2, 14.9, 15.05, 14.95, 15.0, 15.1, 14.9,
            15.0, 15.05, 14.95, 15.0, 15.0, 14.95, 15.05, 15.0, 14.95, 15.0]

    def test_good_step_passes(self):
        res = evaluate_dwell(self.GOOD, target_deg=15.0)
        assert res.passed
        assert res.steady_error_deg == pytest.approx(0.0, abs=0.1)

    def test_steady_error_over_tolerance_fails(self):
        samples = [10.0] + [12.0] * 19  # arrives 3 deg short, stays there
        res = evaluate_dwell(samples, target_deg=15.0)
        assert not res.passed
        assert res.steady_error_deg == pytest.approx(3.0, abs=0.01)

    def test_oscillation_fails(self):
        samples = [10.0, 14.0] + [15.0, 13.0] * 9  # never settles, 2deg pp
        res = evaluate_dwell(samples, target_deg=15.0)
        assert not res.passed

    def test_overshoot_reported(self):
        samples = [10.0, 16.5] + [15.0] * 18  # 1.5 deg overshoot then settle
        res = evaluate_dwell(samples, target_deg=15.0)
        assert res.overshoot_deg == pytest.approx(1.5, abs=0.01)
        assert res.passed  # small overshoot that settles is reported, not failed

    def test_big_overshoot_fails(self):
        samples = [10.0, 19.0] + [15.0] * 18  # 4 deg overshoot = unsafe swing
        res = evaluate_dwell(samples, target_deg=15.0)
        assert not res.passed

from lerobot_robot_grpc.follower.joint_smoke import evaluate_freeze  # noqa: E402


class TestEvaluateFreeze:
    def test_holds_position(self):
        baseline = [15.0, 15.05, 14.95]
        silence = [15.0, 15.1, 14.9, 15.05] * 8  # 3s of stream-cut obs
        assert evaluate_freeze(baseline, silence).passed

    def test_drift_fails(self):
        baseline = [15.0, 15.0, 15.0]
        silence = [15.0] * 10 + [15.8] * 20  # arm sags 0.8 deg mid-silence
        res = evaluate_freeze(baseline, silence)
        assert not res.passed
        assert res.max_drift_deg == pytest.approx(0.8, abs=0.01)

from lerobot_robot_grpc.follower.joint_smoke import (  # noqa: E402
    check_base_headroom,
    retrying_rpc,
)


class TestCheckBaseHeadroom:
    LIMITS = {j: (-90.0, 90.0) for j in BODY_SCAN_ORDER}
    BASE = {j: 0.0 for j in BODY_SCAN_ORDER}

    def _joint(self, results, name):
        return next(r for r in results if r.joint == name)

    def test_mid_range_passes_everywhere(self):
        results = check_base_headroom(self.BASE, self.LIMITS)
        assert len(results) == len(BODY_SCAN_ORDER)
        assert all(r.passed for r in results)
        assert all(r.min_headroom_deg == pytest.approx(90.0) for r in results)

    def test_round1_elbow_wall_fails_via_hand_travel(self):
        # Calibrated range +/-90 says room to spare, but the hand test found
        # the elbow parked ON a wall: 1 deg of room up, 40 down.  Must fail.
        measured = {"elbow_flex": (40.0, 1.0)}
        results = check_base_headroom(self.BASE, self.LIMITS, measured)
        elbow = self._joint(results, "elbow_flex")
        assert not elbow.passed
        assert elbow.min_headroom_deg == pytest.approx(1.0)
        assert elbow.travel_deg == (40.0, 1.0)

    def test_short_travel_in_either_direction_fails(self):
        measured = {"wrist_flex": (8.0, 50.0)}  # 8 < 15 required
        results = check_base_headroom(self.BASE, self.LIMITS, measured)
        assert not self._joint(results, "wrist_flex").passed

    def test_calibrated_wall_fails_without_hand_data(self):
        base = dict(self.BASE, wrist_roll=85.0)  # only 5 deg to calibrated hi
        results = check_base_headroom(base, self.LIMITS)
        wr = self._joint(results, "wrist_roll")
        assert not wr.passed
        assert wr.travel_deg is None

    def test_missing_limit_fails_never_passes(self):
        limits = {k: v for k, v in self.LIMITS.items() if k != "shoulder_pan"}
        results = check_base_headroom(self.BASE, limits)
        pan = self._joint(results, "shoulder_pan")
        assert not pan.passed
        assert pan.min_headroom_deg == float("-inf")

    def test_custom_required_deg(self):
        measured = {"elbow_flex": (12.0, 40.0)}
        results = check_base_headroom(self.BASE, self.LIMITS, measured, required_deg=10.0)
        assert self._joint(results, "elbow_flex").passed


class TestRetryingRpc:
    @staticmethod
    def _no_sleep(_s: float) -> None:
        pass

    def test_success_first_try_never_reconnects(self):
        def reconnect():
            raise AssertionError("healthy RPC must not trigger reconnect")
        assert retrying_rpc(lambda: 42, reconnect, sleep=self._no_sleep) == 42

    def test_camera_death_heals_after_one_reconnect(self):
        state = {"failing": True, "reconnects": 0}
        notes: list[BaseException] = []

        def fn():
            if state["failing"]:
                state["failing"] = False
                raise RuntimeError("read thread is not running")
            return "ok"

        def reconnect():
            state["reconnects"] += 1

        assert retrying_rpc(fn, reconnect, sleep=self._no_sleep,
                            on_retry=notes.append) == "ok"
        assert state["reconnects"] == 1 and len(notes) == 1

    def test_exhausted_attempts_reraise_original(self):
        def fn():
            raise RuntimeError("still down")
        with pytest.raises(RuntimeError, match="still down"):
            retrying_rpc(fn, lambda: None, attempts=3, sleep=self._no_sleep)

    def test_failing_reconnect_does_not_mask_original(self):
        def fn():
            raise RuntimeError("rpc down")
        def reconnect():
            raise RuntimeError("reconnect also down")
        with pytest.raises(RuntimeError, match="rpc down"):
            retrying_rpc(fn, reconnect, attempts=2, sleep=self._no_sleep)
