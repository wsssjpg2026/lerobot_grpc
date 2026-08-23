"""auto_reference mode tests — clients without an alignment step.

``PikaSenseServicer(auto_reference=True)`` blocks Connect until fresh optical
samples are stable, then latches ``T_begin`` and engages.  There is no unsafe
first-action lazy fallback, so a client that never calls ``SetReference`` —
e.g. ``lerobot-teleoperate`` — still starts with an identity delta.
Default mode keeps the #10 contract: the session starts disengaged and
``SetReference`` is what engages teleop.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from lerobot_robot_grpc.leader.pika_sense_leader_server import PikaSenseServicer
from lerobot_robot_grpc.leader.tracker_readiness import TrackerReadinessGate
from lerobot_robot_grpc.protos import device_pb2

pika_sense = pytest.importorskip("pika.sense")


class _FakePose:
    def __init__(
        self,
        position,
        quat,
        timestamp,
        *,
        optical_timestamp_s,
        optical_age_s,
        optical_measurement_count,
        optical_lighthouse_count,
        optical_axis_count,
        pose_confidence,
        raw_optical_timestamp_s=0.0,
        raw_optical_age_s=0.0,
        raw_optical_measurement_count=12,
        raw_optical_event_sequence=0,
        optical_event_sequence=0,
    ):
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(quat, dtype=float)
        self.timestamp = float(timestamp)
        self.optical_timestamp_s = float(optical_timestamp_s)
        self.optical_age_s = float(optical_age_s)
        self.optical_measurement_count = int(optical_measurement_count)
        self.optical_lighthouse_count = int(optical_lighthouse_count)
        self.optical_axis_count = int(optical_axis_count)
        self.pose_confidence = float(pose_confidence)
        self.raw_optical_timestamp_s = float(raw_optical_timestamp_s)
        self.raw_optical_age_s = float(raw_optical_age_s)
        self.raw_optical_measurement_count = int(raw_optical_measurement_count)
        self.raw_optical_event_sequence = int(raw_optical_event_sequence)
        self.optical_event_sequence = int(optical_event_sequence)


class _FakeSense:
    """Stand-in for ``pika.sense.Sense`` covering the Connect path too."""

    def __init__(self):
        self.pose_position = np.zeros(3)
        self.pose_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper = 30.0
        self.devices = ["FAKE"]
        self.timestamp = 0.0
        self.advance_timestamp = True
        self.timestamp_step_s = 0.01
        self.optical_timestamp_s = 0.0
        self.advance_optical_timestamp = True
        self.optical_timestamp_step_s = 0.01
        self.raw_optical_timestamp_s = 0.0
        self.advance_raw_optical_timestamp = True
        self.raw_optical_timestamp_step_s = 0.01
        self.raw_optical_age_s = 0.0
        self.raw_optical_measurement_count = 12
        self.raw_optical_event_sequence = 0
        self.optical_event_sequence = 0
        self.optical_age_s = 0.0
        self.optical_measurement_count = 12
        self.optical_lighthouse_count = 2
        self.optical_axis_count = 4
        self.pose_confidence = 100.0
        self.connect_calls = 0
        self.restart_tracker_calls = 0

    def connect(self):
        self.connect_calls += 1

    def get_vive_tracker(self):
        pass

    def restart_vive_tracker(self):
        self.restart_tracker_calls += 1
        return True

    def get_tracker_devices(self):
        return self.devices

    def get_pose(self, device):
        if self.advance_timestamp:
            self.timestamp += self.timestamp_step_s
        if self.advance_optical_timestamp:
            self.optical_timestamp_s += self.optical_timestamp_step_s
            self.optical_age_s = 0.0
            self.optical_event_sequence += 1
        if self.advance_raw_optical_timestamp:
            self.raw_optical_timestamp_s += self.raw_optical_timestamp_step_s
            self.raw_optical_age_s = 0.0
            self.raw_optical_event_sequence += 1
        return _FakePose(
            self.pose_position,
            self.pose_quat,
            self.timestamp,
            optical_timestamp_s=self.optical_timestamp_s,
            optical_age_s=self.optical_age_s,
            optical_measurement_count=self.optical_measurement_count,
            optical_lighthouse_count=self.optical_lighthouse_count,
            optical_axis_count=self.optical_axis_count,
            pose_confidence=self.pose_confidence,
            raw_optical_timestamp_s=self.raw_optical_timestamp_s,
            raw_optical_age_s=self.raw_optical_age_s,
            raw_optical_measurement_count=self.raw_optical_measurement_count,
            raw_optical_event_sequence=self.raw_optical_event_sequence,
            optical_event_sequence=self.optical_event_sequence,
        )

    def get_tracker_health(self, device=None):
        return {
            "bridge_available": True,
            "context_epoch": 1,
            "global_scene_generation": 1,
            "lighthouse_cohort_generation": 2,
            "discovered_lighthouses": ("LH0", "LH1"),
            "lighthouses": {
                "LH0": {
                    "position": (0.0, 0.0, 0.0),
                    "rotation": (1.0, 0.0, 0.0, 0.0),
                },
                "LH1": {
                    "position": (2.0, 0.0, 0.0),
                    "rotation": (1.0, 0.0, 0.0, 0.0),
                },
            },
            "raw_optical_timestamp_s": self.raw_optical_timestamp_s,
            "raw_optical_age_s": self.raw_optical_age_s,
            "raw_optical_measurement_count": self.raw_optical_measurement_count,
            "raw_optical_event_sequence": self.raw_optical_event_sequence,
            "optical_timestamp_s": self.optical_timestamp_s,
            "optical_age_s": self.optical_age_s,
            "optical_measurement_count": self.optical_measurement_count,
            "optical_lighthouse_count": self.optical_lighthouse_count,
            "optical_event_sequence": self.optical_event_sequence,
            "pose_confidence": self.pose_confidence,
        }

    def get_gripper_distance(self):
        return self.gripper


def _make_leader(
    tmp_path,
    auto_reference,
    *,
    arm_prefix=None,
    cumulative_clutch=False,
    command_state_provider=None,
):
    servicer = PikaSenseServicer(
        port="/dev/null",
        R_lh2base=np.eye(3),
        calibration_dir=str(tmp_path),
        command_state_provider=command_state_provider or (lambda: 1),
        auto_reference=auto_reference,
        arm_prefix=arm_prefix,
        cumulative_clutch=cumulative_clutch,
        tracker_recheck_window_s=0.0,
        tracker_recheck_samples=1,
        tracker_health_enabled=False,
    )
    servicer._device = _FakeSense()
    servicer._tracker_device = "FAKE"
    servicer._readiness_gate = TrackerReadinessGate(
        cohort_stable_s=0.0,
        map_stable_s=0.0,
        stable_window_s=0.0,
        stable_samples=1,
        position_spread_m=1.0,
        rotation_spread_rad=np.pi,
    )
    return servicer


def _delta_pos(action: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            action["hand.delta_pos.x"],
            action["hand.delta_pos.y"],
            action["hand.delta_pos.z"],
        ],
        dtype=float,
    )


class TestAutoReference:
    def test_connect_latches_and_engages(self, tmp_path):
        """Connect auto-latches T_begin at the current pose and engages, so a
        later hand movement publishes a live delta without SetReference."""
        servicer = _make_leader(tmp_path, auto_reference=True)
        servicer.Connect(None, None)
        assert servicer._t_begin_pos is not None
        assert servicer._clutched is True

        servicer._device.pose_position = np.array([0.6, 0.0, 0.0])
        action = servicer._compute_action()
        assert _delta_pos(action)[0] == pytest.approx(0.6, abs=1e-6)

    def test_latch_frame_publishes_zero(self, tmp_path):
        """The frame at the latch itself publishes ~zero: the Connect pose is
        the delta origin, so the first command cannot snap the arm."""
        servicer = _make_leader(tmp_path, auto_reference=True)
        servicer.Connect(None, None)
        action = servicer._compute_action()
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)

    def test_connect_is_hardware_only_and_readiness_is_polled_separately(self, tmp_path):
        servicer = _make_leader(tmp_path, auto_reference=True)
        servicer._device.advance_optical_timestamp = False
        servicer.Connect(None, None)

        assert servicer._connected is True
        assert servicer._device.connect_calls == 1

    def test_readiness_sampling_does_not_depend_on_rpc_poll_rate(self, tmp_path):
        """A slow collection client must observe, not drive, convergence."""
        servicer = _make_leader(tmp_path, auto_reference=False)
        servicer._readiness_gate = TrackerReadinessGate(
            cohort_stable_s=0.0,
            map_stable_s=0.0,
            stable_window_s=0.05,
            stable_samples=4,
            position_spread_m=1.0,
            rotation_spread_rad=np.pi,
        )
        servicer._readiness_sample_period_s = 0.005

        # A non-None context models the real gRPC Connect path.  Do not call
        # GetTrackingReadiness while the leader-owned sampler converges.
        servicer.Connect(None, object())
        try:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                if servicer._last_readiness.state == (
                    device_pb2.TrackingReadinessState.
                    TRACKING_READINESS_STATE_READY
                ):
                    break
                time.sleep(0.005)
            readiness = servicer.GetTrackingReadiness(None, object())
            servicer.Disconnect(None, None)
            assert servicer._readiness_thread is not None
            assert servicer._readiness_thread.is_alive()
        finally:
            servicer._stop_readiness_sampler()

        assert readiness.state == (
            device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_READY
        )
        assert readiness.stable_sample_count >= 4
        assert readiness.stable_window_s >= 0.05
        assert servicer._readiness_thread is None

    def test_runtime_stale_pose_holds_until_stable_button_recovery(self, tmp_path):
        command_state = [0]
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
            command_state_provider=lambda: command_state[0],
        )
        servicer._tracker_health_enabled = True
        servicer._tracker_recheck_samples = 1
        servicer._tracker_recheck_window_s = 0.0
        servicer.Connect(None, None)

        servicer._device.advance_timestamp = False
        servicer._device.advance_optical_timestamp = False
        servicer._last_fresh_received_monotonic -= 0.6
        frozen = servicer._compute_action()
        assert servicer._reference_required is True
        assert servicer._clutched is False

        assert (
            servicer._last_action_quality
            == device_pb2.FrameQuality.FRAME_QUALITY_STALE
        )
        np.testing.assert_allclose(_delta_pos(frozen), [0.0, 0.0, 0.0], atol=1e-9)

        wire = list(servicer.GetAction(None, None))
        assert wire
        assert all(
            feature.quality == device_pb2.FrameQuality.FRAME_QUALITY_STALE
            for feature in wire
        )

        servicer._device.advance_timestamp = True
        servicer._device.advance_optical_timestamp = True
        servicer._compute_action()
        assert servicer._tracker_recovery_ready is True
        command_state[0] = 1
        resumed = servicer._compute_action()
        assert servicer._reference_required is False
        assert servicer._clutched is True
        assert (
            servicer._last_action_quality
            == device_pb2.FrameQuality.FRAME_QUALITY_GOOD
        )
        np.testing.assert_allclose(_delta_pos(resumed), [0.0, 0.0, 0.0], atol=1e-9)

    def test_runtime_brief_optical_gap_holds_and_requires_confirmation(
        self, tmp_path
    ):
        """Any hold-worthy optical gap invalidates the session reference."""
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer._tracker_recheck_samples = 1
        servicer._tracker_recheck_window_s = 0.0
        servicer.Connect(None, None)

        servicer._device.advance_timestamp = False
        servicer._device.advance_optical_timestamp = False
        servicer._last_fresh_received_monotonic -= 0.2
        held = list(servicer.GetAction(None, None))

        assert servicer._reference_required is True
        assert servicer._clutched is False
        assert held
        assert all(
            feature.quality == device_pb2.FrameQuality.FRAME_QUALITY_STALE
            for feature in held
        )

        servicer._device.advance_timestamp = True
        servicer._device.advance_optical_timestamp = True
        recovered = list(servicer.GetAction(None, None))
        assert servicer._reference_required is True
        assert all(
            feature.tracking_state
            == device_pb2.TrackingState.TRACKING_STATE_CONFIRM_REQUIRED
            for feature in recovered
        )

    def test_coherent_fast_motion_never_reports_tracking_loss(self, tmp_path):
        """Consecutive fast hand motion is delayed briefly, not called a loss.

        The 2026-08-21 hardware trace repeatedly exceeded the old 1 m/s
        adjacent-sample gate while both Lighthouse channels remained active.
        Every frame must remain READY; the follower's own rate limiter remains
        responsible for safely approaching the confirmed target.
        """
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()

        wire = []
        for step in range(1, 7):
            # 20 mm / 10 ms = 2 m/s: deliberately above the old soft gate.
            servicer._device.pose_position[0] = 0.020 * step
            wire = list(servicer.GetAction(None, None))
            assert all(
                feature.tracking_state
                == device_pb2.TrackingState.TRACKING_STATE_READY
                for feature in wire
            )

        assert servicer._reference_required is False
        assert servicer._bad_tracker_samples == 0
        assert servicer._published_pos[0] == pytest.approx(0.120)
        assert all(
            feature.quality == device_pb2.FrameQuality.FRAME_QUALITY_GOOD
            for feature in wire
        )

    def test_single_pose_outlier_snapback_is_hidden_from_robot(self, tmp_path):
        """A one-frame solver excursion is discarded when the pose snaps back."""
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()

        servicer._device.pose_position[0] = 0.060
        quarantined = list(servicer.GetAction(None, None))
        assert all(
            feature.tracking_state
            == device_pb2.TrackingState.TRACKING_STATE_READY
            for feature in quarantined
        )
        assert servicer._published_pos[0] == pytest.approx(0.0)

        servicer._device.pose_position[0] = 0.001
        recovered = list(servicer.GetAction(None, None))
        assert all(
            feature.tracking_state
            == device_pb2.TrackingState.TRACKING_STATE_READY
            for feature in recovered
        )
        assert servicer._published_pos[0] == pytest.approx(0.001)
        assert servicer._reference_required is False

    def test_pose_kinematics_uses_source_timestamp_not_optical_sweep_time(
        self, tmp_path
    ):
        """A healthy fused pose must be judged at its own production time."""
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()

        # The fused pose advances by 40 ms while the latest optical sweep only
        # advances by 10 ms.  30 mm / 40 ms is plausible; dividing by the sweep
        # interval would incorrectly classify it as a jump.
        servicer._device.timestamp_step_s = 0.04
        servicer._device.optical_timestamp_step_s = 0.01
        servicer._device.pose_position[0] = 0.030
        wire = list(servicer.GetAction(None, None))

        assert servicer._published_pos[0] == pytest.approx(0.030)
        assert all(
            feature.tracking_state
            == device_pb2.TrackingState.TRACKING_STATE_READY
            for feature in wire
        )

    def test_extreme_pose_discontinuity_still_requires_reference(self, tmp_path):
        """Temporal confirmation must not weaken the absolute safety gate."""
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()

        servicer._device.pose_position[0] = 0.250
        wire = list(servicer.GetAction(None, None))

        assert servicer._reference_required is True
        assert servicer._clutched is False
        assert servicer._published_pos[0] == pytest.approx(0.0)
        assert all(
            feature.tracking_state
            != device_pb2.TrackingState.TRACKING_STATE_READY
            for feature in wire
        )

    def test_healthy_optical_pose_jump_has_pose_specific_lifecycle(self, tmp_path):
        """A fused-pose reset must not be reported as Lighthouse loss.

        This reproduces the measured 138 -> 153 -> 157 mm sequence: optical
        timestamps, event counts and Lighthouse support remain healthy, while
        the fused pose is too discontinuous to expose directly to the robot.
        """
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer._tracker_recheck_samples = 2
        servicer._tracker_recheck_window_s = 0.0
        servicer.Connect(None, None)
        servicer._compute_action()

        servicer._device.timestamp_step_s = 0.032
        servicer._device.optical_timestamp_step_s = 0.032
        for position_m in (0.1377, 0.1530, 0.1574):
            servicer._device.pose_position[0] = position_m
            quarantined = list(servicer.GetAction(None, None))

        assert servicer._reference_required is True
        assert servicer._published_pos[0] == pytest.approx(0.0)
        assert {
            feature.tracking_state for feature in quarantined
        } == {device_pb2.TrackingState.TRACKING_STATE_POSE_DISCONTINUITY}

        stable = list(servicer.GetAction(None, None))
        assert {
            feature.tracking_state for feature in stable
        } == {device_pb2.TrackingState.TRACKING_STATE_POSE_CONFIRM_REQUIRED}
        assert all(
            feature.quality == device_pb2.FrameQuality.FRAME_QUALITY_STALE
            for feature in stable
        )

    def test_clutch_reposition_bypasses_motion_gate_but_not_optical_gate(
        self, tmp_path
    ):
        """Large intentional repositioning while held is not a pose fault."""
        command_state = [1]
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
            command_state_provider=lambda: command_state[0],
        )
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()

        command_state[0] = 0
        servicer._compute_action()
        assert servicer._clutched is False

        servicer._device.pose_position[0] = 0.500
        held = servicer._compute_action()
        assert servicer._reference_required is False
        assert _delta_pos(held)[0] == pytest.approx(0.0)

        command_state[0] = 1
        resumed = servicer._compute_action()
        assert servicer._reference_required is False
        assert servicer._clutched is True
        assert _delta_pos(resumed)[0] == pytest.approx(0.0)

    def test_runtime_recovery_exposes_confirmation_required_on_wire(self, tmp_path):
        """The collection client must know when a recovery squeeze can succeed."""
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer._tracker_recheck_samples = 1
        servicer._tracker_recheck_window_s = 0.0
        servicer.Connect(None, None)

        servicer._device.advance_timestamp = False
        servicer._device.advance_optical_timestamp = False
        servicer._last_fresh_received_monotonic -= 0.6
        list(servicer.GetAction(None, None))
        assert servicer._reference_required is True

        servicer._device.advance_timestamp = True
        servicer._device.advance_optical_timestamp = True
        recovering = list(servicer.GetAction(None, None))

        assert recovering
        assert all(
            feature.tracking_state
            == device_pb2.TrackingState.TRACKING_STATE_CONFIRM_REQUIRED
            for feature in recovering
        )

    def test_raw_light_return_reports_recovering_before_decoded_pose(
        self, tmp_path
    ):
        """Lighthouse visibility must be surfaced while Gen2 decoding relocks."""
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer._tracker_recheck_samples = 1
        servicer._tracker_recheck_window_s = 0.0
        servicer.Connect(None, None)
        servicer._compute_action()

        servicer._device.advance_optical_timestamp = False
        servicer._device.advance_raw_optical_timestamp = False
        servicer._device.optical_age_s = 0.2
        servicer._device.raw_optical_age_s = 0.2
        servicer._last_fresh_received_monotonic -= 0.6
        lost = list(servicer.GetAction(None, None))
        assert {
            feature.tracking_state for feature in lost
        } == {device_pb2.TrackingState.TRACKING_STATE_LOST}

        # The tracker is visible again, but libsurvive has not emitted a new
        # decoded sync/sweep or a trustworthy optical pose yet.
        servicer._device.advance_raw_optical_timestamp = True
        servicer._device.raw_optical_age_s = 0.0
        reacquiring = list(servicer.GetAction(None, None))
        assert {
            feature.tracking_state for feature in reacquiring
        } == {device_pb2.TrackingState.TRACKING_STATE_RECOVERING}
        assert servicer._reference_required is True
        assert servicer._tracker_recovery_ready is False

        # Robot control remains held until decoded optical poses are stable.
        servicer._device.advance_optical_timestamp = True
        servicer._device.optical_age_s = 0.0
        confirmable = list(servicer.GetAction(None, None))
        assert {
            feature.tracking_state for feature in confirmable
        } == {device_pb2.TrackingState.TRACKING_STATE_CONFIRM_REQUIRED}

    def test_raw_light_without_decoded_events_restarts_tracker_backend(
        self, tmp_path
    ):
        """A stuck Gen2 decoder is rebuilt only after raw light has returned."""
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()

        servicer._device.advance_optical_timestamp = False
        servicer._device.advance_raw_optical_timestamp = False
        servicer._device.optical_age_s = 0.2
        servicer._device.raw_optical_age_s = 0.2
        servicer._last_fresh_received_monotonic -= 0.6
        list(servicer.GetAction(None, None))
        assert servicer._device.restart_tracker_calls == 0

        # Physical light is back, but the decoded stream remains frozen past
        # the bounded recovery grace period.
        servicer._device.advance_raw_optical_timestamp = True
        servicer._device.raw_optical_age_s = 0.0
        servicer._decoder_restart_after_s = 0.0
        recovering = list(servicer.GetAction(None, None))

        assert servicer._device.restart_tracker_calls == 1
        assert {
            feature.tracking_state for feature in recovering
        } == {device_pb2.TrackingState.TRACKING_STATE_RECOVERING}
        assert servicer._reference_required is True
        assert servicer._clutched is False

        # The replacement context briefly re-enumerates T20.  That bounded
        # discovery gap remains RECOVERING rather than flickering back to LOST.
        servicer._device.devices = []
        rediscovering = list(servicer.GetAction(None, None))
        assert {
            feature.tracking_state for feature in rediscovering
        } == {device_pb2.TrackingState.TRACKING_STATE_RECOVERING}

    def test_prolonged_total_optical_silence_restarts_tracker_backend(
        self, tmp_path
    ):
        """A wedged raw monitor must not leave recovery stuck forever.

        The real failure trace had neither decoded events nor raw-light
        events after the Tracker returned to view.  While the follower is
        already held, one bounded context rebuild is safer than waiting
        indefinitely for a raw event that the wedged monitor cannot report.
        """
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()

        servicer._device.advance_timestamp = False
        servicer._device.advance_optical_timestamp = False
        servicer._device.advance_raw_optical_timestamp = False
        servicer._device.optical_age_s = 0.2
        servicer._device.raw_optical_age_s = 0.2
        servicer._device.raw_optical_measurement_count = 0
        servicer._last_fresh_received_monotonic -= 0.6
        servicer._decoder_restart_without_raw_after_s = 0.0

        recovering = list(servicer.GetAction(None, None))

        assert servicer._device.restart_tracker_calls == 1
        assert servicer._reference_required is True
        assert servicer._clutched is False
        assert {
            feature.tracking_state for feature in recovering
        } == {device_pb2.TrackingState.TRACKING_STATE_RECOVERING}

    def test_collection_recovery_stays_confirmable_while_operator_repositions(
        self, tmp_path
    ):
        """After optical convergence, the operator may align Pika before squeezing."""
        command_state = [0]
        servicer = _make_leader(
            tmp_path,
            auto_reference=False,
            cumulative_clutch=True,
            command_state_provider=lambda: command_state[0],
        )
        servicer._tracker_health_enabled = True
        servicer._tracker_recheck_samples = 2
        servicer._tracker_recheck_window_s = 0.0
        servicer.Connect(None, None)
        servicer.SetReference(None, None)
        servicer._compute_action()  # establish command-state baseline

        servicer._device.advance_timestamp = False
        servicer._device.advance_optical_timestamp = False
        servicer._last_fresh_received_monotonic -= 0.6
        list(servicer.GetAction(None, None))

        servicer._device.advance_timestamp = True
        servicer._device.advance_optical_timestamp = True
        list(servicer.GetAction(None, None))
        converged = list(servicer.GetAction(None, None))
        assert all(
            feature.tracking_state
            == device_pb2.TrackingState.TRACKING_STATE_CONFIRM_REQUIRED
            for feature in converged
        )

        # Repositioning is intentional after localization has converged.  It
        # must not revoke the prompt while optical health remains fresh.
        servicer._device.pose_position[0] += 0.2
        repositioned = list(servicer.GetAction(None, None))
        assert all(
            feature.tracking_state
            == device_pb2.TrackingState.TRACKING_STATE_CONFIRM_REQUIRED
            for feature in repositioned
        )

        command_state[0] = 1
        pending = list(servicer.GetAction(None, None))
        assert servicer._reference_required is True
        assert all(
            feature.tracking_state
            == device_pb2.TrackingState.TRACKING_STATE_REFERENCE_PENDING
            for feature in pending
        )
        assert all(
            feature.quality == device_pb2.FrameQuality.FRAME_QUALITY_DEGRADED
            for feature in pending
        )

    def test_runtime_imu_drift_during_optical_occlusion_is_never_published(self, tmp_path):
        """A fused IMU pose is not a fresh optical observation.

        libsurvive keeps advancing ``PoseData.timestamp`` from IMU updates
        after the tracker is covered.  The old implementation accepted a
        bounded 6 mm/sample drift until it became a large arm target.  The
        optical timestamp must freeze the output and force re-reference.
        """
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
        )
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)

        servicer._device.advance_optical_timestamp = False
        servicer._device.optical_age_s = 0.2
        servicer._last_fresh_received_monotonic -= 0.2
        for _ in range(50):
            # The fused timestamp continues and every individual movement is
            # below the existing adjacent-sample jump threshold.
            servicer._device.pose_position[0] += 0.006
            action = servicer._compute_action()

        assert servicer._device.timestamp > servicer._device.optical_timestamp_s
        # Fused IMU drift is held immediately and the old session reference
        # is invalidated. Recovery requires a fresh optical window and an
        # explicit operator confirmation.
        assert servicer._reference_required is True
        assert servicer._clutched is False
        assert (
            servicer._last_action_quality
            == device_pb2.FrameQuality.FRAME_QUALITY_STALE
        )
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)

    def test_get_action_never_lazy_latches_before_connect(self, tmp_path):
        """GetAction cannot bypass hardware connection/readiness."""
        servicer = _make_leader(tmp_path, auto_reference=True)
        action = servicer._compute_action()
        assert servicer._t_begin_pos is None
        assert servicer._clutched is False
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)


class TestLateTrackerDiscovery:
    def test_tracker_discovered_late_heals_pose_channel(self, tmp_path):
        """Cold-start race (05 号议题实测): the tracker registers with
        pysurvive after hardware Connect, so _tracker_device initially stays
        None. The pose channel used to be dead for the process lifetime.
        _read_tracker_pose now re-scans lazily until the tracker appears."""
        servicer = _make_leader(tmp_path, auto_reference=True)
        servicer._tracker_device = None            # Connect gave up
        servicer._device.devices = ["LH0", "LH1"]  # only lighthouses so far

        action = servicer._compute_action()        # pose channel dead: zeros
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)
        assert servicer._tracker_device is None

        servicer._device.devices = ["LH0", "LH1", "T20"]  # tracker lands late
        action = servicer._compute_action()        # lazy scan, but no unsafe latch
        assert servicer._tracker_device == "T20"
        assert servicer._t_begin_pos is None
        assert servicer._clutched is False
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)


class TestDefaultContract:
    def test_periodic_optical_age_uses_monotonic_clock(self, tmp_path, caplog):
        """The operator diagnostic must report milliseconds, not epoch time."""
        servicer = _make_leader(tmp_path, auto_reference=False)
        servicer.Connect(None, None)
        caplog.clear()
        caplog.set_level(
            logging.INFO,
            logger=(
                "lerobot_robot_grpc.leader.pika_sense_leader_server"
            ),
        )

        servicer._compute_action()

        tracker_log = next(
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("TRACKER:")
        )
        optical_age_ms = int(
            tracker_log.split("optical_age=", 1)[1].split("ms", 1)[0]
        )
        assert optical_age_ms < 10_000

    def test_collection_connect_checks_optical_health_not_hand_stillness(self, tmp_path):
        """Collection performs its own later Enter/reference alignment.

        Its initial leader Connect must accept healthy optical samples while
        the operator is holding or moving Pika; physical stillness is only
        required for the short SetReference capture.
        """
        servicer = _make_leader(tmp_path, auto_reference=False)
        servicer._tracker_health_enabled = True
        original_get_pose = servicer._device.get_pose

        def moving_pose(device):
            servicer._device.pose_position[0] += 0.01
            return original_get_pose(device)

        servicer._device.get_pose = moving_pose

        servicer.Connect(None, None)

        assert servicer._connected is True
        assert servicer._t_begin_pos is None
        assert servicer._clutched is False

    def test_default_mode_still_waits_for_set_reference(self, tmp_path):
        """Without auto_reference the #10 contract is unchanged: Connect
        leaves the session disengaged and movement publishes zeros until the
        client's SetReference lands."""
        servicer = _make_leader(tmp_path, auto_reference=False)
        servicer.Connect(None, None)
        assert servicer._clutched is False
        assert servicer._t_begin_pos is None

        servicer._device.pose_position = np.array([0.6, 0.0, 0.0])
        action = servicer._compute_action()
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)

        servicer._device.pose_position = np.zeros(3)
        servicer.SetReference(None, None)
        servicer._device.pose_position = np.array([0.6, 0.0, 0.0])
        action = servicer._compute_action()
        assert _delta_pos(action)[0] == pytest.approx(0.6, abs=1e-6)

    def test_set_reference_accepts_large_intentional_alignment_move(self, tmp_path):
        """Alignment movement is a new trust boundary, not a runtime jump.

        Collection connects the leader, lets the operator reposition Pika,
        then calls SetReference.  No GetAction calls occur while the operator
        aligns, so the current pose can legitimately be far from the last
        runtime-health sample.
        """
        servicer = _make_leader(tmp_path, auto_reference=False)
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()  # GRPCLeader warmup sample before alignment.

        servicer._device.pose_position = np.array([0.3, 0.0, 0.0])
        servicer.SetReference(None, None)

        np.testing.assert_allclose(servicer._t_begin_pos, [0.3, 0.0, 0.0])
        assert servicer._reference_required is False
        assert servicer._clutched is True
        np.testing.assert_allclose(_delta_pos(servicer._compute_action()), [0.0, 0.0, 0.0])

    def test_set_reference_rejects_cached_pose_without_new_optical_samples(self, tmp_path):
        servicer = _make_leader(tmp_path, auto_reference=False)
        servicer._tracker_health_enabled = True
        servicer.Connect(None, None)
        servicer._compute_action()
        servicer._device.advance_timestamp = False
        servicer._device.advance_optical_timestamp = False
        servicer._tracker_reference_timeout_s = 0.03

        with pytest.raises(RuntimeError, match="no fresh stable tracker pose"):
            servicer.SetReference(None, None)

        assert servicer._reference_required is True
        assert servicer._reference_confirmation_pending is False
        assert servicer._tracker_recovery_ready is False
        assert servicer._clutched is False

    def test_set_reference_accepts_small_handheld_spread(self, tmp_path):
        """Reference capture tolerates normal hand tremor within identity limits."""
        servicer = _make_leader(tmp_path, auto_reference=False)
        servicer._tracker_health_enabled = True
        servicer._tracker_recheck_window_s = 0.03
        servicer._tracker_recheck_samples = 5
        servicer._tracker_reference_timeout_s = 0.10
        servicer.Connect(None, None)

        original_get_pose = servicer._device.get_pose
        count = 0

        def hand_held_pose(device):
            nonlocal count
            count += 1
            servicer._device.pose_position = np.array(
                [0.0015 if count % 2 else -0.0015, 0.0, 0.0]
            )
            return original_get_pose(device)

        servicer._device.get_pose = hand_held_pose
        servicer.SetReference(None, None)

        assert servicer._clutched is True
        assert servicer._reference_required is False


class TestS1DebugAdapter:
    def test_arm_prefix_namespaces_the_advertised_and_emitted_action(self, tmp_path):
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            arm_prefix="left",
        )
        servicer.Connect(None, None)

        advertised = {
            feature.key for feature in servicer.GetInfo(None, None).action_features
        }
        action = servicer._compute_action()

        assert advertised == set(action)
        assert advertised == {
            "left.hand.delta_pos.x",
            "left.hand.delta_pos.y",
            "left.hand.delta_pos.z",
            "left.hand.delta_rot.qx",
            "left.hand.delta_rot.qy",
            "left.hand.delta_rot.qz",
            "left.hand.delta_rot.qw",
            "left.gripper.distance",
        }

    def test_cumulative_clutch_resumes_without_a_pose_jump(self, tmp_path):
        command_state = [1]
        servicer = _make_leader(
            tmp_path,
            auto_reference=True,
            cumulative_clutch=True,
            command_state_provider=lambda: command_state[0],
        )
        servicer.Connect(None, None)

        servicer._device.pose_position = np.array([0.2, 0.0, 0.0])
        moving = servicer._compute_action()
        assert _delta_pos(moving)[0] == pytest.approx(0.2, abs=1e-6)

        command_state[0] = 0
        frozen = servicer._compute_action()
        servicer._device.pose_position = np.array([5.0, 0.0, 0.0])
        servicer._device.gripper = 12.0
        frozen_while_repositioning = servicer._compute_action()
        assert _delta_pos(frozen)[0] == pytest.approx(0.2, abs=1e-6)
        assert _delta_pos(frozen_while_repositioning)[0] == pytest.approx(
            0.2, abs=1e-6
        )
        assert frozen_while_repositioning["gripper.distance"] == pytest.approx(12.0)

        command_state[0] = 1
        resumed = servicer._compute_action()
        assert _delta_pos(resumed)[0] == pytest.approx(0.2, abs=1e-6)

        servicer._device.pose_position = np.array([5.1, 0.0, 0.0])
        continued = servicer._compute_action()
        assert _delta_pos(continued)[0] == pytest.approx(0.3, abs=1e-6)

    def test_set_reference_consumes_command_edge_seen_while_waiting(
        self, tmp_path, monkeypatch
    ):
        """A squeeze release during reference settling must not disengage again."""
        command_state = [0]
        servicer = _make_leader(
            tmp_path,
            auto_reference=False,
            cumulative_clutch=True,
            command_state_provider=lambda: command_state[0],
        )
        servicer.Connect(None, None)
        servicer._compute_action()  # prime Command=0

        command_state[0] = 1
        servicer._compute_action()  # operator requests re-engage
        assert servicer._clutched is True

        reference_sample = servicer._read_tracker_sample()
        assert reference_sample is not None

        def settle_reference(_context):
            # The same physical squeeze has returned to its resting Command
            # state while SetReference was collecting stable optical samples.
            command_state[0] = 0
            return reference_sample

        monkeypatch.setattr(servicer, "_await_reference_sample", settle_reference)
        servicer.SetReference(None, None)

        servicer._compute_action()
        assert servicer._clutched is True
        assert servicer._pending_relatch is False
