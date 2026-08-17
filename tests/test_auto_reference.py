"""auto_reference mode tests — clients without an alignment step.

``PikaSenseServicer(auto_reference=True)`` latches ``T_begin`` and engages on
Connect (with a lazy fallback on the first action when the solver had not
converged by then), so a client that never calls ``SetReference`` — e.g.
``lerobot-teleoperate`` — still gets live deltas instead of zeros forever.
Default mode keeps the #10 contract: the session starts disengaged and
``SetReference`` is what engages teleop.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot_robot_grpc.leader.pika_sense_leader_server import PikaSenseServicer

pika_sense = pytest.importorskip("pika.sense")


class _FakePose:
    def __init__(self, position, quat):
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(quat, dtype=float)


class _FakeSense:
    """Stand-in for ``pika.sense.Sense`` covering the Connect path too."""

    def __init__(self):
        self.pose_position = np.zeros(3)
        self.pose_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper = 30.0
        self.devices = ["FAKE"]

    def connect(self):
        pass

    def get_vive_tracker(self):
        pass

    def get_tracker_devices(self):
        return self.devices

    def get_pose(self, device):
        return _FakePose(self.pose_position, self.pose_quat)

    def get_gripper_distance(self):
        return self.gripper


def _make_leader(tmp_path, auto_reference):
    servicer = PikaSenseServicer(
        port="/dev/null",
        R_lh2base=np.eye(3),
        calibration_dir=str(tmp_path),
        command_state_provider=lambda: 1,
        ema_alpha=1.0,
        ema_alpha_fast=1.0,
        dead_zone_mm=0.0,
        dead_zone_deg=0.0,
        max_delta_mm=1000.0,
        home_capture_mm=0.0,
        jump_mm=10000.0,
        auto_reference=auto_reference,
    )
    servicer._device = _FakeSense()
    servicer._tracker_device = "FAKE"
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
        assert _delta_pos(action)[0] == pytest.approx(0.6 * 0.45, abs=1e-6)

    def test_latch_frame_publishes_zero(self, tmp_path):
        """The frame at the latch itself publishes ~zero: the Connect pose is
        the delta origin, so the first command cannot snap the arm."""
        servicer = _make_leader(tmp_path, auto_reference=True)
        servicer.Connect(None, None)
        action = servicer._compute_action()
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)

    def test_lazy_latch_on_first_action(self, tmp_path):
        """Connect without a converged solver latches at the first tracker
        sample instead — the session still comes up engaged."""
        servicer = _make_leader(tmp_path, auto_reference=True)
        # No Connect: T_begin unset, as if the solver had not converged yet.
        action = servicer._compute_action()
        assert servicer._t_begin_pos is not None
        assert servicer._clutched is True
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)

        servicer._device.pose_position = np.array([0.0, 0.6, 0.0])
        action = servicer._compute_action()
        assert _delta_pos(action)[1] == pytest.approx(0.6 * 0.45, abs=1e-6)


class TestLateTrackerDiscovery:
    def test_tracker_discovered_late_heals_pose_channel(self, tmp_path):
        """Cold-start race (05 号议题实测): the tracker registers with
        pysurvive after Connect's 10s deadline, so _tracker_device stays
        None and the pose channel used to be dead for the process lifetime.
        _read_tracker_pose now re-scans lazily until the tracker appears."""
        servicer = _make_leader(tmp_path, auto_reference=True)
        servicer._tracker_device = None            # Connect gave up
        servicer._device.devices = ["LH0", "LH1"]  # only lighthouses so far

        action = servicer._compute_action()        # pose channel dead: zeros
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)
        assert servicer._tracker_device is None

        servicer._device.devices = ["LH0", "LH1", "T20"]  # tracker lands late
        action = servicer._compute_action()        # lazy scan + auto-latch
        assert servicer._tracker_device == "T20"
        assert servicer._t_begin_pos is not None
        assert servicer._clutched is True
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)

        servicer._device.pose_position = np.array([0.0, 0.0, 0.6])
        action = servicer._compute_action()
        assert _delta_pos(action)[2] == pytest.approx(0.6 * 0.45, abs=1e-6)


class TestDefaultContract:
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
        assert _delta_pos(action)[0] == pytest.approx(0.6 * 0.45, abs=1e-6)
