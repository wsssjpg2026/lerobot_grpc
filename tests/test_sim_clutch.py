"""Sim clutch tests (wayfinder #10) — double-click follow/hold semantics.

Two halves of the gate, tested without gRPC or real hardware:

1. **Leader** (`PikaSenseServicer` with a fake device): a command-state edge
   toggles follow/hold; while disengaged the published offset is frozen (a
   big tracker jump must not change the action); the re-engage edge stays
   frozen until ``SetReference`` lands, after which ΔT ≈ 0.

2. **Follower** (`MuJoCoSO101Servicer` + MuJoCo): ``SetReference`` re-locks
   ``T_arm_ref`` at the current FK — after re-engage the solve maps onto the
   stop pose, *not* the Connect home; nothing moves at the re-latch itself,
   and a disengaged arm holds its joints when no action arrives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot_robot_grpc.leader.pika_sense_leader_server import (
    PikaSenseServicer,
    command_state_edge,
)

pika_sense = pytest.importorskip("pika.sense")

from lerobot_robot_grpc.follower.mujoco_follower_server import (  # noqa: E402
    BODY_JOINTS,
    MuJoCoSO101Servicer,
)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"

pytest.importorskip("mujoco")

# ---------------------------------------------------------------------------
# Fake Pika hardware
# ---------------------------------------------------------------------------


class _FakePose:
    def __init__(self, position, quat, timestamp):
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(quat, dtype=float)
        self.timestamp = float(timestamp)


class _FakeSense:
    """Minimal stand-in for ``pika.sense.Sense`` — pose + gripper only."""

    def __init__(self):
        self.pose_position = np.zeros(3)
        self.pose_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper = 30.0
        self.timestamp = 0.0

    def get_pose(self, device):
        self.timestamp += 0.01
        return _FakePose(self.pose_position, self.pose_quat, self.timestamp)

    def get_gripper_distance(self):
        return self.gripper


def _make_leader(tmp_path, command_provider):
    servicer = PikaSenseServicer(
        port="/dev/null",
        R_lh2base=np.eye(3),
        calibration_dir=str(tmp_path),
        command_state_provider=command_provider,
        tracker_health_enabled=False,
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


# ---------------------------------------------------------------------------
# Pure edge helper
# ---------------------------------------------------------------------------


class TestCommandStateEdge:
    def test_none_is_never_an_edge(self):
        assert not command_state_edge(None, 0)
        assert not command_state_edge(1, None)
        assert not command_state_edge(None, None)

    def test_no_change_is_not_an_edge(self):
        assert not command_state_edge(1, 1)
        assert not command_state_edge(0, 0)

    def test_rising_and_falling_are_edges(self):
        assert command_state_edge(1, 0)
        assert command_state_edge(0, 1)


# ---------------------------------------------------------------------------
# Leader clutch state machine (fake hardware)
# ---------------------------------------------------------------------------


class TestLeaderClutch:
    def test_disengaged_before_first_set_reference(self, tmp_path):
        """Before the Enter alignment nothing may be published, even when the
        tracker moves — teleop starts holding, not following."""
        state = {"v": 1}
        servicer = _make_leader(tmp_path, lambda: state["v"])
        servicer._compute_action()  # first sample: latch prev command state
        servicer._device.pose_position = np.array([0.6, 0.0, 0.0])
        action = servicer._compute_action()
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)

    def test_set_reference_engages_and_follows(self, tmp_path):
        state = {"v": 1}
        servicer = _make_leader(tmp_path, lambda: state["v"])
        servicer._compute_action()
        servicer._device.pose_position = np.array([0.5, 0.0, 0.0])
        servicer.SetReference(None, None)
        assert servicer._clutched
        servicer._compute_action()
        servicer._device.pose_position = np.array([0.6, 0.0, 0.0])  # +100 mm
        action = servicer._compute_action()
        np.testing.assert_allclose(
            _delta_pos(action), [0.1, 0.0, 0.0], atol=1e-9
        )

    def test_disengage_edge_freezes_publish_across_tracker_jump(self, tmp_path):
        """Clutch off: the action stays frozen even if the tracker teleports —
        the follower would hold its joints (ticket: 断开期间大跳,关节不变)."""
        state = {"v": 1}
        servicer = _make_leader(tmp_path, lambda: state["v"])
        servicer._compute_action()
        servicer.SetReference(None, None)
        servicer._compute_action()
        servicer._device.pose_position = np.array([0.6, 0.0, 0.0])
        action = servicer._compute_action()
        frozen = _delta_pos(action)

        state["v"] = 0  # double-click: disengage
        action = servicer._compute_action()
        assert not servicer._clutched

        servicer._device.pose_position = np.array([1.0, 0.0, 0.0])  # +0.4 m jump
        action = servicer._compute_action()
        np.testing.assert_allclose(_delta_pos(action), frozen, atol=1e-9)

    def test_reengage_stays_frozen_until_set_reference(self, tmp_path):
        """The engage edge alone must not unfreeze: the client sequences
        follower.SetReference → leader.SetReference before fresh offsets
        flow, so the arm never sees zero against a stale T_zero."""
        state = {"v": 1}
        servicer = _make_leader(tmp_path, lambda: state["v"])
        servicer._compute_action()
        servicer.SetReference(None, None)
        servicer._compute_action()
        servicer._device.pose_position = np.array([0.6, 0.0, 0.0])
        action = servicer._compute_action()
        frozen = _delta_pos(action)

        state["v"] = 0
        servicer._compute_action()
        state["v"] = 1  # double-click: re-engage
        action = servicer._compute_action()
        assert servicer._clutched
        assert servicer._pending_relatch
        np.testing.assert_allclose(_delta_pos(action), frozen, atol=1e-9)

        servicer.SetReference(None, None)  # client sequence lands
        assert not servicer._pending_relatch
        action = servicer._compute_action()
        np.testing.assert_allclose(_delta_pos(action), [0.0, 0.0, 0.0], atol=1e-9)

        # Following continues from the new latch (current hand = current arm).
        servicer._device.pose_position = np.array([0.65, 0.0, 0.0])  # +50 mm
        action = servicer._compute_action()
        np.testing.assert_allclose(
            _delta_pos(action), [0.05, 0.0, 0.0], atol=1e-9
        )


# ---------------------------------------------------------------------------
# Follower re-latch (MuJoCo, no hardware)
# ---------------------------------------------------------------------------


def _make_follower() -> MuJoCoSO101Servicer:
    servicer = MuJoCoSO101Servicer(
        xml_path=str(XML_PATH),
        action_mode="pose_delta",
        render=False,
    )
    servicer.Connect(None, None)
    return servicer


def _delta(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict[str, float]:
    return {
        "hand.delta_pos.x": dx,
        "hand.delta_pos.y": dy,
        "hand.delta_pos.z": dz,
        "hand.delta_rot.qx": 0.0,
        "hand.delta_rot.qy": 0.0,
        "hand.delta_rot.qz": 0.0,
        "hand.delta_rot.qw": 1.0,
        "gripper.distance": 0.0,
    }


def _send_delta(servicer: MuJoCoSO101Servicer, dx: float = 0.0) -> None:
    """Mimic SendAction: compute the joint action and store the ctrl target."""
    joint_action = servicer._pose_delta_to_joint_action(_delta(dx))
    servicer._target_ctrl = servicer._joint_action_to_ctrl(joint_action)


def _walk_delta(servicer: MuJoCoSO101Servicer, dx: float, frames: int = 30) -> None:
    """Send the same delta repeatedly so the per-frame step cap can walk the
    commanded joints all the way to the target (single-frame large offsets
    are capped at 6.7 deg by design — the official interpolation role)."""
    for _ in range(frames):
        _send_delta(servicer, dx)


def _settle_physics(servicer: MuJoCoSO101Servicer, steps: int = 600) -> None:
    """Drive the PD toward ``_target_ctrl`` like GetObservation's EMA ramp."""
    servicer._data.ctrl[:] = servicer._target_ctrl
    for _ in range(steps):
        servicer._mj.mj_step(servicer._model, servicer._data)
    servicer._mj.mj_forward(servicer._model, servicer._data)


def _fk_pos(servicer: MuJoCoSO101Servicer) -> np.ndarray:
    sid = servicer._law.ik_solver._site_id
    servicer._mj.mj_forward(servicer._model, servicer._data)
    return servicer._data.site_xpos[sid].copy()


class TestFollowerRelatch:
    def test_set_reference_relocks_arm_ref_at_stop_pose(self):
        """Clutch pass criterion: re-engage must map Δ=0 onto the *current*
        pose — the solve never crawls back toward the Connect home."""
        servicer = _make_follower()
        home_fk = _fk_pos(servicer)

        # Teleop: follow +50 mm (walked — the per-frame cap limits each
        # step), let physics settle at the stop pose.
        _walk_delta(servicer, 0.050)
        _settle_physics(servicer)
        stop_fk = _fk_pos(servicer)
        assert np.linalg.norm(stop_fk - home_fk) > 0.03, "arm did not move"

        qpos_before = servicer._data.qpos.copy()
        servicer.SetReference(None, None)
        # Re-latch must not move anything by itself.
        np.testing.assert_allclose(servicer._data.qpos, qpos_before, atol=1e-12)
        np.testing.assert_allclose(
            servicer._law.arm_reference[:3, 3], stop_fk, atol=1e-6
        )

        # Δ=0 after re-engage → the solve holds the stop pose (commanded
        # joints ≈ settled joints), not the Connect home.
        joint_action = servicer._pose_delta_to_joint_action(_delta(0.0))
        commanded = np.array([joint_action[f"{j}.pos"] for j in BODY_JOINTS])
        settled_deg = np.degrees(servicer._data.qpos[: len(BODY_JOINTS)])
        worst = float(np.abs(commanded - settled_deg).max())
        assert worst < 1.5, f"Δ=0 solve moved {worst:.2f}° off the stop pose"

    def test_disengaged_arm_holds_without_new_actions(self):
        """Clutch off = client stops sending; the follower must hold joints."""
        servicer = _make_follower()
        _walk_delta(servicer, 0.040)
        _settle_physics(servicer)
        qpos_hold = servicer._data.qpos.copy()
        # No SendAction at all (disengaged): more physics steps, then compare.
        for _ in range(600):
            servicer._mj.mj_step(servicer._model, servicer._data)
        servicer._mj.mj_forward(servicer._model, servicer._data)
        drift_deg = max(
            abs(np.degrees(float(servicer._data.qpos[i] - qpos_hold[i])))
            for i in range(len(BODY_JOINTS))
        )
        assert drift_deg < 0.2, f"disengaged arm drifted {drift_deg:.3f}°"

    def test_following_continues_from_relatched_pose(self):
        """After re-engage, a fresh +30 mm body-frame offset moves the solve
        from the stop pose (composition base stays the relatched pose)."""
        servicer = _make_follower()
        _walk_delta(servicer, 0.040)
        _settle_physics(servicer)
        stop_fk = _fk_pos(servicer)
        servicer.SetReference(None, None)
        zero = servicer._pose_delta_to_joint_action(_delta(0.0))
        moved = servicer._pose_delta_to_joint_action(_delta(0.030))
        worst = max(
            abs(moved[f"{j}.pos"] - zero[f"{j}.pos"]) for j in BODY_JOINTS
        )
        assert worst > 0.5, "a fresh 30 mm offset did not move the solve"
        np.testing.assert_allclose(
            servicer._law.arm_reference[:3, 3], stop_fk, atol=1e-6
        )
