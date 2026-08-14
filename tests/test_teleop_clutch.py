"""Client-side clutch loop tests (wayfinder #12).

Pins two bench findings and the official grip decision:

1. **Stale-action discard** — on the engage edge the client must throw away
   the action fetched before ``relatch()`` (frozen at the pre-hold offset)
   and send a freshly fetched one, otherwise the freshly re-latched follower
   chases the old offset direction for one frame (#11 bench: 42.9/29.1 mm
   frozen offsets, ~1 cm twitch capped by the follower's cmd slew).
2. **Official grip semantics** — in ``auto`` mode the action stream keeps
   flowing while holding: the leader freezes the arm offset and the gripper
   stays live.  The follower must hold its joints under the frozen offset
   while still updating the gripper.  The ``keyboard`` fallback freezes
   everything by not sending.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_robot_grpc.follower.mujoco_follower_server import (  # noqa: E402
    BODY_JOINTS,
    MuJoCoSO101Servicer,
)
from lerobot_robot_grpc.protos import device_pb2  # noqa: E402
from lerobot_robot_grpc.teleop_clutch import (  # noqa: E402
    auto_clutch_step,
    keyboard_clutch_step,
)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"


def _stale() -> dict[str, float]:
    return {"hand.delta_pos.x": 0.0429, "gripper.distance": 17.0}


def _fresh() -> dict[str, float]:
    return {"hand.delta_pos.x": 0.0, "gripper.distance": 20.0}


# ---------------------------------------------------------------------------
# Pure client-side clutch helpers
# ---------------------------------------------------------------------------


class TestAutoClutchStep:
    def test_engage_edge_relatches_then_refetches(self):
        """#12: the pre-relatch (frozen) action must never be sent."""
        calls: list[str] = []
        engaged, action, should_send = auto_clutch_step(
            status=device_pb2.DeviceStatus.COLLECTION,
            engaged=False,
            raw_action=_stale(),
            fetch_action=_fresh,
            relatch=lambda: calls.append("relatch"),
        )
        assert engaged is True
        assert should_send is True
        assert calls == ["relatch"]
        assert action == _fresh()  # stale discarded, fresh sent

    def test_hold_keeps_sending(self):
        """Official grip semantics: IDLE still sends — the leader freezes the
        arm offset itself; the client just transports the gripper."""
        engaged, action, should_send = auto_clutch_step(
            status=device_pb2.DeviceStatus.IDLE,
            engaged=True,
            raw_action=_stale(),
            fetch_action=_fresh,
            relatch=lambda: None,
        )
        assert engaged is False
        assert should_send is True
        assert action == _stale()  # frozen arm offset + live gripper

    def test_follow_passes_action_through(self):
        engaged, action, should_send = auto_clutch_step(
            status=device_pb2.DeviceStatus.COLLECTION,
            engaged=True,
            raw_action=_stale(),
            fetch_action=_fresh,
            relatch=lambda: None,
        )
        assert engaged is True
        assert should_send is True
        assert action == _stale()

    def test_fatal_never_sends(self):
        engaged, action, should_send = auto_clutch_step(
            status=device_pb2.DeviceStatus.FATAL,
            engaged=False,
            raw_action=_stale(),
            fetch_action=_fresh,
            relatch=lambda: None,
        )
        assert engaged is False
        assert should_send is False


class TestKeyboardClutchStep:
    def test_toggle_off_stops_sending(self):
        engaged, action, should_send = keyboard_clutch_step(
            engaged=True,
            key_toggled=True,
            raw_action=_stale(),
            fetch_action=_fresh,
            relatch=lambda: None,
        )
        assert engaged is False
        assert should_send is False  # fallback: everything freezes
        assert action == _stale()

    def test_toggle_on_relatches_then_refetches(self):
        calls: list[str] = []
        engaged, action, should_send = keyboard_clutch_step(
            engaged=False,
            key_toggled=True,
            raw_action=_stale(),
            fetch_action=_fresh,
            relatch=lambda: calls.append("relatch"),
        )
        assert engaged is True
        assert should_send is True
        assert calls == ["relatch"]
        assert action == _fresh()  # #12 discipline applies here too

    def test_no_toggle_passes_through(self):
        engaged, action, should_send = keyboard_clutch_step(
            engaged=True,
            key_toggled=False,
            raw_action=_stale(),
            fetch_action=_fresh,
            relatch=lambda: None,
        )
        assert engaged is True
        assert should_send is True
        assert action == _stale()


# ---------------------------------------------------------------------------
# Follower: frozen offset keeps arriving during hold (official grip semantics)
# ---------------------------------------------------------------------------


def _make_follower() -> MuJoCoSO101Servicer:
    servicer = MuJoCoSO101Servicer(
        xml_path=str(XML_PATH),
        action_mode="pose_delta",
        render=False,
        # Production deadbands (1.0 mm / 0.005 rad): a repeated frozen offset
        # must hit the deadband path so micro IK re-seeds cannot nudge the arm.
        position_deadband_m=0.001,
        rotation_deadband_rad=0.005,
        workspace_radius_m=0.0,  # unlimited slew — tests talk in offsets
    )
    servicer.Connect(None, None)
    return servicer


def _delta(dx: float, gripper: float) -> dict[str, float]:
    return {
        "hand.delta_pos.x": dx,
        "hand.delta_pos.y": 0.0,
        "hand.delta_pos.z": 0.0,
        "hand.delta_rot.qx": 0.0,
        "hand.delta_rot.qy": 0.0,
        "hand.delta_rot.qz": 0.0,
        "hand.delta_rot.qw": 1.0,
        "gripper.distance": gripper,
    }


def _send_delta(servicer: MuJoCoSO101Servicer, dx: float, gripper: float) -> None:
    joint_action = servicer._pose_delta_to_joint_action(_delta(dx, gripper))
    servicer._target_ctrl = servicer._joint_action_to_ctrl(joint_action)


def _settle_physics(servicer: MuJoCoSO101Servicer, steps: int = 600) -> None:
    servicer._data.ctrl[:] = servicer._target_ctrl
    for _ in range(steps):
        servicer._mj.mj_step(servicer._model, servicer._data)
    servicer._mj.mj_forward(servicer._model, servicer._data)


class TestFollowerHoldWithLiveGripper:
    def test_frozen_offset_holds_arm_while_gripper_moves(self):
        """Hold = the SAME frozen offset keeps arriving; the arm must not move
        but the gripper must follow the hand (official PikaAnyArm clutch)."""
        servicer = _make_follower()
        _send_delta(servicer, 0.040, gripper=20.0)
        _settle_physics(servicer)
        qpos_hold = servicer._data.qpos.copy()
        body_ctrl_hold = servicer._target_ctrl[: len(BODY_JOINTS)].copy()
        grip_ctrl_hold = float(servicer._target_ctrl[-1])

        # Frozen offset keeps arriving; only the gripper changes.
        _send_delta(servicer, 0.040, gripper=50.0)
        assert abs(float(servicer._target_ctrl[-1]) - grip_ctrl_hold) > 1e-3
        np.testing.assert_allclose(
            servicer._target_ctrl[: len(BODY_JOINTS)], body_ctrl_hold, atol=1e-9
        )

        _settle_physics(servicer)
        drift_deg = max(
            abs(math.degrees(float(servicer._data.qpos[i] - qpos_hold[i])))
            for i in range(len(BODY_JOINTS))
        )
        assert drift_deg < 0.2, f"arm drifted {drift_deg:.3f}° under a frozen offset"
        assert abs(float(servicer._data.qpos[len(BODY_JOINTS)] - qpos_hold[len(BODY_JOINTS)])) > 0.01, (
            "gripper did not move during hold"
        )
