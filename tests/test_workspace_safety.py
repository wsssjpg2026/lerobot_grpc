"""Workspace safety tests (wayfinder #13).

Two bench findings, both handled in the follower (which owns the URDF):

1. **Auto workspace bubble** — the leader's tracking volume is far larger than
   the ~251 mm SO-101; the follower samples the reachable workspace from the
   URDF joint ranges at startup and clamps |ΔT| to a conservative fraction
   (0.6 × max reach).  The arm saturates at the bubble edge instead of being
   commanded beyond reach, and follows back as soon as the hand returns.
2. **Limit escape** — warm-started DLS is a local search: a flipped elbow
   reaches most bubble targets too, so it never unfolds once folded (bench:
   elbow pinned at -96.8° for 11 s with a reachable intent).  When the warm
   solution is flipped / limit-saturated, the follower re-solves from the
   home posture and takes it when its accuracy is comparable.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_robot_grpc.follower.mujoco_follower_server import (  # noqa: E402
    MuJoCoSO101Servicer,
)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"

HOME_RAD = np.radians(np.array([0.0, -20.0, 60.0, -40.0, 0.0]))
FLIPPED_RAD = np.radians(np.array([-14.0, 95.0, -96.0, 85.0, -10.0]))


def _make_follower(**kwargs) -> MuJoCoSO101Servicer:
    defaults = dict(
        xml_path=str(XML_PATH),
        action_mode="pose_delta",
        render=False,
        workspace_radius_m=0.0,  # unlimited slew — tests talk in offsets
        position_deadband_m=0.0,
        rotation_deadband_rad=0.0,
    )
    defaults.update(kwargs)
    servicer = MuJoCoSO101Servicer(**defaults)
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
    joint_action = servicer._pose_delta_to_joint_action(_delta(dx))
    servicer._target_ctrl = servicer._joint_action_to_ctrl(joint_action)


def _intent_offset(servicer: MuJoCoSO101Servicer) -> np.ndarray:
    return servicer._target_pose[:3, 3] - servicer._t_zero[:3, 3]


# ---------------------------------------------------------------------------
# Auto workspace bubble
# ---------------------------------------------------------------------------


class TestAutoBubble:
    def test_auto_bubble_derived_from_urdf(self):
        servicer = _make_follower()  # workspace_bubble_m=None → auto
        reach = servicer._max_reach_m
        assert 0.45 < reach < 0.65, f"unexpected max reach {reach * 1000:.0f}mm"
        servicer._lock_t_zero_from_fk()  # lock at the home FK → bubble recomputes
        tz = float(np.linalg.norm(servicer._t_zero[:3, 3]))
        assert servicer._workspace_bubble_m == pytest.approx(
            0.6 * (reach - tz), rel=1e-9
        )
        assert 0.05 < servicer._workspace_bubble_m < 0.25, (
            f"bubble {servicer._workspace_bubble_m * 1000:.0f}mm out of range"
        )

    def test_clamp_caps_intent(self):
        servicer = _make_follower()
        _send_delta(servicer, dx=0.300)  # 300 mm — far beyond reach
        bubble = servicer._workspace_bubble_m
        np.testing.assert_allclose(_intent_offset(servicer), [bubble, 0.0, 0.0], atol=1e-9)

    def test_small_delta_passes_through(self):
        servicer = _make_follower()
        _send_delta(servicer, dx=0.050)
        np.testing.assert_allclose(_intent_offset(servicer), [0.050, 0.0, 0.0], atol=1e-9)

    def test_hand_return_recovers_without_hysteresis(self):
        servicer = _make_follower()
        _send_delta(servicer, dx=0.300)  # clamped to the bubble
        _send_delta(servicer, dx=0.040)  # hand comes back → intent follows
        np.testing.assert_allclose(_intent_offset(servicer), [0.040, 0.0, 0.0], atol=1e-9)

    def test_bubble_recomputes_on_relatch(self):
        """An arm re-latched away from home gets a new clearance-based bubble."""
        servicer = _make_follower()
        _send_delta(servicer, dx=0.0)  # lock T_zero at the home FK
        bubble_home = servicer._workspace_bubble_m
        # Extend the arm and settle physics at +150 mm.
        _send_delta(servicer, dx=0.150)
        servicer._data.ctrl[:] = servicer._target_ctrl
        for _ in range(600):
            servicer._mj.mj_step(servicer._model, servicer._data)
        servicer._mj.mj_forward(servicer._model, servicer._data)
        servicer.SetReference(None, None)  # re-latch at the extended pose
        tz = float(np.linalg.norm(servicer._t_zero[:3, 3]))
        assert servicer._workspace_bubble_m == pytest.approx(
            0.6 * (servicer._max_reach_m - tz), rel=1e-9
        )
        assert abs(servicer._workspace_bubble_m - bubble_home) > 0.001, (
            "bubble did not update after the re-latch"
        )

    def test_manual_override(self):
        servicer = _make_follower(workspace_bubble_m=0.100)
        _send_delta(servicer, dx=0.300)
        np.testing.assert_allclose(_intent_offset(servicer), [0.100, 0.0, 0.0], atol=1e-9)

    def test_disabled(self):
        servicer = _make_follower(workspace_bubble_m=0.0)
        _send_delta(servicer, dx=0.300)
        np.testing.assert_allclose(_intent_offset(servicer), [0.300, 0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Limit escape (flipped elbow → home-side re-seed)
# ---------------------------------------------------------------------------


class TestLimitEscape:
    def test_flipped_elbow_escapes_to_home_side(self):
        """The bench's stuck state: warm start flipped, intent reachable — the
        arm must fold back to the home side instead of staying pinned."""
        servicer = _make_follower()
        servicer._last_ik_rad = FLIPPED_RAD.copy()
        joint_action = servicer._pose_delta_to_joint_action(_delta(dx=0.040))
        assert servicer._escaped, "flipped warm start should trigger the escape"
        elbow_deg = math.degrees(float(servicer._last_ik_rad[2]))
        assert elbow_deg > 0.0, f"elbow stayed flipped at {elbow_deg:.1f}°"
        assert servicer.last_reach_err_m < 0.008, (
            f"escaped solve residual {servicer.last_reach_err_m * 1000:.1f}mm too large"
        )
        assert joint_action["elbow_flex.pos"] > 0.0

    def test_no_escape_from_home_side_seed(self):
        servicer = _make_follower()
        servicer._last_ik_rad = HOME_RAD.copy()
        _send_delta(servicer, dx=0.040)
        assert not servicer._escaped
        assert math.degrees(float(servicer._last_ik_rad[2])) > 0.0

    def test_escape_disabled_keeps_warm_solution(self):
        """Pre-fix behavior pinned: without the escape the flipped warm start
        stays flipped even though the target is reachable from home."""
        servicer = _make_follower(workspace_escape=False)
        servicer._last_ik_rad = FLIPPED_RAD.copy()
        servicer._pose_delta_to_joint_action(_delta(dx=0.040))
        assert not servicer._escaped
        assert math.degrees(float(servicer._last_ik_rad[2])) < 0.0, (
            "warm-started solve unfolded without the escape"
        )
