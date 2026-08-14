"""Recovery and latch-once tests for MuJoCo SO-101 pose-delta IK.

ΔT is the current offset from T_zero (SetReference / Connect home), not a
per-frame increment.  Sending the same offset twice must hold; sending
identity must return the arm to home.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot_robot_grpc.follower.mujoco_follower_server import (
    BODY_JOINTS,
    HOME_JOINTS_DEG,
    MuJoCoSO101Servicer,
)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"

pytest.importorskip("mujoco")


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


def _make_servicer() -> MuJoCoSO101Servicer:
    # The #13 safety features (workspace bubble clamp + limit escape) are
    # pinned separately in tests/test_workspace_safety.py — disable them here
    # so this file keeps testing the raw DLS recovery mechanics (intent is not
    # rewritten by overshoot, elbow side locking, pan-limit reversal).
    servicer = MuJoCoSO101Servicer(
        xml_path=str(XML_PATH),
        action_mode="pose_delta",
        render=False,
        position_deadband_m=0.0,
        rotation_deadband_rad=0.0,
        workspace_radius_m=0.0,  # unlimited slew — tests talk in offsets
        workspace_bubble_m=0.0,  # bubble off
        workspace_escape=False,  # escape off
    )
    servicer.Connect(None, None)
    return servicer


def _apply(servicer: MuJoCoSO101Servicer, dx: float = 0.0, dy: float = 0.0, n: int = 1):
    action = None
    for _ in range(n):
        action = servicer._pose_delta_to_joint_action(_delta(dx, dy))
    return action


class TestLatchOnceAssign:
    def test_same_offset_does_not_crawl(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        first = _apply(servicer, 0.050)
        x0 = float(servicer._target_pose[0, 3])
        second = _apply(servicer, 0.050)
        x1 = float(servicer._target_pose[0, 3])
        assert abs(x1 - x0) < 1e-9
        joint_delta = max(
            abs(second[f"{j}.pos"] - first[f"{j}.pos"]) for j in BODY_JOINTS
        )
        assert joint_delta < 1.0, f"same offset crawled joints by {joint_delta:.2f}°"

    def test_identity_returns_ee_and_joints_home(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        home_q = {name: HOME_JOINTS_DEG[i] for i, name in enumerate(BODY_JOINTS)}
        _apply(servicer, 0.080, n=5)
        back = _apply(servicer, 0.0, n=8)
        home_err = float(
            np.linalg.norm(servicer._target_pose[:3, 3] - servicer._t_zero[:3, 3])
        )
        assert home_err < 1e-9
        assert servicer.last_home_err_m < 1e-9
        ee_err = float(
            np.linalg.norm(servicer._last_achieved_pos - servicer._t_zero[:3, 3])
        )
        assert ee_err < 0.008, f"EE did not return home ({ee_err * 1000:.1f} mm)"
        for name in BODY_JOINTS:
            assert abs(back[f"{name}.pos"] - home_q[name]) < 5.0, (
                f"{name} {back[f'{name}.pos']:.1f}° not near home {home_q[name]:.1f}°"
            )


class TestOvershootDoesNotFlipElbow:
    def test_first_unreachable_frame_keeps_home_side_elbow(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        action = _apply(servicer, 0.40)
        assert action["elbow_flex.pos"] > 20.0, (
            f"one unreachable frame flipped elbow to {action['elbow_flex.pos']:.1f}°"
        )


class TestOverstretchThenHome:
    def test_intent_stays_at_commanded_offset(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        _apply(servicer, 0.40, n=4)
        offset = servicer._target_pose[:3, 3] - servicer._t_zero[:3, 3]
        np.testing.assert_allclose(offset, [0.40, 0.0, 0.0], atol=1e-9)
        assert servicer.last_overshoot or servicer.last_reach_err_m > 0.008

    def test_return_to_zero_after_overstretch(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        _apply(servicer, 0.40, n=4)
        back = _apply(servicer, 0.0, n=12)
        ee_err = float(
            np.linalg.norm(servicer._last_achieved_pos - servicer._t_zero[:3, 3])
        )
        assert ee_err < 0.008, f"overstretch return EE {ee_err * 1000:.1f} mm from home"
        assert back["elbow_flex.pos"] > 20.0, (
            f"elbow stayed flipped ({back['elbow_flex.pos']:.1f}°)"
        )


class TestLateralTranslation:
    def test_plus_y_moves_shoulder_pan(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        action = _apply(servicer, dy=0.040, n=3)
        assert abs(action["shoulder_pan.pos"]) > 5.0

    def test_pan_limit_reverses_on_smaller_offset(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        at_limit = _apply(servicer, dy=0.35, n=8)
        pan0 = at_limit["shoulder_pan.pos"]
        assert abs(pan0) > 40.0, f"pan never left home ({pan0:.1f}°)"
        reversed_ = _apply(servicer, dy=0.20, n=4)
        pan1 = reversed_["shoulder_pan.pos"]
        assert abs(pan1) < abs(pan0) - 0.5, (
            f"pan stuck at limit: {pan0:.1f}° → {pan1:.1f}°"
        )


class TestDLSSolveInfo:
    def test_solve_reports_achieved_pose(self):
        servicer = _make_servicer()
        solver = servicer._ik_solver
        seed = servicer._data.qpos.copy()
        servicer._mj.mj_forward(servicer._model, servicer._data)
        sid = solver._site_id
        pos = servicer._data.site_xpos[sid].copy()
        rot = servicer._data.site_xmat[sid].reshape(3, 3).copy()
        result = solver.solve(pos + np.array([0.01, 0.0, 0.0]), rot, seed, rot_weight=0.01)
        assert result.qpos.shape == (5,)
        assert np.isfinite(result.pos_err)
        assert result.achieved_pos.shape == (3,)
        assert result.achieved_rot.shape == (3, 3)
        assert result.manipulability >= 0.0
