"""Recovery and latch-once tests for MuJoCo SO-101 pose-delta IK.

Δ is the current offset from the latched arm reference (SetReference /
Connect home), not a per-frame increment.  Sending the same offset twice
must not accumulate; sending identity must return the arm home.  The
per-frame step cap (official stack) walks large offsets in over several
frames — the helpers below therefore apply an offset repeatedly until the
commanded joints settle.
"""

from __future__ import annotations

import math
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

_FK_MODEL = None


def _fk_of_deg(q_deg) -> float:
    """EE radius (m) of a body-joint vector in degrees, on a separate model."""
    import mujoco

    global _FK_MODEL
    if _FK_MODEL is None:
        _FK_MODEL = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(_FK_MODEL)
    data.qpos[: len(BODY_JOINTS)] = np.radians(np.asarray(q_deg, dtype=float))
    mujoco.mj_forward(_FK_MODEL, data)
    sid = mujoco.mj_name2id(_FK_MODEL, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    return float(np.linalg.norm(data.site_xpos[sid]))


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
    # The official safety stack (FK consistency, jump reset, frame cap) is
    # pinned in tests/test_pose_delta_law.py — this file drives the raw DLS
    # recovery mechanics through repeated frames.
    servicer = MuJoCoSO101Servicer(
        xml_path=str(XML_PATH),
        action_mode="pose_delta",
        render=False,
    )
    servicer.Connect(None, None)
    return servicer


def _apply(servicer: MuJoCoSO101Servicer, dx: float = 0.0, dy: float = 0.0, n: int = 1):
    action = None
    for _ in range(n):
        action = servicer._pose_delta_to_joint_action(_delta(dx, dy))
    return action


def _apply_base_frame(servicer: MuJoCoSO101Servicer, base_vec, n: int = 1):
    """Apply a BASE-frame displacement intent: fold it through R_ref^T into
    the body frame the leader actually publishes (official composition)."""
    r_ref = servicer._law.arm_reference[:3, :3]
    body = r_ref.T @ np.asarray(base_vec, dtype=float)
    return _apply(servicer, float(body[0]), float(body[1]), n=n)


def _joints(action) -> np.ndarray:
    return np.array([action[f"{j}.pos"] for j in BODY_JOINTS])


class TestLatchOnceAssign:
    def test_same_offset_does_not_crawl(self):
        """Latch-once: once the commanded joints have converged to an offset,
        re-sending the SAME offset must not accumulate further motion."""
        servicer = _make_servicer()
        _apply(servicer, 0.0, n=5)
        _apply(servicer, 0.050, n=40)  # walk to the offset (frame-capped)
        first = _apply(servicer, 0.050)
        second = _apply(servicer, 0.050)
        joint_delta = float(np.abs(_joints(second) - _joints(first)).max())
        assert joint_delta < 1.0, f"same offset crawled joints by {joint_delta:.2f}°"

    def test_identity_returns_joints_home(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0, n=5)
        home_q = {name: HOME_JOINTS_DEG[i] for i, name in enumerate(BODY_JOINTS)}
        _apply(servicer, 0.080, n=30)
        back = _apply(servicer, 0.0, n=30)
        for name in BODY_JOINTS:
            assert abs(back[f"{name}.pos"] - home_q[name]) < 5.0, (
                f"{name} {back[f'{name}.pos']:.1f}° not near home {home_q[name]:.1f}°"
            )


class TestOvershootDoesNotFlipElbow:
    def test_first_unreachable_frame_keeps_home_side_elbow(self):
        """An overstretch intent on the FIRST frame is frame-capped off the
        home posture — the elbow cannot flip configuration in one frame."""
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        action = _apply(servicer, 0.40)
        assert action["elbow_flex.pos"] > 20.0, (
            f"one unreachable frame flipped elbow to {action['elbow_flex.pos']:.1f}°"
        )


class TestOverstretchThenHome:
    def test_moderate_overstretch_walks_outward_then_returns(self):
        """A 0.25 m overstretch stays inside the official FK-consistency band
        (solution FK within 0.3 m of the target): repeated frames keep
        walking outward (the intent is never rewritten); returning to
        identity walks the joints home with the elbow on the home side."""
        servicer = _make_servicer()
        _apply(servicer, 0.0, n=5)
        r0 = _fk_of_deg(_joints(_apply(servicer, 0.0)))
        r_mid = _fk_of_deg(_joints(_apply(servicer, 0.25, n=4)))
        r_far = _fk_of_deg(_joints(_apply(servicer, 0.25, n=30)))
        assert r_far > r_mid > r0, "repeated overstretch frames stopped walking"
        back = _apply(servicer, 0.0, n=30)
        assert _fk_of_deg(_joints(back)) == pytest.approx(r0, abs=0.008)
        assert back["elbow_flex.pos"] > 20.0, (
            f"elbow stayed flipped ({back['elbow_flex.pos']:.1f}°)"
        )

    def test_deep_overstretch_rejected_by_fk_consistency(self):
        """A 0.40 m overstretch puts the closest reachable solution >0.3 m
        from the target — the official check REJECTS the solve and the
        commanded joints freeze at the last accepted action (no walk)."""
        servicer = _make_servicer()
        _apply(servicer, 0.0, n=5)
        settled = _apply(servicer, 0.0, n=2)
        r_settled = _fk_of_deg(_joints(settled))
        frozen = _apply(servicer, 0.40, n=8)
        assert _fk_of_deg(_joints(frozen)) == pytest.approx(r_settled, abs=0.002)


class TestLateralTranslation:
    def test_plus_y_moves_shoulder_pan(self):
        """A base-frame +Y displacement drives pan around the base.  The
        converged pan is smaller than the old translation-only-solve value:
        with the fixed official rotation weight the 5-DOF solver trades pan
        against the soft orientation task, so the assertion pins the
        converged behavior (~3.8 deg for 40 mm)."""
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        action = _apply_base_frame(servicer, [0.0, 0.040, 0.0], n=8)
        assert abs(action["shoulder_pan.pos"]) > 3.0

    def test_pan_limit_reverses_on_smaller_offset(self):
        servicer = _make_servicer()
        _apply(servicer, 0.0)
        at_limit = _apply_base_frame(servicer, [0.0, 0.35, 0.0], n=25)
        pan0 = at_limit["shoulder_pan.pos"]
        assert abs(pan0) > 30.0, f"pan never left home ({pan0:.1f}°)"
        reversed_ = _apply_base_frame(servicer, [0.0, 0.20, 0.0], n=6)
        pan1 = reversed_["shoulder_pan.pos"]
        assert abs(pan1) < abs(pan0) - 0.5, (
            f"pan stuck at limit: {pan0:.1f}° → {pan1:.1f}°"
        )


class TestDLSSolveInfo:
    def test_solve_reports_achieved_pose(self):
        servicer = _make_servicer()
        solver = servicer._law.ik_solver
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
