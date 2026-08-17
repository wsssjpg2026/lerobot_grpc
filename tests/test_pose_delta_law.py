"""Seam tests for the shared PoseDeltaLaw (wayfinder pika-sense-real #03).

These prove the core design decision of #03: the entire assign + DLS +
workspace-safety + slew + hold pipeline lives in PoseDeltaLaw behind a tiny
interface, drivable with NO servicer and NO backend -- just a MuJoCo model and a
joint vector.  This is exactly how the real Feetech servicer (#04) will drive it:
read Present_Position -> qpos -> law.solve -> send_action.  The MuJoCo servicer
is merely the first (sim) adapter; these tests exercise the law directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from lerobot_robot_grpc.follower.pose_delta_law import (  # noqa: E402
    BaseSafetySphere,
    ClearanceBubble,
    FixedBubble,
    JointSolution,
    NoBubble,
    PoseDeltaLaw,
    WorkspacePolicy,
    workspace_policy_from_legacy,
)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"

BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
HOME_DEG = (0.0, -20.0, 60.0, -40.0, 0.0)


def _model():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


def _home_qpos() -> np.ndarray:
    q = np.zeros(6)
    q[:5] = np.radians(np.array(HOME_DEG, dtype=float))
    return q


def _delta(dx=0.0, dy=0.0, dz=0.0, grip=0.0) -> dict[str, float]:
    return {
        "hand.delta_pos.x": dx,
        "hand.delta_pos.y": dy,
        "hand.delta_pos.z": dz,
        "hand.delta_rot.qx": 0.0,
        "hand.delta_rot.qy": 0.0,
        "hand.delta_rot.qz": 0.0,
        "hand.delta_rot.qw": 1.0,
        "gripper.distance": grip,
    }


def _law(
    workspace_policy: WorkspacePolicy = ClearanceBubble(),
    residual_hold_m: float | None = None,
    max_reach_override_m: float | None = None,
) -> PoseDeltaLaw:
    """A law with no slew / no deadband so tests speak in raw offsets."""
    return PoseDeltaLaw(
        _model(),
        site_name="gripperframe",
        body_dofs=[0, 1, 2, 3, 4],
        body_joint_names=BODY,
        home_joints_deg=HOME_DEG,
        workspace_policy=workspace_policy,
        workspace_radius_m=0.0,
        position_deadband_m=0.0,
        rotation_deadband_rad=0.0,
        residual_hold_m=residual_hold_m,
        max_reach_override_m=max_reach_override_m,
    )


# ---------------------------------------------------------------------------
# The interface is two methods + a dataclass
# ---------------------------------------------------------------------------


class TestInterface:
    def test_two_methods_plus_solution(self):
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)
        sol = law.solve(_delta(0.03), qpos)
        assert isinstance(sol, JointSolution)
        # joint_action is lerobot-normalised -- exactly what a backend writes.
        assert set(sol.joint_action) == {
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        }
        assert sol.joint_action["gripper.pos"] == 0.0

    def test_law_needs_no_servicer(self):
        """The seam: a bare model + a qpos vector drive the whole pipeline."""
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)          # SetReference equivalent
        sol = law.solve(_delta(0.04), qpos)  # SendAction equivalent
        assert sol.pos_err_m < 0.01       # a 40 mm offset is reachable
        assert not sol.held and not sol.stale


# ---------------------------------------------------------------------------
# WorkspacePolicy is the pluggable safety seam
# ---------------------------------------------------------------------------


class TestWorkspacePolicy:
    def test_clearance_clamps_relative_to_t_zero(self):
        pol = ClearanceBubble(ratio=0.5, floor_m=0.0)
        tz = np.array([0.3, 0.0, 0.0])
        max_reach = 0.5
        # clearance = 0.5 - 0.3 = 0.2 -> radius 0.1; a 0.3 m delta clamps to 0.1
        raw = tz + np.array([0.3, 0.0, 0.0])
        out = pol.clamp(raw, tz, max_reach)
        np.testing.assert_allclose(out, tz + np.array([0.1, 0.0, 0.0]), atol=1e-9)

    def test_base_sphere_clamps_absolute_intent(self):
        pol = BaseSafetySphere(ratio=0.72)
        max_reach = 0.5
        # intent at 0.6 m beyond the 0.36 m sphere -> pulled back to the sphere
        raw = np.array([0.6, 0.0, 0.0])
        out = pol.clamp(raw, np.zeros(3), max_reach)
        np.testing.assert_allclose(np.linalg.norm(out), 0.36, atol=1e-9)

    def test_no_bubble_passes_through(self):
        pol = NoBubble()
        raw = np.array([9.0, 9.0, 9.0])
        out = pol.clamp(raw, np.zeros(3), 0.5)
        assert out is raw

    def test_legacy_builder_maps_three_modes(self):
        assert isinstance(workspace_policy_from_legacy(None), ClearanceBubble)
        assert isinstance(workspace_policy_from_legacy(0.0), NoBubble)
        assert isinstance(workspace_policy_from_legacy(0.123), FixedBubble)
        assert workspace_policy_from_legacy(0.123).effective_radius(None, 0.5) == 0.123

    def test_policies_are_workspace_policies(self):
        for p in (NoBubble(), ClearanceBubble(), FixedBubble(0.1), BaseSafetySphere()):
            assert isinstance(p, WorkspacePolicy)

    def test_base_sphere_rejects_full_extension_through_law(self):
        """A base sphere sized to 72% of reach must keep the intent inside it."""
        law = _law(workspace_policy=BaseSafetySphere(0.72))
        qpos = _home_qpos()
        law.lock_reference(qpos)
        law.solve(_delta(dx=2.0), qpos)  # huge outward push
        intent = law.target_pose[:3, 3]
        radius = 0.72 * law.max_reach_m
        assert float(np.linalg.norm(intent)) <= radius + 1e-9


# ---------------------------------------------------------------------------
# Holds are law-internal and parameterised
# ---------------------------------------------------------------------------


class TestHolds:
    def test_stale_flag_holds_last_action(self):
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)
        fresh = law.solve(_delta(0.04, grip=10.0), qpos).joint_action
        # leader stream goes stale -> the arm freezes at the last command
        held = law.solve(_delta(0.20, grip=50.0), qpos, stale=True)
        assert held.held and held.stale
        # body joints are the frozen last solve; only the gripper could move
        for j in BODY:
            assert held.joint_action[f"{j}.pos"] == fresh[f"{j}.pos"]


class TestEscapeFlippedThreshold:
    """escape_flipped_deg parameterises the elbow sign-flip leg of the
    workspace escape (sim default -15 deg, i.e. sign-vs-the-+60-sim-home).
    None disables the flip leg entirely -- the joint-limit leg stays.  The
    real arm's whole working range is negative elbow (#07: the wall caps
    +2 deg), so a sim-sign threshold would fire every frame mid-work."""

    def _law_with(self, escape_flipped_deg, home_deg=HOME_DEG):
        return PoseDeltaLaw(
            _model(),
            site_name="gripperframe",
            body_dofs=[0, 1, 2, 3, 4],
            body_joint_names=BODY,
            home_joints_deg=home_deg,
            workspace_policy=ClearanceBubble(),
            workspace_radius_m=0.0,
            position_deadband_m=0.0,
            rotation_deadband_rad=0.0,
            escape_flipped_deg=escape_flipped_deg,
        )

    def _droop_elbow_qpos(self, elbow_deg: float) -> np.ndarray:
        q = _home_qpos()
        q[2] = np.radians(elbow_deg)
        return q

    def test_default_threshold_triggers_on_sim_sign_flip(self):
        law = self._law_with(escape_flipped_deg=-15.0)
        qpos = self._droop_elbow_qpos(-20.0)
        law.lock_reference(qpos)
        law.solve(_delta(0.01), qpos)
        assert law.escaped, "elbow -20 deg is 'flipped' vs the sim home (+60)"

    def test_none_disables_the_flip_leg(self):
        law = self._law_with(escape_flipped_deg=None)
        qpos = self._droop_elbow_qpos(-20.0)
        law.lock_reference(qpos)
        law.solve(_delta(0.01), qpos)
        assert not law.escaped, "flip leg off: a negative-elbow solve is normal"

    def test_custom_threshold_moves_the_trip_point(self):
        # Home AT the seed posture so the nullspace rest task cannot drag the
        # solve across the threshold -- the trip point alone decides.
        law = self._law_with(
            escape_flipped_deg=-60.0, home_deg=(0.0, -20.0, -20.0, -40.0, 0.0)
        )
        qpos = self._droop_elbow_qpos(-20.0)
        law.lock_reference(qpos)
        law.solve(_delta(0.01), qpos)
        assert not law.escaped, "-20 deg is above the -60 deg threshold"

    def test_custom_threshold_still_trips_deep_extension(self):
        law = self._law_with(
            escape_flipped_deg=-60.0, home_deg=(0.0, -20.0, -65.0, -40.0, 0.0)
        )
        qpos = self._droop_elbow_qpos(-65.0)
        law.lock_reference(qpos)
        law.solve(_delta(0.01), qpos)
        assert law.escaped, "-65 deg is below the -60 deg threshold"

    def test_residual_hold_returns_last_when_unreachable(self):
        # residual hold on; with the bubble off, a target far beyond reach
        # genuinely cannot be reached -> the law holds the previous action.
        law = _law(residual_hold_m=0.015, workspace_policy=NoBubble())
        qpos = _home_qpos()
        law.lock_reference(qpos)
        first = law.solve(_delta(0.03), qpos).joint_action
        unreachable = law.solve(_delta(dx=2.0), qpos)
        assert unreachable.pos_err_m > 0.015  # confirmed unreachable
        assert unreachable.held
        for j in BODY:
            assert unreachable.joint_action[f"{j}.pos"] == first[f"{j}.pos"]

    def test_residual_hold_off_walks_toward_unreachable(self):
        # residual hold off (sim default): an unreachable target is
        # overshoot-limited (clipped toward last joints) but NOT held.
        law = _law(residual_hold_m=None, workspace_policy=NoBubble())
        qpos = _home_qpos()
        law.lock_reference(qpos)
        sol = law.solve(_delta(dx=2.0), qpos)
        assert not sol.held


# ---------------------------------------------------------------------------
# Backend injection: the same law, a different qpos source
# ---------------------------------------------------------------------------


class TestBackendInjection:
    def test_fk_seed_comes_from_qpos_not_live_state(self):
        """The law never reads a live sim; FK is derived from the qpos argument.
        Feeding a deliberately wrong qpos moves T_zero accordingly -- this is
        the real servicer's injection point (Present_Position -> rad)."""
        law = _law()
        tilted = _home_qpos().copy()
        tilted[0] += np.radians(30.0)  # pan the base 30 deg
        law.lock_reference(tilted)
        # T_zero moved laterally vs the home lock
        home_lock = _law()
        home_lock.lock_reference(_home_qpos())
        assert not np.allclose(law.t_zero[:3, 3], home_lock.t_zero[:3, 3])

    def test_max_reach_override_flows_to_sphere(self):
        """A calibration-derived max_reach reaches the base sphere radius."""
        law = _law(
            workspace_policy=BaseSafetySphere(0.72),
            max_reach_override_m=0.300,  # pretend calibration says 300 mm
        )
        assert law.max_reach_m == pytest.approx(0.300)
        qpos = _home_qpos()
        law.lock_reference(qpos)
        assert law.workspace_bubble_m == pytest.approx(0.72 * 0.300)
