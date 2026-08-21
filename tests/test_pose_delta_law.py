"""Seam tests for the shared PoseDeltaLaw — PikaAnyArm official alignment.

The entire compose + DLS + official-safety-check pipeline lives in PoseDeltaLaw
behind a tiny interface, drivable with NO servicer and NO backend -- just a
MuJoCo model and a joint vector.  This is exactly how both the sim and the
real Feetech servicer drive it: read qpos -> law.solve -> write joints.

Under test (align-official-decisions.md decisions 2/4/6):

- the official composition ``T_target = T_arm_ref @ Δ`` (translation follows
  the reference orientation, not the room axes);
- the official follower safety stack: FK consistency 0.3 m reject, 30° jump
  warm-start reset, per-joint per-frame step cap, self-collision gate
  (auto-bypassed on SO-101's visual-only meshes);
- the stale hold (leader-stream gap freeze).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from lerobot_robot_grpc.follower.pose_delta_law import (  # noqa: E402
    JointSolution,
    PoseDeltaLaw,
)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"

BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
HOME_DEG = (0.0, -20.0, 60.0, -40.0, 0.0)

# Synthetic 3-link chain WITH collision geometry (unlike the SO-101 visual
# meshes, every link here carries a default-collision capsule) for the
# self-collision gate tests.  All three links are jointed so every geom maps
# onto the solved chain.
_COLLISION_XML = """
<mujoco>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.2 0 0" size="0.01"
            contype="1" conaffinity="1"/>
      <body name="link2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.15 0 0" size="0.01"
              contype="1" conaffinity="1"/>
        <body name="link3" pos="0.15 0 0">
          <joint name="j3" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <geom name="g2" type="capsule" fromto="0 0 0 0.15 0 0" size="0.01"
                contype="1" conaffinity="1"/>
          <site name="ee" pos="0.15 0 0" size="0.005"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


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


def _law(**kwargs) -> PoseDeltaLaw:
    return PoseDeltaLaw(
        _model(),
        site_name="gripperframe",
        body_dofs=[0, 1, 2, 3, 4],
        body_joint_names=BODY,
        home_joints_deg=HOME_DEG,
        **kwargs,
    )


def _fk_pos(law: PoseDeltaLaw, qpos) -> np.ndarray:
    pose = law._fk(np.asarray(qpos, dtype=float))
    return pose[:3, 3]


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
# Official composition: T_target = T_arm_ref @ Δ
# ---------------------------------------------------------------------------


class TestOfficialComposition:
    def test_body_delta_rotates_through_reference(self):
        """A body-frame +X delta must move the EE along the reference's X axis
        in the base frame (translation follows the hand orientation), not
        along the base X axis."""
        # Reference latched with the base panned +90 deg: R_ref @ x̂ = +ŷ.
        panned = _home_qpos().copy()
        panned[0] += math.pi / 2.0
        law = _law()
        law.lock_reference(panned)
        ref = law.arm_reference[:3, 3]

        sol = law.solve(_delta(dx=0.05), panned)
        moved = _fk_pos(law, [math.radians(sol.joint_action[f"{j}.pos"]) for j in BODY]
                        + [0.0])
        offset = moved - ref
        # The dominant motion is +Y (base left), NOT +X (the old room-axis
        # semantics would have moved +X for a world-frame dx).
        assert abs(offset[1]) > 3.0 * abs(offset[0])

    def test_zero_delta_holds_the_reference(self):
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)
        sol = law.solve(_delta(), qpos)
        sent = [math.radians(sol.joint_action[f"{j}.pos"]) for j in BODY]
        np.testing.assert_allclose(sent, qpos[:5], atol=np.radians(1.5))


# ---------------------------------------------------------------------------
# Official safety stack
# ---------------------------------------------------------------------------


class TestFkConsistency:
    def test_far_unreachable_target_rejected_and_held(self):
        """Official piper_IK check: a solution whose FK deviates >0.3 m from
        the target is rejected -- hold the last published joints."""
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)
        first = law.solve(_delta(0.03), qpos).joint_action
        far = law.solve(_delta(dx=2.0), qpos)
        assert far.rejected and far.held
        for j in BODY:
            assert far.joint_action[f"{j}.pos"] == first[f"{j}.pos"]

    def test_reachable_target_not_rejected(self):
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)
        sol = law.solve(_delta(0.05), qpos)
        assert not sol.rejected and not sol.held


class TestJumpReset:
    def test_solution_jump_is_rejected_and_holds_the_committed_command(
        self, monkeypatch
    ):
        """A >30 degree IK branch change is not a motion command.

        A frame cap limits velocity but does not make a discontinuous IK
        branch safe.  The law must keep the last committed command and report
        the rejected jump so the session layer can request re-alignment.
        """
        law = _law(reject_branch_jumps=True)
        qpos = _home_qpos()
        law.lock_reference(qpos)
        law.solve(_delta(0.02), qpos)
        assert law._warm_seed is not None
        committed = law._last_sent.copy()

        # Make every deterministic seed converge accurately onto the same
        # discontinuous branch.  This isolates branch acceptance from DLS
        # convergence details while still exercising the law's public solve
        # seam, collision gate, state commit, and hold result.
        def discontinuous_solve(target_pos, target_rot, _seed):
            raw = law._last_solved.copy()
            raw[0] += math.radians(45.0)
            return type("AccurateJump", (), {
                "qpos": raw,
                "pos_err": 0.0,
                "rot_err": 0.0,
                "manipulability": 1.0,
                "achieved_pos": np.asarray(target_pos),
                "achieved_rot": np.asarray(target_rot),
            })()

        monkeypatch.setattr(law._ik_solver, "solve", discontinuous_solve)
        jumped = law.solve(_delta(dz=-0.15, dy=0.10), qpos)
        assert jumped.held and jumped.rejected and jumped.jumped
        assert jumped.reason == "ik-branch-jump"
        np.testing.assert_allclose(law._last_sent, committed)
        assert law._warm_seed is None


class TestFrameCap:
    def test_published_step_capped_per_joint(self):
        """The official >30 deg 200 Hz interpolation equivalent: no published
        joint step exceeds max_dq_frame_deg, even toward a large retarget."""
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)
        prev = np.array(qpos[:5], dtype=float)
        for _ in range(3):
            sol = law.solve(_delta(dz=0.12, dx=0.10), qpos)
            sent = np.radians(
                [sol.joint_action[f"{j}.pos"] for j in BODY]
            )
            assert float(np.abs(sent - prev).max()) <= math.radians(6.7) + 1e-9
            prev = sent
            if sol.held:
                break

    def test_first_step_capped_from_measured_qpos(self):
        """Official interpolation seeds its baseline from the measured joints
        -- the first published step after a latch is capped too."""
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)
        sol = law.solve(_delta(dx=0.3), qpos)  # large first intent
        sent = np.radians([sol.joint_action[f"{j}.pos"] for j in BODY])
        assert float(np.abs(sent - qpos[:5]).max()) <= math.radians(6.7) + 1e-9


class TestStaleHold:
    def test_stale_flag_holds_last_action(self):
        law = _law()
        qpos = _home_qpos()
        law.lock_reference(qpos)
        law.solve(_delta(0.04, grip=10.0), qpos)
        # leader stream goes stale -> re-anchor the hold at measured qpos so a
        # delayed controller target cannot resume after the gap.
        held = law.solve(_delta(0.20, grip=50.0), qpos, stale=True)
        assert held.held and held.stale
        for index, joint in enumerate(BODY):
            assert held.joint_action[f"{joint}.pos"] == pytest.approx(
                np.degrees(qpos[index])
            )


class TestCollisionHysteresis:
    def test_collision_status_releases_only_after_the_outer_margin(
        self, monkeypatch
    ):
        """A 10 mm entry gate must not chatter READY before 15 mm release."""

        class Checker:
            supports_hysteresis = True

            @staticmethod
            def check(qpos, *, release=False):
                clearance = float(qpos[0])
                threshold = 0.015 if release else 0.010
                return type(
                    "Collision",
                    (),
                    {
                        "collided": clearance <= threshold,
                        "body_a": "left_gripper_base_link",
                        "body_b": "right_gripper_base_link",
                        "distance_m": clearance,
                    },
                )()

        law = _law(collision_checker=Checker())
        qpos = _home_qpos()
        qpos[0] = 0.020
        law.lock_reference(qpos)
        next_clearance = 0.009

        def scripted_solve(target_pos, target_rot, _seed):
            raw = qpos[:5].copy()
            raw[0] = next_clearance
            return type(
                "Candidate",
                (),
                {
                    "qpos": raw,
                    "pos_err": 0.0,
                    "rot_err": 0.0,
                    "manipulability": 1.0,
                    "achieved_pos": np.asarray(target_pos),
                    "achieved_rot": np.asarray(target_rot),
                },
            )()

        monkeypatch.setattr(law._ik_solver, "solve", scripted_solve)

        entered = law.solve(_delta(), qpos)
        assert entered.collided

        # The measured arm has escaped beyond 10 mm, but remains inside the
        # 15 mm release margin.  An outward command is allowed while the
        # safety receipt remains collision-limited.
        qpos[0] = 0.012
        next_clearance = 0.013
        escaping = law.solve(_delta(), qpos)
        assert not escaping.held
        assert escaping.collided
        assert escaping.reason == "collision-hysteresis"

        qpos[0] = 0.016
        next_clearance = 0.017
        released = law.solve(_delta(), qpos)
        assert not released.held
        assert not released.collided

    def test_gripper_precheck_does_not_block_arm_escape(self, monkeypatch):
        """An unchanged arm-body pair must reach IK so the arm can move away."""

        class Checker:
            supports_hysteresis = True

            @staticmethod
            def check(qpos, *, release=False):
                clearance = float(qpos[0])
                threshold = 0.015 if release else 0.010
                return type(
                    "Collision",
                    (),
                    {
                        "collided": clearance <= threshold,
                        "body_a": "left_arm_link2",
                        "body_b": "torso_base_link",
                        "distance_m": clearance,
                    },
                )()

        def gripper_candidate(state, gripper_open):
            candidate = np.asarray(state, dtype=float).copy()
            candidate[-1] = float(gripper_open) / 100.0
            return candidate

        law = _law(
            collision_checker=Checker(),
            candidate_qpos_adapter=gripper_candidate,
        )
        qpos = _home_qpos()
        qpos[0] = 0.012
        law.lock_reference(qpos)
        law._collision_latched = True

        def escaping_solve(target_pos, target_rot, _seed):
            raw = qpos[:5].copy()
            raw[0] = 0.013
            return type(
                "Candidate",
                (),
                {
                    "qpos": raw,
                    "pos_err": 0.0,
                    "rot_err": 0.0,
                    "manipulability": 1.0,
                    "achieved_pos": np.asarray(target_pos),
                    "achieved_rot": np.asarray(target_rot),
                },
            )()

        monkeypatch.setattr(law._ik_solver, "solve", escaping_solve)

        escaping = law.solve(_delta(grip=30.0), qpos)
        assert not escaping.held
        assert escaping.collided
        assert escaping.reason == "collision-hysteresis"


# ---------------------------------------------------------------------------
# Self-collision gate: capability auto-detect
# ---------------------------------------------------------------------------


class TestSelfCollisionGate:
    def test_so101_scene_enables_the_gate(self):
        """The SO-101 scene carries a collision-geom group (default
        contype/conaffinity mesh geoms on every arm link) alongside the
        visual-only meshes — the auto-detect finds 43 non-adjacent pairs and
        the gate is ON.  Verified semantics on this model: no false
        positives across the ordinary workspace, true positives on deep
        folds (shoulder vs lower-arm/wrist penetration)."""
        law = _law()
        assert law.collision_enabled
        home = np.zeros(6)
        home[:5] = np.radians(np.array(HOME_DEG, dtype=float))
        assert not law._in_self_collision(home)
        real_rest = np.zeros(6)
        real_rest[:5] = np.radians(np.array((0.0, 30.0, -20.0, 0.0, 0.0)))
        assert not law._in_self_collision(real_rest)
        # Deep fold drives the arm into the shoulder (measured 2026-08-18:
        # shoulder vs lower_arm/wrist penetrations of 3-16 mm).
        folded = np.radians(np.array((-80.0, 10.0, 95.0, 80.0, 0.0)).astype(float))
        probe = np.zeros(6)
        probe[:5] = folded
        assert law._in_self_collision(probe)

    def test_collision_model_enables_and_detects(self):
        model = mujoco.MjModel.from_xml_string(_COLLISION_XML)
        law = PoseDeltaLaw(
            model,
            site_name="ee",
            body_dofs=[0, 1, 2],
            body_joint_names=("j1", "j2", "j3"),
            home_joints_deg=(0.0, 0.0, 0.0),
        )
        assert law.collision_enabled
        # g0 (link1) x g2 (link3) is the only non-adjacent pair with bits;
        # g0-g1 and g1-g2 sit on parent/child links and are excluded.
        assert law._collision_pairs == {frozenset((0, 2))}

        straight = np.zeros(3)                      # chain extended: no contact
        assert not law._in_self_collision(straight)
        folded = np.array([0.0, math.pi, 0.0])      # link3 folded back onto link1
        assert law._in_self_collision(folded)


# ---------------------------------------------------------------------------
# Backend injection: the same law, a different qpos source
# ---------------------------------------------------------------------------


class TestBackendInjection:
    def test_fk_seed_comes_from_qpos_not_live_state(self):
        """The law never reads a live sim; FK is derived from the qpos argument.
        Feeding a deliberately wrong qpos moves T_arm_ref accordingly -- this is
        the real servicer's injection point (Present_Position -> rad)."""
        law = _law()
        tilted = _home_qpos().copy()
        tilted[0] += np.radians(30.0)  # pan the base 30 deg
        law.lock_reference(tilted)
        # T_arm_ref moved laterally vs the home lock
        home_lock = _law()
        home_lock.lock_reference(_home_qpos())
        assert not np.allclose(
            law.arm_reference[:3, 3], home_lock.arm_reference[:3, 3]
        )
