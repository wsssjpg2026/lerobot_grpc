"""Sim/real-shared pose_delta control law — PikaAnyArm official alignment.

The leader publishes the RAW relative transform of the tracker from its
reference latch (no filtering, 1:1):

    Δ = inv(T_tracker_ref) @ T_tracker_now
      = [ R_ref^T @ R_now , R_ref^T @ (p_now - p_ref) ]

i.e. hand.delta_pos is the position offset expressed in the tracker
reference's BODY frame, and hand.delta_rot is the body-frame rotation offset.
This law composes the official target (PikaAnyArm teleop_piper_publish):

    T_target = T_arm_ref @ Δ
    p_target = p_ref + R_ref @ Δp        (translation follows the hand's
    R_target = R_ref @ ΔR                 orientation, not the room axes)

No lighthouse→base calibration (R_lh2base) participates — the construction
eliminates it.

Follower safety stack (shared by sim and real adapters):

1. Input freshness/finite checks and a robot-supplied Cartesian workspace
   prefilter reject before IK.
2. IK hard joint limits — DLSIKSolver clips every Newton step into the model
   range, narrowed by the measured calibration range where available.
3. A deterministic bounded multi-seed fallback explores measured, committed,
   home/rest and alternate-elbow branches when the fast candidate fails or
   jumps.
4. IK acceptance — the legacy 0.3 m per-axis FK check remains a corruption
   guard, while a candidate with more than 10 mm operational position
   residual is not allowed to become a fallback motion branch.
5. Per-joint per-frame step cap — plays the rate-constraint role of the
   official >30° 200 Hz linear interpolation: a walked-to solution advances
   at most max_dq_frame_deg per action frame.
6. Collision gating checks the raw IK endpoint and the actual capped path.
   A robot adapter may inject semantic full-body geometry and gripper coupling;
   otherwise model-capability auto-detection supplies the legacy self-check.
7. A >30° branch jump triggers same-frame fallback; if no continuous branch
   remains, the command is rejected and the last safe target is held.  A
   velocity cap cannot turn a discontinuous IK branch into a safe command.

Kept from the previous stack (official-equivalent or bench UX):

- stale hold: a leader-stream gap freezes the arm at the last action (the
  official piper publisher drops >1 s stale messages);
- the law is driven identically by the MuJoCo servicer and the real Feetech
  servicer — only the qpos source and the joint write-out differ.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .dls_ik import DLSIKSolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JointSolution:
    """Result of one pose_delta solve, including diagnostics.

    joint_action is lerobot-normalised (body joints in degrees, gripper.pos in
    0-100) -- exactly what both backends write.
    """

    joint_action: dict[str, float]
    pos_err_m: float
    held: bool
    stale: bool
    rejected: bool
    collided: bool
    jumped: bool
    manipulability: float
    reason: str = ""
    frame_capped: bool = False
    rot_err_rad: float = 0.0


class PoseDeltaLaw:
    """The shared pose_delta control law behind a single solve() method.

    Owns the arm reference latch (T_arm_ref), the DLS IK solver and its warm
    start, and the official follower safety checks.  A servicer drives it by:

    1. reading its current joint vector (sim data.qpos / real
       Present_Position to rad) and passing it to lock_reference / solve;
    2. writing the returned joint_action to its backend (MuJoCo ctrl / Feetech
       send_action).

    FK and IK both run on the same MuJoCo model passed at construction -- the
    real arm loads it purely as a kinematics oracle, seeded with measured
    joints.  Nothing in this module reads a live sim state directly: every FK
    comes from the qpos_rad argument, which is the seam.

    Parameters
    ----------
    model
        mujoco.MjModel -- the kinematics engine.  Same instance the DLSIKSolver
        uses.
    site_name
        MuJoCo site whose world pose is tracked (e.g. 'gripperframe').
    body_dofs
        qpos indices of the body joints the solver may move (gripper excluded).
    body_joint_names
        Names of those body joints, in order -- used to build the "<joint>.pos"
        keys of the returned action dict.
    home_joints_deg
        Rest posture (degrees) for the body joints -- the DLS null-space bias.
    rot_weight
        DLS rotation weight (fixed; the official solver uses fixed weights).
    rest_gain
        DLS null-space rest gain (solver configuration preference, not a
        safety mechanism).
    max_dq_deg
        DLS per-iteration per-joint step clip (solver stabilisation).
    fk_consistency_m
        Official FK consistency threshold: reject a solution whose FK position
        deviates from the commanded target by more than this on ANY axis.
    jump_reset_deg
        Official jump rejection: max per-joint difference between consecutive
        solutions before the warm start resets.
    max_dq_frame_deg
        Per-joint per-action-frame cap on the published step.  The generic
        default is 6.7°; S1 passes 2.292° for 1.2 rad/s at 30 Hz.
    gripper_max_distance_mm
        Full-open gripper distance for the distance→0-100 mapping.
    """

    def __init__(
        self,
        model,
        *,
        site_name,
        body_dofs,
        body_joint_names,
        home_joints_deg,
        rot_weight: float = 0.3,
        rest_gain: float = 0.08,
        max_dq_deg: float = 6.0,
        fk_consistency_m: float = 0.3,
        ik_accept_pos_err_m: float | None = None,
        jump_reset_deg: float = 30.0,
        reject_branch_jumps: bool = False,
        max_dq_frame_deg: float = 6.7,
        gripper_max_distance_mm: float = 60.0,
        collision_checker: Callable[[np.ndarray], bool] | None = None,
        candidate_qpos_adapter: Callable[[np.ndarray, float], np.ndarray] | None = None,
        workspace_delta_m: float | None = None,
        workspace_checker: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    ):
        import mujoco

        self._mj = mujoco
        self._model = model
        self._body_dofs = tuple(int(d) for d in body_dofs)
        self._body_joint_names = tuple(body_joint_names)
        self._n_body = len(self._body_dofs)
        self._rot_weight = float(rot_weight)
        self._gripper_max_distance_mm = float(gripper_max_distance_mm)
        self._fk_consistency_m = float(fk_consistency_m)
        self._ik_accept_pos_err_m = (
            None
            if ik_accept_pos_err_m is None
            else float(ik_accept_pos_err_m)
        )
        if (
            self._ik_accept_pos_err_m is not None
            and self._ik_accept_pos_err_m <= 0.0
        ):
            raise ValueError("ik_accept_pos_err_m must be positive")
        self._jump_reset_rad = math.radians(float(jump_reset_deg))
        self._reject_branch_jumps = bool(reject_branch_jumps)
        self._max_dq_frame_rad = math.radians(float(max_dq_frame_deg))
        self._home_qpos = np.radians(np.asarray(home_joints_deg, dtype=float))
        self._max_ik_candidates = 5
        self._ik_deadline_s = 0.010
        self._path_step_rad = math.radians(1.0)
        self._collision_checker = collision_checker
        self._candidate_qpos_adapter = candidate_qpos_adapter
        self._workspace_delta_m = (
            None if workspace_delta_m is None else float(workspace_delta_m)
        )
        self._workspace_checker = workspace_checker

        # FK / collision scratch (never the sim live data -- seeded per call).
        self._fk_data = mujoco.MjData(model)

        self._ik_solver = DLSIKSolver(
            model,
            site_name=site_name,
            body_dofs=list(self._body_dofs),
            rot_weight=rot_weight,
            rest_qpos=self._home_qpos,
            rest_gain=rest_gain,
            max_dq_rad=math.radians(max_dq_deg),
        )
        self._site_id = self._ik_solver.site_id

        # --- Self-collision capability (auto-detect) ----------------------
        self._collision_pairs = self._detect_collision_pairs()
        if self._collision_checker is not None:
            logger.info("External collision candidate checker ENABLED.")
        elif self._collision_pairs:
            logger.info(
                "Self-collision gate ENABLED: %d geom pairs under watch.",
                len(self._collision_pairs),
            )
        else:
            logger.info(
                "Self-collision gate BYPASSED: model has no usable collision "
                "geometry on non-adjacent arm links (SO-101 meshes are "
                "visual-only). Enable by adding contype/conaffinity geoms."
            )

        # --- Mutable solve state (reset on Connect) -------------------------
        self._reset_state()

        # Throttle helpers for event logs.
        self._last_jump_log_ts = 0.0
        self._last_reject_log_ts = 0.0
        self._last_ik_debug_ts = 0.0

        logger.info(
            "PoseDeltaLaw ready: site=%r dofs=%s rot_weight=%.3f "
            "fk_consistency=%.1fm ik_accept_pos_err=%s "
            "jump_reset=%.0fdeg frame_cap=%.1fdeg "
            "collision_pairs=%d",
            site_name, self._body_dofs, rot_weight,
            self._fk_consistency_m,
            (
                "disabled"
                if self._ik_accept_pos_err_m is None
                else f"{self._ik_accept_pos_err_m * 1000.0:.0f}mm"
            ),
            jump_reset_deg, max_dq_frame_deg,
            len(self._collision_pairs),
        )

    # ------------------------------------------------------------------
    # Self-collision capability detection
    # ------------------------------------------------------------------

    def _detect_collision_pairs(self) -> set[frozenset[int]]:
        """Geom pairs eligible for self-collision gating.

        A pair qualifies when both geoms carry collision bits
        (contype|conaffinity != 0), the bits are mutually compatible
        (MuJoCo's own filter rule), and the geoms sit on non-adjacent links
        of the solved chain.  A geom's link is the nearest ancestor body
        carrying one of the solved DOFs (welded children -- fingers on a
        wrist -- count as their carrier link).  World geoms (the floor) are
        excluded: this gate is self-collision only.  Empty result means the
        model cannot support the gate (SO-101: all meshes contype=0).
        """
        mj = self._mj
        model = self._model
        dof_bodies = sorted({int(model.dof_bodyid[d]) for d in self._body_dofs})
        dof_body_set = set(dof_bodies)

        # link(body) = nearest ancestor-or-self carrying a solved DOF.
        def link_of(body_id: int) -> int | None:
            b = body_id
            while b != 0:
                if b in dof_body_set:
                    return b
                b = int(model.body_parentid[b])
            return None

        # Adjacency in the REDUCED link chain (link1 is link2's parent or
        # vice versa in the body tree).
        link_parent: dict[int, int] = {}
        for b in dof_bodies:
            link_parent[b] = link_of(int(model.body_parentid[b]))

        geoms: list[tuple[int, int]] = []  # (geom_id, link)
        for gid in range(model.ngeom):
            link = link_of(int(model.geom_bodyid[gid]))
            if link is None:
                continue  # world / unrelated body
            ct, ca = int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])
            if ct == 0 and ca == 0:
                continue  # visual-only
            geoms.append((gid, link))

        pairs: set[frozenset[int]] = set()
        for i in range(len(geoms)):
            gid1, link1 = geoms[i]
            for j in range(i + 1, len(geoms)):
                gid2, link2 = geoms[j]
                if link1 == link2:
                    continue  # same link
                if link_parent.get(link1) == link2 or link_parent.get(link2) == link1:
                    continue  # adjacent links always touch at the joint
                ct1, ca1 = int(model.geom_contype[gid1]), int(model.geom_conaffinity[gid1])
                ct2, ca2 = int(model.geom_contype[gid2]), int(model.geom_conaffinity[gid2])
                if (ct1 & ca2) or (ct2 & ca1):
                    pairs.add(frozenset((gid1, gid2)))
        _ = mj  # detection is pure model inspection; mj kept for symmetry
        return pairs

    def _in_self_collision(self, qpos_rad: np.ndarray) -> bool:
        """mj_forward the candidate joints; True on any watched pair contact."""
        self._fk_data.qpos[:] = qpos_rad
        self._mj.mj_forward(self._model, self._fk_data)
        for k in range(self._fk_data.ncon):
            c = self._fk_data.contact[k]
            if frozenset((int(c.geom1), int(c.geom2))) in self._collision_pairs:
                return True
        return False

    # ------------------------------------------------------------------
    # Public state / diagnostics (read by servicers, tests, 1 Hz logs)
    # ------------------------------------------------------------------

    @property
    def ik_solver(self):
        return self._ik_solver

    @property
    def arm_reference(self) -> np.ndarray:
        """Locked arm reference pose T_arm_ref (4x4). After lock_reference()."""
        assert self._arm_ref is not None, "lock_reference() first"
        return self._arm_ref

    @property
    def collision_enabled(self) -> bool:
        """Whether the self-collision gate is active for this model."""
        return self._collision_checker is not None or bool(self._collision_pairs)

    # ------------------------------------------------------------------
    # Reset / latch
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._arm_ref = None
        self._warm_seed: np.ndarray | None = None  # None -> seed from measured qpos
        self._last_solved: np.ndarray | None = None
        self._last_sent: np.ndarray | None = None
        self._last_joint_action: dict[str, float] | None = None
        self.last_pos_err_m = 0.0
        self.last_rot_err_rad = 0.0
        self.last_manipulability = 0.0
        self.last_collision_pair: tuple[str, str] = ("", "")
        self.last_collision_distance_m = float("inf")
        self._collision_latched = False

    def reset(self):
        """Clear all latch/solve state (call on Connect)."""
        self._reset_state()

    def _fk(self, qpos_rad) -> np.ndarray:
        """4x4 EE pose from a full joint vector via the shared model."""
        self._fk_data.qpos[:] = qpos_rad
        self._mj.mj_forward(self._model, self._fk_data)
        pose = np.eye(4)
        pose[:3, 3] = self._fk_data.site_xpos[self._site_id].copy()
        pose[:3, :3] = self._fk_data.site_xmat[self._site_id].reshape(3, 3).copy()
        return pose

    def lock_reference(self, qpos_rad):
        """Latch T_arm_ref at the FK of the given joint vector.

        Clutch re-engage contract: the servicer calls this (SetReference)
        with the current joints so the next Δ=0 maps onto the arm's current
        pose.  Nothing here moves the arm -- it only re-anchors the
        reference; solve state (warm start, last action) is cleared so the
        first post-latch action is a fresh solve from the measured pose.
        """
        self._arm_ref = self._fk(qpos_rad)
        self._warm_seed = None
        self._last_solved = None
        self._last_sent = None
        self._last_joint_action = None
        logger.info(
            "PoseDeltaLaw reference locked: T_arm_ref at pos=[%.3f %.3f %.3f] "
            "|p|=%.0fmm",
            *self._arm_ref[:3, 3],
            float(np.linalg.norm(self._arm_ref[:3, 3])) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Unit helpers
    # ------------------------------------------------------------------

    def _gripper_distance_to_0_100(self, distance_mm):
        return float(np.clip(distance_mm / self._gripper_max_distance_mm * 100.0, 0.0, 100.0))

    def _identity_action(self, qpos_rad, gripper_0_100: float) -> dict[str, float]:
        """Action holding the measured joints (engage-frame / reject fallback)."""
        action = {
            "%s.pos" % joint: math.degrees(float(qpos_rad[dof]))
            for joint, dof in zip(
                self._body_joint_names, self._body_dofs, strict=True
            )
        }
        action["gripper.pos"] = gripper_0_100
        return action

    # ------------------------------------------------------------------
    # The law
    # ------------------------------------------------------------------

    def _hold_solution(
        self,
        qpos_rad: np.ndarray,
        gripper_0_100: float,
        reason: str,
        *,
        pos_err_m: float,
        manipulability: float,
        rot_err_rad: float | None = None,
        stale: bool = False,
        collided: bool = False,
        jumped: bool = False,
    ) -> JointSolution:
        # A failed candidate must never remain the next frame's preferred IK
        # branch.  Committed targets stay intact; only the speculative seed is
        # invalidated.
        self._warm_seed = None
        now = time.time()
        reported_rot_err = (
            self.last_rot_err_rad
            if rot_err_rad is None
            else float(rot_err_rad)
        )
        if now - self._last_reject_log_ts > 1.0:
            self._last_reject_log_ts = now
            collision_detail = ""
            if reason.startswith("collision") and self.last_collision_pair != ("", ""):
                collision_detail = " pair=%s<->%s distance=%.1fmm" % (
                    self.last_collision_pair[0],
                    self.last_collision_pair[1],
                    self.last_collision_distance_m * 1000.0,
                )
            logger.warning(
                "SOLVE rejected (%s): pos_err=%.1fmm rot_err=%.1fdeg%s "
                "-- holding last action.",
                reason,
                pos_err_m * 1000.0,
                math.degrees(reported_rot_err),
                collision_detail,
            )
        if self._last_joint_action is not None:
            action = dict(self._last_joint_action)
        else:
            action = self._identity_action(qpos_rad, gripper_0_100)
        action["gripper.pos"] = gripper_0_100
        return JointSolution(
            action,
            pos_err_m,
            True,
            stale,
            not stale,
            collided,
            jumped,
            manipulability,
            reason,
            False,
            reported_rot_err,
        )

    def _adapt_candidate_state(
        self, qpos_rad: np.ndarray, gripper_0_100: float
    ) -> np.ndarray:
        state = np.asarray(qpos_rad, dtype=float).copy()
        if self._candidate_qpos_adapter is not None:
            state = np.asarray(
                self._candidate_qpos_adapter(state, gripper_0_100), dtype=float
            )
        return state

    def _candidate_collides(
        self,
        qpos_rad: np.ndarray,
        *,
        escape_from: np.ndarray | None = None,
        release: bool | None = None,
        allow_non_worsening_escape: bool = False,
    ) -> bool:
        if self._collision_checker is not None:
            checker = self._collision_checker
            if hasattr(checker, "check"):
                use_release = self._collision_latched if release is None else release
                if getattr(checker, "supports_hysteresis", False):
                    result = checker.check(qpos_rad, release=use_release)
                else:
                    result = checker.check(qpos_rad)
                if result.collided:
                    # A model can start inside a safety margin because of
                    # assembly tolerances or a previously interrupted move.
                    # Ordinary arm candidates must strictly increase the
                    # same-pair distance so hold cannot deadlock the escape
                    # direction.  A gripper-only precheck may also keep the
                    # distance unchanged: it is only proving that the gripper
                    # did not introduce/worsen the existing arm-body pair,
                    # after which the arm IK still has to pass this gate.
                    if escape_from is not None:
                        if getattr(checker, "supports_hysteresis", False):
                            current = checker.check(
                                escape_from, release=use_release
                            )
                        else:
                            current = checker.check(escape_from)
                        if (
                            current.collided
                            and (result.body_a, result.body_b)
                            == (current.body_a, current.body_b)
                        ):
                            distance_change = (
                                float(result.distance_m)
                                - float(current.distance_m)
                            )
                            if distance_change > 1e-6 or (
                                allow_non_worsening_escape
                                and distance_change >= -1e-6
                            ):
                                return False
                    self.last_collision_pair = (result.body_a, result.body_b)
                    self.last_collision_distance_m = float(result.distance_m)
                return bool(result.collided)
            return bool(checker(qpos_rad))
        if self._collision_pairs:
            return self._in_self_collision(qpos_rad)
        return False

    def _refresh_collision_latch(self, qpos_rad: np.ndarray) -> None:
        """Release collision state only after the measured pose clears it.

        Candidate checks may safely permit a distance-increasing escape while
        still inside the outer margin.  Keeping the latch tied to measured
        qpos prevents LIMIT/READY chatter at the inner threshold.
        """
        if not self._collision_latched or self._collision_checker is None:
            return
        checker = self._collision_checker
        if not (
            hasattr(checker, "check")
            and getattr(checker, "supports_hysteresis", False)
        ):
            self._collision_latched = False
            return
        result = checker.check(qpos_rad, release=True)
        if result.collided:
            self.last_collision_pair = (result.body_a, result.body_b)
            self.last_collision_distance_m = float(result.distance_m)
        else:
            self._collision_latched = False

    def _path_collides(
        self,
        start_qpos_rad: np.ndarray,
        end_qpos_rad: np.ndarray,
    ) -> bool:
        if not self.collision_enabled:
            return False
        start = np.asarray(start_qpos_rad, dtype=float)
        end = np.asarray(end_qpos_rad, dtype=float)
        distance = float(np.abs(end - start).max())
        steps = max(1, int(math.ceil(distance / self._path_step_rad)))
        for step in range(1, steps + 1):
            fraction = step / steps
            probe = start + fraction * (end - start)
            if self._candidate_collides(probe, escape_from=start):
                return True
        return False

    def _last_safe_path_state(
        self,
        start_qpos_rad: np.ndarray,
        end_qpos_rad: np.ndarray,
    ) -> np.ndarray | None:
        """Return the furthest collision-free prefix of a joint-space step.

        A Cartesian IK endpoint may lie beyond a safety margin even though
        part of the rate-capped step toward it is safe.  Rejecting the whole
        endpoint makes the arm appear to stop prematurely.  Walk the actual
        published path, then bisect only the first safe/colliding interval so
        the arm approaches—but never enters—the boundary.
        """
        start = np.asarray(start_qpos_rad, dtype=float)
        end = np.asarray(end_qpos_rad, dtype=float)
        distance = float(np.abs(end - start).max())
        if distance <= 1e-9:
            return None

        steps = max(1, int(math.ceil(distance / self._path_step_rad)))
        previous_fraction = 0.0
        for step in range(1, steps + 1):
            fraction = step / steps
            probe = start + fraction * (end - start)
            if not self._candidate_collides(probe, escape_from=start):
                previous_fraction = fraction
                continue

            low = previous_fraction
            high = fraction
            for _ in range(10):
                middle = 0.5 * (low + high)
                middle_probe = start + middle * (end - start)
                if self._candidate_collides(middle_probe, escape_from=start):
                    high = middle
                else:
                    low = middle
            if low * distance <= 1e-4:
                return None
            return start + low * (end - start)
        return end.copy()

    def _candidate_seeds(self, qpos_rad: np.ndarray) -> list[np.ndarray]:
        measured_arm = qpos_rad[list(self._body_dofs)]
        arm_seeds: list[np.ndarray] = []

        def add(arm_seed: np.ndarray) -> None:
            arm_seed = np.asarray(arm_seed, dtype=float)
            if any(np.allclose(arm_seed, existing, atol=1e-9, rtol=0.0) for existing in arm_seeds):
                return
            arm_seeds.append(arm_seed.copy())

        add(self._last_sent if self._last_sent is not None else measured_arm)
        add(measured_arm)
        if (
            self._last_solved is not None
            and float(np.abs(self._last_solved - measured_arm).max())
            <= self._jump_reset_rad
        ):
            add(self._last_solved)
        home = self._home_qpos.copy()
        home[0] = measured_arm[0]
        add(home)
        alternate = measured_arm.copy()
        if len(alternate) >= 4:
            alternate[3] = -alternate[3]
        add(alternate)

        seeds: list[np.ndarray] = []
        for arm_seed in arm_seeds[: self._max_ik_candidates]:
            seed = qpos_rad.copy()
            seed[list(self._body_dofs)] = arm_seed
            seeds.append(seed)
        return seeds

    def solve(self, delta_action, qpos_rad, *, stale: bool = False) -> JointSolution:
        """Compose T_target = T_arm_ref @ Δ, solve DLS IK, apply official checks.

        Pipeline: validate/stale/workspace -> deterministic multi-seed DLS ->
        FK/raw endpoint collision -> branch ranking -> per-frame cap -> capped
        path collision -> atomic state commit and publish.
        qpos_rad is the current full joint vector (the only backend-injected
        input).
        """
        from lerobot.utils.rotation import Rotation as Rot

        delta_pos = np.array(
            [
                delta_action["hand.delta_pos.x"],
                delta_action["hand.delta_pos.y"],
                delta_action["hand.delta_pos.z"],
            ],
            dtype=float,
        )
        delta_quat = np.array(
            [
                delta_action["hand.delta_rot.qx"],
                delta_action["hand.delta_rot.qy"],
                delta_action["hand.delta_rot.qz"],
                delta_action["hand.delta_rot.qw"],
            ],
            dtype=float,
        )
        gripper_distance = delta_action["gripper.distance"]
        gripper_0_100 = self._gripper_distance_to_0_100(gripper_distance)
        qpos_rad = np.asarray(qpos_rad, dtype=float)
        self.last_collision_pair = ("", "")
        self.last_collision_distance_m = float("inf")

        if not (
            np.isfinite(delta_pos).all()
            and np.isfinite(delta_quat).all()
            and np.isfinite(gripper_distance)
            and float(np.linalg.norm(delta_quat)) > 1e-8
        ):
            return self._hold_solution(
                qpos_rad,
                gripper_0_100,
                "input-invalid",
                pos_err_m=self.last_pos_err_m,
                manipulability=self.last_manipulability,
            )

        try:
            self._refresh_collision_latch(qpos_rad)
        except Exception:
            logger.exception("Collision hysteresis checker failed closed")
            return self._hold_solution(
                qpos_rad,
                gripper_0_100,
                "checker-error",
                pos_err_m=self.last_pos_err_m,
                manipulability=self.last_manipulability,
                collided=True,
            )

        if self._arm_ref is None:
            self.lock_reference(qpos_rad)
        arm_ref = self._arm_ref
        assert arm_ref is not None

        # --- Stale hold: leader stream died -> freeze at the last action ----
        if stale and self._last_joint_action is not None:
            measured_arm = qpos_rad[list(self._body_dofs)].copy()
            self._last_sent = measured_arm
            self._last_solved = None
            self._last_joint_action = self._identity_action(
                qpos_rad, gripper_0_100
            )
            return self._hold_solution(
                qpos_rad,
                gripper_0_100,
                "stale",
                pos_err_m=self.last_pos_err_m,
                manipulability=self.last_manipulability,
                stale=True,
            )

        # --- Official composition -------------------------------------------
        # T_target = T_arm_ref @ Δ.  The leader's Δp is already expressed in
        # the tracker reference's body frame, so it rotates into the base
        # frame through R_ref -- translation follows the hand orientation.
        target_pos = arm_ref[:3, 3] + arm_ref[:3, :3] @ delta_pos
        r_delta = Rot.from_quat(delta_quat).as_matrix()
        target_rot = arm_ref[:3, :3] @ r_delta

        # Apply the requested gripper coupling to a full candidate state before
        # the arm prefilters.  Thus a recoverable workspace/IK arm hold may let
        # the gripper move only after the new finger pose itself passed safety.
        gripper_candidate = self._adapt_candidate_state(qpos_rad, gripper_0_100)
        if self._candidate_qpos_adapter is not None and self.collision_enabled:
            try:
                if self._candidate_collides(
                    gripper_candidate,
                    escape_from=qpos_rad,
                    allow_non_worsening_escape=True,
                ):
                    return self._hold_solution(
                        qpos_rad,
                        gripper_0_100,
                        "collision-gripper",
                        pos_err_m=self.last_pos_err_m,
                        manipulability=self.last_manipulability,
                        collided=True,
                    )
            except Exception:
                logger.exception("Gripper collision checker failed closed")
                return self._hold_solution(
                    qpos_rad,
                    gripper_0_100,
                    "checker-error",
                    pos_err_m=self.last_pos_err_m,
                    manipulability=self.last_manipulability,
                    collided=True,
                )

        if (
            self._workspace_delta_m is not None
            and float(np.abs(delta_pos).max()) > self._workspace_delta_m
        ):
            return self._hold_solution(
                qpos_rad,
                gripper_0_100,
                "workspace-delta",
                pos_err_m=self.last_pos_err_m,
                manipulability=self.last_manipulability,
            )
        if self._workspace_checker is not None and not self._workspace_checker(
            target_pos, qpos_rad
        ):
            return self._hold_solution(
                qpos_rad,
                gripper_0_100,
                "workspace-arm-base",
                pos_err_m=self.last_pos_err_m,
                manipulability=self.last_manipulability,
            )

        # --- Deterministic multi-seed IK -----------------------------------
        # The fast seed is the last committed/published command, never an IK
        # goal that may be several capped frames ahead of the real robot.
        solve_started = time.perf_counter()
        measured_arm = qpos_rad[list(self._body_dofs)]
        step_start = (
            self._last_sent.copy()
            if self._last_sent is not None
            else measured_arm.copy()
        )
        candidates = []
        last_failure = "ik-nan"
        # Preserve the most informative failure across deterministic seeds.
        # A late inaccurate fallback must not overwrite an accurate solution
        # rejected by collision, which hid the real cause in live logs.
        failure_priority = {
            "checker-error": 100,
            "collision": 90,
            "collision-path": 90,
            "ik-branch-jump": 80,
            "fk-consistency": 70,
            "ik-residual": 60,
            "ik-nan": 50,
            "ik-deadline": 40,
        }
        failures: list[tuple[int, str, float, float, float]] = []

        def remember_failure(reason: str, result=None) -> None:
            failures.append(
                (
                    failure_priority.get(reason, 0),
                    reason,
                    self.last_pos_err_m if result is None else float(result.pos_err),
                    self.last_rot_err_rad if result is None else float(result.rot_err),
                    (
                        self.last_manipulability
                        if result is None
                        else float(result.manipulability)
                    ),
                )
            )

        checker_failed = False
        for seed_index, seed in enumerate(self._candidate_seeds(qpos_rad)):
            if (
                seed_index > 0
                and time.perf_counter() - solve_started >= self._ik_deadline_s
            ):
                last_failure = "ik-deadline"
                remember_failure(last_failure)
                break
            result = self._ik_solver.solve(target_pos, target_rot, seed)
            raw_solution = np.asarray(result.qpos, dtype=float)
            if not np.isfinite(raw_solution).all():
                last_failure = "ik-nan"
                remember_failure(last_failure, result)
                continue
            fk_err = np.abs(result.achieved_pos - target_pos)
            if float(fk_err.max()) > self._fk_consistency_m:
                last_failure = "fk-consistency"
                remember_failure(last_failure, result)
                continue
            # The 0.3 m check above catches corrupt/incompatible kinematics.
            # It is intentionally not an operational accuracy threshold: an
            # inaccurate fallback can be on a completely different branch and
            # was the source of the observed fixed-target oscillation.
            if (
                self._ik_accept_pos_err_m is not None
                and float(result.pos_err) > self._ik_accept_pos_err_m
            ):
                last_failure = "ik-residual"
                remember_failure(last_failure, result)
                continue

            branch_distance = (
                0.0
                if self._last_solved is None
                else float(np.abs(raw_solution - self._last_solved).max())
            )
            jumped_candidate = branch_distance > self._jump_reset_rad
            if jumped_candidate and self._reject_branch_jumps:
                last_failure = "ik-branch-jump"
                remember_failure(last_failure, result)
                continue

            probe = gripper_candidate.copy()
            probe[list(self._body_dofs)] = raw_solution
            target_collided = False
            try:
                if self._candidate_collides(probe, escape_from=qpos_rad):
                    last_failure = "collision"
                    remember_failure(last_failure, result)
                    target_collided = True
            except Exception:
                logger.exception("Collision checker failed closed")
                checker_failed = True
                last_failure = "checker-error"
                remember_failure(last_failure, result)
                break
            clipped_dq = np.clip(
                raw_solution - step_start,
                -self._max_dq_frame_rad,
                self._max_dq_frame_rad,
            )
            step_solution = step_start + clipped_dq
            step_probe = gripper_candidate.copy()
            step_probe[list(self._body_dofs)] = step_solution
            collision_clipped = False
            try:
                if target_collided:
                    safe_probe = self._last_safe_path_state(qpos_rad, step_probe)
                    if safe_probe is None:
                        continue
                    collision_clipped = not np.allclose(
                        safe_probe, step_probe, rtol=0.0, atol=1e-12
                    )
                    step_probe = safe_probe
                    step_solution = safe_probe[list(self._body_dofs)].copy()
                elif self._path_collides(qpos_rad, step_probe):
                    last_failure = "collision-path"
                    remember_failure(last_failure, result)
                    continue
            except Exception:
                logger.exception("Collision path checker failed closed")
                checker_failed = True
                last_failure = "checker-error"
                remember_failure(last_failure, result)
                break

            normalized_distance = float(
                np.linalg.norm(raw_solution - measured_arm) / max(len(raw_solution), 1)
            )
            score = (
                collision_clipped,
                jumped_candidate,
                float(result.pos_err),
                self._rot_weight * float(result.rot_err),
                normalized_distance,
                -float(result.manipulability),
                seed_index,
            )
            candidates.append(
                (
                    score,
                    result,
                    raw_solution,
                    step_solution,
                    jumped_candidate,
                    collision_clipped,
                    target_collided,
                )
            )
            # Normal operation remains a sub-millisecond one-solve fast path.
            if (
                seed_index == 0
                and not collision_clipped
                and not jumped_candidate
                and float(result.pos_err) <= 0.010
            ):
                break

        if not candidates:
            self._warm_seed = None
            if failures:
                _, last_failure, failure_pos_err, failure_rot_err, failure_manip = max(
                    failures, key=lambda failure: failure[0]
                )
            else:
                failure_pos_err = self.last_pos_err_m
                failure_rot_err = self.last_rot_err_rad
                failure_manip = self.last_manipulability
            if last_failure.startswith("collision"):
                self._collision_latched = True
            return self._hold_solution(
                qpos_rad,
                gripper_0_100,
                last_failure,
                pos_err_m=failure_pos_err,
                rot_err_rad=failure_rot_err,
                manipulability=failure_manip,
                collided=last_failure.startswith("collision") or checker_failed,
                jumped=last_failure == "ik-branch-jump",
            )

        (
            _,
            result,
            raw_solution,
            sol_rad,
            jumped,
            collision_clipped,
            target_collided,
        ) = min(
            candidates, key=lambda candidate: candidate[0]
        )
        frame_capped = not np.allclose(
            raw_solution, sol_rad, rtol=0.0, atol=1e-12
        )
        if jumped:
            now = time.time()
            if now - self._last_jump_log_ts > 1.0:
                self._last_jump_log_ts = now
                logger.warning(
                    "IK branch jump %.1fdeg; publishing only the capped, "
                    "path-checked step.",
                    math.degrees(
                        float(np.abs(raw_solution - self._last_solved).max())
                    ) if self._last_solved is not None else 0.0,
                )

        # Commit all cross-frame solve state atomically after the winner and
        # its actual next-step path have passed every gate.
        self._last_solved = (
            sol_rad.copy() if target_collided else raw_solution.copy()
        )
        self._warm_seed = sol_rad.copy()

        self._last_sent = sol_rad.copy()
        self.last_pos_err_m = result.pos_err
        self.last_rot_err_rad = result.rot_err
        self.last_manipulability = result.manipulability

        now = time.time()
        if now - self._last_ik_debug_ts > 1.0:
            self._last_ik_debug_ts = now
            logger.info(
                "IK: pos_err=%.1fmm manip=%.4f q_deg=[%s]",
                result.pos_err * 1000.0, result.manipulability,
                ", ".join("%.1f" % math.degrees(q) for q in sol_rad),
            )

        joint_action = {
            "%s.pos" % joint: math.degrees(float(sol_rad[i]))
            for i, joint in enumerate(self._body_joint_names)
        }
        joint_action["gripper.pos"] = gripper_0_100
        self._last_joint_action = dict(joint_action)
        if collision_clipped:
            self._collision_latched = True
        collision_status = collision_clipped or self._collision_latched
        return JointSolution(
            joint_action,
            result.pos_err,
            False,
            False,
            False,
            collision_status,
            jumped,
            result.manipulability,
            (
                "collision-clipped"
                if collision_clipped
                else "collision-hysteresis"
                if self._collision_latched
                else ""
            ),
            frame_capped,
            result.rot_err,
        )
