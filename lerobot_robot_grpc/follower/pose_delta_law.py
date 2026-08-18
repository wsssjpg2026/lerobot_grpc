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

Follower safety stack (PikaAnyArm piper_IK mechanisms, sim + real identical):

1. IK hard joint limits — DLSIKSolver clips every Newton step into the model
   range, narrowed by the measured calibration range where available.
2. 30° jump rejection — a solution >30° per joint from the previous solved
   one resets the warm start (official ik_fun resets init_data); the next
   solve re-anchors on the measured arm pose instead of chasing the jump.
3. FK consistency — a solution whose FK position deviates >0.3 m from the
   commanded target on any axis is REJECTED (hold last action), the official
   arm_end_pose_callback check.
4. Per-joint per-frame step cap — plays the rate-constraint role of the
   official >30° 200 Hz linear interpolation: a walked-to solution advances
   at most max_dq_frame_deg per action frame (default 6.7° ≈ 200°/s at the
   30 Hz action loop).
5. Self-collision gate — model-capability auto-detect: if the model carries
   usable collision geometry (contype/conaffinity bits on non-adjacent arm
   links), every candidate solution is mj_forward'd and rejected on contact;
   otherwise the gate bypasses with one log line (SO-101 meshes are
   visual-only).  A future robot XML with collision geometry enables it
   with no code change.

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
        Per-joint per-action-frame cap on the published step — the official
        >30° 200 Hz interpolation equivalent.  6.7° = 200°/s at 30 Hz.
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
        jump_reset_deg: float = 30.0,
        max_dq_frame_deg: float = 6.7,
        gripper_max_distance_mm: float = 60.0,
    ):
        import mujoco

        self._mj = mujoco
        self._model = model
        self._body_dofs = tuple(int(d) for d in body_dofs)
        self._body_joint_names = tuple(body_joint_names)
        self._n_body = len(self._body_dofs)
        self._gripper_max_distance_mm = float(gripper_max_distance_mm)
        self._fk_consistency_m = float(fk_consistency_m)
        self._jump_reset_rad = math.radians(float(jump_reset_deg))
        self._max_dq_frame_rad = math.radians(float(max_dq_frame_deg))

        # FK / collision scratch (never the sim live data -- seeded per call).
        self._fk_data = mujoco.MjData(model)

        self._ik_solver = DLSIKSolver(
            model,
            site_name=site_name,
            body_dofs=list(self._body_dofs),
            rot_weight=rot_weight,
            rest_qpos=np.radians(np.asarray(home_joints_deg, dtype=float)),
            rest_gain=rest_gain,
            max_dq_rad=math.radians(max_dq_deg),
        )
        self._site_id = self._ik_solver.site_id

        # --- Self-collision capability (auto-detect) ----------------------
        self._collision_pairs = self._detect_collision_pairs()
        if self._collision_pairs:
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
            "fk_consistency=%.1fm jump_reset=%.0fdeg frame_cap=%.1fdeg "
            "collision_pairs=%d",
            site_name, self._body_dofs, rot_weight,
            self._fk_consistency_m, jump_reset_deg, max_dq_frame_deg,
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
        return bool(self._collision_pairs)

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
        self.last_manipulability = 0.0

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
            "%s.pos" % joint: math.degrees(float(qpos_rad[i]))
            for i, joint in enumerate(self._body_joint_names)
        }
        action["gripper.pos"] = gripper_0_100
        return action

    # ------------------------------------------------------------------
    # The law
    # ------------------------------------------------------------------

    def solve(self, delta_action, qpos_rad, *, stale: bool = False) -> JointSolution:
        """Compose T_target = T_arm_ref @ Δ, solve DLS IK, apply official checks.

        Pipeline: compose target -> DLS solve (warm start) -> FK consistency
        (0.3 m, reject/hold) -> self-collision gate (reject/hold) -> 30° jump
        warm-start reset -> per-joint per-frame step cap -> publish.
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

        if self._arm_ref is None:
            self.lock_reference(qpos_rad)
        arm_ref = self._arm_ref
        assert arm_ref is not None

        # --- Stale hold: leader stream died -> freeze at the last action ----
        if stale and self._last_joint_action is not None:
            ja = dict(self._last_joint_action)
            ja["gripper.pos"] = gripper_0_100
            return JointSolution(
                ja, self.last_pos_err_m, True, True, False, False, False,
                self.last_manipulability,
            )

        # --- Official composition -------------------------------------------
        # T_target = T_arm_ref @ Δ.  The leader's Δp is already expressed in
        # the tracker reference's body frame, so it rotates into the base
        # frame through R_ref -- translation follows the hand orientation.
        target_pos = arm_ref[:3, 3] + arm_ref[:3, :3] @ delta_pos
        r_delta = Rot.from_quat(delta_quat).as_matrix()
        target_rot = arm_ref[:3, :3] @ r_delta

        # --- DLS solve, warm-started ----------------------------------------
        seed = np.array(qpos_rad, dtype=float)
        if self._warm_seed is not None:
            seed[: len(self._warm_seed)] = self._warm_seed
        result = self._ik_solver.solve(target_pos, target_rot, seed)
        sol_rad = np.asarray(result.qpos, dtype=float)

        def _hold(reason: str, *, collided: bool = False) -> JointSolution:
            """Reject the solution: keep the last published joints."""
            now = time.time()
            if now - self._last_reject_log_ts > 1.0:
                self._last_reject_log_ts = now
                logger.warning(
                    "SOLVE rejected (%s): pos_err=%.0fmm -- holding last action.",
                    reason, result.pos_err * 1000.0,
                )
            if self._last_joint_action is not None:
                ja = dict(self._last_joint_action)
            else:
                ja = self._identity_action(qpos_rad, gripper_0_100)
            ja["gripper.pos"] = gripper_0_100
            return JointSolution(
                ja, result.pos_err, True, False, True, collided,
                False, result.manipulability,
            )

        if not np.isfinite(sol_rad).all():
            return _hold("ik-nan")

        # --- Official check 3: FK consistency (reject >0.3 m off-target) ----
        fk_err = np.abs(result.achieved_pos - target_pos)
        if float(fk_err.max()) > self._fk_consistency_m:
            return _hold("fk-consistency")

        # --- Official check 5: self-collision gate (capability-gated) -------
        if self._collision_pairs:
            probe = np.array(qpos_rad, dtype=float)
            probe[: len(sol_rad)] = sol_rad
            if self._in_self_collision(probe):
                return _hold("collision", collided=True)

        # --- Official check 2: 30° jump rejection (warm-start reset) --------
        jumped = False
        if (
            self._last_solved is not None
            and float(np.abs(sol_rad - self._last_solved).max()) > self._jump_reset_rad
        ):
            jumped = True
            self._warm_seed = None  # next solve re-anchors on measured qpos
            now = time.time()
            if now - self._last_jump_log_ts > 1.0:
                self._last_jump_log_ts = now
                logger.warning(
                    "IK jump: solution %.1f° from the previous — warm start "
                    "reset (official 30° rule).",
                    math.degrees(float(np.abs(sol_rad - self._last_solved).max())),
                )
        else:
            self._warm_seed = sol_rad.copy()
        self._last_solved = sol_rad.copy()

        # --- Official check 4: per-joint per-frame step cap -----------------
        # Cap base = last PUBLISHED step; on the first frame after a latch the
        # official interpolation path seeds from the MEASURED joints
        # (piper_IK arm_joint_state_ctrl_linear_interpolation), so the first
        # published step is capped from the arm's actual pose.
        if self._last_sent is None:
            self._last_sent = np.array(qpos_rad[: self._n_body], dtype=float).copy()
        dq = sol_rad - self._last_sent
        sol_rad = self._last_sent + np.clip(dq, -self._max_dq_frame_rad, self._max_dq_frame_rad)

        self._last_sent = sol_rad.copy()
        self.last_pos_err_m = result.pos_err
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
        return JointSolution(
            joint_action, result.pos_err, False, False, False, False, jumped,
            result.manipulability,
        )
