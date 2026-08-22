"""MuJoCo-backed Galaxy General Galbot S1 pose-delta follower.

The public robot interface is deliberately small: a namespaced single-hand
pose intent enters through ``SendAction`` and the complete independently
controlled S1 state leaves through ``GetObservation`` in native SI units.
All MuJoCo qpos and actuator addresses are resolved by name because the S1
model intentionally uses different qpos and actuator orderings.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np
from google.protobuf.empty_pb2 import Empty

from lerobot_robot_grpc.action_safety import AppliedGroup, SafetyFlag, groups_for_arm
from lerobot_robot_grpc.follower.collision_distance import CollisionConstraint
from lerobot_robot_grpc.follower.follower_server import FollowerServicer
from lerobot_robot_grpc.follower.pose_delta_law import PoseDeltaLaw
from lerobot_robot_grpc.follower.utils import encode_feature, load_feature
from lerobot_robot_grpc.pose_delta_schema import (
    ACTION_KEYS,
    action_keys,
    build_pose_delta_feature_info,
)
from lerobot_robot_grpc.protos import device_pb2

logger = logging.getLogger(__name__)

LEFT_HOME_RAD: tuple[float, ...] = (
    1.1459823,
    -1.6041308,
    -0.4108652,
    -2.2274825,
    -0.0996052,
    0.0939316,
    -0.1995033,
)
RIGHT_HOME_RAD: tuple[float, ...] = tuple(-value for value in LEFT_HOME_RAD)

ARM_JOINTS: dict[str, tuple[str, ...]] = {
    side: tuple(f"{side}_arm_joint{i}" for i in range(1, 8))
    for side in ("left", "right")
}
EFFECTIVE_ACTION_JOINTS: dict[str, tuple[str, ...]] = {
    side: (*ARM_JOINTS[side], f"{side}_active_joint1")
    for side in ("left", "right")
}
ACTIVE_GRIPPERS: tuple[str, ...] = (
    "left_active_joint1",
    "right_active_joint1",
)
OBSERVATION_JOINTS: tuple[str, ...] = (
    *ARM_JOINTS["left"],
    *ARM_JOINTS["right"],
    *ACTIVE_GRIPPERS,
    "head_joint1",
    "head_joint2",
    "torso_lift_joint1",
    "base_x_joint",
    "base_y_joint",
    "base_yaw_joint",
)

_PHYSICS_PERIOD_S = 1.0 / 50.0
GRIPPER_CLOSED_RAD = 1.6357
DEFAULT_GRIPPER_MAX_DISTANCE_MM = 60.0
S1_TELEOP_REACH_M = 1.0844833988
_NEAR_TORSO_ENTRY_MARGIN_M = 0.002
_NEAR_TORSO_RELEASE_MARGIN_M = 0.005


def _scalar_feature_info(key: str) -> device_pb2.OneFeatureInfo:
    return device_pb2.OneFeatureInfo(
        key=key,
        criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
        watchdog=device_pb2.WatchDogLevel.WATCH_DOG_LEVEL_A,
        type=device_pb2.DataType.FLOAT32,
        shape=device_pb2.ImageShape(H=1, W=1, C=1),
        encoding=device_pb2.Encoding.RAW,
        img_quality=100,
    )


def _load_model_with_teleop_sites(xml_path: str | Path):
    """Compile the untouched vendored MJCF with runtime-only EE sites."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml_path))
    for side in ("left", "right"):
        body = spec.body(f"{side}_gripper_base_link")
        if body is None:
            raise ValueError(f"S1 body '{side}_gripper_base_link' not found")
        body.add_site(
            name=f"{side}_teleop_site",
            pos=(0.0, 0.0, 0.0),
            size=(0.005,),
        )
    return spec.compile()


def _arm_collision_link(body_name: str, arm: str) -> int | None:
    """Map an S1 collision body to its reduced arm-link number."""
    prefix = f"{arm}_arm_link"
    if body_name.startswith(prefix):
        suffix = body_name.removeprefix(prefix)
        if suffix.isdigit():
            return int(suffix)
    if body_name.startswith(
        (
            f"{arm}_arm_camera",
            f"{arm}_adapter_flange",
            f"{arm}_gripper_base",
            f"{arm}_active_link",
            f"{arm}_passive_link",
        )
    ):
        return 7
    return None


@dataclass(frozen=True)
class CollisionResult:
    collided: bool
    body_a: str = ""
    body_b: str = ""
    distance_m: float = float("inf")


class S1CollisionChecker:
    """Independent semantic full-body S1 collision candidate checker.

    The upstream S1 MJCF deliberately disables robot self-contact.  Enabling
    every robot geom in the live physics model creates unrelated contacts
    (notably inside each coupled gripper), so safety uses a second model with
    Every upstream group-3 collision geom is included, plus a floor plane.
    Only pairs involving the selected arm/wrist/fingers can reject a command;
    fixed-fixed assembly and wheel-floor contacts therefore cannot deadlock
    teleoperation.  The live simulation's contact masks remain untouched.
    """

    supports_hysteresis = True

    def __init__(
        self,
        xml_path: str | Path,
        *,
        arm: str,
        margin_m: float = 0.005,
        self_soft_distance_ratio: float = 0.06,
        cross_arm_soft_distance_ratio: float = 0.10,
        reach_m: float = S1_TELEOP_REACH_M,
    ) -> None:
        if arm not in ARM_JOINTS:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        if margin_m < 0.0:
            raise ValueError("collision margin must be non-negative")
        if reach_m <= 0.0:
            raise ValueError("collision reach must be positive")
        if self_soft_distance_ratio <= 0.0:
            raise ValueError("self soft-distance ratio must be positive")
        if cross_arm_soft_distance_ratio <= 0.0:
            raise ValueError("cross-arm soft-distance ratio must be positive")

        import mujoco

        opposite = "right" if arm == "left" else "left"
        spec = mujoco.MjSpec.from_file(str(xml_path))
        # MuJoCo must generate contacts out to the *release* threshold so the
        # controller can implement a Schmitt trigger: ordinary pairs enter at
        # 5 mm and release at 8 mm; cross-arm pairs enter at 10 mm and release
        # at 15 mm.  ``check(..., release=False)`` filters these generated
        # contacts back down to the entry threshold.
        self._entry_margin_m = float(margin_m)
        self._release_margin_m = max(0.008, self._entry_margin_m)
        self._cross_entry_margin_m = max(0.010, self._entry_margin_m)
        self._cross_release_margin_m = max(0.015, self._cross_entry_margin_m)
        # The official link2/torso collision proxies are deliberately much
        # more conservative than the rendered shells around the shoulder.
        # Give this known near-field pair its own Schmitt band instead of
        # weakening every body, floor and cross-arm collision threshold.
        self._near_torso_entry_margin_m = _NEAR_TORSO_ENTRY_MARGIN_M
        self._near_torso_release_margin_m = _NEAR_TORSO_RELEASE_MARGIN_M
        self._self_soft_distance_m = float(reach_m) * float(
            self_soft_distance_ratio
        )
        self._cross_soft_distance_m = float(reach_m) * float(
            cross_arm_soft_distance_ratio
        )
        if self._self_soft_distance_m < self._release_margin_m:
            raise ValueError("self soft distance must cover the hard release margin")
        if self._cross_soft_distance_m < self._cross_release_margin_m:
            raise ValueError(
                "cross-arm soft distance must cover the hard release margin"
            )
        # The hard model only generates millimetre-range contacts.  A private
        # soft copy below uses the larger predictive band so swept-path checks
        # do not repeatedly pay the soft broad-phase cost.
        moving_margin = self._release_margin_m / 2.0
        fixed_margin = self._release_margin_m - moving_margin
        opposite_margin = max(
            self._cross_release_margin_m - moving_margin,
            fixed_margin,
        )
        selected_count = 0
        moving_body_links: dict[str, int] = {}
        fixed_body_names: set[str] = set()
        for geom in spec.geoms:
            body_name = geom.parent.name
            is_collision_geom = int(geom.group) == 3
            moving = (
                is_collision_geom
                and _arm_collision_link(body_name, arm) is not None
            )
            fixed = is_collision_geom and not moving
            # bit 1: moving/moving; bit 2 against bit-1 affinity:
            # moving/fixed.  Fixed/fixed cannot collide.
            geom.contype = 1 if moving else 2 if fixed else 0
            geom.conaffinity = 1 if moving or fixed else 0
            if moving:
                geom.margin = moving_margin
                moving_body_links[body_name] = int(
                    _arm_collision_link(body_name, arm)
                )
            elif fixed and _arm_collision_link(body_name, opposite) is not None:
                geom.margin = opposite_margin
                fixed_body_names.add(body_name)
            elif fixed:
                geom.margin = fixed_margin
                fixed_body_names.add(body_name)
            else:
                geom.margin = 0.0
            if moving or fixed:
                selected_count += 1

        # Do not ask MuJoCo broad phase to generate contacts that the semantic
        # policy will always discard.  This is especially important with a
        # centimetre-scale soft band: the wrist rigid assembly and coupled
        # gripper otherwise create hundreds of irrelevant predictive contacts.
        moving_bodies = sorted(moving_body_links)
        exclude_index = 0
        for index, body_a in enumerate(moving_bodies):
            for body_b in moving_bodies[index + 1 :]:
                if abs(moving_body_links[body_a] - moving_body_links[body_b]) < 3:
                    spec.add_exclude(
                        name=f"teleop_allow_{exclude_index}",
                        bodyname1=body_a,
                        bodyname2=body_b,
                    )
                    exclude_index += 1
        link1_bodies = [
            body
            for body, link in moving_body_links.items()
            if link == 1
        ]
        for body_a in sorted(link1_bodies):
            for body_b in (f"{arm}_arm_base_link", "torso_base_link"):
                if body_b not in fixed_body_names:
                    continue
                spec.add_exclude(
                    name=f"teleop_allow_{exclude_index}",
                    bodyname1=body_a,
                    bodyname2=body_b,
                )
                exclude_index += 1

        floor = spec.worldbody.add_geom(
            name="teleop_safety_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=(0.0, 0.0, 0.0),
            size=(3.0, 3.0, 0.1),
            contype=2,
            conaffinity=1,
            margin=fixed_margin,
            group=3,
        )
        del floor
        selected_count += 1

        self._mj = mujoco
        soft_spec = spec.copy()
        soft_moving_margin = self._self_soft_distance_m / 2.0
        soft_fixed_margin = self._self_soft_distance_m - soft_moving_margin
        soft_opposite_margin = max(
            self._cross_soft_distance_m - soft_moving_margin,
            soft_fixed_margin,
        )
        for geom in soft_spec.geoms:
            body_name = geom.parent.name
            if int(geom.contype) == 1:
                geom.margin = soft_moving_margin
            elif (
                int(geom.contype) == 2
                and _arm_collision_link(body_name, opposite) is not None
            ):
                geom.margin = soft_opposite_margin
            elif int(geom.contype) == 2:
                geom.margin = soft_fixed_margin
            else:
                geom.margin = 0.0
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        self._soft_model = soft_spec.compile()
        self._soft_data = mujoco.MjData(self._soft_model)
        self._arm = arm
        self._opposite = opposite
        self._controlled_dofs = np.asarray(
            [
                int(
                    self._model.jnt_dofadr[
                        mujoco.mj_name2id(
                            self._model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    ]
                )
                for name in ARM_JOINTS[arm]
            ],
            dtype=int,
        )
        self._moving_link_by_geom: dict[int, int] = {}
        self._fixed_geoms: set[int] = set()
        self._body_name_by_geom: dict[int, str] = {}
        for geom_id in range(self._model.ngeom):
            body_id = int(self._model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if body_name is None:
                continue
            self._body_name_by_geom[geom_id] = body_name
            link = _arm_collision_link(body_name, arm)
            if link is not None and int(self._model.geom_contype[geom_id]) == 1:
                self._moving_link_by_geom[geom_id] = link
            elif int(self._model.geom_contype[geom_id]) == 2:
                self._fixed_geoms.add(geom_id)

        if not self._moving_link_by_geom or not self._fixed_geoms:
            raise ValueError("S1 collision model did not expose the expected geoms")
        logger.info(
            "S1 collision checker ready: arm=%s entry/release=%.1f/%.1fmm "
            "near-torso=%.1f/%.1fmm cross-arm=%.1f/%.1fmm geoms=%d",
            arm,
            self._entry_margin_m * 1000.0,
            self._release_margin_m * 1000.0,
            self._near_torso_entry_margin_m * 1000.0,
            self._near_torso_release_margin_m * 1000.0,
            self._cross_entry_margin_m * 1000.0,
            self._cross_release_margin_m * 1000.0,
            selected_count,
        )

    def active_constraints(
        self, qpos_rad: np.ndarray
    ) -> tuple[CollisionConstraint, ...]:
        """Return all semantic S1 pairs currently inside a soft distance.

        MuJoCo's predictive contacts supply a mutually consistent signed
        distance, midpoint, and normal.  Reconstructed surface points and
        their point Jacobians then map the normal into the seven controlled
        arm joints.  Multiple convex pieces of the same body pair collapse to
        the most severe constraint.
        """
        qpos = np.asarray(qpos_rad, dtype=float)
        if qpos.shape != (self._soft_model.nq,):
            raise ValueError(
                f"expected S1 qpos shape {(self._soft_model.nq,)}, got {qpos.shape}"
            )
        self._soft_data.qpos[:] = qpos
        self._mj.mj_forward(self._soft_model, self._soft_data)
        active: dict[tuple[str, str], CollisionConstraint] = {}
        nv = self._soft_model.nv
        for contact_id in range(self._soft_data.ncon):
            contact = self._soft_data.contact[contact_id]
            oriented = self._oriented_semantic_pair(
                int(contact.geom1), int(contact.geom2)
            )
            if oriented is None:
                continue
            geom_a, geom_b, body_a, body_b, moving_link, cross_arm = oriented
            activation = (
                self._cross_soft_distance_m
                if cross_arm
                else self._self_soft_distance_m
            )
            # The contact is authoritative.  Re-querying a predictive mesh
            # contact with mj_geomDistance can select a different convex
            # piece and occasionally return a false zero distance.  That
            # discontinuity makes the null-space avoidance term amplify
            # sub-millimetre tracker noise into multi-degree joint motion.
            distance = float(contact.dist)
            if distance > activation:
                continue
            midpoint = np.asarray(contact.pos, dtype=float)
            normal = np.asarray(contact.frame[:3], dtype=float)
            if (geom_a, geom_b) != (int(contact.geom1), int(contact.geom2)):
                normal = -normal
            if not (
                np.isfinite(distance)
                and np.isfinite(midpoint).all()
                and np.isfinite(normal).all()
            ):
                raise ValueError("MuJoCo produced a non-finite predictive contact")
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-12:
                raise ValueError("MuJoCo produced a zero-length contact normal")
            normal /= normal_norm
            point_a = midpoint - 0.5 * distance * normal
            point_b = midpoint + 0.5 * distance * normal
            jac_a = np.zeros((3, nv), dtype=float)
            jac_b = np.zeros((3, nv), dtype=float)
            body_a_id = int(self._soft_model.geom_bodyid[geom_a])
            body_b_id = int(self._soft_model.geom_bodyid[geom_b])
            self._mj.mj_jac(
                self._soft_model,
                self._soft_data,
                jac_a,
                None,
                point_a,
                body_a_id,
            )
            self._mj.mj_jac(
                self._soft_model,
                self._soft_data,
                jac_b,
                None,
                point_b,
                body_b_id,
            )
            gradient = normal @ (jac_b - jac_a)
            minimum = self._hard_entry_distance(moving_link, body_b, cross_arm)
            constraint = CollisionConstraint(
                body_a=body_a,
                body_b=body_b,
                distance_m=distance,
                activation_distance_m=activation,
                minimum_distance_m=minimum,
                gradient=np.asarray(gradient[self._controlled_dofs], dtype=float),
            )
            key = (body_a, body_b)
            previous = active.get(key)
            if previous is None or constraint.distance_m < previous.distance_m:
                active[key] = constraint
        return tuple(
            sorted(
                active.values(),
                key=lambda item: (
                    item.distance_m / item.activation_distance_m,
                    item.body_a,
                    item.body_b,
                ),
            )
        )

    def _oriented_semantic_pair(
        self, geom1: int, geom2: int
    ) -> tuple[int, int, str, str, int, bool] | None:
        link1 = self._moving_link_by_geom.get(geom1)
        link2 = self._moving_link_by_geom.get(geom2)
        body1 = self._body_name_by_geom.get(geom1, "world")
        body2 = self._body_name_by_geom.get(geom2, "world")
        if link1 is not None and geom2 in self._fixed_geoms:
            if self._allowed_mount_pair(link1, body2):
                return None
            return (
                geom1,
                geom2,
                body1,
                body2,
                link1,
                _arm_collision_link(body2, self._opposite) is not None,
            )
        if link2 is not None and geom1 in self._fixed_geoms:
            if self._allowed_mount_pair(link2, body1):
                return None
            return (
                geom2,
                geom1,
                body2,
                body1,
                link2,
                _arm_collision_link(body1, self._opposite) is not None,
            )
        if link1 is not None and link2 is not None and abs(link1 - link2) >= 3:
            if (body2, body1) < (body1, body2):
                return geom2, geom1, body2, body1, link2, False
            return geom1, geom2, body1, body2, link1, False
        return None

    def _hard_entry_distance(
        self, moving_link: int, fixed_body: str, cross_arm: bool
    ) -> float:
        if cross_arm:
            return self._cross_entry_margin_m
        if moving_link == 2 and fixed_body == "torso_base_link":
            return self._near_torso_entry_margin_m
        return self._entry_margin_m

    def check(
        self,
        qpos_rad: np.ndarray,
        *,
        release: bool = False,
    ) -> CollisionResult:
        qpos = np.asarray(qpos_rad, dtype=float)
        if qpos.shape != (self._model.nq,):
            raise ValueError(
                f"expected S1 qpos shape {(self._model.nq,)}, got {qpos.shape}"
            )
        self._data.qpos[:] = qpos
        self._mj.mj_forward(self._model, self._data)
        collisions: list[CollisionResult] = []
        for contact_id in range(self._data.ncon):
            contact = self._data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            link1 = self._moving_link_by_geom.get(geom1)
            link2 = self._moving_link_by_geom.get(geom2)
            body1 = self._body_name_by_geom.get(geom1, "world")
            body2 = self._body_name_by_geom.get(geom2, "world")
            if link1 is not None and geom2 in self._fixed_geoms:
                if (
                    not self._allowed_mount_pair(link1, body2)
                    and self._within_threshold(
                        float(contact.dist),
                        moving_link=link1,
                        fixed_body=body2,
                        release=release,
                    )
                ):
                    collisions.append(
                        CollisionResult(True, body1, body2, float(contact.dist))
                    )
            if link2 is not None and geom1 in self._fixed_geoms:
                if (
                    not self._allowed_mount_pair(link2, body1)
                    and self._within_threshold(
                        float(contact.dist),
                        moving_link=link2,
                        fixed_body=body1,
                        release=release,
                    )
                ):
                    collisions.append(
                        CollisionResult(True, body2, body1, float(contact.dist))
                    )
            # S1's compact wrist has deliberate overlap between links 5 and
            # 7 at the safe home.  Treat links within two chain steps as
            # kinematic neighbours; more distant links are self-collision.
            if (
                link1 is not None
                and link2 is not None
                and abs(link1 - link2) >= 3
                and float(contact.dist)
                <= (
                    self._release_margin_m
                    if release
                    else self._entry_margin_m
                )
            ):
                collisions.append(
                    CollisionResult(True, body1, body2, float(contact.dist))
                )
        if not collisions:
            return CollisionResult(False)
        # Stable, most-severe diagnostic.  Pair names break exact-distance
        # ties deterministically across platforms.
        return min(
            collisions,
            key=lambda item: (item.distance_m, item.body_a, item.body_b),
        )

    def _within_threshold(
        self,
        distance_m: float,
        moving_link: int,
        fixed_body: str,
        *,
        release: bool,
    ) -> bool:
        cross_arm = _arm_collision_link(fixed_body, self._opposite) is not None
        if cross_arm:
            threshold = (
                self._cross_release_margin_m
                if release
                else self._cross_entry_margin_m
            )
        elif moving_link == 2 and fixed_body == "torso_base_link":
            threshold = (
                self._near_torso_release_margin_m
                if release
                else self._near_torso_entry_margin_m
            )
        else:
            threshold = (
                self._release_margin_m
                if release
                else self._entry_margin_m
            )
        return distance_m <= threshold

    def _allowed_mount_pair(self, moving_link: int, fixed_body: str) -> bool:
        # Normal shoulder assembly contacts.  All other selected-arm contacts
        # with chassis, wheels, column, torso, head, opposite arm or floor are
        # safety-relevant.
        return moving_link == 1 and fixed_body in {
            f"{self._arm}_arm_base_link",
            "torso_base_link",
        }

    def __call__(self, qpos_rad: np.ndarray) -> bool:
        return self.check(qpos_rad).collided


class S1ArmWorkspace:
    """Conservative arm-base-local Cartesian prefilter.

    Ratios are normalized by the maximum arm-base to teleop-site reach, so
    the same policy can be carried to another embodiment after supplying its
    reach.  This is only a cheap prefilter; IK, limits and collision remain
    authoritative for each candidate.
    """

    def __init__(
        self,
        model,
        *,
        arm: str,
        reach_m: float = S1_TELEOP_REACH_M,
        xy_limit_ratio: float = 0.85,
        z_min_ratio: float = -0.60,
        z_max_ratio: float = 0.85,
        radius_min_ratio: float = 0.20,
        radius_max_ratio: float = 0.85,
    ) -> None:
        import mujoco

        if reach_m <= 0.0:
            raise ValueError("workspace reach must be positive")
        self._mj = mujoco
        self._model = model
        self._data = mujoco.MjData(model)
        self._body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_arm_base_link"
        )
        if self._body_id < 0:
            raise ValueError(f"S1 {arm}_arm_base_link not found")
        self._reach_m = float(reach_m)
        self._xy_limit_ratio = float(xy_limit_ratio)
        self._z_min_ratio = float(z_min_ratio)
        self._z_max_ratio = float(z_max_ratio)
        self._radius_min_ratio = float(radius_min_ratio)
        self._radius_max_ratio = float(radius_max_ratio)

    def local_position(
        self, target_world_m: np.ndarray, qpos_rad: np.ndarray
    ) -> np.ndarray:
        self._data.qpos[:] = np.asarray(qpos_rad, dtype=float)
        self._mj.mj_forward(self._model, self._data)
        position = self._data.xpos[self._body_id]
        rotation = self._data.xmat[self._body_id].reshape(3, 3)
        return rotation.T @ (np.asarray(target_world_m, dtype=float) - position)

    def __call__(self, target_world_m: np.ndarray, qpos_rad: np.ndarray) -> bool:
        local_n = self.local_position(target_world_m, qpos_rad) / self._reach_m
        radius_n = float(np.linalg.norm(local_n))
        return bool(
            abs(float(local_n[0])) <= self._xy_limit_ratio
            and abs(float(local_n[1])) <= self._xy_limit_ratio
            and self._z_min_ratio <= float(local_n[2]) <= self._z_max_ratio
            and self._radius_min_ratio <= radius_n <= self._radius_max_ratio
        )


class MuJoCoS1Servicer(FollowerServicer):
    """Single-arm pose-delta adapter over the full S1 MuJoCo model."""

    def __init__(
        self,
        xml_path: str | Path,
        *,
        arm: str = "left",
        render: bool = False,
        reset_on_connect: bool = False,
        ctrl_smoothing_alpha: float = 0.20,
        rot_weight: float = 0.3,
        max_dq_deg: float = 6.0,
        max_dq_frame_deg: float = 2.291831,
        gripper_max_distance_mm: float = DEFAULT_GRIPPER_MAX_DISTANCE_MM,
        collision_margin_m: float = 0.005,
        self_soft_distance_ratio: float = 0.06,
        cross_arm_soft_distance_ratio: float = 0.10,
        stale_timeout_s: float = 0.5,
        torso_home_m: float = 0.6,
        collision_aware_ik: bool = True,
        reference_grace_s: float = 0.5,
    ) -> None:
        if arm not in ARM_JOINTS:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")

        import mujoco

        self._mj = mujoco
        self._model = _load_model_with_teleop_sites(xml_path)
        self._data = mujoco.MjData(self._model)
        self._arm = arm
        self._lock = threading.Lock()
        self._connected = False
        self._reset_on_connect = bool(reset_on_connect)
        self._ctrl_smoothing_alpha = float(ctrl_smoothing_alpha)
        if stale_timeout_s < 0.0:
            raise ValueError("stale timeout must be non-negative")
        self._stale_timeout_s = float(stale_timeout_s)
        if reference_grace_s < 0.0:
            raise ValueError("reference grace must be non-negative")
        self._collision_aware_ik = bool(collision_aware_ik)
        self._reference_grace_s = float(reference_grace_s)
        self._reference_locked_monotonic = time.monotonic()
        self._last_action_monotonic: float | None = None
        self._watchdog_held = False
        self._session_hold_latched = False
        self._hold_epoch = 0
        self._reference_epoch = 0
        self._hold_reference_epoch = 0
        self._hold_reason = ""

        self._qpos_by_joint = {
            name: self._qpos_address(name) for name in OBSERVATION_JOINTS
        }
        self._coupled_gripper_qpos = {
            name: self._qpos_address(name)
            for name in (
                f"{arm}_active_joint1",
                f"{arm}_active_joint2",
                f"{arm}_passive_joint",
            )
        }
        self._actuator_by_name = {
            name: self._actuator_id(name) for name in OBSERVATION_JOINTS
        }
        self._obs_ft_info = {
            f"{name}.pos": _scalar_feature_info(f"{name}.pos")
            for name in OBSERVATION_JOINTS
        }
        torso_joint_id = self._mj.mj_name2id(
            self._model,
            self._mj.mjtObj.mjOBJ_JOINT,
            "torso_lift_joint1",
        )
        torso_min_m, torso_max_m = self._model.jnt_range[torso_joint_id]
        if not torso_min_m <= torso_home_m <= torso_max_m:
            raise ValueError(
                "torso home must be within "
                f"[{torso_min_m:.3f}, {torso_max_m:.3f}] m, got {torso_home_m:.3f}"
            )
        self._torso_home_m = float(torso_home_m)
        self._act_ft_info = build_pose_delta_feature_info(prefix=arm)
        self._effective_act_ft_info = {
            f"{name}.pos": _scalar_feature_info(f"{name}.pos")
            for name in EFFECTIVE_ACTION_JOINTS[arm]
        }
        self._identity_action = {
            key: (
                1.0
                if key.endswith("delta_rot.qw")
                else gripper_max_distance_mm
                if key.endswith("gripper.distance")
                else 0.0
            )
            for key in self._act_ft_info
        }
        self._latest_action = dict(self._identity_action)

        self._target_ctrl = np.zeros(self._model.nu, dtype=float)
        self._set_home()
        self._reject_streak = 0
        self._reject_started_monotonic: float | None = None
        self._last_safety = device_pb2.SafetyReport()

        arm_home = LEFT_HOME_RAD if arm == "left" else RIGHT_HOME_RAD
        self._collision_checker = S1CollisionChecker(
            xml_path,
            arm=arm,
            margin_m=collision_margin_m,
            self_soft_distance_ratio=self_soft_distance_ratio,
            cross_arm_soft_distance_ratio=cross_arm_soft_distance_ratio,
        )
        self._workspace = S1ArmWorkspace(self._model, arm=arm)
        self._law = PoseDeltaLaw(
            self._model,
            site_name=f"{arm}_teleop_site",
            body_dofs=[self._qpos_by_joint[name] for name in ARM_JOINTS[arm]],
            body_joint_names=ARM_JOINTS[arm],
            home_joints_deg=np.degrees(np.asarray(arm_home, dtype=float)),
            rot_weight=rot_weight,
            max_dq_deg=max_dq_deg,
            ik_accept_pos_err_m=0.010,
            reject_branch_jumps=True,
            max_dq_frame_deg=max_dq_frame_deg,
            gripper_max_distance_mm=gripper_max_distance_mm,
            collision_checker=self._collision_checker,
            collision_distance_provider=self._collision_checker,
            collision_aware_ik=collision_aware_ik,
            candidate_qpos_adapter=self._with_gripper_candidate,
            workspace_checker=self._workspace,
        )
        self._teleop_site_id = self._mj.mj_name2id(
            self._model,
            self._mj.mjtObj.mjOBJ_SITE,
            f"{arm}_teleop_site",
        )
        if self._teleop_site_id < 0:
            raise ValueError(f"S1 {arm}_teleop_site not found")
        self._last_control_debug_monotonic = 0.0
        self._law.lock_reference(self._data.qpos.copy())
        self._reference_locked_monotonic = time.monotonic()
        logger.info(
            "S1 collision-aware IK %s: self/cross soft bands=%.1f%%/%.1f%% "
            "reference grace=%.3fs (hard collision gate always enabled)",
            "ENABLED" if self._collision_aware_ik else "DISABLED",
            self_soft_distance_ratio * 100.0,
            cross_arm_soft_distance_ratio * 100.0,
            self._reference_grace_s,
        )

        self._viewer = None
        if render:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def _qpos_address(self, joint_name: str) -> int:
        jid = self._mj.mj_name2id(
            self._model, self._mj.mjtObj.mjOBJ_JOINT, joint_name
        )
        if jid < 0:
            raise ValueError(f"S1 joint {joint_name!r} not found")
        return int(self._model.jnt_qposadr[jid])

    def _actuator_id(self, actuator_name: str) -> int:
        aid = self._mj.mj_name2id(
            self._model, self._mj.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if aid < 0:
            raise ValueError(f"S1 actuator {actuator_name!r} not found")
        return int(aid)

    def _set_home(self) -> None:
        self._mj.mj_resetData(self._model, self._data)
        home_by_joint = {
            **dict(zip(ARM_JOINTS["left"], LEFT_HOME_RAD, strict=True)),
            **dict(zip(ARM_JOINTS["right"], RIGHT_HOME_RAD, strict=True)),
            "torso_lift_joint1": self._torso_home_m,
        }
        for name in OBSERVATION_JOINTS:
            value = float(home_by_joint.get(name, 0.0))
            self._data.qpos[self._qpos_by_joint[name]] = value
            actuator = self._actuator_by_name[name]
            # The three base actuators are velocity actuators; zero is their
            # hold command. All other exposed actuators are position targets.
            self._target_ctrl[actuator] = 0.0 if name.startswith("base_") else value
        self._data.ctrl[:] = self._target_ctrl
        self._mj.mj_forward(self._model, self._data)

    def _with_gripper_candidate(
        self, qpos_rad: np.ndarray, gripper_open_0_100: float
    ) -> np.ndarray:
        """Apply the S1 master/mimic gripper coupling to a collision probe."""
        probe = np.asarray(qpos_rad, dtype=float).copy()
        master = (
            1.0 - np.clip(gripper_open_0_100, 0.0, 100.0) / 100.0
        ) * GRIPPER_CLOSED_RAD
        probe[self._coupled_gripper_qpos[f"{self._arm}_active_joint1"]] = master
        probe[self._coupled_gripper_qpos[f"{self._arm}_active_joint2"]] = -master
        probe[self._coupled_gripper_qpos[f"{self._arm}_passive_joint"]] = master
        return probe

    def _observation(self) -> dict[str, float]:
        return {
            f"{name}.pos": float(self._data.qpos[address])
            for name, address in self._qpos_by_joint.items()
        }

    def GetInfo(self, request, context):
        return device_pb2.GetInfoResponse(
            observation_features=list(self._obs_ft_info.values()),
            action_features=list(self._act_ft_info.values()),
            feedback_features=list(self._act_ft_info.values()),
            effective_action_features=list(self._effective_act_ft_info.values()),
        )

    def Connect(self, request, context):
        with self._lock:
            if self._reset_on_connect:
                self._set_home()
                self._latest_action = dict(self._identity_action)
            self._connected = True
            self._law.reset()
            self._law.lock_reference(self._data.qpos.copy())
            self._reference_locked_monotonic = time.monotonic()
            self._last_action_monotonic = None
            self._watchdog_held = False
            self._session_hold_latched = False
            self._hold_reason = ""
            self._reference_epoch += 1
            self._reject_streak = 0
            self._reject_started_monotonic = None
            self._last_safety = device_pb2.SafetyReport()
        return device_pb2.CalibrationInfo(
            status=device_pb2.CalibrationStatus.CALIBRATED
        )

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(
            status=device_pb2.CalibrationStatus.CALIBRATED
        )

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        with self._lock:
            self._connected = False
        return Empty()

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(
            status=device_pb2.DeviceStatus.COLLECTION
            if self._connected
            else device_pb2.DeviceStatus.IDLE
        )

    def GetObservation(self, request, context):
        substeps = max(
            1, int(round(_PHYSICS_PERIOD_S / self._model.opt.timestep))
        )
        while context.is_active():
            with self._lock:
                now = time.monotonic()
                if (
                    self._connected
                    and self._last_action_monotonic is not None
                    and now - self._last_action_monotonic > self._stale_timeout_s
                    and not self._watchdog_held
                ):
                    for joint_name in EFFECTIVE_ACTION_JOINTS[self._arm]:
                        self._target_ctrl[self._actuator_by_name[joint_name]] = (
                            self._data.qpos[self._qpos_by_joint[joint_name]]
                        )
                    self._watchdog_held = True
                    logger.error(
                        "ACTION WATCHDOG: no command for %.3fs; selected arm "
                        "and gripper atomically held at measured positions.",
                        now - self._last_action_monotonic,
                    )
                alpha = self._ctrl_smoothing_alpha
                self._data.ctrl[:] = (
                    alpha * self._target_ctrl + (1.0 - alpha) * self._data.ctrl
                )
                for _ in range(substeps):
                    self._mj.mj_step(self._model, self._data)
                if self._viewer is not None:
                    self._viewer.sync()
                observation = self._observation()
            yield from encode_feature(self._obs_ft_info, observation)
            time.sleep(_PHYSICS_PERIOD_S)

    def SendAction(self, request, context):
        action: dict[str, float] = {}
        for feature in request.features:
            load_feature(feature, self._act_ft_info, action, aux_behavior="ignore")
        with self._lock:
            if self._session_hold_latched:
                self._last_safety = device_pb2.SafetyReport(
                    flags=int(SafetyFlag.HELD | SafetyFlag.SESSION_HOLD),
                    applied_mask=int(AppliedGroup.NONE),
                    reason=self._hold_reason or "session-hold",
                )
                effective = {
                    f"{joint_name}.pos": float(
                        self._target_ctrl[self._actuator_by_name[joint_name]]
                    )
                    for joint_name in EFFECTIVE_ACTION_JOINTS[self._arm]
                }
                return device_pb2.ActionResult(
                    features=list(
                        encode_feature(self._effective_act_ft_info, effective)
                    ),
                    safety=self._last_safety,
                )
            self._latest_action.update(action)
            prefixed = action_keys(self._arm)
            generic_action = {
                generic: self._latest_action[prefixed_key]
                for generic, prefixed_key in zip(ACTION_KEYS, prefixed, strict=True)
            }
            now = time.monotonic()
            stale = (
                self._watchdog_held
                or (
                    self._last_action_monotonic is not None
                    and now - self._last_action_monotonic > self._stale_timeout_s
                )
            )
            solution = self._law.solve(
                generic_action,
                self._data.qpos.copy(),
                stale=stale,
                allow_collision_avoidance=bool(
                    self._collision_aware_ik
                    and not stale
                    and now - self._reference_locked_monotonic
                    >= self._reference_grace_s
                ),
            )
            self._last_action_monotonic = now
            if solution.stale:
                for joint_name in EFFECTIVE_ACTION_JOINTS[self._arm]:
                    self._target_ctrl[self._actuator_by_name[joint_name]] = (
                        self._data.qpos[self._qpos_by_joint[joint_name]]
                    )
            self._watchdog_held = False
            for joint_name in ARM_JOINTS[self._arm]:
                actuator = self._actuator_by_name[joint_name]
                if not solution.stale:
                    self._target_ctrl[actuator] = np.radians(
                        solution.joint_action[f"{joint_name}.pos"]
                    )
            # Stale input and any collision touching an as-yet-unclassified
            # body are atomic holds.  Recoverable arm-only workspace/IK/FK
            # rejects may still apply a fresh finite gripper command.
            atomic_hold = bool(
                solution.stale
                or (solution.held and solution.collided)
                or solution.reason in {"input-invalid", "checker-error"}
            )
            gripper_actuator = self._actuator_by_name[
                f"{self._arm}_active_joint1"
            ]
            if not atomic_hold:
                gripper_open_0_100 = solution.joint_action["gripper.pos"]
                gripper_target = (
                    1.0 - np.clip(gripper_open_0_100, 0.0, 100.0) / 100.0
                ) * GRIPPER_CLOSED_RAD
                self._target_ctrl[gripper_actuator] = gripper_target

            flags = SafetyFlag.NONE
            if solution.held:
                flags |= SafetyFlag.HELD
            if solution.stale:
                flags |= SafetyFlag.STALE
            if solution.collided and solution.reason != "checker-error":
                flags |= SafetyFlag.COLLISION
            if solution.jumped:
                flags |= SafetyFlag.IK_JUMP
            if solution.frame_capped:
                flags |= SafetyFlag.FRAME_CAPPED
            if solution.reason in {"workspace-delta", "workspace-arm-base"}:
                flags |= SafetyFlag.WORKSPACE
            elif solution.reason in {
                "ik-nan", "ik-deadline", "ik-residual", "ik-branch-jump"
            }:
                flags |= SafetyFlag.IK
            elif solution.reason == "fk-consistency":
                flags |= SafetyFlag.FK
            elif solution.reason == "input-invalid":
                flags |= SafetyFlag.INPUT_INVALID
            elif solution.reason == "checker-error":
                flags |= SafetyFlag.CHECKER_ERROR

            arm_group, gripper_group = groups_for_arm(self._arm)
            if solution.held:
                applied = AppliedGroup.NONE if atomic_hold else gripper_group
                if self._reject_streak == 0:
                    self._reject_started_monotonic = now
                self._reject_streak += 1
            else:
                applied = arm_group | gripper_group
                self._reject_streak = 0
                self._reject_started_monotonic = None
            reject_duration_s = (
                0.0
                if self._reject_started_monotonic is None
                else now - self._reject_started_monotonic
            )
            self._last_safety = device_pb2.SafetyReport(
                flags=int(flags),
                applied_mask=int(applied),
                reason=solution.reason,
                collision_pair_a=self._law.last_collision_pair[0],
                collision_pair_b=self._law.last_collision_pair[1],
                min_distance_m=(
                    float(self._law.last_collision_distance_m)
                    if np.isfinite(self._law.last_collision_distance_m)
                    else 0.0
                ),
                pos_err_m=float(solution.pos_err_m),
                rot_err_rad=float(solution.rot_err_rad),
                manipulability=float(solution.manipulability),
                reject_streak=self._reject_streak,
                reject_duration_s=reject_duration_s,
            )
            effective = {
                f"{joint_name}.pos": float(
                    self._target_ctrl[self._actuator_by_name[joint_name]]
                )
                for joint_name in EFFECTIVE_ACTION_JOINTS[self._arm]
            }
            if now - self._last_control_debug_monotonic >= 1.0:
                self._last_control_debug_monotonic = now
                tcp_actual = self._data.site_xpos[self._teleop_site_id]
                gripper_actual = self._data.qpos[
                    self._qpos_by_joint[f"{self._arm}_active_joint1"]
                ]
                logger.info(
                    "S1_CONTROL: arm=%s tcp_actual_m=[%.3f,%.3f,%.3f] "
                    "gripper_cmd_mm=%.1f gripper_target_rad=%.3f "
                    "gripper_actual_rad=%.3f safety_flags=%d applied_mask=%d "
                    "reason=%s",
                    self._arm,
                    *tcp_actual,
                    float(generic_action["gripper.distance"]),
                    float(self._target_ctrl[gripper_actuator]),
                    float(gripper_actual),
                    int(self._last_safety.flags),
                    int(self._last_safety.applied_mask),
                    self._last_safety.reason or "-",
                )
        return device_pb2.ActionResult(
            features=list(
                encode_feature(self._effective_act_ft_info, effective)
            ),
            safety=self._last_safety,
        )

    def GetFeedback(self, request, context):
        with self._lock:
            return encode_feature(self._act_ft_info, dict(self._latest_action))

    def SetReference(self, request, context):
        with self._lock:
            self._law.lock_reference(self._data.qpos.copy())
            self._reference_locked_monotonic = time.monotonic()
            self._reference_epoch += 1
        return Empty()

    def Hold(self, request, context):
        with self._lock:
            if not self._session_hold_latched:
                self._hold_epoch += 1
                self._hold_reference_epoch = self._reference_epoch
            self._session_hold_latched = True
            self._hold_reason = request.reason or "session-hold"
            for joint_name in EFFECTIVE_ACTION_JOINTS[self._arm]:
                self._target_ctrl[self._actuator_by_name[joint_name]] = (
                    self._data.qpos[self._qpos_by_joint[joint_name]]
                )
            self._watchdog_held = True
            self._last_safety = device_pb2.SafetyReport(
                flags=int(SafetyFlag.HELD | SafetyFlag.SESSION_HOLD),
                applied_mask=int(AppliedGroup.NONE),
                reason=self._hold_reason,
            )
            return device_pb2.HoldResponse(
                held=True, hold_epoch=self._hold_epoch
            )

    def Resume(self, request, context):
        with self._lock:
            if not self._session_hold_latched:
                return device_pb2.ResumeResponse(
                    resumed=True, hold_epoch=self._hold_epoch
                )
            if self._reference_epoch <= self._hold_reference_epoch:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "Resume requires a newer reference captured after Hold",
                )
            self._session_hold_latched = False
            self._hold_reason = ""
            self._watchdog_held = False
            self._last_action_monotonic = None
            self._last_safety = device_pb2.SafetyReport()
            return device_pb2.ResumeResponse(
                resumed=True, hold_epoch=self._hold_epoch
            )
