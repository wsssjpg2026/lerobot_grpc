"""Public-seam tests for the packaged MuJoCo Galbot S1 follower."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from google.protobuf.empty_pb2 import Empty

mujoco = pytest.importorskip("mujoco")

from lerobot_robot_grpc.action_safety import AppliedGroup, SafetyFlag  # noqa: E402
from lerobot_robot_grpc.follower.mujoco_s1_follower_server import (  # noqa: E402
    ARM_JOINTS,
    GRIPPER_CLOSED_RAD,
    LEFT_HOME_RAD,
    RIGHT_HOME_RAD,
    S1ArmWorkspace,
    S1_TELEOP_REACH_M,
    MuJoCoS1Servicer,
    S1CollisionChecker,
)
from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig  # noqa: E402
from lerobot_robot_grpc.follower.follower_server import (  # noqa: E402
    FollowerServer,
    FollowerServerConfig,
)
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower  # noqa: E402
from lerobot.utils.errors import DeviceNotConnectedError  # noqa: E402
from lerobot_robot_grpc.follower.utils import encode_feature, load_feature  # noqa: E402
from lerobot_robot_grpc.protos import device_pb2  # noqa: E402
from tests.response.backends import free_port  # noqa: E402


PKG_ROOT = Path(__file__).resolve().parents[1]
S1_ASSET_ROOT = PKG_ROOT / "assets" / "s1"
S1_XML = S1_ASSET_ROOT / "mjcf" / "galbot_s1.xml"

LEFT_ACTION_KEYS = {
    "left.hand.delta_pos.x",
    "left.hand.delta_pos.y",
    "left.hand.delta_pos.z",
    "left.hand.delta_rot.qx",
    "left.hand.delta_rot.qy",
    "left.hand.delta_rot.qz",
    "left.hand.delta_rot.qw",
    "left.gripper.distance",
}
LEFT_EFFECTIVE_ACTION_KEYS = {
    *(f"left_arm_joint{i}.pos" for i in range(1, 8)),
    "left_active_joint1.pos",
}
OBSERVATION_KEYS = {
    *(f"left_arm_joint{i}.pos" for i in range(1, 8)),
    *(f"right_arm_joint{i}.pos" for i in range(1, 8)),
    "left_active_joint1.pos",
    "right_active_joint1.pos",
    "head_joint1.pos",
    "head_joint2.pos",
    "torso_lift_joint1.pos",
    "base_x_joint.pos",
    "base_y_joint.pos",
    "base_yaw_joint.pos",
}


class _OneFrameContext:
    def __init__(self):
        self._first = True

    def is_active(self):
        if self._first:
            self._first = False
            return True
        return False


def _one_observation(servicer: MuJoCoS1Servicer) -> dict[str, float]:
    info = servicer.GetInfo(None, None)
    schema = {feature.key: feature for feature in info.observation_features}
    observation: dict[str, float] = {}
    for feature in servicer.GetObservation(Empty(), _OneFrameContext()):
        load_feature(feature, schema, observation)
    return observation


def _left_action(
    *,
    dx=0.0,
    dy=0.0,
    dz=0.0,
    quat=(0.0, 0.0, 0.0, 1.0),
    gripper_mm=60.0,
):
    return {
        "left.hand.delta_pos.x": dx,
        "left.hand.delta_pos.y": dy,
        "left.hand.delta_pos.z": dz,
        "left.hand.delta_rot.qx": quat[0],
        "left.hand.delta_rot.qy": quat[1],
        "left.hand.delta_rot.qz": quat[2],
        "left.hand.delta_rot.qw": quat[3],
        "left.gripper.distance": gripper_mm,
    }


def _send_action(servicer: MuJoCoS1Servicer, action: dict[str, float]):
    info = servicer.GetInfo(None, None)
    schema = {feature.key: feature for feature in info.action_features}
    request = device_pb2.Action(features=list(encode_feature(schema, action)))
    return servicer.SendAction(request, None)


def _left_ee_position(observation: dict[str, float]) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(S1_XML))
    data = mujoco.MjData(model)
    for name in OBSERVATION_KEYS:
        joint_name = name.removesuffix(".pos")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = observation[name]
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_base_link"
    )
    return data.xpos[body_id].copy()


def _left_ee_rotation(observation: dict[str, float]) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(S1_XML))
    data = mujoco.MjData(model)
    for name in OBSERVATION_KEYS:
        joint_name = name.removesuffix(".pos")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = observation[name]
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_base_link"
    )
    return data.xmat[body_id].reshape(3, 3).copy()


def test_packaged_s1_model_and_license_are_self_contained():
    """The default follower asset loads without the external source checkout."""
    assert (S1_ASSET_ROOT / "LICENSE").is_file()
    model = mujoco.MjModel.from_xml_path(str(S1_XML))
    assert (model.nq, model.nu) == (26, 22)


def test_s1_collision_checker_has_a_clean_home_and_detects_cross_arm_contact():
    checker = S1CollisionChecker(S1_XML, arm="left", margin_m=0.005)
    model = mujoco.MjModel.from_xml_path(str(S1_XML))
    qpos = np.zeros(model.nq)

    def set_arm(side: str, values):
        for joint_name, value in zip(ARM_JOINTS[side], values, strict=True):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos[model.jnt_qposadr[joint_id]] = value

    set_arm("left", LEFT_HOME_RAD)
    set_arm("right", RIGHT_HOME_RAD)
    torso_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "torso_lift_joint1"
    )
    qpos[model.jnt_qposadr[torso_joint_id]] = 0.6
    assert checker(qpos) is False

    # A deterministic in-limit posture whose left wrist intersects the fixed
    # right arm.
    set_arm(
        "left",
        (
            2.8972103532,
            -1.7848371780,
            0.9279999274,
            -1.9759436424,
            -1.2052651571,
            -0.1035295326,
            0.1054998094,
        ),
    )
    assert checker(qpos) is True


def test_s1_cross_arm_collision_uses_10mm_entry_and_15mm_release():
    checker = S1CollisionChecker(S1_XML, arm="left", margin_m=0.005)
    model = mujoco.MjModel.from_xml_path(str(S1_XML))
    qpos = np.zeros(model.nq)

    def set_arm(side: str, values):
        for joint_name, value in zip(ARM_JOINTS[side], values, strict=True):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos[model.jnt_qposadr[joint_id]] = value

    set_arm("right", RIGHT_HOME_RAD)
    set_arm(
        "left",
        (
            -2.740171843556137,
            1.5692704245705758,
            -2.4330455789586254,
            -2.158771819326527,
            -1.103794410060629,
            -0.3274900341972172,
            0.17020363794239235,
        ),
    )
    torso_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "torso_lift_joint1"
    )
    qpos[model.jnt_qposadr[torso_joint_id]] = 0.6

    entry = checker.check(qpos)
    release = checker.check(qpos, release=True)
    assert not entry.collided
    assert release.collided
    assert 0.010 < release.distance_m < 0.015
    assert release.body_a.startswith("left_")
    assert release.body_b.startswith("right_")


def test_s1_near_torso_pair_uses_2mm_entry_and_5mm_release():
    """The conservative link2 proxy gets a pair-specific near-field band.

    This rounded joint target was captured from the 2026-08-21 live session.
    The official collision proxies are about 4.98 mm apart even though the
    rendered meshes still have a visibly larger gap.  It must not enter a new
    collision latch at that distance, but an already-latched controller must
    keep holding until the pair clears the 5 mm release threshold.
    """
    checker = S1CollisionChecker(S1_XML, arm="left", margin_m=0.005)
    model = mujoco.MjModel.from_xml_path(str(S1_XML))
    qpos = np.zeros(model.nq)

    def set_arm(side: str, values):
        for joint_name, value in zip(ARM_JOINTS[side], values, strict=True):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos[model.jnt_qposadr[joint_id]] = value

    set_arm("right", RIGHT_HOME_RAD)
    set_arm(
        "left",
        np.radians([39.4, -110.9, -39.5, -96.9, -1.7, -26.2, -9.3]),
    )
    torso_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "torso_lift_joint1"
    )
    qpos[model.jnt_qposadr[torso_joint_id]] = 0.6

    entry = checker.check(qpos)
    release = checker.check(qpos, release=True)
    assert not entry.collided
    assert release.collided
    assert release.body_a == "left_arm_link2"
    assert release.body_b == "torso_base_link"
    assert 0.002 < release.distance_m < 0.005


def test_s1_distance_provider_reports_a_distance_increasing_joint_gradient():
    """Soft collision constraints expose a usable whole-arm gradient.

    The expected sign is verified against an independent finite perturbation
    of the packaged S1 model rather than by repeating the adapter formula.
    """
    checker = S1CollisionChecker(
        S1_XML,
        arm="left",
        margin_m=0.005,
        self_soft_distance_ratio=0.06,
        cross_arm_soft_distance_ratio=0.10,
    )
    model = mujoco.MjModel.from_xml_path(str(S1_XML))
    qpos = np.zeros(model.nq)

    for side, values in (
        ("left", np.radians([39.4, -110.9, -39.5, -96.9, -1.7, -26.2, -9.3])),
        ("right", RIGHT_HOME_RAD),
    ):
        for joint_name, value in zip(ARM_JOINTS[side], values, strict=True):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos[model.jnt_qposadr[joint_id]] = value
    torso_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "torso_lift_joint1"
    )
    qpos[model.jnt_qposadr[torso_joint_id]] = 0.6

    constraints = checker.active_constraints(qpos)
    torso = next(
        item
        for item in constraints
        if (item.body_a, item.body_b)
        == ("left_arm_link2", "torso_base_link")
    )
    assert torso.activation_distance_m == pytest.approx(
        0.06 * S1_TELEOP_REACH_M
    )
    assert 0.002 < torso.distance_m < 0.005
    assert torso.gradient.shape == (7,)
    assert np.linalg.norm(torso.gradient) > 1e-4

    arm_dofs = []
    for name in ARM_JOINTS["left"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        arm_dofs.append(int(model.jnt_qposadr[joint_id]))
    escaped = qpos.copy()
    escaped[arm_dofs] += 1e-4 * torso.gradient / np.linalg.norm(torso.gradient)
    escaped_torso = next(
        item
        for item in checker.active_constraints(escaped)
        if (item.body_a, item.body_b)
        == ("left_arm_link2", "torso_base_link")
    )
    assert escaped_torso.distance_m > torso.distance_m


def test_s1_distance_provider_uses_the_larger_cross_arm_soft_band():
    checker = S1CollisionChecker(
        S1_XML,
        arm="left",
        margin_m=0.005,
        self_soft_distance_ratio=0.06,
        cross_arm_soft_distance_ratio=0.10,
    )
    model = mujoco.MjModel.from_xml_path(str(S1_XML))
    qpos = np.zeros(model.nq)
    for side, values in (
        (
            "left",
            (
                -2.740171843556137,
                1.5692704245705758,
                -2.4330455789586254,
                -2.158771819326527,
                -1.103794410060629,
                -0.3274900341972172,
                0.17020363794239235,
            ),
        ),
        ("right", RIGHT_HOME_RAD),
    ):
        for joint_name, value in zip(ARM_JOINTS[side], values, strict=True):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos[model.jnt_qposadr[joint_id]] = value
    torso_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "torso_lift_joint1"
    )
    qpos[model.jnt_qposadr[torso_joint_id]] = 0.6

    cross_arm = [
        item
        for item in checker.active_constraints(qpos)
        if item.body_a.startswith("left_") and item.body_b.startswith("right_")
    ]
    assert cross_arm
    assert all(
        item.activation_distance_m
        == pytest.approx(0.10 * S1_TELEOP_REACH_M)
        for item in cross_arm
    )
    assert min(item.distance_m for item in cross_arm) < 0.015


def test_collision_aware_ik_increases_clearance_while_preserving_the_tcp_pose():
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML),
        arm="left",
        render=False,
        collision_aware_ik=True,
        stale_timeout_s=10.0,
    )
    body_dofs = [
        servicer._qpos_by_joint[name] for name in ARM_JOINTS["left"]
    ]
    qpos = servicer._data.qpos.copy()
    qpos[body_dofs] = np.radians(
        [39.4, -110.9, -39.5, -96.9, -1.7, -26.2, -9.3]
    )
    servicer._law.lock_reference(qpos)
    before_pose = servicer._law._fk(qpos)
    before = next(
        item
        for item in servicer._collision_checker.active_constraints(qpos)
        if (item.body_a, item.body_b)
        == ("left_arm_link2", "torso_base_link")
    )

    solution = servicer._law.solve(
        {
            "hand.delta_pos.x": 0.0,
            "hand.delta_pos.y": 0.0,
            "hand.delta_pos.z": 0.0,
            "hand.delta_rot.qx": 0.0,
            "hand.delta_rot.qy": 0.0,
            "hand.delta_rot.qz": 0.0,
            "hand.delta_rot.qw": 1.0,
            "gripper.distance": 60.0,
        },
        qpos,
    )
    candidate = qpos.copy()
    candidate[body_dofs] = np.radians(
        [
            solution.joint_action[f"{name}.pos"]
            for name in ARM_JOINTS["left"]
        ]
    )
    after = next(
        item
        for item in servicer._collision_checker.active_constraints(candidate)
        if (item.body_a, item.body_b)
        == ("left_arm_link2", "torso_base_link")
    )
    after_pose = servicer._law._fk(candidate)

    assert not solution.held
    assert after.distance_m > before.distance_m + 1e-5
    assert np.linalg.norm(after_pose[:3, 3] - before_pose[:3, 3]) <= 0.005
    relative = before_pose[:3, :3].T @ after_pose[:3, :3]
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    assert angle <= np.radians(3.0)
    assert float(np.abs(candidate[body_dofs] - qpos[body_dofs]).max()) <= np.radians(
        2.291831
    ) + 1e-9


def test_collision_aware_ik_converges_without_fixed_target_joint_oscillation():
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML),
        arm="left",
        render=False,
        collision_aware_ik=True,
        stale_timeout_s=10.0,
    )
    body_dofs = [
        servicer._qpos_by_joint[name] for name in ARM_JOINTS["left"]
    ]
    qpos = servicer._data.qpos.copy()
    qpos[body_dofs] = np.radians(
        [39.4, -110.9, -39.5, -96.9, -1.7, -26.2, -9.3]
    )
    servicer._law.lock_reference(qpos)
    command = {
        "hand.delta_pos.x": 0.0,
        "hand.delta_pos.y": 0.0,
        "hand.delta_pos.z": 0.0,
        "hand.delta_rot.qx": 0.0,
        "hand.delta_rot.qy": 0.0,
        "hand.delta_rot.qz": 0.0,
        "hand.delta_rot.qw": 1.0,
        "gripper.distance": 60.0,
    }

    targets = []
    barrier_costs = []
    for _ in range(80):
        solution = servicer._law.solve(command, qpos)
        assert not solution.held
        qpos[body_dofs] = np.radians(
            [
                solution.joint_action[f"{name}.pos"]
                for name in ARM_JOINTS["left"]
            ]
        )
        targets.append(qpos[body_dofs].copy())
        constraints = servicer._collision_checker.active_constraints(qpos)
        barrier_costs.append(
            sum(
                np.clip(
                    (item.activation_distance_m - item.distance_m)
                    / (item.activation_distance_m - item.minimum_distance_m),
                    0.0,
                    1.0,
                )
                ** 2
                for item in constraints
            )
        )

    assert barrier_costs[-1] < 0.8 * barrier_costs[0]
    tail = np.asarray(targets[-10:])
    assert float(np.ptp(tail, axis=0).max()) < np.radians(0.02)


def test_collision_aware_ik_routes_the_whole_arm_around_the_torso_boundary():
    """The captured live target should not become a permanent hard-gate hold."""
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML),
        arm="left",
        render=False,
        torso_home_m=0.6,
        collision_aware_ik=True,
        stale_timeout_s=10.0,
    )
    body_dofs = [
        servicer._qpos_by_joint[name] for name in ARM_JOINTS["left"]
    ]
    qpos = servicer._data.qpos.copy()
    qpos[body_dofs] = np.radians(
        [62.1, -108.8, -6.5, -112.9, 9.5, -21.6, -30.3]
    )
    servicer._law.lock_reference(qpos)
    command = {
        "hand.delta_pos.x": -0.04868578443587338,
        "hand.delta_pos.y": -0.0384126000770656,
        "hand.delta_pos.z": -0.04315918889128535,
        "hand.delta_rot.qx": 0.026754698286012266,
        "hand.delta_rot.qy": 0.008380012753899059,
        "hand.delta_rot.qz": 0.010672686999462904,
        "hand.delta_rot.qw": 0.9995499263458931,
        "gripper.distance": 60.0,
    }

    held = 0
    for _ in range(60):
        solution = servicer._law.solve(command, qpos)
        held += int(solution.held)
        qpos[body_dofs] = np.radians(
            [
                solution.joint_action[f"{name}.pos"]
                for name in ARM_JOINTS["left"]
            ]
        )

    torso = next(
        item
        for item in servicer._collision_checker.active_constraints(qpos)
        if (item.body_a, item.body_b)
        == ("left_arm_link2", "torso_base_link")
    )
    assert held == 0
    assert torso.distance_m > 0.010
    assert not servicer._collision_checker.check(qpos).collided


def test_collision_aware_ik_meets_the_30hz_compute_budget():
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML),
        arm="left",
        render=False,
        collision_aware_ik=True,
        stale_timeout_s=10.0,
    )
    body_dofs = [
        servicer._qpos_by_joint[name] for name in ARM_JOINTS["left"]
    ]
    qpos = servicer._data.qpos.copy()
    servicer._law.lock_reference(qpos)
    identity = {
        "hand.delta_pos.x": 0.0,
        "hand.delta_pos.y": 0.0,
        "hand.delta_pos.z": 0.0,
        "hand.delta_rot.qx": 0.0,
        "hand.delta_rot.qy": 0.0,
        "hand.delta_rot.qz": 0.0,
        "hand.delta_rot.qw": 1.0,
        "gripper.distance": 60.0,
    }
    elapsed = []
    for frame in range(160):
        started = time.perf_counter()
        solution = servicer._law.solve(identity, qpos)
        duration = time.perf_counter() - started
        qpos[body_dofs] = np.radians(
            [
                solution.joint_action[f"{name}.pos"]
                for name in ARM_JOINTS["left"]
            ]
        )
        if frame >= 40:
            elapsed.append(duration)
    assert np.percentile(elapsed, 95) <= 0.005

    qpos = servicer._data.qpos.copy()
    qpos[body_dofs] = np.radians(
        [62.1, -108.8, -6.5, -112.9, 9.5, -21.6, -30.3]
    )
    servicer._law.lock_reference(qpos)
    collision_target = {
        "hand.delta_pos.x": -0.04868578443587338,
        "hand.delta_pos.y": -0.0384126000770656,
        "hand.delta_pos.z": -0.04315918889128535,
        "hand.delta_rot.qx": 0.026754698286012266,
        "hand.delta_rot.qy": 0.008380012753899059,
        "hand.delta_rot.qz": 0.010672686999462904,
        "hand.delta_rot.qw": 0.9995499263458931,
        "gripper.distance": 60.0,
    }
    elapsed = []
    for _ in range(40):
        started = time.perf_counter()
        servicer._law.solve(collision_target, qpos)
        elapsed.append(time.perf_counter() - started)
    assert np.percentile(elapsed, 99) <= 0.020


def test_arm_base_workspace_uses_normalized_local_coordinates():
    servicer = MuJoCoS1Servicer(xml_path=str(S1_XML), arm="left", render=False)
    workspace = S1ArmWorkspace(servicer._model, arm="left")
    qpos = servicer._data.qpos.copy()
    workspace._data.qpos[:] = qpos
    mujoco.mj_forward(workspace._model, workspace._data)
    base_pos = workspace._data.xpos[workspace._body_id].copy()
    base_rot = workspace._data.xmat[workspace._body_id].reshape(3, 3).copy()

    inside = base_pos + base_rot @ np.array([0.50 * S1_TELEOP_REACH_M, 0.0, 0.0])
    outside_x = base_pos + base_rot @ np.array([0.86 * S1_TELEOP_REACH_M, 0.0, 0.0])
    outside_z = base_pos + base_rot @ np.array([0.0, 0.0, -0.61 * S1_TELEOP_REACH_M])
    assert workspace(inside, qpos)
    assert not workspace(outside_x, qpos)
    assert not workspace(outside_z, qpos)


def test_connect_exposes_namespaced_actions_and_full_si_home_state():
    servicer = MuJoCoS1Servicer(xml_path=str(S1_XML), arm="left", render=False)
    servicer.Connect(Empty(), None)

    info = servicer.GetInfo(None, None)
    assert {feature.key for feature in info.action_features} == LEFT_ACTION_KEYS
    assert {
        feature.key for feature in info.effective_action_features
    } == LEFT_EFFECTIVE_ACTION_KEYS
    assert {feature.key for feature in info.observation_features} == OBSERVATION_KEYS

    observation = _one_observation(servicer)
    assert set(observation) == OBSERVATION_KEYS
    np.testing.assert_allclose(
        [observation[f"left_arm_joint{i}.pos"] for i in range(1, 8)],
        LEFT_HOME_RAD,
        atol=0.02,
    )
    np.testing.assert_allclose(
        [observation[f"right_arm_joint{i}.pos"] for i in range(1, 8)],
        -np.asarray(LEFT_HOME_RAD),
        atol=0.02,
    )
    for key in OBSERVATION_KEYS - {
        *(f"left_arm_joint{i}.pos" for i in range(1, 8)),
        *(f"right_arm_joint{i}.pos" for i in range(1, 8)),
        "torso_lift_joint1.pos",
    }:
        assert observation[key] == pytest.approx(0.0, abs=0.02)
    assert observation["torso_lift_joint1.pos"] == pytest.approx(0.6, abs=0.02)


def test_small_pose_delta_moves_only_the_selected_arm_through_public_rpcs():
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML), arm="left", render=False, stale_timeout_s=10.0
    )
    baseline = MuJoCoS1Servicer(
        xml_path=str(S1_XML), arm="left", render=False, stale_timeout_s=10.0
    )
    servicer.Connect(Empty(), None)
    baseline.Connect(Empty(), None)

    _send_action(servicer, _left_action(dx=0.02))
    after = _one_observation(servicer)
    unchanged = _one_observation(baseline)
    for _ in range(30):
        after = _one_observation(servicer)
        unchanged = _one_observation(baseline)

    left_change = max(
        abs(after[f"left_arm_joint{i}.pos"] - unchanged[f"left_arm_joint{i}.pos"])
        for i in range(1, 8)
    )
    right_change = max(
        abs(after[f"right_arm_joint{i}.pos"] - unchanged[f"right_arm_joint{i}.pos"])
        for i in range(1, 8)
    )
    assert left_change > 0.005
    assert np.linalg.norm(
        _left_ee_position(after) - _left_ee_position(unchanged)
    ) > 0.005
    assert right_change < 0.003


def test_rotation_and_gripper_are_controlled_through_public_rpcs():
    servicer = MuJoCoS1Servicer(xml_path=str(S1_XML), arm="left", render=False)
    baseline = MuJoCoS1Servicer(xml_path=str(S1_XML), arm="left", render=False)
    servicer.Connect(Empty(), None)
    baseline.Connect(Empty(), None)

    half_angle = np.radians(7.5)
    command = _left_action(
        quat=(0.0, 0.0, np.sin(half_angle), np.cos(half_angle)),
        gripper_mm=0.0,
    )
    _send_action(servicer, command)
    for _ in range(80):
        # A real teleoperation client refreshes the command at 30 Hz.  Keep
        # this dynamics test alive rather than intentionally exercising the
        # independent 0.5 s follower watchdog.
        _send_action(servicer, command)
        after = _one_observation(servicer)
        unchanged = _one_observation(baseline)

    relative_rot = _left_ee_rotation(unchanged).T @ _left_ee_rotation(after)
    angle = np.arccos(np.clip((np.trace(relative_rot) - 1.0) / 2.0, -1.0, 1.0))
    assert angle > np.radians(3.0)
    assert after["left_active_joint1.pos"] > 0.5
    assert unchanged["left_active_joint1.pos"] == pytest.approx(0.0, abs=0.05)
    assert after["right_active_joint1.pos"] == pytest.approx(
        unchanged["right_active_joint1.pos"], abs=0.05
    )


def test_reset_on_connect_restores_home_after_a_disconnected_session():
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML),
        arm="left",
        render=False,
        reset_on_connect=True,
    )
    servicer.Connect(Empty(), None)
    _send_action(servicer, _left_action(dx=0.04))
    for _ in range(40):
        moved = _one_observation(servicer)
    assert max(
        abs(moved[f"left_arm_joint{i}.pos"] - LEFT_HOME_RAD[i - 1])
        for i in range(1, 8)
    ) > 0.01

    servicer.Disconnect(Empty(), None)
    servicer.Connect(Empty(), None)
    reset = _one_observation(servicer)
    np.testing.assert_allclose(
        [reset[f"left_arm_joint{i}.pos"] for i in range(1, 8)],
        LEFT_HOME_RAD,
        atol=0.02,
    )


def test_target_outside_arm_base_workspace_is_held():
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML), arm="left", render=False, stale_timeout_s=10.0
    )
    baseline = MuJoCoS1Servicer(
        xml_path=str(S1_XML), arm="left", render=False, stale_timeout_s=10.0
    )
    servicer.Connect(Empty(), None)
    baseline.Connect(Empty(), None)

    response = _send_action(servicer, _left_action(dx=1.0, gripper_mm=30.0))
    assert response.safety.flags & int(SafetyFlag.WORKSPACE)
    assert response.safety.applied_mask == int(AppliedGroup.LEFT_GRIPPER)
    for _ in range(30):
        after = _one_observation(servicer)
        unchanged = _one_observation(baseline)

    assert max(
        abs(after[f"left_arm_joint{i}.pos"] - unchanged[f"left_arm_joint{i}.pos"])
        for i in range(1, 8)
    ) < 0.003


def test_fixed_pose_target_holds_instead_of_oscillating_between_ik_branches():
    """A frozen Pika target must not make effective joints move indefinitely."""
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML), arm="left", render=False, stale_timeout_s=10.0
    )
    command = {
        "hand.delta_pos.x": -0.26805012925541416,
        "hand.delta_pos.y": 0.2232640721294734,
        "hand.delta_pos.z": 0.30906283856521843,
        "hand.delta_rot.qx": -0.4734024639799082,
        "hand.delta_rot.qy": 0.6092764269248978,
        "hand.delta_rot.qz": 0.46629205889639314,
        "hand.delta_rot.qw": 0.43271706518410363,
        "gripper.distance": 60.0,
    }
    qpos = servicer._data.qpos.copy()
    body_dofs = [
        servicer._qpos_by_joint[name] for name in ARM_JOINTS["left"]
    ]

    targets = []
    held_frames = 0
    for _ in range(120):
        solution = servicer._law.solve(command, qpos)
        target = np.radians(
            [
                solution.joint_action[f"{name}.pos"]
                for name in ARM_JOINTS["left"]
            ]
        )
        targets.append(target)
        held_frames += solution.held
        # Deterministic perfect-actuator feedback is the minimal condition
        # that exposed the production branch-selection loop.
        qpos[body_dofs] = target

    # This target reaches a collision-path boundary.  The safe behavior is to
    # hold the final accepted joint target; selecting a distant fallback IK
    # branch used to create a deterministic, perpetual back-and-forth motion.
    tail = np.asarray(targets[-40:])
    assert held_frames > 0
    assert float(np.ptp(tail, axis=0).max()) < 1e-6


def test_near_torso_target_advances_to_a_safe_collision_prefix():
    """A colliding Cartesian goal may advance only to its last safe prefix.

    This is the deterministic pose captured from the 2026-08-20 recovery
    session.  Accurate IK endpoints enter the 2 mm link2/torso margin while a
    late fallback merely has a large residual.  The operator must receive the
    collision diagnosis, and the arm should approach the boundary without
    ever publishing a colliding joint state.
    """
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML),
        arm="left",
        render=False,
        torso_home_m=0.6,
        stale_timeout_s=10.0,
    )
    body_dofs = [
        servicer._qpos_by_joint[name] for name in ARM_JOINTS["left"]
    ]
    qpos = servicer._data.qpos.copy()
    qpos[body_dofs] = np.radians(
        [62.1, -108.8, -6.5, -112.9, 9.5, -21.6, -30.3]
    )
    servicer._law.lock_reference(qpos)
    command = {
        "hand.delta_pos.x": -0.04868578443587338,
        "hand.delta_pos.y": -0.0384126000770656,
        "hand.delta_pos.z": -0.04315918889128535,
        "hand.delta_rot.qx": 0.026754698286012266,
        "hand.delta_rot.qy": 0.008380012753899059,
        "hand.delta_rot.qz": 0.010672686999462904,
        "hand.delta_rot.qw": 0.9995499263458931,
        "gripper.distance": 60.0,
    }

    start = qpos[body_dofs].copy()
    solution = servicer._law.solve(command, qpos)
    sent = np.radians(
        [
            solution.joint_action[f"{name}.pos"]
            for name in ARM_JOINTS["left"]
        ]
    )
    published = qpos.copy()
    published[body_dofs] = sent

    assert not solution.held
    assert solution.collided
    assert solution.reason == "collision-clipped"
    assert servicer._law.last_collision_pair == (
        "left_arm_link2",
        "torso_base_link",
    )
    assert float(np.abs(sent - start).max()) > np.radians(0.05)
    assert not servicer._collision_checker.check(published).collided
    start_position = servicer._law._fk(qpos)[:3, 3]
    sent_position = servicer._law._fk(published)[:3, 3]
    requested_offset = servicer._law.arm_reference[:3, :3] @ np.array(
        [
            command["hand.delta_pos.x"],
            command["hand.delta_pos.y"],
            command["hand.delta_pos.z"],
        ]
    )
    fraction = float(
        np.dot(sent_position - start_position, requested_offset)
        / np.dot(requested_offset, requested_offset)
    )
    line_position = start_position + np.clip(fraction, 0.0, 1.0) * requested_offset
    assert 0.0 < fraction < 1.0
    assert np.linalg.norm(sent_position - line_position) <= 0.005


def test_first_action_after_a_stale_gap_is_discarded_and_holds():
    servicer = MuJoCoS1Servicer(
        xml_path=str(S1_XML),
        arm="left",
        render=False,
        stale_timeout_s=0.0,
    )
    baseline = MuJoCoS1Servicer(
        xml_path=str(S1_XML),
        arm="left",
        render=False,
        stale_timeout_s=0.0,
    )
    servicer.Connect(Empty(), None)
    baseline.Connect(Empty(), None)
    _send_action(servicer, _left_action(dx=0.02))
    _send_action(baseline, _left_action(dx=0.02))
    for _ in range(20):
        _one_observation(servicer)
        _one_observation(baseline)

    # With a zero timeout, this retarget is the first action after a stale
    # gap and must be discarded.  The comparison instance receives no new
    # action, so both simulations should continue holding the same target.
    response = _send_action(
        servicer, _left_action(dx=-0.02, gripper_mm=0.0)
    )
    effective_schema = {
        feature.key: feature
        for feature in servicer.GetInfo(None, None).effective_action_features
    }
    effective = {}
    for feature in response.features:
        load_feature(feature, effective_schema, effective)
    assert response.safety.flags & int(SafetyFlag.STALE)
    assert response.safety.applied_mask == int(AppliedGroup.NONE)
    assert effective["left_active_joint1.pos"] == pytest.approx(0.0, abs=1e-4)
    for _ in range(20):
        after = _one_observation(servicer)
        held = _one_observation(baseline)
    np.testing.assert_allclose(
        [after[f"left_arm_joint{i}.pos"] for i in range(1, 8)],
        [held[f"left_arm_joint{i}.pos"] for i in range(1, 8)],
        atol=0.01,
    )


def test_standard_grpc_follower_client_negotiates_the_s1_schema():
    servicer = MuJoCoS1Servicer(xml_path=str(S1_XML), arm="left", render=False)
    server = FollowerServer(
        FollowerServerConfig(
            address=f"127.0.0.1:{free_port()}", server_grace_period_s=0.1
        ),
        servicer,
    )
    server.start()
    client = GRPCFollower(
        GRPCFollowerConfig(address=server.address, need_warmup=False)
    )
    try:
        client.connect(calibrate=False)
        assert set(client.action_features) == LEFT_ACTION_KEYS
        assert set(client.effective_action_features) == LEFT_EFFECTIVE_ACTION_KEYS
        assert set(client.observation_features) == OBSERVATION_KEYS
        effective = client.send_action(_left_action(dx=0.005, gripper_mm=30.0))
        assert set(effective) == LEFT_EFFECTIVE_ACTION_KEYS
        assert effective["left_active_joint1.pos"] == pytest.approx(
            GRIPPER_CLOSED_RAD / 2.0, abs=1e-5
        )
        observation = client.get_observation()
        assert set(observation) == OBSERVATION_KEYS

        hold_epoch = client.hold("leader tracking lost")
        assert hold_epoch > 0
        held = client.send_action(_left_action(dx=0.20, gripper_mm=0.0))
        assert client.last_action_safety.flags & int(SafetyFlag.HELD)
        assert client.last_action_safety.flags & int(SafetyFlag.SESSION_HOLD)
        assert client.last_action_safety.applied_mask == int(AppliedGroup.NONE)
        # A caller cannot accidentally release a stale Cartesian reference.
        with pytest.raises(DeviceNotConnectedError, match="newer reference"):
            client.resume()

        client.set_reference()
        assert client.resume() == hold_epoch
        resumed = client.send_action(_left_action())
        assert not (client.last_action_safety.flags & int(SafetyFlag.SESSION_HOLD))
        assert set(resumed) == set(held)
    finally:
        if client.is_connected:
            client.disconnect()
        server.stop()
