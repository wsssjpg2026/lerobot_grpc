"""End-to-end pytest suite for the MuJoCo SO-101 follower in pose_delta mode.

Starts a :class:`MuJoCoSO101Servicer` in-process behind a real gRPC server
(random localhost port), connects the real ``GRPCFollower`` client, and
validates the full delta-pose -> FK -> DLS IK -> MuJoCo physics pipeline:

1. ``GetInfo`` reports the 8 shared pose-delta action features.
2. Sending a small position delta moves the MuJoCo arm's joints and the EE
   lands on the composed target (independent-model FK ground truth).
3. Gripper distance mapping opens the gripper.
4. Holding a forward offset then sending identity walks the arm home.

EE ground truth is test-side forward kinematics on assets/so101/scene.xml
(separate model — the same pattern as test_real_pose_delta.py; no placo
dependency).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
import mujoco  # noqa: E402

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig  # noqa: E402
from lerobot_robot_grpc.follower.follower_server import (  # noqa: E402
    FollowerServer,
    FollowerServerConfig,
)
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower  # noqa: E402
from lerobot_robot_grpc.follower.mujoco_follower_server import (  # noqa: E402
    BODY_JOINTS,
    MuJoCoSO101Servicer,
)
from lerobot_robot_grpc.pose_delta_schema import ACTION_KEYS  # noqa: E402
from tests.response.backends import free_port  # noqa: E402

PKG_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PKG_ROOT / "assets" / "so101" / "scene.xml"


def _fk_ee(joint_obs: dict) -> np.ndarray:
    """Independent-model FK: joint observation dict -> EE position."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    qpos = [math.radians(joint_obs[f"{j}.pos"]) for j in BODY_JOINTS]
    grip = float(joint_obs["gripper.pos"])
    qpos.append((grip / 100.0) * (1.74533 - (-0.17453)) + (-0.17453))
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    return data.site_xpos[sid].copy()


def _fk_of_commanded(joint_action: dict) -> np.ndarray:
    """FK of a lerobot-normalised joint action on a separate model."""
    return _fk_ee(
        {f"{j}.pos": joint_action[f"{j}.pos"] for j in BODY_JOINTS}
        | {"gripper.pos": joint_action.get("gripper.pos", 0.0)}
    )


def _delta_action(dx=0.0, dy=0.0, dz=0.0, gripper_mm=30.0) -> dict[str, float]:
    return {
        "hand.delta_pos.x": dx,
        "hand.delta_pos.y": dy,
        "hand.delta_pos.z": dz,
        "hand.delta_rot.qx": 0.0,
        "hand.delta_rot.qy": 0.0,
        "hand.delta_rot.qz": 0.0,
        "hand.delta_rot.qw": 1.0,
        "gripper.distance": gripper_mm,
    }


@pytest.fixture(scope="module")
def server():
    """The servicer + real gRPC server on a random port (module lifetime)."""
    servicer = MuJoCoSO101Servicer(
        xml_path=str(XML_PATH),
        action_mode="pose_delta",
        render=False,
    )
    grpc_server = FollowerServer(
        FollowerServerConfig(address=f"127.0.0.1:{free_port()}",
                             server_grace_period_s=1.0),
        servicer,
    )
    grpc_server.start()
    yield grpc_server
    grpc_server.stop()


@pytest.fixture(scope="module")
def client(server):
    follower = GRPCFollower(
        GRPCFollowerConfig(address=server.address, need_warmup=False)
    )
    follower.connect(calibrate=False)
    yield follower
    follower.disconnect()


def test_getinfo_reports_the_shared_pose_delta_schema(client):
    assert set(client.action_features.keys()) == set(ACTION_KEYS)
    assert len(client.action_features) == 8


def test_initial_observation_starts_at_home(client):
    obs = client.get_observation()
    for joint, deg in zip(BODY_JOINTS, (0.0, -20.0, 60.0, -40.0, 0.0)):
        assert obs[f"{joint}.pos"] == pytest.approx(deg, abs=0.5)
    assert np.isfinite(_fk_ee(obs)).all()


def test_small_delta_moves_arm_toward_intent(client):
    obs0 = client.get_observation()
    ee0 = _fk_ee(obs0)

    client.send_action(_delta_action(dx=0.005))
    time.sleep(0.5)  # let MuJoCo physics settle toward the new targets

    obs1 = client.get_observation()
    ee1 = _fk_ee(obs1)
    joint_delta = max(abs(obs1[f"{j}.pos"] - obs0[f"{j}.pos"]) for j in BODY_JOINTS)
    ee_shift = np.linalg.norm(ee1 - ee0)

    assert joint_delta > 0.1, (
        f"joints barely moved ({joint_delta:.4f} deg) — IK or ctrl may be broken"
    )
    assert ee_shift > 0.002, f"EE moved only {ee_shift * 1000:.2f} mm (commanded 5 mm)"
    # Direction: toward the reference-frame +X the law composed (the home
    # reference is near-identity, so base +X is a good approximation).
    assert ee1[0] > ee0[0], "EE did not move along the reference +X intent"


def test_gripper_distance_opens_the_gripper(client):
    client.send_action(_delta_action(gripper_mm=30.0))
    time.sleep(0.3)
    gripper_val = client.get_observation()["gripper.pos"]
    assert gripper_val > 1.0, f"gripper didn't open ({gripper_val:.1f})"
    assert gripper_val == pytest.approx(30.0 / 60.0 * 100.0, abs=10.0)


def test_identity_returns_the_arm_toward_home():
    """Latch-once semantics on the law seam: a +10 cm offset walked out,
    then identity offsets, must converge the commanded EE back onto the
    latched reference (within the walk budget of 8 identity frames)."""
    local = MuJoCoSO101Servicer(
        xml_path=str(XML_PATH), action_mode="pose_delta", render=False
    )
    local.Connect(None, None)
    ref_pos = local._law.arm_reference[:3, 3]

    fwd = _delta_action(dx=0.10)
    for _ in range(5):
        extended = local._pose_delta_to_joint_action(fwd)
    out_err = float(np.linalg.norm(_fk_of_commanded(extended) - ref_pos))
    assert out_err > 0.03, f"+10 cm walk moved only {out_err * 1000:.1f} mm out"
    for _ in range(8):
        retracted = local._pose_delta_to_joint_action(_delta_action())
    ee_err = float(np.linalg.norm(_fk_of_commanded(retracted) - ref_pos))
    assert ee_err < 0.008, (
        f"identity offset did not return home ({ee_err * 1000:.1f} mm)"
    )
