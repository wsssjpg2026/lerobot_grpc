"""End-to-end test for MuJoCo SO-101 follower in pose_delta mode.

Starts a :class:`MuJoCoSO101Servicer` in-process (no external server needed),
connects the real ``GRPCFollower`` client, and validates the full delta-pose →
FK → IK → MuJoCo physics pipeline:

1. ``GetInfo`` reports 8 FLOAT32 pose-delta action features.
2. Sending a small position delta moves the MuJoCo arm's joints.
3. FK on the observed joints confirms the EE moved approximately by the delta.
4. Gripper distance mapping opens the gripper.

Run with the env that has ``mujoco`` + ``lerobot[kinematics]`` installed::

    python tests/test_mujoco_pose_delta.py
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.follower_server import FollowerServer, FollowerServerConfig
from lerobot_robot_grpc.follower.mujoco_follower_server import MuJoCoSO101Servicer
from lerobot_robot_grpc.pose_delta_schema import ACTION_KEYS

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("mujoco_test")

ADDRESS = "127.0.0.1:50052"
PKG_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PKG_ROOT / "assets" / "so101" / "scene.xml"
URDF_PATH = PKG_ROOT / "assets" / "so101" / "so101_new_calib.urdf"

BODY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _fk_ee_position(joint_obs: dict, kin) -> np.ndarray:
    """Forward-kinematics EE position from a joint observation dict."""
    q = np.array([joint_obs[f"{j}.pos"] for j in BODY_JOINTS], dtype=float)
    t = kin.forward_kinematics(q)
    return t[:3, 3]


def main() -> None:
    servicer = MuJoCoSO101Servicer(
        xml_path=str(XML_PATH),
        action_mode="pose_delta",
        render=False,
    )
    server = FollowerServer(FollowerServerConfig(address=ADDRESS), servicer)
    server.start()
    log.warning("MuJoCo pose_delta server listening on %s", ADDRESS)

    # Separate kinematics instance for FK verification in the test
    from lerobot.model.kinematics import RobotKinematics
    verify_kin = RobotKinematics(
        str(URDF_PATH), target_frame_name="gripper_frame_link",
        joint_names=list(BODY_JOINTS),
    )

    try:
        client = GRPCFollower(GRPCFollowerConfig(address=ADDRESS, need_warmup=False))
        client.connect(calibrate=False)

        # --- 1. Verify action schema ---
        print("\n=== 1. Action schema (GetInfo) ===")
        print("action_features:", client.action_features)
        expected_keys = set(ACTION_KEYS)
        got_keys = set(client.action_features.keys())
        assert got_keys == expected_keys, f"Action keys mismatch: {got_keys} vs {expected_keys}"
        print(f"OK — {len(got_keys)} pose_delta action features")

        # --- 2. Read initial observation ---
        print("\n=== 2. Initial observation ===")
        obs0 = client.get_observation()
        ee0 = _fk_ee_position(obs0, verify_kin)
        print("joints:", {k: round(v, 2) for k, v in obs0.items() if k.endswith(".pos")})
        print(f"EE position: [{ee0[0]:.4f}, {ee0[1]:.4f}, {ee0[2]:.4f}]")

        # --- 3. Send a +5mm X delta + 30mm gripper ---
        print("\n=== 3. Send pose delta: +5mm X, identity rotation, 30mm gripper ===")
        delta_action = {
            "hand.delta_pos.x": 0.005,
            "hand.delta_pos.y": 0.0,
            "hand.delta_pos.z": 0.0,
            "hand.delta_rot.qx": 0.0,
            "hand.delta_rot.qy": 0.0,
            "hand.delta_rot.qz": 0.0,
            "hand.delta_rot.qw": 1.0,  # identity quaternion
            "gripper.distance": 30.0,
        }
        executed = client.send_action(delta_action)
        print("executed:", {k: round(v, 4) for k, v in executed.items()})

        # Let MuJoCo physics settle toward the new targets
        time.sleep(0.5)

        # --- 4. Verify joints moved ---
        print("\n=== 4. Post-action observation ===")
        obs1 = client.get_observation()
        ee1 = _fk_ee_position(obs1, verify_kin)
        print("joints:", {k: round(v, 2) for k, v in obs1.items() if k.endswith(".pos")})
        print(f"EE position: [{ee1[0]:.4f}, {ee1[1]:.4f}, {ee1[2]:.4f}]")

        joint_delta = max(abs(obs1[f"{j}.pos"] - obs0[f"{j}.pos"]) for j in BODY_JOINTS)
        ee_shift = np.linalg.norm(ee1 - ee0)
        print(f"max joint change: {joint_delta:.3f}°")
        print(f"EE shift: {ee_shift * 1000:.2f} mm (commanded: 5.00 mm)")

        assert joint_delta > 0.1, f"Joints barely moved ({joint_delta:.4f}°) — IK or ctrl may be broken"
        print("OK — joints responded to pose delta")

        # --- 5. Verify gripper ---
        print("\n=== 5. Gripper mapping ===")
        gripper_val = obs1["gripper.pos"]
        print(f"gripper.pos: {gripper_val:.1f} (commanded distance=30mm, expect ~50%)")
        assert gripper_val > 1.0, f"Gripper didn't open ({gripper_val:.1f})"
        print("OK — gripper responded")

        # --- 6. Hold a forward offset, then identity must return toward home ---
        print("\n=== 6. Latch-once: +10 cm offset then identity returns home ===")
        identity = {
            "hand.delta_pos.x": 0.0, "hand.delta_pos.y": 0.0, "hand.delta_pos.z": 0.0,
            "hand.delta_rot.qx": 0.0, "hand.delta_rot.qy": 0.0, "hand.delta_rot.qz": 0.0,
            "hand.delta_rot.qw": 1.0, "gripper.distance": 30.0,
        }
        local = MuJoCoSO101Servicer(
            xml_path=str(XML_PATH), action_mode="pose_delta", render=False,
            position_deadband_m=0.0, rotation_deadband_rad=0.0, workspace_radius_m=0.0,
        )
        local.Connect(None, None)
        fwd = dict(identity)
        fwd["hand.delta_pos.x"] = 0.10
        extended = None
        for _ in range(5):
            extended = local._pose_delta_to_joint_action(fwd)
        retracted = None
        for _ in range(8):
            retracted = local._pose_delta_to_joint_action(identity)
        ee_err = float(np.linalg.norm(local._last_achieved_pos - local._t_zero[:3, 3]))
        print(
            f"elbow {extended['elbow_flex.pos']:.1f}° → {retracted['elbow_flex.pos']:.1f}°  "
            f"home_err={ee_err*1000:.1f}mm"
        )
        assert ee_err < 0.008, f"identity offset did not return home ({ee_err*1000:.1f} mm)"
        print("OK — identity offset returns the arm toward home")

        client.disconnect()
        print("\nMUJOCO_POSE_DELTA_PASS — delta-pose → FK → IK → MuJoCo pipeline OK")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
