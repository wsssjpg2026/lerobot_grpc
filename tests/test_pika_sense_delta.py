"""Unit tests for the Pika Sense leader delta computation.

Tests the pure helper functions from ``pika_sense_leader_server`` — no Pika
hardware or gRPC required.  Validates the PikaAnyArm official conventions
(align-official-decisions.md decisions 3/4): the published offset is the raw
relative transform ``inv(T_ref) @ T_now`` — body-frame translation and
body-frame rotation, 1:1, no filtering.
"""

import numpy as np
import pytest

from lerobot_robot_grpc.leader.pika_sense_leader_server import (
    build_R_lh2base,
    command_state_edge,
    compute_position_delta_body,
    compute_rotation_delta_rotvec,
)


class TestPositionDeltaBody:
    """compute_position_delta_body: R_ref^T @ (p_now - p_ref)."""

    def test_identity_reference_is_subtraction(self):
        pos_now = np.array([0.1, 0.2, 0.3])
        pos_ref = np.array([0.05, 0.15, 0.25])
        delta = compute_position_delta_body(pos_now, pos_ref, np.eye(3))
        np.testing.assert_allclose(delta, [0.05, 0.05, 0.05])

    def test_reference_yaw_rotates_delta_into_body_frame(self):
        """World +X hand motion with the reference yawed 90° reads as −Y in
        the reference body frame (R_ref^T @ x̂ = −ŷ)."""
        theta = np.pi / 2
        rot_ref = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],
        ])
        delta = compute_position_delta_body(
            np.array([0.1, 0.0, 0.0]), np.zeros(3), rot_ref
        )
        np.testing.assert_allclose(delta, [0.0, -0.1, 0.0], atol=1e-10)

    def test_no_lighthouse_to_base_rotation_involved(self):
        """The official composition needs no R_lh2base: a 90° yaw of the
        WHOLE scene (rotating ref and now together) must not change the
        published body-frame delta."""
        theta = np.radians(37.0)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],
        ])
        move = np.array([0.08, -0.02, 0.04])
        delta = compute_position_delta_body(R @ (np.array([0.3, 0.1, 0.2]) + move),
                                            R @ np.array([0.3, 0.1, 0.2]), R)
        np.testing.assert_allclose(delta, move, atol=1e-12)


class TestRotationDelta:
    """compute_rotation_delta_rotvec: body-frame delta R_ref^T @ R_now."""

    def test_identity_rotations(self):
        rot = np.eye(3)
        rotvec = compute_rotation_delta_rotvec(rot, rot)
        np.testing.assert_allclose(rotvec, [0, 0, 0], atol=1e-12)

    def test_pure_z_rotation(self):
        from lerobot.utils.rotation import Rotation as R

        angle = np.radians(30)
        rotvec = compute_rotation_delta_rotvec(
            R.from_rotvec([0, 0, angle]).as_matrix(), np.eye(3)
        )
        np.testing.assert_allclose(rotvec, [0, 0, angle], atol=1e-10)

    def test_scene_yaw_does_not_change_body_delta(self):
        """Rotating tracker and reference together leaves the body-frame
        delta unchanged (the whole point of the body-frame composition)."""
        from lerobot.utils.rotation import Rotation as R

        scene = R.from_rotvec([0.2, -0.4, 1.1]).as_matrix()
        body = R.from_rotvec([0.1, 0.05, -0.3]).as_matrix()
        rotvec = compute_rotation_delta_rotvec(scene @ body, scene)
        expected = R.from_matrix(body).as_rotvec()
        np.testing.assert_allclose(rotvec, expected, atol=1e-10)


class TestCommandStateEdge:
    """command_state_edge: any 0/1 change toggles; None never does."""

    def test_none_is_never_an_edge(self):
        assert not command_state_edge(None, None)
        assert not command_state_edge(1, None)
        assert not command_state_edge(None, 0)

    def test_edges_toggle(self):
        assert command_state_edge(1, 0)
        assert command_state_edge(0, 1)

    def test_same_state_is_no_edge(self):
        assert not command_state_edge(0, 0)
        assert not command_state_edge(1, 1)


class TestBuildR_lh2base:
    """Calibration-side helper: still measured and persisted (the teleop path
    no longer consumes it — official composition), so it must stay correct."""

    def test_orthonormal_from_orthogonal_inputs(self):
        fwd = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        R = build_R_lh2base(fwd, up)
        np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_non_orthogonal_inputs_reorthogonalised(self):
        fwd = np.array([1.0, 0.1, 0.0])
        up = np.array([0.0, 0.05, 1.0])
        R = build_R_lh2base(fwd / np.linalg.norm(fwd), up / np.linalg.norm(up))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
