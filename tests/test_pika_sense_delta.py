"""Unit tests for the Pika Sense leader delta-computation pipeline.

Tests the pure helper functions from ``pika_sense_leader_server`` — no Pika
hardware or gRPC required.  Validates the coordinate-frame conventions
(wayfinder #05) and smoothing/dead-zone behaviour (#06).
"""

import numpy as np
import pytest

from lerobot_robot_grpc.leader.pika_sense_leader_server import (
    apply_dead_zone,
    apply_ema,
    compute_position_delta,
    compute_rotation_delta_rotvec,
)


class TestPositionDelta:
    """compute_position_delta: lighthouse→base rotation + subtraction."""

    def test_identity_rotation(self):
        """With R_lh2base = I, delta is simple subtraction."""
        pos_now = np.array([0.1, 0.2, 0.3])
        pos_ref = np.array([0.05, 0.15, 0.25])
        R = np.eye(3)
        delta = compute_position_delta(pos_now, pos_ref, R)
        np.testing.assert_allclose(delta, [0.05, 0.05, 0.05])

    def test_axis_swap(self):
        """X↔Y swap: lighthouse [0.1, 0, 0] → base [0, 0.1, 0]."""
        R = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1]], dtype=float)
        pos_now = np.array([0.1, 0.0, 0.0])
        pos_ref = np.zeros(3)
        delta = compute_position_delta(pos_now, pos_ref, R)
        np.testing.assert_allclose(delta, [0.0, 0.1, 0.0])

    def test_90deg_yaw(self):
        """90° yaw: lighthouse X → base -Y."""
        theta = np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta),  np.cos(theta), 0],
                       [0,              0,             1]])
        pos_now = np.array([0.1, 0.0, 0.0])
        pos_ref = np.zeros(3)
        delta = compute_position_delta(pos_now, pos_ref, R)
        np.testing.assert_allclose(delta, [0.0, 0.1, 0.0], atol=1e-10)


class TestRotationDelta:
    """compute_rotation_delta_rotvec: body-frame delta + axis remapping."""

    def test_identity_rotations(self):
        """No rotation change → zero rotvec."""
        rot = np.eye(3)
        R_lh2base = np.eye(3)
        rotvec = compute_rotation_delta_rotvec(rot, rot, R_lh2base)
        np.testing.assert_allclose(rotvec, [0, 0, 0], atol=1e-12)

    def test_pure_z_rotation(self):
        """30° rotation about Z → rotvec [0, 0, 30°]."""
        from lerobot.utils.rotation import Rotation as R

        angle = np.radians(30)
        rot_ref = np.eye(3)
        rot_now = R.from_rotvec([0, 0, angle]).as_matrix()
        R_lh2base = np.eye(3)
        rotvec = compute_rotation_delta_rotvec(rot_now, rot_ref, R_lh2base)
        np.testing.assert_allclose(rotvec, [0, 0, angle], atol=1e-10)

    def test_axis_swap_remaps_rotvec(self):
        """X↔Y swap: rotation about X (lighthouse) → rotation about Y (base).

        A rotation about X remapped through an X↔Y axis swap becomes a
        rotation about Y.
        """
        from lerobot.utils.rotation import Rotation as R

        angle = np.radians(20)
        rot_ref = np.eye(3)
        rot_now = R.from_rotvec([angle, 0, 0]).as_matrix()
        R_swap = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1]], dtype=float)
        rotvec = compute_rotation_delta_rotvec(rot_now, rot_ref, R_swap)
        # X-rotation remapped through X↔Y swap → Y-axis
        np.testing.assert_allclose(rotvec[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(abs(rotvec[1]), angle, atol=1e-10)
        np.testing.assert_allclose(rotvec[2], 0.0, atol=1e-10)


class TestDeadZone:
    """apply_dead_zone: below-threshold → zeros."""

    def test_below_threshold_zeros(self):
        delta = np.array([0.001, 0.0, 0.0])  # 1mm
        result = apply_dead_zone(delta, threshold=0.002)  # 2mm
        np.testing.assert_allclose(result, [0, 0, 0])

    def test_above_threshold_passes(self):
        delta = np.array([0.01, 0.0, 0.0])  # 10mm
        result = apply_dead_zone(delta, threshold=0.002)
        np.testing.assert_allclose(result, delta)

    def test_exactly_at_threshold(self):
        """Norm equal to threshold passes through (strict <, not <=)."""
        delta = np.array([0.002, 0.0, 0.0])
        result = apply_dead_zone(delta, threshold=0.002)
        np.testing.assert_allclose(result, delta)


class TestEMA:
    """apply_ema: exponential moving average update."""

    def test_first_update(self):
        raw = np.array([1.0, 0.0, 0.0])
        filtered = np.zeros(3)
        result = apply_ema(raw, filtered, alpha=0.25)
        # 0.25 * 1.0 + 0.75 * 0.0 = 0.25
        np.testing.assert_allclose(result, [0.25, 0.0, 0.0])

    def test_steady_state(self):
        """Repeated same input converges to that value."""
        raw = np.array([1.0, 0.0, 0.0])
        filtered = np.zeros(3)
        for _ in range(100):
            filtered = apply_ema(raw, filtered, alpha=0.25)
        np.testing.assert_allclose(filtered, [1.0, 0.0, 0.0], atol=1e-6)

    def test_decay(self):
        """After input stops, output decays geometrically."""
        filtered = np.array([1.0, 0.0, 0.0])
        raw = np.zeros(3)
        result = apply_ema(raw, filtered, alpha=0.25)
        # 0.25 * 0 + 0.75 * 1.0 = 0.75
        np.testing.assert_allclose(result, [0.75, 0.0, 0.0])
