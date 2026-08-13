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
    build_R_lh2base,
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

    def test_axis_swap_does_not_remap_rotvec(self):
        """Body-frame rotvec is returned directly — R_lh2base is ignored
        because the rotvec is in the body frame, not the lighthouse frame."""
        from lerobot.utils.rotation import Rotation as R

        angle = np.radians(20)
        rot_ref = np.eye(3)
        rot_now = R.from_rotvec([angle, 0, 0]).as_matrix()
        R_swap = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1]], dtype=float)
        rotvec = compute_rotation_delta_rotvec(rot_now, rot_ref, R_swap)
        # X-rotation stays on X — no remapping through R_lh2base
        np.testing.assert_allclose(rotvec, [angle, 0, 0], atol=1e-10)


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


class TestBuildRlh2base:
    """build_R_lh2base: construct rotation from measured axis directions."""

    def test_identity_axes(self):
        """Forward=[1,0,0], up=[0,0,1] → identity rotation."""
        forward = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        R = build_R_lh2base(forward, up)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_non_symmetric_rotation(self):
        """A 35° yaw rotation: verify R @ delta_lh projects correctly.

        This is the critical test — symmetric/identity matrices can mask
        row_stack vs column_stack bugs.  A non-trivial rotation exposes them.
        """
        from lerobot.utils.rotation import Rotation as Rot

        angle = np.radians(35)
        Q = Rot.from_rotvec([0, 0, angle]).as_matrix()

        # Rows of Q are base axes in lighthouse frame.
        forward_lh = Q[0].copy()
        up_lh = Q[2].copy()

        R = build_R_lh2base(forward_lh, up_lh)

        # R should reconstruct Q (rows = [forward, left, up]).
        np.testing.assert_allclose(R, Q, atol=1e-12)

        # Projection check: a lighthouse-frame displacement along forward_lh
        # should map to base-X.
        delta_lh = forward_lh * 0.05  # 5cm forward
        delta_base = R @ delta_lh
        np.testing.assert_allclose(delta_base, [0.05, 0.0, 0.0], atol=1e-12)

        # Up displacement → base-Z.
        delta_lh = up_lh * 0.03
        delta_base = R @ delta_lh
        np.testing.assert_allclose(delta_base, [0.0, 0.0, 0.03], atol=1e-12)

    def test_det_is_positive_one(self):
        """Output must be a proper rotation (det = +1)."""
        from lerobot.utils.rotation import Rotation as Rot

        rng = np.random.default_rng(42)
        for _ in range(50):
            # Random rotation
            Q = Rot.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
            forward_lh = Q[0].copy()
            up_lh = Q[2].copy()
            R = build_R_lh2base(forward_lh, up_lh)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_noisy_inputs_still_orthogonal(self):
        """Slightly non-orthogonal measured axes still produce a valid rotation."""
        forward = np.array([0.99, 0.01, 0.02])
        forward /= np.linalg.norm(forward)
        up = np.array([0.01, 0.02, 0.98])
        up /= np.linalg.norm(up)
        R = build_R_lh2base(forward, up)
        # Output rows should be orthonormal.
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10
