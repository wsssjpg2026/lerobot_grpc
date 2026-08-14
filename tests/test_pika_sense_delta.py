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
    adaptive_ema_alpha,
    apply_rate_limit,
    apply_soft_dead_zone,
    clamp_vector,
    build_R_lh2base,
    compute_position_delta,
    compute_rotation_delta_rotvec,
    filter_pose,
    consume_tracker_jump,
    is_pose_jump,
    publish_latch_offset,
    publish_latch_rotation,
    slerp_rotation,
    slew_vector,
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


class TestSoftDeadZone:
    """apply_soft_dead_zone: zero below d, linear ramp d→2d, pass-through above 2d."""

    def test_below_threshold_zeros(self):
        delta = np.array([0.001, 0.0, 0.0])  # 1mm
        result = apply_soft_dead_zone(delta, threshold=0.002)
        np.testing.assert_allclose(result, [0, 0, 0])

    def test_mid_ramp_scales(self):
        """At 1.5 d the output magnitude is 0.5 * 1.5 d = 0.75 d."""
        d = 0.002
        delta = np.array([1.5 * d, 0.0, 0.0])
        result = apply_soft_dead_zone(delta, threshold=d)
        np.testing.assert_allclose(result, [0.75 * d, 0.0, 0.0], atol=1e-12)

    def test_above_twice_threshold_passes(self):
        delta = np.array([0.01, 0.0, 0.0])  # 10mm > 4mm
        result = apply_soft_dead_zone(delta, threshold=0.002)
        np.testing.assert_allclose(result, delta)

    def test_continuous_at_twice_threshold(self):
        d = 0.002
        delta = np.array([2.0 * d, 0.0, 0.0])
        result = apply_soft_dead_zone(delta, threshold=d)
        np.testing.assert_allclose(result, delta, atol=1e-12)


class TestClampAndAdaptiveEma:
    def test_clamp_passes_short_vector(self):
        v = np.array([0.01, 0.0, 0.0])
        np.testing.assert_allclose(clamp_vector(v, 0.02), v)

    def test_clamp_scales_long_vector(self):
        v = np.array([0.04, 0.0, 0.0])
        out = clamp_vector(v, 0.02)
        np.testing.assert_allclose(np.linalg.norm(out), 0.02, atol=1e-12)
        np.testing.assert_allclose(out / np.linalg.norm(out), v / np.linalg.norm(v))

    def test_adaptive_alpha_still_and_fast(self):
        assert adaptive_ema_alpha(0.0, 0.15, 0.70, 0.008) == 0.15
        assert adaptive_ema_alpha(0.008, 0.15, 0.70, 0.008) == 0.70
        mid = adaptive_ema_alpha(0.004, 0.15, 0.70, 0.008)
        np.testing.assert_allclose(mid, 0.425)


class TestRateLimit:
    def test_clips_upward_step(self):
        assert apply_rate_limit(10.0, prev=0.0, max_step=3.0) == 3.0

    def test_clips_downward_step(self):
        assert apply_rate_limit(0.0, prev=10.0, max_step=3.0) == 7.0

    def test_passes_small_step(self):
        assert apply_rate_limit(1.5, prev=0.0, max_step=3.0) == 1.5


class TestSlerpAndPoseFilter:
    def test_slerp_endpoints(self):
        from lerobot.utils.rotation import Rotation as Rot

        R0 = np.eye(3)
        R1 = Rot.from_rotvec([0.0, 0.0, np.radians(40)]).as_matrix()
        np.testing.assert_allclose(slerp_rotation(R0, R1, 0.0), R0, atol=1e-12)
        np.testing.assert_allclose(slerp_rotation(R0, R1, 1.0), R1, atol=1e-12)

    def test_slerp_halfway_angle(self):
        from lerobot.utils.rotation import Rotation as Rot

        angle = np.radians(40)
        R0 = np.eye(3)
        R1 = Rot.from_rotvec([0.0, 0.0, angle]).as_matrix()
        R_mid = slerp_rotation(R0, R1, 0.5)
        mid_angle = np.linalg.norm(Rot.from_matrix(R_mid).as_rotvec())
        np.testing.assert_allclose(mid_angle, 0.5 * angle, atol=1e-10)

    def test_filter_pose_first_sample_passthrough(self):
        pos = np.array([0.1, 0.2, 0.3])
        rot = np.eye(3)
        fpos, frot = filter_pose(pos, rot, None, None, alpha=0.25)
        np.testing.assert_allclose(fpos, pos)
        np.testing.assert_allclose(frot, rot)

    def test_filter_pose_ema_position(self):
        pos0 = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        rot = np.eye(3)
        fpos, _ = filter_pose(pos1, rot, pos0, rot, alpha=0.25)
        np.testing.assert_allclose(fpos, [0.25, 0.0, 0.0])


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


class TestPoseJump:
    def test_none_prev_is_not_a_jump(self):
        assert not is_pose_jump(np.array([0.4, 0.0, 0.0]), None, 0.05)

    def test_small_step_is_not_a_jump(self):
        assert not is_pose_jump(np.zeros(3), np.array([0.01, 0.0, 0.0]), 0.05)

    def test_forty_cm_is_a_jump(self):
        assert is_pose_jump(np.zeros(3), np.array([0.40, 0.0, 0.0]), 0.05)

    def test_consecutive_far_samples_do_not_latch_the_anchor(self):
        """Walking 8 cm in 1 cm steps after leaving the origin must keep accepting.

        The old last-accepted-anchor compared every sample to the first
        accepted pose and froze once the hand was 5 cm away.
        """
        last = np.zeros(3)
        begin = np.zeros(3)
        for i in range(1, 9):
            raw = np.array([0.01 * i, 0.0, 0.0])
            jumped, last, begin, _ = consume_tracker_jump(raw, last, begin, 0.05)
            assert not jumped, f"step {i} falsely jumped"
            np.testing.assert_allclose(last, raw)
        np.testing.assert_allclose(begin, [0.0, 0.0, 0.0])

    def test_single_jump_rebases_t_begin_and_advances_last_raw(self):
        last = np.array([0.10, 0.0, 0.0])
        begin = np.array([0.10, 0.0, 0.0])
        raw = np.array([0.38, 0.0, 0.0])  # +28 cm
        jumped, last, begin, shift = consume_tracker_jump(raw, last, begin, 0.05)
        assert jumped
        np.testing.assert_allclose(last, raw)
        np.testing.assert_allclose(shift, [0.28, 0.0, 0.0])
        np.testing.assert_allclose(begin, [0.38, 0.0, 0.0])
        # Next stable sample at the new world pose is not a jump.
        jumped2, last2, begin2, _ = consume_tracker_jump(raw, last, begin, 0.05)
        assert not jumped2
        np.testing.assert_allclose(begin2, begin)


class TestLatchPublish:
    def test_closed_square_returns_to_zero(self):
        published = np.zeros(3)
        path = []
        for _ in range(20):
            path.append(path[-1] + np.array([0.005, 0.0, 0.0]) if path else np.array([0.005, 0.0, 0.0]))
        corner = path[-1].copy()
        for _ in range(20):
            corner = corner + np.array([0.0, 0.005, 0.0])
            path.append(corner.copy())
        for _ in range(20):
            corner = corner + np.array([-0.005, 0.0, 0.0])
            path.append(corner.copy())
        for _ in range(20):
            corner = corner + np.array([0.0, -0.005, 0.0])
            path.append(corner.copy())
        assert np.linalg.norm(path[-1]) < 1e-12
        for desired in path:
            published = publish_latch_offset(
                desired, published,
                dead_zone_m=0.002, max_delta_m=0.020, home_capture_m=0.040,
            )
        assert np.linalg.norm(published) < 0.003

    def test_dead_zone_steps_still_return_to_zero(self):
        published = np.zeros(3)
        desired = np.zeros(3)
        step = np.array([0.0015, 0.0, 0.0])
        for _ in range(80):
            desired = desired + step
            published = publish_latch_offset(
                desired, published,
                dead_zone_m=0.002, max_delta_m=0.020, home_capture_m=0.040,
            )
        for _ in range(80):
            desired = desired - step
            published = publish_latch_offset(
                desired, published,
                dead_zone_m=0.002, max_delta_m=0.020, home_capture_m=0.040,
            )
        np.testing.assert_allclose(desired, [0.0, 0.0, 0.0], atol=1e-12)
        assert np.linalg.norm(published) < 0.003

    def test_far_tiny_error_is_held(self):
        desired = np.array([0.10, 0.0, 0.0])
        published = np.array([0.1005, 0.0, 0.0])  # 0.5 mm error, far from origin
        out = publish_latch_offset(
            desired, published,
            dead_zone_m=0.002, max_delta_m=0.020, home_capture_m=0.040,
        )
        np.testing.assert_allclose(out, published)

    def test_near_origin_closes_sub_deadzone_error(self):
        desired = np.zeros(3)
        published = np.array([0.0015, 0.0, 0.0])
        out = publish_latch_offset(
            desired, published,
            dead_zone_m=0.002, max_delta_m=0.020, home_capture_m=0.040,
        )
        np.testing.assert_allclose(out, desired, atol=1e-12)

    def test_slew_caps_step(self):
        out = slew_vector(np.zeros(3), np.array([0.10, 0.0, 0.0]), 0.020)
        np.testing.assert_allclose(out, [0.020, 0.0, 0.0], atol=1e-12)

    def test_rotation_closed_path(self):
        from lerobot.utils.rotation import Rotation as Rot

        published = np.eye(3)
        angle = np.radians(20)
        desired = Rot.from_rotvec([0.0, 0.0, angle]).as_matrix()
        published = publish_latch_rotation(
            desired, published,
            dead_zone_rad=np.radians(0.5),
            max_delta_rad=np.radians(10),
            home_capture_m=0.040,
            desired_pos_norm=0.10,
        )
        published = publish_latch_rotation(
            desired, published,
            dead_zone_rad=np.radians(0.5),
            max_delta_rad=np.radians(10),
            home_capture_m=0.040,
            desired_pos_norm=0.10,
        )
        mid = Rot.from_matrix(published).as_rotvec()
        np.testing.assert_allclose(np.linalg.norm(mid), angle, atol=1e-6)
        published = publish_latch_rotation(
            np.eye(3), published,
            dead_zone_rad=np.radians(0.5),
            max_delta_rad=np.radians(10),
            home_capture_m=0.040,
            desired_pos_norm=0.0,
        )
        published = publish_latch_rotation(
            np.eye(3), published,
            dead_zone_rad=np.radians(0.5),
            max_delta_rad=np.radians(10),
            home_capture_m=0.040,
            desired_pos_norm=0.0,
        )
        back = Rot.from_matrix(published).as_rotvec()
        assert np.linalg.norm(back) < 1e-6
