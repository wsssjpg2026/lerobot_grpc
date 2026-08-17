"""Real SO-101 servicer pose_delta mode tests (wayfinder pika-sense-real #04).

The real SO101FollowerServicer is the second thin adapter over the shared
PoseDeltaLaw (the sim MuJoCoSO101Servicer was the first).  These tests
drive its **public RPC surface** with a mock Feetech bus (no hardware, no gRPC
socket): Connect / SendAction / SetReference / GetObservation / GetFeedback.

Safety posture under test (map Notes, locked): base safety sphere at
max_reach x ratio (URDF-nominal 543 mm), residual-hold 15-20 mm, slew ~5
mm/frame, stale-hold > 1 s (real-only), IK limits from the measured
calibration range.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")  # Any-typed: no py.typed upstream

from lerobot.motors import MotorCalibration  # noqa: E402
from lerobot_robot_grpc.follower.mujoco_follower_server import (  # noqa: E402
    BODY_JOINTS,
    HOME_JOINTS_DEG,
    JOINTS,
    MuJoCoSO101Servicer,
)
from lerobot_robot_grpc.follower.pose_delta_law import BaseSafetySphere  # noqa: E402
from lerobot_robot_grpc.follower.so101_follower_server import (  # noqa: E402
    REAL_REST_POSTURE_DEG,
    SO101FollowerServicer,
)
from lerobot_robot_grpc.pose_delta_schema import ACTION_KEYS  # noqa: E402
from lerobot_robot_grpc.protos import device_pb2  # noqa: E402

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"

# Sim home pose as lerobot-normalised values (degrees body, 0-100 gripper) --
# the pose the mock arm sits in unless a test moves it.
HOME_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 60.0,
    "wrist_flex": -40.0,
    "wrist_roll": 0.0,
    "gripper": 10.0,
}


# ---------------------------------------------------------------------------
# Mock-bus robot (duck-typed SO101FollowerAdapted -- no hardware)
# ---------------------------------------------------------------------------


class _FakeBus:
    """Stands in for FeetechMotorsBus: sync_read returns fixed normalised
    positions (degrees body / 0-100 gripper, as the real bus does after
    calibration is registered)."""

    def __init__(self, positions):
        self._positions = dict(positions)
        self.sync_read_calls = 0

    def sync_read(self, data_name, motors=None, normalize=True, num_retry=0):
        assert data_name == "Present_Position"
        self.sync_read_calls += 1
        return dict(self._positions)


class _FakeRobotConfig:
    cameras: dict = {}
    num_read_retries = 2


class _FakeRobot:
    """Mock-bus SO-101: send_action teleports the arm to the commanded
    joints (a perfect position servo), so later reads see the new pose."""

    def __init__(self, positions=None, calibrated=True):
        self.bus = _FakeBus(positions or HOME_POS)
        self.config = _FakeRobotConfig()
        self.is_connected = False
        self.is_calibrated = calibrated
        self.calibration: dict | None = None
        self.sent_actions = []
        self.latest_action = None

    @property
    def observation_features(self):
        return {f"{j}.pos": float for j in JOINTS}

    @property
    def action_features(self):
        return {f"{j}.pos": float for j in JOINTS}

    def connect(self, calibrate=False):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        return {f"{j}.pos": self.bus._positions[j] for j in JOINTS}

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        for key, val in action.items():
            if key.endswith(".pos"):
                self.bus._positions[key[:-len(".pos")]] = val
        return action


def _make_servicer(robot=None, **kwargs) -> SO101FollowerServicer:
    # The fake is a deliberate duck type, not an SO101FollowerAdapted subclass.
    return SO101FollowerServicer(cast(Any, robot or _FakeRobot()), **kwargs)


def _law_of(servicer: SO101FollowerServicer):
    """Narrows the Optional law for tests (pose_delta servicers always have one)."""
    assert servicer._law is not None
    return servicer._law


def _connect(servicer) -> None:
    resp = servicer.Connect(None, None)
    assert resp.status == device_pb2.CalibrationStatus.CALIBRATED


# ---------------------------------------------------------------------------
# Slice 1 -- action_mode construction
# ---------------------------------------------------------------------------


class TestActionModeConstruction:
    def test_default_is_joint_mode_with_no_law(self):
        servicer = _make_servicer()
        assert servicer._action_mode == "joint"
        assert servicer._law is None

    def test_invalid_action_mode_raises(self):
        with pytest.raises(ValueError, match="action_mode"):
            _make_servicer(action_mode="cartesian_teleport")

    def test_pose_delta_builds_law_with_real_safety_posture(self):
        servicer = _make_servicer(action_mode="pose_delta")
        law = _law_of(servicer)
        assert isinstance(law._workspace_policy, BaseSafetySphere)
        # URDF-nominal max reach (#02) until #05/#07 calibration says otherwise.
        assert law.max_reach_m == pytest.approx(0.543)
        assert law._residual_hold_m == pytest.approx(0.018)
        # Real-arm slew: ~5 mm/frame, not the sim 15 mm debug speed.
        assert law._workspace_radius_m == pytest.approx(0.005)

    def test_pose_delta_action_schema_is_the_shared_pose_delta_keys(self):
        servicer = _make_servicer(action_mode="pose_delta")
        info = servicer.GetInfo(None, None)
        assert {f.key for f in info.action_features} == set(ACTION_KEYS)

    def test_joint_mode_action_schema_stays_joint_space(self):
        servicer = _make_servicer()
        info = servicer.GetInfo(None, None)
        assert {f.key for f in info.action_features} == {f"{j}.pos" for j in JOINTS}


# ---------------------------------------------------------------------------
# Slice 2 -- Connect locks the law reference from Present_Position
# ---------------------------------------------------------------------------


def _independent_fk_pos(positions) -> np.ndarray:
    """FK of the gripperframe site computed on a SEPARATE MuJoCo model -- the
    independent source of truth for what the law should have latched."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    qpos = [math.radians(positions[j]) for j in BODY_JOINTS]
    gripper_rad = (positions["gripper"] / 100.0) * (1.74533 - (-0.17453)) + (-0.17453)
    qpos.append(gripper_rad)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    return data.site_xpos[sid].copy()


class TestConnectLocksReference:
    def test_connect_locks_t_zero_at_fk_of_present_position(self):
        robot = _FakeRobot()
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        np.testing.assert_allclose(
            _law_of(servicer).t_zero[:3, 3], _independent_fk_pos(HOME_POS), atol=1e-6
        )

    def test_connect_reads_the_bus_under_the_lock(self):
        robot = _FakeRobot()
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        assert robot.bus.sync_read_calls >= 1

    def test_uncalibrated_connect_defers_the_latch(self):
        """Without calibration the bus cannot normalise Present_Position, so
        Connect must not latch; the first solve auto-latches instead."""
        robot = _FakeRobot(calibrated=False)
        servicer = _make_servicer(robot, action_mode="pose_delta")
        resp = servicer.Connect(None, None)
        assert resp.status == device_pb2.CalibrationStatus.NEED_TO_CALIBRATE
        with pytest.raises(AssertionError, match="lock_reference"):
            _ = _law_of(servicer).t_zero

    def test_joint_mode_connect_never_touches_the_law(self):
        robot = _FakeRobot()
        servicer = _make_servicer(robot)
        _connect(servicer)
        assert servicer._law is None



# ---------------------------------------------------------------------------
# Slice 3 -- SendAction: protobuf delta -> law -> send_action (one bus hold)
# ---------------------------------------------------------------------------


def _delta(dx=0.0, dy=0.0, dz=0.0, gripper_mm=30.0):
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


def _action_request(servicer, action_dict):
    """Encode a dict onto the wire format -- a REAL protobuf Action request."""
    from lerobot_robot_grpc.follower.utils import encode_feature

    return device_pb2.Action(
        features=list(encode_feature(servicer._act_ft_info, action_dict))
    )


def _decode_response(servicer, action_msg):
    from lerobot_robot_grpc.follower.utils import load_feature

    out = {}
    for feat in action_msg.features:
        load_feature(feat, servicer._act_ft_info, out, aux_behavior="ignore")
    return out


def _fk_of_commanded(joint_action) -> np.ndarray:
    """FK of a lerobot-normalised joint action on a separate model."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    qpos = [math.radians(joint_action[f"{j}.pos"]) for j in BODY_JOINTS]
    qpos.append(
        (joint_action["gripper.pos"] / 100.0) * (1.74533 - (-0.17453)) + (-0.17453)
    )
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    return data.site_xpos[sid].copy()


class TestSendActionPoseDelta:
    def test_delta_action_writes_lerobot_normalised_joints(self):
        robot = _FakeRobot()
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        assert len(robot.sent_actions) == 1
        sent = robot.sent_actions[0]
        # The bus only ever sees lerobot-normalised joint targets.
        assert set(sent) == {f"{j}.pos" for j in JOINTS}
        assert 0.0 <= sent["gripper.pos"] <= 100.0

    def test_commanded_joints_reach_the_intent(self):
        """A +10 mm x intent (unlimited slew) lands the commanded FK ~10 mm
        ahead of the latched home pose -- checked on an independent model."""
        robot = _FakeRobot()
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        moved = _fk_of_commanded(robot.sent_actions[0]) - _independent_fk_pos(HOME_POS)
        np.testing.assert_allclose(moved, [0.010, 0.0, 0.0], atol=2e-3)

    def test_slew_limits_one_frame_to_five_mm(self):
        """Real-arm slew: a 30 mm intent moves the first command only ~5 mm."""
        robot = _FakeRobot()
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.030)), None)
        dist = float(
            np.linalg.norm(
                _fk_of_commanded(robot.sent_actions[0]) - _independent_fk_pos(HOME_POS)
            )
        )
        assert dist == pytest.approx(0.005, abs=2e-3)

    def test_base_safety_sphere_clamps_beyond_reach_intent(self):
        """The base sphere caps the absolute intent at ratio x max_reach even
        when the leader asks for 2 m away."""
        robot = _FakeRobot()
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=2.0)), None)
        radius = float(np.linalg.norm(_fk_of_commanded(robot.sent_actions[0])))
        assert radius <= 0.72 * 0.543 + 0.02  # sphere + IK residual slack

    def test_response_echoes_the_wire_action(self):
        robot = _FakeRobot()
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        resp = servicer.SendAction(
            _action_request(servicer, _delta(dx=0.010, gripper_mm=45.0)), None
        )
        echoed = _decode_response(servicer, resp)
        assert echoed["hand.delta_pos.x"] == pytest.approx(0.010)
        assert echoed["gripper.distance"] == pytest.approx(45.0)

    def test_read_solve_write_happen_under_one_bus_hold(self):
        """Bus contention contract: one SendAction = one lock acquisition
        covering the Present_Position read AND the send_action write, so the
        30 Hz GetObservation stream interleaves between actions, never mid-
        action."""
        robot = _FakeRobot()
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        robot.bus.sync_read_calls = 0
        acquires = []
        orig = servicer._acquire_bus
        def spy(what, timeout=0.0):
            acquires.append(what)
            return orig(what, timeout=timeout)
        servicer._acquire_bus = spy
        try:
            servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        finally:
            servicer._acquire_bus = orig
        assert acquires == ["send_action"]
        assert robot.bus.sync_read_calls == 1  # exactly one read inside that hold

    def test_joint_mode_send_action_is_untouched(self):
        robot = _FakeRobot()
        servicer = _make_servicer(robot)
        _connect(servicer)
        joint_action = {f"{j}.pos": 1.0 for j in JOINTS}
        servicer.SendAction(_action_request(servicer, joint_action), None)
        assert robot.sent_actions[-1] == joint_action



# ---------------------------------------------------------------------------
# Slice 4 -- stale-hold: >1 s of leader silence freezes the arm
# ---------------------------------------------------------------------------


class TestStaleHold:
    def test_action_after_long_silence_holds_last_joints(self):
        """Real-only safety: an action arriving > stale_timeout after the
        previous one is treated as a dead stream -- the body joints hold,
        only the gripper follows."""
        robot = _FakeRobot()
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        held = robot.sent_actions[-1]
        # Simulate >1 s of silence, then a very different intent arrives.
        last = servicer._last_action_monotonic
        assert last is not None
        servicer._last_action_monotonic = last - 2.0
        servicer.SendAction(
            _action_request(servicer, _delta(dx=0.050, gripper_mm=55.0)), None
        )
        for j in BODY_JOINTS:
            assert robot.sent_actions[-1][f"{j}.pos"] == held[f"{j}.pos"]
        # Gripper still tracks (holds are body-only, per the law contract).
        assert robot.sent_actions[-1]["gripper.pos"] != held["gripper.pos"]

    def test_actions_within_timeout_are_not_stale(self):
        robot = _FakeRobot()
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        first = robot.sent_actions[-1]
        servicer.SendAction(_action_request(servicer, _delta(dx=0.020)), None)
        assert robot.sent_actions[-1]["shoulder_pan.pos"] != first["shoulder_pan.pos"] or \
            robot.sent_actions[-1]["shoulder_lift.pos"] != first["shoulder_lift.pos"]

    def test_first_action_after_connect_is_never_stale(self):
        robot = _FakeRobot()
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        moved = _fk_of_commanded(robot.sent_actions[0]) - _independent_fk_pos(HOME_POS)
        assert float(np.linalg.norm(moved)) > 0.005



# ---------------------------------------------------------------------------
# Slice 5 -- SetReference: re-lock T_zero at the current pose, no motion
# ---------------------------------------------------------------------------


class TestSetReference:
    def test_set_reference_relocks_at_current_pose_without_moving(self):
        """Clutch re-engage contract on the real adapter: T_zero re-latches at
        the measured arm pose (not the Connect home), and the re-latch itself
        writes nothing to the bus."""
        robot = _FakeRobot()
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        # Teleop: follow +10 mm; the mock arm teleports to the commanded pose.
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        stop_fk = _independent_fk_pos(
            {j: robot.bus._positions[j] for j in JOINTS}
        )
        assert np.linalg.norm(stop_fk - _independent_fk_pos(HOME_POS)) > 0.005

        actions_before = len(robot.sent_actions)
        servicer.SetReference(None, None)
        assert len(robot.sent_actions) == actions_before  # re-latch moves nothing
        np.testing.assert_allclose(
            _law_of(servicer).t_zero[:3, 3], stop_fk, atol=1e-6
        )

        # A zero delta after re-engage targets the stop pose, not Connect home.
        servicer.SendAction(_action_request(servicer, _delta(dx=0.0)), None)
        np.testing.assert_allclose(
            _law_of(servicer).target_pose[:3, 3], stop_fk, atol=1e-6
        )

    def test_set_reference_joint_mode_is_a_noop(self):
        servicer = _make_servicer()
        _connect(servicer)
        resp = servicer.SetReference(None, None)
        assert resp is not None  # Empty, no UNIMPLEMENTED error



# ---------------------------------------------------------------------------
# Slice 6 -- measured calibration range tightens the IK joint limits
# ---------------------------------------------------------------------------


def _calibration(elbow_range=(1024, 3072)) -> dict:
    """Full-turn (0..4095) everywhere except elbow_flex, which is limited to
    1024..3072 raw ticks = +/-90.02 deg about the DEGREES-mode mid-range."""
    calib = {}
    for i, j in enumerate(JOINTS):
        lo, hi = elbow_range if j == "elbow_flex" else (0, 4095)
        calib[j] = MotorCalibration(
            id=i + 1, drive_mode=0, homing_offset=0, range_min=lo, range_max=hi
        )
    return calib


class TestCalibrationQposLimits:
    def test_connect_tightens_ik_limits_to_measured_range(self):
        robot = _FakeRobot()
        robot.calibration = _calibration()
        servicer = _make_servicer(robot, action_mode="pose_delta", elbow_max_deg=None)
        lo0, hi0 = _law_of(servicer).ik_solver.qpos_limits
        _connect(servicer)
        lo, hi = _law_of(servicer).ik_solver.qpos_limits
        elbow = BODY_JOINTS.index("elbow_flex")
        half_span_rad = math.radians((3072 - 1024) / 2 * 360 / 4095)
        assert hi[elbow] == pytest.approx(half_span_rad, abs=1e-6)
        assert lo[elbow] == pytest.approx(-half_span_rad, abs=1e-6)
        assert hi[elbow] < hi0[elbow]  # tighter than the URDF range (1.69 rad)

    def test_full_turn_calibration_keeps_the_model_range(self):
        """A 0..4095 calibration (+/-180 deg) is WIDER than the model range
        (wrist_roll is asymmetric), so the model limit wins -- limits only
        ever tighten."""
        robot = _FakeRobot()
        robot.calibration = _calibration()
        servicer = _make_servicer(robot, action_mode="pose_delta")
        lo0, hi0 = _law_of(servicer).ik_solver.qpos_limits
        _connect(servicer)
        lo, hi = _law_of(servicer).ik_solver.qpos_limits
        wr = BODY_JOINTS.index("wrist_roll")
        assert hi[wr] == pytest.approx(hi0[wr])
        assert lo[wr] == pytest.approx(lo0[wr])

    def test_no_calibration_keeps_model_limits(self):
        robot = _FakeRobot()  # calibration None despite is_calibrated flag
        servicer = _make_servicer(robot, action_mode="pose_delta")
        lo0, hi0 = _law_of(servicer).ik_solver.qpos_limits
        _connect(servicer)
        lo, hi = _law_of(servicer).ik_solver.qpos_limits
        np.testing.assert_allclose(lo, lo0)
        np.testing.assert_allclose(hi, hi0)

    def test_ik_solution_respects_the_measured_elbow_limit(self):
        """Behavioural: with the elbow calibrated to +/-90 deg, a far intent
        never commands the elbow past the measured range."""
        robot = _FakeRobot()
        robot.calibration = _calibration()
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dz=-0.20)), None)
        sent = robot.sent_actions[-1]
        half_span_deg = (3072 - 1024) / 2 * 360 / 4095
        assert abs(sent["elbow_flex.pos"]) <= half_span_deg + 0.5



# ---------------------------------------------------------------------------
# Slice 7 -- characterization: protobuf SendAction -> GetObservation readback
# (locks the PUBLIC contract end-to-end; sim + real share the helpers)
# ---------------------------------------------------------------------------


class _StreamContext:
    """Minimal server-streaming context stand-in (always active, no peer)."""

    def is_active(self):
        return True


def _obs_feature_info(servicer):
    """Observation schema fetched through the PUBLIC GetInfo RPC."""
    info = servicer.GetInfo(None, None)
    return {f.key: f for f in info.observation_features}


def _drain_observation(servicer, max_ticks):
    """Pull up to max_ticks of the GetObservation stream and decode the joint
    features from the LAST tick via the public schema."""
    from lerobot_robot_grpc.follower.utils import load_feature

    info = _obs_feature_info(servicer)
    gen = servicer.GetObservation(None, _StreamContext())
    obs: dict = {}
    ticks = 0
    try:
        for feat in gen:
            load_feature(feat, info, obs, aux_behavior="ignore")
            if len([k for k in obs if k.endswith(".pos")]) >= len(JOINTS):
                ticks += 1
                if ticks >= max_ticks:
                    break
    finally:
        gen.close()
    return obs


class TestCharacterizationRealServicer:
    def test_protobuf_send_action_then_observation_reads_back(self):
        """End-to-end on the public surface: a wire-format pose_delta action
        moves the arm; the observation stream reads the new joints back.
        Seeded at the measured R1 droop pose -- a REAL-reachable park (the
        sim HOME_POS elbow +60 sits past the over-fold wall, so the real
        adapter's default rest posture biases solves away from it)."""
        robot = _FakeRobot(DROOP_POS)
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        obs = _drain_observation(servicer, max_ticks=2)
        # The mock arm teleported to the command; the stream must report it.
        for j in JOINTS:
            assert obs[f"{j}.pos"] == pytest.approx(
                robot.sent_actions[0][f"{j}.pos"], abs=1e-6
            )
        moved = _fk_of_commanded(obs) - _independent_fk_pos(DROOP_POS)
        np.testing.assert_allclose(moved, [0.010, 0.0, 0.0], atol=2e-3)


class TestCharacterizationSimServicer:
    def test_protobuf_send_action_then_observation_reads_back(self):
        """Same contract on the sim adapter: wire action in, joints out of the
        observation stream, converging to the +10 mm intent."""
        servicer = MuJoCoSO101Servicer(
            xml_path=str(XML_PATH),
            action_mode="pose_delta",
            render=False,
            position_deadband_m=0.0,
            rotation_deadband_rad=0.0,
            workspace_radius_m=0.0,
        )
        servicer.Connect(None, None)
        # Draining the sim stream IS settling: physics only advances while the
        # generator is iterated.
        before = _drain_observation(servicer, max_ticks=1)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        obs = _drain_observation(servicer, max_ticks=120)
        assert obs["elbow_flex.pos"] != pytest.approx(before["elbow_flex.pos"], abs=0.5) or \
            obs["shoulder_lift.pos"] != pytest.approx(before["shoulder_lift.pos"], abs=0.5)
        moved = _fk_of_commanded(obs) - _independent_fk_pos(HOME_POS)
        np.testing.assert_allclose(moved, [0.010, 0.0, 0.0], atol=4e-3)



# ---------------------------------------------------------------------------
# Review-driven additions (code-review of #04)
# ---------------------------------------------------------------------------


class TestRealArmSlewBand:
    def test_joint_slew_stays_in_the_real_arm_band(self):
        """Spec: real-arm joint slew is 3-6 deg/tick (with the ~5 mm/frame
        Cartesian slew).  A 30 mm intent must move no joint past the band top
        in one action."""
        robot = _FakeRobot()
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.030)), None)
        sent = robot.sent_actions[-1]
        for j in BODY_JOINTS:
            delta_deg = abs(sent[f"{j}.pos"] - HOME_POS[j])
            assert delta_deg <= 6.0, f"{j} moved {delta_deg:.2f} deg in one tick"


class TestResidualHoldWiring:
    def test_residual_hold_reaches_the_law_on_the_real_adapter(self):
        """The 15-20 mm residual-hold is law-side and tested there; this locks
        the WIRING: with an absurdly tight threshold (0.0 -> any nonzero IK
        residual holds), a second distinct intent returns the held joints."""
        robot = _FakeRobot()
        servicer = _make_servicer(
            robot,
            action_mode="pose_delta",
            workspace_radius_m=0.0,
            residual_hold_m=0.0,  # any nonzero residual holds (never exactly 0)
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dx=0.010)), None)
        held = robot.sent_actions[-1]  # first solve: no previous action, no hold
        servicer.SendAction(_action_request(servicer, _delta(dx=0.020)), None)
        for j in BODY_JOINTS:
            assert robot.sent_actions[-1][f"{j}.pos"] == held[f"{j}.pos"]


class TestRecalibrationWidensLimits:
    def test_reconnect_after_looser_recalibration_widens_limits_back(self):
        """set_qpos_limits only narrows -- so Connect must reset to the model
        range first, or a re-calibration with a WIDER measured range would
        silently keep the old tight limits for the server lifetime."""
        robot = _FakeRobot()
        robot.calibration = _calibration(elbow_range=(1024, 3072))  # +/-90 deg
        servicer = _make_servicer(robot, action_mode="pose_delta", elbow_max_deg=None)
        _connect(servicer)
        elbow = BODY_JOINTS.index("elbow_flex")
        _, hi_tight = _law_of(servicer).ik_solver.qpos_limits
        assert hi_tight[elbow] == pytest.approx(1.5712, abs=1e-3)

        robot.calibration = _calibration(elbow_range=(0, 4095))  # full turn
        robot.is_connected = False  # simulate a fresh session
        _connect(servicer)
        _, hi = _law_of(servicer).ik_solver.qpos_limits
        assert hi[elbow] == pytest.approx(1.69)  # model range restored


# ---------------------------------------------------------------------------
# Slice 8 -- elbow over-fold hard wall (#05 bench finding, #07 mandate)
# ---------------------------------------------------------------------------


class TestElbowHardWall:
    """The physical arm binds at ~+3-4 deg past the folded-rest calibration
    zero (over-fold direction); the recorded calibration range overstates that
    side.  The IK elbow ceiling must be cut to the wall so no solve commands
    the arm into it (positive = over-fold, model qpos == normalised degrees)."""

    def test_default_wall_caps_elbow_ceiling_at_two_deg(self):
        robot = _FakeRobot()
        robot.calibration = _calibration()  # +/-90 deg recorded
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        lo, hi = _law_of(servicer).ik_solver.qpos_limits
        elbow = BODY_JOINTS.index("elbow_flex")
        assert hi[elbow] == pytest.approx(math.radians(2.0), abs=1e-6)
        # The under-fold side keeps the measured calibration range.
        half_span_rad = math.radians((3072 - 1024) / 2 * 360 / 4095)
        assert lo[elbow] == pytest.approx(-half_span_rad, abs=1e-6)

    def test_wall_wins_over_full_turn_calibration(self):
        """A full-turn recording (+/-180 deg, the actual follower.json range
        class #05 called wrong) still cannot widen past the wall."""
        robot = _FakeRobot()
        robot.calibration = _calibration(elbow_range=(0, 4095))
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        _, hi = _law_of(servicer).ik_solver.qpos_limits
        elbow = BODY_JOINTS.index("elbow_flex")
        assert hi[elbow] == pytest.approx(math.radians(2.0), abs=1e-6)

    def test_elbow_max_deg_none_disables_the_wall(self):
        robot = _FakeRobot()
        robot.calibration = _calibration()
        servicer = _make_servicer(robot, action_mode="pose_delta", elbow_max_deg=None)
        _connect(servicer)
        _, hi = _law_of(servicer).ik_solver.qpos_limits
        elbow = BODY_JOINTS.index("elbow_flex")
        half_span_rad = math.radians((3072 - 1024) / 2 * 360 / 4095)
        assert hi[elbow] == pytest.approx(half_span_rad, abs=1e-6)

    def test_wall_survives_a_wider_recalibration(self):
        robot = _FakeRobot()
        robot.calibration = _calibration(elbow_range=(1024, 3072))
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        robot.calibration = _calibration(elbow_range=(0, 4095))
        robot.is_connected = False
        _connect(servicer)
        _, hi = _law_of(servicer).ik_solver.qpos_limits
        elbow = BODY_JOINTS.index("elbow_flex")
        assert hi[elbow] == pytest.approx(math.radians(2.0), abs=1e-6)

    def test_ik_never_commands_the_elbow_past_the_wall(self):
        """Behavioural: with a full-turn recording the solver may want elbow
        past +2 deg, but the commanded joint action stays under the wall."""
        robot = _FakeRobot()
        robot.calibration = _calibration(elbow_range=(0, 4095))
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        servicer.SendAction(_action_request(servicer, _delta(dz=-0.20)), None)
        sent = robot.sent_actions[-1]
        assert sent["elbow_flex.pos"] <= 2.0 + 0.5


# ---------------------------------------------------------------------------
# Regression: Calibrate raced the observation stream on the serial bus
# (lerobot-calibrate on an uncalibrated real arm -> "Port is in use!").
# The handler logged is_calibrated BEFORE taking the bus lock; each
# is_calibrated read is a raw EEPROM read on the same port the streaming
# GetObservation loop reads under the lock.
# ---------------------------------------------------------------------------

import threading  # noqa: E402

from lerobot.utils.errors import DeviceNotConnectedError  # noqa: E402


class _GuardingBus(_FakeBus):
    """Fake bus that TRIPS on overlapping access (the real SDK's 'Port is in
    use!') and blocks inside sync_read until released -- so the test controls
    the interleave instead of rolling dice."""

    def __init__(self, positions):
        super().__init__(positions)
        self.release = threading.Event()
        self.inside = threading.Event()
        self.violations: list[str] = []
        self._inside = False
        self._guard = threading.Lock()

    def _enter(self, what: str):
        with self._guard:
            if self._inside:
                self.violations.append(what)
                raise RuntimeError(f"concurrent bus access: {what}")
            self._inside = True
        self.inside.set()

    def _exit(self):
        with self._guard:
            self._inside = False
        self.inside.clear()

    def sync_read(self, data_name, motors=None, normalize=True, num_retry=0):
        self._enter("sync_read")
        try:
            self.release.wait(timeout=5.0)
        finally:
            self._exit()
        return dict(self._positions)

    def read(self, data_name, motor, normalize=True, num_retry=0):
        self._enter(f"read:{data_name}")
        try:
            return 0
        finally:
            self._exit()


class _UncalibratedRobot(_FakeRobot):
    """Uncalibrated fake whose bus accesses mirror the real robot: observation
    goes through bus.sync_read, is_calibrated through a raw EEPROM read."""

    def __init__(self):
        super().__init__(calibrated=False)
        self.bus = _GuardingBus(HOME_POS)

    def get_observation(self):
        pos = self.bus.sync_read("Present_Position")
        return {f"{j}.pos": pos[j] for j in JOINTS}

    @property
    def is_calibrated(self):  # like FeetechMotorsBus.is_calibrated: a bus read
        self.bus.read("Min_Position_Limit", "shoulder_pan", normalize=False)
        return False

    @is_calibrated.setter
    def is_calibrated(self, _value):  # _FakeRobot.__init__ assigns; ignore
        pass


class _Ctx:
    def __init__(self):
        self._stop = threading.Event()

    def is_active(self):
        return not self._stop.is_set()

    def peer(self):
        return "test"

    def stop(self):
        self._stop.set()


def _drain_obs_stream(servicer, ctx):
    """Consume one observation snapshot (blocks inside the bus read until the
    test releases it), then stop the stream."""
    gen = servicer.GetObservation(None, ctx)
    next(gen)  # hold the bus mid-read
    ctx.stop()
    gen.close()


def test_calibrate_never_touches_bus_outside_the_lock():
    servicer = _make_servicer(robot=_UncalibratedRobot(), bus_call_timeout_s=0.3)
    bus = cast(_GuardingBus, servicer.robot.bus)
    servicer.Connect(None, None)
    ctx = _Ctx()
    obs_thread = threading.Thread(target=_drain_obs_stream, args=(servicer, ctx), daemon=True)
    obs_thread.start()
    # Wait until the obs stream is genuinely inside its bus read.
    assert bus.inside.wait(timeout=2.0)

    # With the bug this read races the stream (RuntimeError from the guard);
    # fixed, Calibrate waits on the bus lock and times out cleanly instead.
    with pytest.raises(DeviceNotConnectedError, match="bus lock busy"):
        servicer.Calibrate(device_pb2.CalibrateRequest(force=False), ctx)
    assert bus.violations == []

    bus.release.set()
    obs_thread.join(timeout=5.0)
    assert not obs_thread.is_alive()


# ---------------------------------------------------------------------------
# Slice 9 -- real IK rest posture: the R1 droop-freeze fix (#07)
# ---------------------------------------------------------------------------

# R1 measured gravity-droop park pose (CSV last row, deg / 0-100): near
# singular (manip 0.0138), elbow riding the over-fold wall.
DROOP_POS = {
    "shoulder_pan": 6.286,
    "shoulder_lift": -2.418,
    "elbow_flex": 1.099,
    "wrist_flex": 46.418,
    "wrist_roll": -5.582,
    "gripper": 0.943,
}


class TestRealRestPosture:
    """R1 freeze root cause: the law's rest posture (nullspace attractor +
    limit-escape re-seed) was the sim home with elbow +60 deg -- unreachable
    on the real arm (over-fold wall at +2 deg), so every solve pinned into the
    wall/singularity and the body froze while the gripper followed.

    The real adapter defaults to a reachable mid-range rest posture instead.
    It is solver bias ONLY -- the arm is never commanded there (Connect and
    SetReference write nothing; teleop starts from the unpowered droop pose
    the user parks the arm in)."""

    def test_real_default_rest_is_reachable_not_sim_home(self):
        servicer = _make_servicer(action_mode="pose_delta")
        law = _law_of(servicer)
        assert law._home_joints_deg == REAL_REST_POSTURE_DEG
        assert REAL_REST_POSTURE_DEG != HOME_JOINTS_DEG

    def test_sim_adapter_keeps_the_sim_home(self):
        servicer = MuJoCoSO101Servicer(
            xml_path=str(XML_PATH), action_mode="pose_delta", render=False
        )
        assert servicer._law is not None
        assert servicer._law._home_joints_deg == HOME_JOINTS_DEG

    def test_rest_lies_within_wall_tightened_limits(self):
        """The rest posture must be INSIDE the effective limits the real arm
        solves under (measured calibration intersected with the elbow wall)."""
        robot = _FakeRobot()
        robot.calibration = _calibration(elbow_range=(0, 4095))  # full turn + wall
        servicer = _make_servicer(robot, action_mode="pose_delta")
        _connect(servicer)
        lo, hi = _law_of(servicer).ik_solver.qpos_limits
        rest = np.radians(np.asarray(REAL_REST_POSTURE_DEG, dtype=float))
        assert np.all(rest >= lo)
        assert np.all(rest <= hi)
        elbow = BODY_JOINTS.index("elbow_flex")
        assert rest[elbow] <= math.radians(2.0)  # under the over-fold wall

    def test_droop_seed_tracks_the_intent(self):
        """Behavioural regression of the R1 freeze: arm parked at the measured
        droop pose, full-turn calibration + default wall, a 15 mm x intent --
        the commanded FK must land ON the intent.  With the old sim-home rest
        the solve pinned ~27 mm off the intent and froze there (body frozen,
        gripper following)."""
        robot = _FakeRobot(DROOP_POS)
        robot.calibration = _calibration(elbow_range=(0, 4095))
        servicer = _make_servicer(
            robot, action_mode="pose_delta", workspace_radius_m=0.0
        )
        _connect(servicer)
        for _ in range(6):
            servicer.SendAction(_action_request(servicer, _delta(dx=0.015)), None)
        target = _independent_fk_pos(DROOP_POS) + np.array([0.015, 0.0, 0.0])
        cmd_fk = _fk_of_commanded(robot.sent_actions[-1])
        assert float(np.linalg.norm(cmd_fk - target)) < 0.005

    def test_real_adapter_wires_solver_bias_params(self):
        """No reachable home to snap back to: the near-reference rest gain
        matches the far gain (the sim's 0.40 snap would crawl a real arm),
        and the elbow-flip escape threshold is re-based -- the sim's -15 deg
        (sign vs the +60 sim home) sits inside the real arm's all-negative
        working range and would fire every frame."""
        servicer = _make_servicer(action_mode="pose_delta")
        law = _law_of(servicer)
        assert law._home_rest_gain == pytest.approx(0.08)
        assert law._escape_flipped_deg == pytest.approx(math.radians(-60.0))
