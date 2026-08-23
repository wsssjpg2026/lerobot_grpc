"""Hardware-free acceptance tests for Tracker/global-scene readiness."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lerobot_robot_grpc.leader.tracker_readiness import TrackerReadinessGate
from lerobot_robot_grpc.protos import device_pb2


def _health(
    *,
    epoch: int = 1,
    scene_generation: int = 1,
    cohort=("LH0", "LH1"),
    positions=None,
):
    positions = positions or {
        "LH0": (0.0, 0.0, 0.0),
        "LH1": (2.0, 0.0, 0.0),
    }
    return {
        "bridge_available": True,
        "context_epoch": epoch,
        "global_scene_generation": scene_generation,
        "lighthouse_cohort_generation": len(cohort),
        "discovered_lighthouses": cohort,
        "lighthouses": {
            name: {
                "position": positions[name],
                "rotation": (1.0, 0.0, 0.0, 0.0),
            }
            for name in positions
        },
    }


def _sample(sequence: int, position=(0.2, 0.3, 0.4)):
    return SimpleNamespace(
        optical_event_sequence=sequence,
        optical_age_s=0.0,
        optical_measurement_count=12,
        position=np.asarray(position, dtype=float),
        rotation=np.eye(3),
    )


def _ready(gate: TrackerReadinessGate, health=None):
    health = health or _health()
    gate.update(health, _sample(1), now_s=0.0)
    result = None
    for index in range(20):
        result = gate.update(
            health,
            _sample(index + 2),
            now_s=2.1 + index * 0.06,
        )
    assert result is not None
    assert (
        result.state
        == device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_READY
    )
    assert result.token
    return result


def test_no_alignment_lease_before_new_global_scene_solve() -> None:
    gate = TrackerReadinessGate()
    health = _health(scene_generation=0, positions={})

    gate.update(health, _sample(1), now_s=0.0)
    result = gate.update(health, _sample(2), now_s=2.1)

    assert (
        result.state
        == device_pb2.TrackingReadinessState.
        TRACKING_READINESS_STATE_SOLVING_GLOBAL_SCENE
    )
    assert result.token == ""


def test_ready_requires_stable_cohort_and_distinct_optical_samples() -> None:
    gate = TrackerReadinessGate()
    health = _health()

    first = gate.update(health, _sample(1), now_s=0.0)
    settling = gate.update(health, _sample(2), now_s=1.9)
    ready = _ready(TrackerReadinessGate(), health)

    assert first.state == settling.state == (
        device_pb2.TrackingReadinessState.
        TRACKING_READINESS_STATE_WAITING_LIGHTHOUSE
    )
    assert ready.stable_sample_count >= 20
    assert ready.stable_window_s >= 1.0


def test_operator_motion_after_ready_does_not_revoke_map_lease() -> None:
    gate = TrackerReadinessGate()
    ready = _ready(gate)

    moved = gate.update(_health(), _sample(100, (0.8, -0.4, 1.2)), now_s=5.0)

    assert moved.state == ready.state
    assert moved.token == ready.token


def test_late_lighthouse_revokes_existing_token() -> None:
    gate = TrackerReadinessGate()
    ready = _ready(gate)
    late_health = _health(
        scene_generation=2,
        cohort=("LH0", "LH1", "LH2"),
        positions={
            "LH0": (0.0, 0.0, 0.0),
            "LH1": (2.0, 0.0, 0.0),
            "LH2": (0.0, 2.0, 0.0),
        },
    )

    changed = gate.update(late_health, _sample(100), now_s=5.0)

    assert (
        changed.state
        == device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_MAP_CHANGED
    )
    assert changed.token == ""
    assert not gate.token_is_current(ready.token, changed)


def test_context_restart_revokes_existing_token() -> None:
    gate = TrackerReadinessGate()
    ready = _ready(gate)

    restarted = gate.update(_health(epoch=2), _sample(100), now_s=5.0)

    assert (
        restarted.state
        == device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_MAP_CHANGED
    )
    assert "context changed" in restarted.reason
    assert not gate.token_is_current(ready.token, restarted)


def test_missing_native_bridge_is_fatal_and_never_issues_token() -> None:
    gate = TrackerReadinessGate()

    result = gate.update(
        {
            "bridge_available": False,
            "bridge_error": "missing survive_install_lighthouse_pose_fn",
        },
        None,
        now_s=0.0,
    )

    assert (
        result.state
        == device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_ERROR
    )
    assert result.token == ""
    assert "missing" in result.reason
