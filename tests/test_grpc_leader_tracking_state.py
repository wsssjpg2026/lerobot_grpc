"""Tracking-state propagation through the generic gRPC leader adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import grpc
import pytest

from lerobot_robot_grpc.leader.grpc_leader import (
    GRPCLeader,
    ReferenceNotReadyError,
)
from lerobot_robot_grpc.leader.utils import encode_feature
from lerobot_robot_grpc.protos import device_pb2


def _leader_with_snapshot(*, quality: int, tracking_state: int) -> GRPCLeader:
    feature_info = device_pb2.OneFeatureInfo(
        key="left.hand.delta_pos.x",
        criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
        watchdog=device_pb2.WatchDogLevel.WATCH_DOG_LEVEL_A,
        type=device_pb2.DataType.FLOAT32,
        shape=device_pb2.ImageShape(H=1, W=1, C=1),
        encoding=device_pb2.Encoding.RAW,
    )
    feature = next(
        iter(
            encode_feature(
                {feature_info.key: feature_info},
                {feature_info.key: 0.125},
            )
        )
    )
    feature.quality = quality
    feature.tracking_state = tracking_state

    leader = GRPCLeader.__new__(GRPCLeader)
    leader._is_connected = True
    leader.data_timeout_s = 0.1
    leader.reference_timeout_s = 0.25
    leader.stub = MagicMock()
    leader.stub.GetAction.return_value = iter([feature])
    leader._act_ft_info = {feature_info.key: feature_info}
    leader._latest_act_ft = {feature_info.key: 0.0}
    leader._last_action_quality = device_pb2.FrameQuality.FRAME_QUALITY_GOOD
    leader._last_tracking_state = (
        device_pb2.TrackingState.TRACKING_STATE_UNSPECIFIED
    )
    leader._last_tracking_readiness = None
    leader.stub.GetTrackingReadiness.return_value = device_pb2.TrackingReadiness(
        state=(
            device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_READY
        ),
        token="lease-123",
    )
    return leader


def test_get_action_propagates_confirmation_required_state() -> None:
    leader = _leader_with_snapshot(
        quality=device_pb2.FrameQuality.FRAME_QUALITY_DEGRADED,
        tracking_state=(
            device_pb2.TrackingState.TRACKING_STATE_CONFIRM_REQUIRED
        ),
    )

    action = leader.get_action()

    assert action["left.hand.delta_pos.x"] == 0.125
    assert (
        leader.last_tracking_state
        == device_pb2.TrackingState.TRACKING_STATE_CONFIRM_REQUIRED
    )


def test_get_action_derives_state_for_legacy_server() -> None:
    leader = _leader_with_snapshot(
        quality=device_pb2.FrameQuality.FRAME_QUALITY_STALE,
        tracking_state=device_pb2.TrackingState.TRACKING_STATE_UNSPECIFIED,
    )

    leader.get_action()

    assert (
        leader.last_tracking_state
        == device_pb2.TrackingState.TRACKING_STATE_LOST
    )


def test_get_action_propagates_reference_pending_state() -> None:
    leader = _leader_with_snapshot(
        quality=device_pb2.FrameQuality.FRAME_QUALITY_DEGRADED,
        tracking_state=(
            device_pb2.TrackingState.TRACKING_STATE_REFERENCE_PENDING
        ),
    )

    leader.get_action()

    assert (
        leader.last_tracking_state
        == device_pb2.TrackingState.TRACKING_STATE_REFERENCE_PENDING
    )


class _FailedPrecondition(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.FAILED_PRECONDITION


class _DeadlineExceeded(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED


def test_set_reference_maps_failed_precondition_to_retryable_error() -> None:
    leader = _leader_with_snapshot(
        quality=device_pb2.FrameQuality.FRAME_QUALITY_GOOD,
        tracking_state=device_pb2.TrackingState.TRACKING_STATE_READY,
    )
    leader.address = "leader"
    leader.stub.SetReference.side_effect = _FailedPrecondition()

    with pytest.raises(ReferenceNotReadyError):
        leader.set_reference()


def test_set_reference_uses_dedicated_reference_deadline() -> None:
    leader = _leader_with_snapshot(
        quality=device_pb2.FrameQuality.FRAME_QUALITY_GOOD,
        tracking_state=device_pb2.TrackingState.TRACKING_STATE_READY,
    )
    leader.address = "leader"

    leader.set_reference()

    assert leader.stub.SetReference.call_args.kwargs["timeout"] == pytest.approx(0.25)
    request = leader.stub.SetReference.call_args.args[0]
    assert request.readiness_token == "lease-123"


def test_set_reference_maps_deadline_to_retryable_error() -> None:
    """A reference capture deadline keeps the follower held and is retryable."""
    leader = _leader_with_snapshot(
        quality=device_pb2.FrameQuality.FRAME_QUALITY_DEGRADED,
        tracking_state=device_pb2.TrackingState.TRACKING_STATE_REFERENCE_PENDING,
    )
    leader.address = "leader"
    leader.stub.SetReference.side_effect = _DeadlineExceeded()

    with pytest.raises(ReferenceNotReadyError):
        leader.set_reference()
