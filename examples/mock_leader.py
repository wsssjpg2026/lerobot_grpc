"""Shared no-hardware mock leader for example scripts (E2E tests, demos).

``MockLeaderServicer`` is a drop-in ``LeaderServicer``: two dummy scalar action
joints that climb with wall clock, ``SetReference`` records the call and zeros
the delta origin. Symmetric to ``mock_follower.MockFollowerServicer`` minus
the camera — the leader side produces actions, not observations.
"""

from __future__ import annotations

import threading
import time

from google.protobuf.empty_pb2 import Empty

from lerobot_robot_grpc.leader.leader_server import LeaderServicer
from lerobot_robot_grpc.follower.utils import (
    encode_feature,
)
from lerobot_robot_grpc.protos import device_pb2

# Must match MockFollowerServicer's ACTION_KEYS for combo schema compatibility.
ACTION_KEYS = ("joint_0.pos", "joint_1.pos")


def scalar_feature_info(key: str) -> device_pb2.OneFeatureInfo:
    return device_pb2.OneFeatureInfo(
        key=key,
        criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
        type=device_pb2.DataType.FLOAT32,
        shape=device_pb2.ImageShape(H=1, W=1, C=1),
        encoding=device_pb2.Encoding.RAW,
        img_quality=100,
    )


class MockLeaderServicer(LeaderServicer):
    """A no-hardware leader: actions stream continuously, SetReference resets origin.

    Before ``SetReference`` is called, actions output absolute joint angles
    (climbing with wall clock). After ``SetReference``, actions output the
    delta relative to the reference pose — mirroring the A-class engage protocol
    where the teleop device outputs relative movement after engagement.
    """

    def __init__(self):
        self._act_ft_info = {k: scalar_feature_info(k) for k in ACTION_KEYS}
        self._fb_ft_info = {k: scalar_feature_info(k) for k in ACTION_KEYS}
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        # SetReference records the wall-clock offset at which the reference was locked.
        self._reference_set = False
        self._reference_ts: float | None = None
        self._set_reference_called = False

    @property
    def set_reference_called(self) -> bool:
        """True after SetReference RPC was received — for test assertions."""
        with self._lock:
            return self._set_reference_called

    # --- feature introspection ---
    def GetInfo(self, request, context):
        return device_pb2.GetInfoResponse(
            observation_features=[],  # leader has no observations
            action_features=list(self._act_ft_info.values()),
            feedback_features=list(self._fb_ft_info.values()),
        )

    # --- lifecycle ---
    def Connect(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        return Empty()

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)

    # --- alignment ---
    def SetReference(self, request, context):
        with self._lock:
            self._set_reference_called = True
            self._reference_set = True
            self._reference_ts = time.monotonic()
        return Empty()

    # --- data flow ---
    def GetAction(self, request, context):
        """Send one snapshot of action joints per call.

        GRPCLeader.get_action() calls GetAction with a timeout and reads features
        until the stream ends. Each call yields one complete set of action joints
        (like the real leader server which sends one snapshot per GetAction call).
        """
        with self._lock:
            tick = time.monotonic() - self._t0
            if self._reference_set and self._reference_ts is not None:
                ref = self._reference_ts - self._t0
                action = {
                    "joint_0.pos": float(tick - ref),
                    "joint_1.pos": (float(tick - ref)) * 0.5,
                }
            else:
                action = {
                    "joint_0.pos": float(tick),
                    "joint_1.pos": float(tick) * 0.5,
                }
        yield from encode_feature(self._act_ft_info, action)

    def GetObservation(self, request, context):
        """Leader has no observations — empty stream (protocol requires the method)."""
        return
        yield  # make it a generator

    def SendFeedback(self, request_iterator, context):
        """Receive feedback — no-op for the mock."""
        for _ in request_iterator:
            pass
        return Empty()
