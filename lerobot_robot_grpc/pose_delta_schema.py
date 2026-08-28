"""Shared pose-delta action schema for delta-pose teleoperation.

Both the leader servicer (Pika Sense) and the follower servicer (SO-101
pose_delta mode) import ``ACTION_KEYS`` and ``build_pose_delta_feature_info()``
so the wire schema is defined in exactly one place — preventing drift between
the two sides of the gRPC link.

The schema encodes the **current** end-effector offset from SetReference
(latch-once, refreshed every frame — not a per-frame velocity) as 8
scalar FLOAT32 features:

- ``hand.delta_pos.{x,y,z}``  — translational offset from ``T_begin`` (metres)
- ``hand.delta_rot.{qx,qy,qz,qw}`` — rotational offset from ``T_begin`` (quaternion)
- ``gripper.distance`` — gripper finger distance (millimetres)

The follower assigns ``T_intent = T_zero ⊕ ΔT``.  Identity ΔT returns the
arm to ``T_zero``.  Sending the same offset twice holds.

All features are ``CRITICAL`` scalars ``(1,1,1) RAW`` at ``WATCH_DOG_LEVEL_A``,
matching the convention used by ``SO101LeaderServicer._encode_feature_info``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _grpc_available

if TYPE_CHECKING or _grpc_available:
    from lerobot_robot_grpc.protos import device_pb2
else:
    device_pb2 = None  # type: ignore[assignment]

# Semantic sub-groups — exposed so callers can iterate position / rotation /
# gripper keys independently (e.g. the leader builds a dict with only these keys).
DELTA_POS_KEYS: tuple[str, ...] = ("hand.delta_pos.x", "hand.delta_pos.y", "hand.delta_pos.z")
DELTA_ROT_KEYS: tuple[str, ...] = ("hand.delta_rot.qx", "hand.delta_rot.qy", "hand.delta_rot.qz", "hand.delta_rot.qw")
GRIPPER_KEYS: tuple[str, ...] = ("gripper.distance",)

# The complete ordered key set — what the leader produces and the follower
# consumes. Order matters for neither side (features are keyed by name on the
# wire), but keeping it stable aids log readability.
ACTION_KEYS: tuple[str, ...] = DELTA_POS_KEYS + DELTA_ROT_KEYS + GRIPPER_KEYS


def action_keys(prefix: str | None = None) -> tuple[str, ...]:
    """Return the pose-delta keys, optionally namespaced to one arm.

    The unprefixed schema remains the compatibility default for SO-101.  S1
    uses ``prefix="left"`` (and later ``"right"``) so recorded intent keeps
    its arm identity and can grow into a bimanual schema without renaming.
    """
    if prefix is None or not prefix.strip():
        return ACTION_KEYS
    normalized = prefix.strip().strip(".")
    return tuple(f"{normalized}.{key}" for key in ACTION_KEYS)


def _scalar_feature_info(key: str) -> "device_pb2.OneFeatureInfo":
    """Builds a CRITICAL FLOAT32 scalar feature info for *key*.

    Mirrors ``mock_leader.scalar_feature_info`` but adds ``WATCH_DOG_LEVEL_A``
    to match the real servicer convention (``SO101LeaderServicer``).
    """
    return device_pb2.OneFeatureInfo(
        key=key,
        criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
        watchdog=device_pb2.WatchDogLevel.WATCH_DOG_LEVEL_A,
        type=device_pb2.DataType.FLOAT32,
        shape=device_pb2.ImageShape(H=1, W=1, C=1),
        encoding=device_pb2.Encoding.RAW,
        img_quality=100,
    )


def build_pose_delta_feature_info(
    prefix: str | None = None,
) -> "dict[str, device_pb2.OneFeatureInfo]":
    """Returns the pose-delta action schema keyed by feature name.

    The returned dict is consumed directly by
    ``encode_feature`` / ``load_feature`` (which subscript by key) and by
    ``GetInfoResponse`` (which wraps ``.values()`` into a list).
    """
    return {key: _scalar_feature_info(key) for key in action_keys(prefix)}
