"""Stable bit assignments for action-application diagnostics.

The protobuf transports the fields as integers so that the control hot path
and the optional LeRobot ``safety`` feature remain compact.  Keep assignments
append-only: recorded datasets may outlive the serving binaries.
"""

from __future__ import annotations

from enum import IntFlag


class SafetyFlag(IntFlag):
    NONE = 0
    HELD = 1 << 0
    STALE = 1 << 1
    WORKSPACE = 1 << 2
    IK = 1 << 3
    FK = 1 << 4
    COLLISION = 1 << 5
    IK_JUMP = 1 << 6
    FRAME_CAPPED = 1 << 7
    INPUT_INVALID = 1 << 8
    CHECKER_ERROR = 1 << 9
    LEADER_DEGRADED = 1 << 10
    LEADER_STALE = 1 << 11
    SESSION_HOLD = 1 << 12


class AppliedGroup(IntFlag):
    NONE = 0
    LEFT_ARM = 1 << 0
    LEFT_GRIPPER = 1 << 1
    RIGHT_ARM = 1 << 2
    RIGHT_GRIPPER = 1 << 3


def groups_for_arm(arm: str) -> tuple[AppliedGroup, AppliedGroup]:
    if arm == "left":
        return AppliedGroup.LEFT_ARM, AppliedGroup.LEFT_GRIPPER
    if arm == "right":
        return AppliedGroup.RIGHT_ARM, AppliedGroup.RIGHT_GRIPPER
    raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
