"""Client-side clutch loop helpers (wayfinder #10/#12).

The teleop client is "只搬运" (pure transport).  The leader owns the freeze:
on a disengage edge it stops updating the published arm offset but keeps
reading the gripper every ``GetAction`` — the official PikaAnyArm clutch
gates the arm only, the gripper stays live.  The client's job is edge
detection and the relatch sequence (follower ``SetReference`` → leader
``SetReference``).

These pure functions pin the per-iteration clutch logic so the bench-found
stale-action bug (#12) is covered by unit tests:

- On the **engage edge** the client MUST discard the action it fetched
  before ``relatch()`` — it is frozen at the pre-hold offset, and sending it
  would yank the freshly re-latched follower toward the old offset direction
  for one frame (the follower slew caps it at ``workspace_radius``, so it
  shows as a ~1 cm twitch instead of a full jump, but it is wrong motion).
"""

from __future__ import annotations

from collections.abc import Callable

from lerobot_robot_grpc.protos import device_pb2

RobotAction = dict[str, float]


def auto_clutch_step(
    *,
    status: device_pb2.DeviceStatus,
    engaged: bool,
    raw_action: RobotAction,
    fetch_action: Callable[[], RobotAction],
    relatch: Callable[[], None],
) -> tuple[bool, RobotAction, bool]:
    """One iteration of the auto (leader-status) clutch.

    Args:
        status: leader ``GetStatus`` result this iteration.
        engaged: client-side follow flag from the previous iteration.
        raw_action: action fetched at the top of this iteration (before the
            status poll).  On the engage edge this is the stale frozen action
            and is discarded (#12).
        fetch_action: callable returning a fresh action from the leader.
        relatch: callable performing follower ``SetReference`` → leader
            ``SetReference`` (order matters, see ``relatch`` in the script).

    Returns:
        ``(engaged, action_to_send, should_send)``.  ``should_send`` is True
        for COLLECTION and IDLE — on IDLE the leader freezes the arm offset
        itself and the client keeps transporting the live gripper (official
        grip semantics) — and False on FATAL (nothing may flow from a broken
        leader).
    """
    engaged_now = status == device_pb2.DeviceStatus.COLLECTION
    if engaged_now and not engaged:
        # Engage edge: re-latch both bases FIRST, then fetch a fresh action.
        # The pre-relatch action is the frozen pre-hold offset (#12).
        relatch()
        raw_action = fetch_action()
    engaged = engaged_now
    should_send = status in (
        device_pb2.DeviceStatus.COLLECTION,
        device_pb2.DeviceStatus.IDLE,
    )
    return engaged, raw_action, should_send


def keyboard_clutch_step(
    *,
    engaged: bool,
    key_toggled: bool,
    raw_action: RobotAction,
    fetch_action: Callable[[], RobotAction],
    relatch: Callable[[], None],
) -> tuple[bool, RobotAction, bool]:
    """One iteration of the keyboard (local) clutch.

    Keyboard mode owns the clutch locally — the leader keeps publishing the
    live offset regardless of the local hold, so on a local hold the client
    must stop sending entirely (arm AND gripper freeze; this fallback mode
    has no button for the leader to read).  The engage edge still sequences
    ``relatch()`` first and discards the stale pre-relatch action (#12).
    """
    if key_toggled:
        engaged = not engaged
        if engaged:
            relatch()
            raw_action = fetch_action()
    return engaged, raw_action, engaged
