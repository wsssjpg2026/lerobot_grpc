"""Tests for the SO101LeaderServicer bus-lock safety mechanism.

These tests verify the bus-lock subsystem (acquire/release/watchdog/poison/recover)
in isolation — the methods under test only manipulate threading state and never
touch the robot hardware, so a mock robot is sufficient.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from lerobot_robot_grpc.leader.so101_leader_server import SO101LeaderServicer


@pytest.fixture
def servicer():
    """A servicer with a mock robot and a short watchdog timeout for fast tests."""
    robot = MagicMock()
    robot.is_connected = False
    s = SO101LeaderServicer(robot, bus_call_timeout_s=0.3)
    yield s
    # Ensure no lingering watchdog trips interfere with subsequent tests
    setattr(s, "_bus_poisoned", True)  # silence watchdog once poisoned


class TestReleaseBusOwnerGuard:
    """Issue: _release_bus must reject non-owner threads (stale-thread safety)."""

    def test_release_by_non_owner_thread_is_ignored(self, servicer):
        """A thread that did not acquire the bus must not release another owner's lock."""
        result = {}

        def acquire_and_block():
            acquired = servicer._acquire_bus("get_action")
            result["acquired"] = acquired
            # Simulate the thread being stuck (never calls _release_bus)
            result["event"] = threading.Event()
            result["event"].wait(timeout=5)

        owner_thread = threading.Thread(target=acquire_and_block, daemon=True)
        owner_thread.start()
        time.sleep(0.1)  # let the owner acquire

        assert result["acquired"] is True
        assert servicer._bus_held is True
        assert servicer._bus_owner == "get_action"

        # A *different* thread (the main test thread) tries to release — must be ignored
        servicer._release_bus()

        # The owner's state must be intact: bus still held by the owner thread
        assert servicer._bus_held is True
        assert servicer._bus_owner == "get_action"

        # Cleanup: let the stuck thread exit
        result["event"].set()
        owner_thread.join(timeout=2)


class TestBusWatchdogPoisons:
    """Issue: watchdog must poison the bus (not force-release the lock).

    Force-releasing lets the next caller write to the same pyserial handle the
    stuck thread is still blocked on — protocol corruption, unsafe motion.
    """

    def test_watchdog_poisons_bus_on_stuck_call(self, servicer):
        """After the watchdog fires on a stuck call, the bus is poisoned and
        further acquisitions are rejected — the lock is NOT force-released."""
        done = threading.Event()

        def acquire_and_block():
            servicer._acquire_bus("get_action")
            done.wait(timeout=5)

        owner_thread = threading.Thread(target=acquire_and_block, daemon=True)
        owner_thread.start()
        time.sleep(0.1)  # let the owner acquire

        assert servicer._bus_held is True

        # Wait for the watchdog to fire (>bus_call_timeout_s, with margin)
        time.sleep(1.0)

        # The bus must be poisoned, not force-released
        assert servicer._bus_poisoned is True

        # A fresh thread must NOT be able to acquire — that would mean the lock
        # was force-released and the stuck owner's serial I/O can be corrupted
        can_acquire = {}

        def try_acquire():
            can_acquire["result"] = servicer._acquire_bus("get_action")

        t = threading.Thread(target=try_acquire, daemon=True)
        t.start()
        t.join(timeout=1.0)
        assert can_acquire["result"] is False, "Bus should be poisoned and reject new acquisitions"

        # Cleanup
        done.set()
        owner_thread.join(timeout=2)


class TestBusLockRecovery:
    """Issue: poisoned bus must be recoverable via _reset_bus_lock_state."""

    def test_reset_clears_poisoned_state(self, servicer):
        """After _reset_bus_lock_state, the bus is un-poisoned and acquisitions work again."""
        done = threading.Event()

        def acquire_and_block():
            servicer._acquire_bus("get_action")
            done.wait(timeout=5)

        owner_thread = threading.Thread(target=acquire_and_block, daemon=True)
        owner_thread.start()
        time.sleep(0.1)

        # Wait for watchdog to poison
        time.sleep(1.0)
        assert servicer._bus_poisoned is True

        # Let the stuck thread exit so the lock is releasable
        done.set()
        owner_thread.join(timeout=2)

        # Reset: the serial port would be reopened on a fresh Connect/Disconnect,
        # so releasing the lock here is safe (no concurrent I/O in flight).
        servicer._reset_bus_lock_state()

        assert servicer._bus_poisoned is False
        assert servicer._bus_held is False

        # Bus must be usable again
        assert servicer._acquire_bus("get_action") is True
        servicer._release_bus()


class TestConnectDisconnectResetBus:
    """Issue: Connect/Disconnect must call _reset_bus_lock_state so a poisoned bus
    recovers when the client reconnects."""

    def test_connect_recovers_poisoned_bus(self, servicer):
        servicer._bus_poisoned = True
        assert servicer._acquire_bus("get_action") is False  # poisoned

        servicer.Connect(request=MagicMock(), context=MagicMock())

        assert servicer._bus_poisoned is False
        assert servicer._acquire_bus("get_action") is True
        servicer._release_bus()

    def test_disconnect_recovers_poisoned_bus(self, servicer):
        servicer._bus_poisoned = True
        assert servicer._acquire_bus("get_action") is False  # poisoned

        servicer.Disconnect(request=MagicMock(), context=MagicMock())

        assert servicer._bus_poisoned is False
        assert servicer._acquire_bus("get_action") is True
        servicer._release_bus()
