"""Follower backends for the response suite — swappable by design.

The sim backend (this round) runs the real :class:`MuJoCoSO101Servicer`
in-process behind a real :class:`FollowerServer` gRPC server bound to a
random localhost port — the full wire stack (GetInfo / Connect /
SendAction / GetObservation / SetReference) is exercised, and the client
is the real :class:`GRPCFollower`.

The real backend is the interface for stage B-3 (human present): a server
started out-of-band via ``examples/serve_so101_follower.py
--action_mode=pose_delta``; the suite attaches to it by address.  Nothing
here ever touches a serial port — starting the real server stays a manual
step.

Because the sim servicer lives in-process, the backend can also expose the
shared law's per-solve :class:`JointSolution` records (a read-only spy over
``law.solve``).  That is the only place the safety-stack state (rejected /
held / jumped / stale flags) is observable — neither adapter surfaces it
over RPC (see the report's findings section).
"""

from __future__ import annotations

import socket
import time
from abc import ABC, abstractmethod
from pathlib import Path

from google.protobuf.empty_pb2 import Empty

from lerobot_robot_grpc.follower.config_grpc import GRPCFollowerConfig
from lerobot_robot_grpc.follower.follower_server import (
    FollowerServer,
    FollowerServerConfig,
)
from lerobot_robot_grpc.follower.grpc_follower import GRPCFollower
from lerobot_robot_grpc.follower.utils import TeleopStats

PKG_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = PKG_ROOT / "assets" / "so101" / "scene.xml"

# Real-backend launch command (stage B-3, human-triggered — never started by
# the test suite).  Kept here so report + docs quote one source of truth.
REAL_LAUNCH_CMD = (
    "conda run -n lerobot-grpc-serve python examples/serve_so101_follower.py "
    "--robot.port=<SERIAL_PORT> --robot.id=follower "
    "--action_mode=pose_delta --address=127.0.0.1:5556"
)


def free_port() -> int:
    """A free localhost TCP port (bind-0 dance; good enough for tests)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FollowerBackend(ABC):
    """One characterized follower behind a gRPC endpoint + one client."""

    name: str = "abstract"
    #: Whether law-side solution records are available (sim only).
    law_spy_available: bool = False

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def make_client(self) -> GRPCFollower:
        """A connected GRPCFollower (observation stream already running)."""

    def reset_session(self, client: GRPCFollower) -> None:
        """Return the follower to a clean latch state for a new sequence.

        Default: a fresh Connect RPC (sim: also teleports home + re-latches
        the law).  The real backend reconnects the client instead.
        """
        client.stub.Connect(Empty(), timeout=client.connect_timeout_s)


class SimFollowerBackend(FollowerBackend):
    """In-process MuJoCo servicer + real gRPC server on a random port."""

    law_spy_available = True

    def __init__(self, xml_path: Path | str = XML_PATH, rot_weight: float | None = None):
        self.xml_path = str(xml_path)
        # Sweep hook (B-2 comparison): rot_weight forwarded to the servicer
        # (same parameter examples/serve_mujoco_follower.py exposes as
        # --rot-weight).  None = the servicer default (0.3).  The backend
        # name carries the override so report directories stay distinct.
        self.rot_weight = rot_weight
        self.name = "sim" if rot_weight is None else f"sim_rw{rot_weight:g}"
        self.servicer = None
        self.server = None
        self.client: GRPCFollower | None = None
        self.stats: TeleopStats | None = None
        # Read-only law.solve records: (wall monotonic, JointSolution).
        self.solutions: list[tuple[float, "object"]] = []

    def start(self) -> None:
        import pytest

        pytest.importorskip("mujoco")
        from lerobot_robot_grpc.follower.mujoco_follower_server import (
            MuJoCoSO101Servicer,
        )

        kwargs = {}
        if self.rot_weight is not None:
            kwargs["rot_weight"] = self.rot_weight
        self.servicer = MuJoCoSO101Servicer(
            xml_path=self.xml_path,
            action_mode="pose_delta",
            render=False,
            **kwargs,
        )
        self._install_law_spy(self.servicer._law)

        port = free_port()
        address = f"127.0.0.1:{port}"
        self.server = FollowerServer(
            FollowerServerConfig(address=address, server_grace_period_s=1.0),
            self.servicer,
        )
        self.server.start()
        self.address = address
        self.client = self.make_client()

    def _install_law_spy(self, law) -> None:
        """Record every law.solve result without touching its behaviour.

        An instance attribute shadowing the bound method — the servicer's
        ``self._law.solve(...)`` calls pick this up unchanged.
        """
        orig_solve = law.solve

        def spy(delta_action, qpos_rad, *, stale: bool = False):
            sol = orig_solve(delta_action, qpos_rad, stale=stale)
            self.solutions.append((time.monotonic(), sol))
            return sol

        law.solve = spy

    def make_client(self) -> GRPCFollower:
        self.stats = TeleopStats()
        client = GRPCFollower(
            GRPCFollowerConfig(
                address=self.address, need_warmup=False, teleop_stats=False
            ),
            stats=self.stats,
        )
        client.connect(calibrate=False)
        return client

    @property
    def law(self):
        assert self.servicer is not None and self.servicer._law is not None
        return self.servicer._law

    def stop(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client = None
        if self.server is not None:
            self.server.stop()
            self.server = None


class RealFollowerBackend(FollowerBackend):
    """Stage-B-3 interface: attach to a human-started real follower server.

    The suite never launches this backend — starting the real server opens
    the serial port, which is a human-in-the-loop step.  With the server
    already listening::

        RESPONSE_TEST_BACKEND=real RESPONSE_REAL_ADDRESS=127.0.0.1:5556 \
            conda run -n lerobot-grpc-serve python -m pytest tests/response -q

    Attach mode only builds the client; law-side solution records are
    unavailable (the law lives in the server process), so law-flag
    assertions auto-skip and the report notes it.
    """

    name = "real"
    law_spy_available = False

    def __init__(self, address: str):
        self.address = address
        self.client: GRPCFollower | None = None
        self.stats: TeleopStats | None = None

    def start(self) -> None:
        # Interface only.  Triggering requires a human-started server:
        #   <REAL_LAUNCH_CMD with --robot.port filled in>
        raise NotImplementedError(
            "The real follower backend must be started by a human (serial "
            f"port access). Launch the server first:  {REAL_LAUNCH_CMD}"
        )

    def attach(self) -> None:
        """Connect the client to an ALREADY-RUNNING real server."""
        self.stats = TeleopStats()
        self.client = self.make_client()

    def make_client(self) -> GRPCFollower:
        assert self.address, "RESPONSE_REAL_ADDRESS must name the server"
        client = GRPCFollower(
            GRPCFollowerConfig(
                address=self.address, need_warmup=False, teleop_stats=False
            ),
            stats=self.stats,
        )
        client.connect(calibrate=False)
        return client

    def reset_session(self, client: GRPCFollower) -> None:
        # The real arm cannot teleport home: re-latch the reference where it
        # stands instead (the clutch re-engage contract).
        client.stub.SetReference(Empty(), timeout=client.data_timeout_s)

    def stop(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client = None
