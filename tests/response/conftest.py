"""Session fixtures for the response suite.

Backend selection (env):

- ``RESPONSE_TEST_BACKEND=sim`` (default) — in-process MuJoCo servicer on a
  real localhost gRPC server; everything runs, report written.
- ``RESPONSE_TEST_BACKEND=real`` — stage B-3: attach to a human-started
  ``serve_so101_follower.py --action_mode=pose_delta`` server at
  ``RESPONSE_REAL_ADDRESS``.  If no address is set the suite SKIPS (never
  launches anything, never touches a serial port).
"""

from __future__ import annotations

import os

import pytest

from .backends import REAL_LAUNCH_CMD, RealFollowerBackend, SimFollowerBackend
from .harness import Runner
from .metrics import TestFK
from .report import ReportCollector


def _make_backend():
    kind = os.environ.get("RESPONSE_TEST_BACKEND", "sim").strip().lower()
    if kind == "sim":
        backend = SimFollowerBackend()
        backend.start()
        return backend
    if kind == "real":
        address = os.environ.get("RESPONSE_REAL_ADDRESS", "").strip()
        if not address:
            pytest.exit(
                "RESPONSE_TEST_BACKEND=real requires RESPONSE_REAL_ADDRESS "
                "(a human-started server; the suite never launches it). "
                f"Launch command: {REAL_LAUNCH_CMD}",
                returncode=4,
            )
        backend = RealFollowerBackend(address)
        backend.attach()
        return backend
    raise ValueError(f"unknown RESPONSE_TEST_BACKEND {kind!r}")


@pytest.fixture(scope="session")
def backend():
    backend = _make_backend()
    yield backend
    backend.stop()


@pytest.fixture(scope="session")
def fk():
    return TestFK()


@pytest.fixture(scope="session")
def report(backend):
    collector = ReportCollector(backend.name)
    if not backend.law_spy_available:
        collector.note(
            "law-side flags unavailable on this backend",
            "The real backend has no in-process law spy; rejected/jumped/"
            "stale assertions are skipped there (law-flag channel is "
            "in-process only — see the RPC-channel finding).",
        )
    yield collector
    out_dir = collector.write()
    print(f"\n[response-suite] report written: {out_dir}")


@pytest.fixture(scope="session")
def runner(backend, fk, report):
    return Runner(backend, fk, collector=report)
