"""Sequence runner: drives one :class:`~.injectors.Sequence` against a
backend's gRPC endpoint and records everything the metrics need.

Per run:

1. ``reset_session`` — fresh Connect (sim: arm teleports home, law
   re-latches) so every sequence starts from the same baseline;
2. handshake — wait for a settled observation, latch the test-side
   reference ``T_ref`` (and snapshot the law's own ``T_arm_ref`` in sim,
   quantifying the handshake offset);
3. 30 Hz action loop — ``frame(t)`` (None = silence) with SendAction RTT
   recorded per packet; optional timed events (e.g. a mid-run
   ``SetReference``) fire on the nearest tick;
4. ~50 Hz observation sampler — the GRPCFollower background stream is the
   SINGLE GetObservation consumer (physics only steps while a stream is
   iterated, so a second stream would double the physics rate — see
   findings); each new tick is timestamped client-side;
5. TeleopStats snapshot diff for the window (frames, latency, RTT).

The runner never asserts — tests interpret :class:`RunResult`; the report
collector receives every run either way.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .backends import FollowerBackend
from .injectors import Sequence
from .metrics import (
    ACTION_HZ,
    Sample,
    SentRecord,
    TestFK,
    gripper_metrics,
    lag_metrics,
    smoothness_metrics,
    solutions_summary,
    step_metrics,
    stream_metrics,
    tracking_metrics,
)

# Minimum wall time between recorded observation samples (the stream ticks
# at ~50 Hz; polling faster and deduping by timestamp keeps the sample grid
# close to the true tick times).
_MIN_SAMPLE_DT_S = 0.014
# Post-run capture window so the final target's response is recorded.
_TAIL_S = 0.35


@dataclass
class RunResult:
    sequence: Sequence
    ref: np.ndarray  # test-side latched reference T_ref (4x4)
    law_ref: np.ndarray | None = None  # sim: the law's own T_arm_ref
    samples: list[Sample] = field(default_factory=list)
    sent: list[SentRecord] = field(default_factory=list)
    solutions: list = field(default_factory=list)  # sim: JointSolution slice
    metrics: dict = field(default_factory=dict)
    events_fired: list = field(default_factory=list)


class Runner:
    def __init__(self, backend: FollowerBackend, fk: TestFK, collector=None):
        self.backend = backend
        self.fk = fk
        self.collector = collector

    # ------------------------------------------------------------------
    # Session plumbing
    # ------------------------------------------------------------------

    def _client(self):
        client = self.backend.client
        if client is None:
            raise RuntimeError("backend not started")
        return client

    def wait_settled(self, timeout_s: float = 2.0, tol_deg: float = 0.05) -> dict:
        """Block until consecutive observation ticks barely move."""
        client = self._client()
        deadline = time.monotonic() + timeout_s
        prev = None
        while time.monotonic() < deadline:
            obs = client.get_observation()
            if prev is not None:
                drift = max(
                    abs(obs[f"{j}.pos"] - prev[f"{j}.pos"])
                    for j in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                              "wrist_flex", "wrist_roll")
                )
                if drift < tol_deg:
                    return obs
            prev = obs
            time.sleep(0.02)
        return prev or client.get_observation()

    def reset_session(self) -> None:
        self.backend.reset_session(self._client())
        # Discard observation ticks that were sampled before the reset RPC
        # completed (a tick mid-flight during Connect carries pre-reset
        # joints — latching T_ref from it skews every later metric).
        time.sleep(0.08)
        self.wait_settled()

    def set_reference(self) -> None:
        """Client-side clutch re-engage: re-latch T_ref at the current pose."""
        client = self._client()
        from google.protobuf.empty_pb2 import Empty

        client.stub.SetReference(Empty(), timeout=client.data_timeout_s)

    # ------------------------------------------------------------------
    # The run
    # ------------------------------------------------------------------

    def run(
        self,
        seq: Sequence,
        *,
        events: dict[float, callable] | None = None,
        reset: bool = True,
        note: str = "",
    ) -> RunResult:
        """Record one sequence; returns the RunResult (also added to the
        report collector, metrics included).

        events maps a sequence time to a zero-arg callback fired on the
        nearest action tick (e.g. t -> self.set_reference for relock probes).
        """
        client = self._client()
        stats = getattr(self.backend, "stats", None)
        solutions_all = getattr(self.backend, "solutions", None)

        if reset:
            self.reset_session()
        # Test-side reference from the settled observation.
        obs0 = client.get_observation()
        pos0, mat0 = self.fk.pose(obs0)
        ref = np.eye(4)
        ref[:3, 3] = pos0
        ref[:3, :3] = mat0
        law_ref = None
        if self.backend.law_spy_available:
            law_ref = self.backend.law.arm_reference.copy()

        stats_before = stats.snapshot() if stats is not None else {}
        sol_start = len(solutions_all) if solutions_all is not None else 0

        result = RunResult(sequence=seq, ref=ref, law_ref=law_ref)
        samples: list[Sample] = result.samples
        sent: list[SentRecord] = result.sent

        stop = threading.Event()

        def sampler() -> None:
            last_rec = -1.0
            while not stop.is_set():
                ts = client._latest_obs_time
                if ts > last_rec + _MIN_SAMPLE_DT_S:
                    last_rec = ts
                    obs = client._latest_obs_ft.copy()
                    if not obs:
                        continue
                    pos, mat = self.fk.pose(obs)
                    qpos = self.fk.qpos_from_obs(obs)
                    samples.append(Sample(ts - t0, obs, pos, mat, qpos))
                time.sleep(0.004)

        t0 = time.monotonic()
        thread = threading.Thread(target=sampler, daemon=True, name="resp-sampler")
        thread.start()
        try:
            pending = sorted(events.items()) if events else []
            n_ticks = int(seq.duration_s * ACTION_HZ) + 1
            for tick in range(n_ticks):
                t_rel = tick / ACTION_HZ
                if t_rel > seq.duration_s:
                    break
                now = time.monotonic()
                delay = (t0 + t_rel) - now
                if delay > 0:
                    time.sleep(delay)
                # Fire any event scheduled for this tick.
                while pending and pending[0][0] <= t_rel + 0.5 / ACTION_HZ:
                    _, fn = pending.pop(0)
                    fn()
                    result.events_fired.append((t_rel, getattr(fn, "__name__", "event")))
                action = seq.frame(t_rel)
                if action is not None:
                    t_send = time.monotonic()
                    client.send_action(action)
                    rtt_ms = (time.monotonic() - t_send) * 1000.0
                    sent.append(SentRecord(t_rel, dict(action), rtt_ms))
            # Tail: let the response to the final frames land in the record.
            stop_wait = time.monotonic() + _TAIL_S
            while time.monotonic() < stop_wait:
                time.sleep(0.02)
        finally:
            stop.set()
            thread.join(timeout=2.0)

        if stats is not None:
            window = stream_metrics(
                _stats_diff(stats_before, stats.snapshot()),
                window_s=seq.duration_s + _TAIL_S,
            )
        else:
            window = {}
        if solutions_all is not None:
            result.solutions = [sol for _, sol in solutions_all[sol_start:]]

        lead_s = float(seq.meta.get("lead_s", 0.3))
        m: dict = {"note": note} if note else {}
        m["ref_law_offset_mm"] = (
            float(np.linalg.norm(ref[:3, 3] - law_ref[:3, 3]) * 1000.0)
            if law_ref is not None else None
        )
        m.update(tracking_metrics(samples, seq.frame, ref, lead_s=lead_s))
        m.update(step_metrics(samples, seq.frame, ref, lead_s=lead_s))
        m.update(lag_metrics(samples, seq.frame, ref, lead_s=lead_s))
        m.update(smoothness_metrics(samples))
        m.update(window)
        m.update(solutions_summary(result.solutions))
        if seq.category == "gripper" and "from_mm" in seq.meta:
            step_sent = [s for s in sent
                         if abs(s.action["gripper.distance"] - seq.meta["to_mm"]) < 1e-6]
            m.update(gripper_metrics(
                samples, step_sent or sent,
                from_mm=seq.meta["from_mm"], to_mm=seq.meta["to_mm"],
            ))
        result.metrics = m

        if self.collector is not None:
            self.collector.add(result)
        return result


def _stats_diff(before: dict, after: dict) -> dict:
    """snapshot() is cumulative — diff the counters over the window."""
    per_key = {}
    for key, (frames, nbytes, lats) in after.get("per_key", {}).items():
        b = before.get("per_key", {}).get(key, (0, 0, []))
        per_key[key] = (frames - b[0], nbytes - b[1], lats[len(b[2]):])
    rtt_b = before.get("action_rtt_ms", [])
    rtt_a = after.get("action_rtt_ms", [])
    return {
        "per_key": per_key,
        "action_rtt_ms": rtt_a[len(rtt_b):],
    }
