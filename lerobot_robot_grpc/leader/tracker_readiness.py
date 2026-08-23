"""Pika Tracker global-scene readiness policy.

The Pika SDK reports facts; this module owns the policy that turns those
facts into a safe, tokenized readiness state.  It deliberately has no gRPC or
hardware dependencies so cold-start, late-Lighthouse and map-change behavior
can be tested deterministically.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lerobot_robot_grpc.protos import device_pb2


@dataclass(frozen=True)
class ReadinessSnapshot:
    state: int
    reason: str
    context_epoch: int = 0
    global_scene_generation: int = 0
    lighthouse_cohort_generation: int = 0
    readiness_generation: int = 0
    expected_lighthouses: tuple[str, ...] = ()
    solved_lighthouses: tuple[str, ...] = ()
    token: str = ""
    stable_sample_count: int = 0
    stable_window_s: float = 0.0
    position_spread_m: float = 0.0
    rotation_spread_rad: float = 0.0

    def to_proto(self) -> device_pb2.TrackingReadiness:
        return device_pb2.TrackingReadiness(
            state=self.state,
            reason=self.reason,
            context_epoch=self.context_epoch,
            global_scene_generation=self.global_scene_generation,
            lighthouse_cohort_generation=self.lighthouse_cohort_generation,
            readiness_generation=self.readiness_generation,
            expected_lighthouses=self.expected_lighthouses,
            solved_lighthouses=self.solved_lighthouses,
            token=self.token,
            stable_sample_count=self.stable_sample_count,
            stable_window_s=self.stable_window_s,
            position_spread_m=self.position_spread_m,
            rotation_spread_rad=self.rotation_spread_rad,
        )


@dataclass(frozen=True)
class _PoseSample:
    sequence: int
    received_s: float
    position: np.ndarray
    rotation: np.ndarray


class TrackerReadinessGate:
    """Convert SDK health snapshots into an opaque readiness lease."""

    def __init__(
        self,
        *,
        cohort_stable_s: float = 2.0,
        map_stable_s: float = 15.0,
        stable_window_s: float = 1.0,
        stable_samples: int = 20,
        position_spread_m: float = 0.005,
        rotation_spread_rad: float = np.radians(2.0),
        optical_stale_s: float = 0.1,
        map_position_delta_m: float = 0.005,
        map_rotation_delta_rad: float = np.radians(1.0),
    ) -> None:
        self._cohort_stable_s = float(cohort_stable_s)
        self._map_stable_s = float(map_stable_s)
        if self._map_stable_s < 0.0:
            raise ValueError("map_stable_s must be non-negative")
        self._stable_window_s = float(stable_window_s)
        self._stable_samples = int(stable_samples)
        self._position_spread_m = float(position_spread_m)
        self._rotation_spread_rad = float(rotation_spread_rad)
        self._optical_stale_s = float(optical_stale_s)
        self._map_position_delta_m = float(map_position_delta_m)
        self._map_rotation_delta_rad = float(map_rotation_delta_rad)
        self._salt = secrets.token_hex(16)
        self._samples: deque[_PoseSample] = deque()
        self._last_sequence: int | None = None
        self._observed_cohort: tuple[str, ...] = ()
        self._cohort_changed_s: float | None = None
        self._observed_map_identity: tuple[int, tuple[str, ...]] | None = None
        self._observed_scene_generation: int | None = None
        self._map_changed_s: float | None = None
        self._ready_identity: tuple[int, tuple[str, ...]] | None = None
        self._ready_scene: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._readiness_generation = 0
        self._ready_token = ""
        self._invalidated_state: int | None = None
        self._invalidated_reason = ""

    def _reset_map_settling(self) -> None:
        self._observed_map_identity = None
        self._observed_scene_generation = None
        self._map_changed_s = None

    def _observe_complete_map(
        self,
        *,
        epoch: int,
        cohort: tuple[str, ...],
        scene_generation: int,
        now_s: float,
    ) -> float:
        """Return how long the current complete global map has been quiet."""
        identity = (epoch, cohort)
        if (
            identity != self._observed_map_identity
            or scene_generation != self._observed_scene_generation
        ):
            self._observed_map_identity = identity
            self._observed_scene_generation = scene_generation
            self._map_changed_s = float(now_s)
            # Tracker stability is meaningful only after the map that defines
            # its coordinates has stopped changing.
            self._samples.clear()
            self._last_sequence = None
        if self._map_changed_s is None:
            self._map_changed_s = float(now_s)
        return max(0.0, float(now_s) - self._map_changed_s)

    @staticmethod
    def _rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
        relative = a.T @ b
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.arccos(cosine))

    @staticmethod
    def _quat_wxyz_matrix(values: Any) -> np.ndarray:
        quat = np.asarray(values, dtype=float)
        if quat.shape != (4,) or not np.isfinite(quat).all():
            raise ValueError("invalid Lighthouse quaternion")
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12:
            raise ValueError("zero Lighthouse quaternion")
        w, x, y, z = quat / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _scene(self, health: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, raw in dict(health.get("lighthouses", {})).items():
            position = np.asarray(raw.get("position"), dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                continue
            try:
                rotation = self._quat_wxyz_matrix(raw.get("rotation"))
            except (TypeError, ValueError):
                continue
            result[str(name)] = (position, rotation)
        return result

    def _invalidate(self, state: int, reason: str) -> None:
        if self._invalidated_state is None:
            self._readiness_generation += 1
        self._invalidated_state = state
        self._invalidated_reason = reason
        self._ready_token = ""
        self._samples.clear()
        self._last_sequence = None

    def _map_change_reason(
        self,
        epoch: int,
        cohort: tuple[str, ...],
        scene: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> str | None:
        if self._ready_identity is None:
            return None
        ready_epoch, ready_cohort = self._ready_identity
        if epoch != ready_epoch:
            return f"libsurvive context changed ({ready_epoch} -> {epoch})"
        if cohort != ready_cohort:
            return f"Lighthouse cohort changed ({ready_cohort} -> {cohort})"
        for name, (old_position, old_rotation) in self._ready_scene.items():
            current = scene.get(name)
            if current is None:
                return f"solved Lighthouse {name} disappeared"
            position_delta = float(np.linalg.norm(current[0] - old_position))
            rotation_delta = self._rotation_distance(old_rotation, current[1])
            if (
                position_delta > self._map_position_delta_m
                or rotation_delta > self._map_rotation_delta_rad
            ):
                return (
                    f"Lighthouse {name} global pose changed by "
                    f"{position_delta * 1000.0:.1f}mm/"
                    f"{np.degrees(rotation_delta):.2f}deg"
                )
        return None

    def _append_sample(self, sample: Any, now_s: float) -> None:
        if sample is None:
            return
        sequence = int(getattr(sample, "optical_event_sequence", 0))
        if sequence <= 0 or sequence == self._last_sequence:
            return
        self._last_sequence = sequence
        self._samples.append(
            _PoseSample(
                sequence=sequence,
                received_s=float(now_s),
                position=np.asarray(sample.position, dtype=float).copy(),
                rotation=np.asarray(sample.rotation, dtype=float).copy(),
            )
        )
        keep_s = self._stable_window_s + 0.25
        while (
            len(self._samples) > 1
            and self._samples[-1].received_s - self._samples[0].received_s > keep_s
        ):
            self._samples.popleft()

    def _stability(self) -> tuple[int, float, float, float, bool]:
        count = len(self._samples)
        window = (
            0.0
            if count < 2
            else self._samples[-1].received_s - self._samples[0].received_s
        )
        if count == 0:
            return 0, window, 0.0, 0.0, False
        positions = np.stack([sample.position for sample in self._samples])
        position_spread = float(
            np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=-1
            ).max()
        )
        rotations = [sample.rotation for sample in self._samples]
        rotation_spread = max(
            self._rotation_distance(a, b)
            for index, a in enumerate(rotations)
            for b in rotations[index:]
        )
        ready = (
            count >= self._stable_samples
            and window >= self._stable_window_s
            and position_spread <= self._position_spread_m
            and rotation_spread <= self._rotation_spread_rad
        )
        return count, window, position_spread, rotation_spread, ready

    def _token(
        self,
        *,
        epoch: int,
        scene_generation: int,
        cohort_generation: int,
        cohort: tuple[str, ...],
    ) -> str:
        payload = json.dumps(
            {
                "salt": self._salt,
                "context_epoch": epoch,
                "global_scene_generation": scene_generation,
                "lighthouse_cohort_generation": cohort_generation,
                "readiness_generation": self._readiness_generation,
                "lighthouses": cohort,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def update(
        self,
        health: Mapping[str, Any],
        sample: Any,
        *,
        now_s: float,
    ) -> ReadinessSnapshot:
        state = device_pb2.TrackingReadinessState
        epoch = int(health.get("context_epoch", 0) or 0)
        scene_generation = int(health.get("global_scene_generation", 0) or 0)
        cohort_generation = int(health.get("lighthouse_cohort_generation", 0) or 0)
        cohort = tuple(sorted(str(v) for v in health.get("discovered_lighthouses", ())))
        scene = self._scene(health)
        solved = tuple(sorted(scene))

        common = {
            "context_epoch": epoch,
            "global_scene_generation": scene_generation,
            "lighthouse_cohort_generation": cohort_generation,
            "readiness_generation": self._readiness_generation,
            "expected_lighthouses": cohort,
            "solved_lighthouses": solved,
        }
        if not bool(health.get("bridge_available", False)):
            self._invalidate(
                state.TRACKING_READINESS_STATE_ERROR,
                str(health.get("bridge_error") or "native tracking bridge unavailable"),
            )
            return ReadinessSnapshot(
                state=state.TRACKING_READINESS_STATE_ERROR,
                reason=self._invalidated_reason,
                **{**common, "readiness_generation": self._readiness_generation},
            )
        if epoch <= 0:
            return ReadinessSnapshot(
                state=state.TRACKING_READINESS_STATE_STARTING,
                reason="waiting for libsurvive context",
                **common,
            )

        if cohort != self._observed_cohort:
            self._observed_cohort = cohort
            self._cohort_changed_s = float(now_s)
            self._reset_map_settling()
            self._samples.clear()
            self._last_sequence = None
            if self._ready_identity is not None and cohort != self._ready_identity[1]:
                self._invalidate(
                    state.TRACKING_READINESS_STATE_MAP_CHANGED,
                    f"Lighthouse cohort changed "
                    f"({self._ready_identity[1]} -> {cohort})",
                )
                self._ready_identity = None
                self._ready_scene = {}
        if not cohort:
            return ReadinessSnapshot(
                state=state.TRACKING_READINESS_STATE_WAITING_LIGHTHOUSE,
                reason="no Lighthouse has been discovered",
                **common,
            )
        cohort_age = (
            0.0
            if self._cohort_changed_s is None
            else max(0.0, float(now_s) - self._cohort_changed_s)
        )
        if cohort_age < self._cohort_stable_s:
            return ReadinessSnapshot(
                state=(
                    self._invalidated_state
                    or state.TRACKING_READINESS_STATE_WAITING_LIGHTHOUSE
                ),
                reason=(
                    self._invalidated_reason
                    or f"Lighthouse cohort settling "
                    f"{cohort_age:.1f}/{self._cohort_stable_s:.1f}s"
                ),
                **{**common, "readiness_generation": self._readiness_generation},
            )

        map_reason = self._map_change_reason(epoch, cohort, scene)
        if map_reason is not None:
            self._invalidate(state.TRACKING_READINESS_STATE_MAP_CHANGED, map_reason)
            self._ready_identity = None
            self._ready_scene = {}
        missing = tuple(name for name in cohort if name not in scene)
        if scene_generation <= 0 or missing:
            current_state = (
                self._invalidated_state
                or state.TRACKING_READINESS_STATE_SOLVING_GLOBAL_SCENE
            )
            reason = self._invalidated_reason if self._invalidated_state else (
                "waiting for a new successful global-scene solve; missing "
                + ", ".join(missing or cohort)
            )
            return ReadinessSnapshot(state=current_state, reason=reason, **common)

        map_age = self._observe_complete_map(
            epoch=epoch,
            cohort=cohort,
            scene_generation=scene_generation,
            now_s=now_s,
        )
        if self._ready_identity is None and map_age < self._map_stable_s:
            current_state = (
                self._invalidated_state
                or state.TRACKING_READINESS_STATE_SOLVING_GLOBAL_SCENE
            )
            reason = self._invalidated_reason if self._invalidated_state else (
                "Lighthouse global map settling "
                f"{map_age:.1f}/{self._map_stable_s:.1f}s after scene "
                f"generation {scene_generation}"
            )
            return ReadinessSnapshot(
                state=current_state,
                reason=reason,
                **{**common, "readiness_generation": self._readiness_generation},
            )

        health_reason = None
        if sample is None:
            health_reason = "no Tracker pose"
        elif float(getattr(sample, "optical_age_s", float("inf"))) > self._optical_stale_s:
            health_reason = "Tracker optical measurements are stale"
        elif int(getattr(sample, "optical_measurement_count", 0)) <= 0:
            health_reason = "no recent decoded optical measurements"
        if health_reason is not None:
            if self._ready_identity is not None:
                self._invalidate(state.TRACKING_READINESS_STATE_LOST, health_reason)
            current_state = (
                self._invalidated_state
                or state.TRACKING_READINESS_STATE_VERIFYING_STABILITY
            )
            return ReadinessSnapshot(
                state=current_state,
                reason=self._invalidated_reason or health_reason,
                **{**common, "readiness_generation": self._readiness_generation},
            )

        # Once convergence has been proven, ordinary operator motion must not
        # revoke readiness.  READY is a lease on the optical map/context, not
        # a requirement that the user's hand remain motionless while aligning.
        if (
            self._ready_identity == (epoch, cohort)
            and self._ready_token
            and self._invalidated_state is None
        ):
            return ReadinessSnapshot(
                state=state.TRACKING_READINESS_STATE_READY,
                reason="global scene and Tracker pose are stable",
                token=self._ready_token,
                **{**common, "readiness_generation": self._readiness_generation},
            )

        self._append_sample(sample, now_s)
        count, window, pos_spread, rot_spread, stable = self._stability()
        metrics = {
            "stable_sample_count": count,
            "stable_window_s": window,
            "position_spread_m": pos_spread,
            "rotation_spread_rad": rot_spread,
        }
        if not stable:
            current_state = (
                self._invalidated_state
                or state.TRACKING_READINESS_STATE_VERIFYING_STABILITY
            )
            reason = self._invalidated_reason if self._invalidated_state else (
                f"verifying Tracker stability: {count}/{self._stable_samples} "
                f"samples, {window:.2f}/{self._stable_window_s:.2f}s"
            )
            return ReadinessSnapshot(
                state=current_state,
                reason=reason,
                **{**common, "readiness_generation": self._readiness_generation},
                **metrics,
            )

        if not self._ready_token:
            self._readiness_generation += 1
            self._ready_token = self._token(
                epoch=epoch,
                scene_generation=scene_generation,
                cohort_generation=cohort_generation,
                cohort=cohort,
            )
        self._ready_identity = (epoch, cohort)
        self._ready_scene = {
            name: (position.copy(), rotation.copy())
            for name, (position, rotation) in scene.items()
            if name in cohort
        }
        self._invalidated_state = None
        self._invalidated_reason = ""
        return ReadinessSnapshot(
            state=state.TRACKING_READINESS_STATE_READY,
            reason="global scene and Tracker pose are stable",
            token=self._ready_token,
            **{**common, "readiness_generation": self._readiness_generation},
            **metrics,
        )

    def token_is_current(self, token: str, snapshot: ReadinessSnapshot) -> bool:
        return bool(
            token
            and snapshot.state
            == device_pb2.TrackingReadinessState.TRACKING_READINESS_STATE_READY
            and token == snapshot.token
            and token == self._ready_token
        )
