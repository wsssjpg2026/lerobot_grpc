"""Pika Sense leader servicer for delta-pose teleoperation.

Wraps the Pika Sense device (Vive Tracker + gripper sensor) as a
:class:`LeaderServicer` that produces pose-delta actions.  After
:meth:`SetReference` the servicer publishes the **current validated relative
transform** from that latch — 1:1 after the optical-freshness and temporal-
coherence gate (PikaAnyArm transform semantics):

    Δ = inv(T_tracker_ref) @ T_tracker_now
    Δp = R_ref^T @ (p_now − p_ref)     # body frame of the reference latch
    ΔR = R_ref^T @ R_now

The follower composes ``T_target = T_arm_ref @ Δ``.  No lighthouse→base
calibration (``R_lh2base``) participates in the teleop path — the
construction eliminates it — so the translation follows the hand's
orientation, not the room axes.  The calibration file is still loaded/saved
(the gripper travel range is needed); its rotation part is retained for
compatibility but unused.

Robot geometry and IK safety remain follower-side.  Sensor correctness stays
at this leader boundary: Connect waits for distinct timestamped poses to soak
and settle, runtime cached/implausible poses freeze publication and require a
stable re-reference, and action-frame quality is propagated to collection.
The quick-gripper-squeeze command-state edge also supplies the clutch freeze.

The Pika SDK (``pika.sense.Sense``) is imported lazily inside ``__init__`` so
the module imports cleanly on machines without the SDK — only construction
fails.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf.empty_pb2 import Empty

from lerobot.utils.rotation import Rotation as Rot

from lerobot_robot_grpc.leader.leader_server import LeaderServicer
from lerobot_robot_grpc.leader.utils import encode_feature
from lerobot_robot_grpc.pose_delta_schema import (
    ACTION_KEYS,
    action_keys,
    build_pose_delta_feature_info,
)
from lerobot_robot_grpc.protos import device_pb2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackerSample:
    position: np.ndarray
    rotation: np.ndarray
    source_timestamp_s: float
    raw_optical_timestamp_s: float
    raw_optical_age_s: float
    raw_optical_measurement_count: int
    raw_optical_event_sequence: int
    optical_timestamp_s: float
    optical_age_s: float
    optical_measurement_count: int
    optical_lighthouse_count: int
    optical_event_sequence: int
    pose_confidence: float | None
    received_monotonic_s: float

# ---------------------------------------------------------------------------
# Pure delta-computation helpers (testable without Pika hardware)
# ---------------------------------------------------------------------------


def compute_position_delta_body(
    pos_now: np.ndarray,
    pos_ref: np.ndarray,
    rot_ref: np.ndarray,
) -> np.ndarray:
    """Body-frame translation delta of the official relative transform.

    ``Δp = R_ref^T @ (p_now − p_ref)`` — the position offset expressed in the
    tracker reference latch's body frame.  The follower rotates it into the
    base frame via ``R_arm_ref`` (official ``T_arm_ref @ inv(T_ref) @ T_now``
    composition); no lighthouse→base rotation is involved on this side.
    """
    return rot_ref.T @ (np.asarray(pos_now, dtype=float) - np.asarray(pos_ref, dtype=float))


def compute_rotation_delta_rotvec(
    rot_now: np.ndarray,
    rot_ref: np.ndarray,
) -> np.ndarray:
    """Body-frame rotation delta as a rotvec.

    ``R_delta = R_ref^T @ R_now`` (body-frame composition).  The rotvec
    is returned directly in the body frame — it matches the follower's
    composition (``R_arm_ref @ R_delta``) without any world-frame
    remapping, because both tracker and end-effector body frames are
    aligned when the user holds the tracker in a natural forward grip.
    """
    r_delta = rot_ref.T @ rot_now
    return Rot.from_matrix(r_delta).as_rotvec()


def command_state_edge(command_now: int | None, command_prev: int | None) -> bool:
    """True when Pika's quick-squeeze command state changed since the last sample.

    ``None`` means "no reading" and is never an edge.  A 0→1 **or** 1→0 change
    toggles the clutch (wayfinder #10) — the official ``/teleop_trigger``
    semantics: serial ``Command`` change → toggle follow/hold.
    """
    if command_now is None or command_prev is None:
        return False
    return int(command_now) != int(command_prev)


def build_R_lh2base(forward_lh: np.ndarray, up_lh: np.ndarray) -> np.ndarray:
    """Build the lighthouse→base rotation from measured axis directions.

    Used only by the 8-step calibration flow, which persists it alongside the
    gripper range.  The teleop path itself no longer consumes it (official
    composition eliminates the lighthouse→base mapping); it is kept in the
    calibration file for compatibility.

    Parameters
    ----------
    forward_lh, up_lh
        Normalised lighthouse-frame direction vectors for the robot's
        forward (base-X) and up (base-Z) axes, as measured during
        calibration.

    Returns
    -------
    (3, 3) rotation matrix.  Rows are base axes expressed in lighthouse
    coordinates, so ``R @ delta_lh = [forward·δ, left·δ, up·δ]`` — the
    correct dot-product projection.  MuJoCo SO-101 uses X=forward,
    Y=left, Z=up (right-handed).
    """
    # Re-orthogonalise into a proper right-handed rotation.
    # left = cross(up, forward), then forward = cross(left, up).
    left_lh = np.cross(up_lh, forward_lh)
    left_lh /= np.linalg.norm(left_lh)
    forward_lh = np.cross(left_lh, up_lh)
    return np.vstack([forward_lh, left_lh, up_lh])


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------


class PikaSenseServicer(LeaderServicer):
    """Leader servicer wrapping Pika Sense for delta-pose teleoperation.

    In each :meth:`GetAction` call the servicer:

    1. Reads the Vive Tracker pose (position + quaternion) and gripper
       distance from the Pika Sense hardware.
    2. Computes the relative transform from ``T_begin`` (SetReference):
       body-frame translation offset + body-frame rotation, 1:1 after the
       optical-freshness and temporal-coherence gate.
    3. Encodes the 8 FLOAT32 pose-delta action features and streams them.

    Clutch (#10): quickly squeezing Pika changes its command state and toggles
    follow / hold.
    While disengaged (or pending a re-engage relatch) the published offset
    is **frozen at its last value** — never identity — so the follower holds
    its stop pose.  ``GetStatus`` reports ``COLLECTION`` (following) /
    ``IDLE`` (clutch off) as the leader→client edge channel; on the engage
    edge the client must call the follower's ``SetReference`` before this
    servicer's ``SetReference`` (current hand = current arm).

    Parameters
    ----------
    port
        Serial port path for the Pika Sense device.
    R_lh2base
        3×3 lighthouse→base rotation, used ONLY by the 8-step calibration
        flow (measured + persisted).  The teleop path never consumes it —
        the official composition eliminates the mapping.
    command_state_provider
        Callable returning Pika's quick-squeeze command state (0/1) or ``None``.
        Defaults to ``device.get_command_state``; injectable so the clutch
        state machine (#10) is testable without hardware.
    auto_reference
        When True, block Connect until fresh timestamped poses settle and the
        configured operator confirmation succeeds, then latch ``T_begin`` and
        engage.  This supports clients without an alignment step, such as
        ``lerobot-teleoperate``.  Default False leaves reference capture to
        the collection session's follower-then-leader SetReference sequence.
    arm_prefix
        Optional action namespace such as ``"left"`` for S1.  The default
        unprefixed schema remains compatible with SO101.
    cumulative_clutch
        When True, a clutch re-engage starts a new tracker-relative segment
        composed onto the frozen published transform.  This gives clients
        without ``SetReference`` coordination a jump-free resume path.
    """

    # Multi-step calibration sequence for the StreamCalibration protocol.
    # Each step shows a prompt in the client terminal; the user presses Enter
    # to capture a data point via CalibrateDone.
    _CALIB_STEPS: list[dict[str, str]] = [
        {"key": "forward_start", "prompt": "[1/8] Hold tracker at START position"},
        {"key": "forward_moved", "prompt": "[2/8] Move FORWARD ~10cm (away from you)"},
        {"key": "right_start", "prompt": "[3/8] Return to NEUTRAL, hold steady"},
        {"key": "right_moved", "prompt": "[4/8] Move RIGHT ~10cm"},
        {"key": "up_start", "prompt": "[5/8] Return to NEUTRAL, hold steady"},
        {"key": "up_moved", "prompt": "[6/8] Move UP ~10cm (lift tracker)"},
        {"key": "gripper_close", "prompt": "[7/8] Fully CLOSE the gripper"},
        {"key": "gripper_open", "prompt": "[8/8] Fully OPEN the gripper"},
    ]

    # Runtime pose motion is deliberately a two-tier gate.  The soft envelope
    # only quarantines a sample for temporal confirmation; it is not evidence
    # of optical loss.  The absolute envelope remains fail-closed for a pose
    # discontinuity that is too large to expose to the follower even briefly.
    _MOTION_TRANSLATION_BASE_M = 0.005
    _MOTION_TRANSLATION_SPEED_M_S = 1.0
    _MOTION_TRANSLATION_SOFT_CAP_M = 0.080
    _MOTION_ROTATION_BASE_RAD = np.radians(3.0)
    _MOTION_ROTATION_SPEED_RAD_S = np.radians(300.0)
    _MOTION_ROTATION_SOFT_CAP_RAD = np.radians(30.0)
    _MOTION_TRANSLATION_ABSOLUTE_M = 0.200
    _MOTION_ROTATION_ABSOLUTE_RAD = np.radians(60.0)
    _MOTION_CONFIRM_SAMPLES = 3
    _MOTION_DIRECTION_COSINE = 0.25
    _MOTION_MIN_SPEED_RATIO = 0.15
    _MOTION_MAX_SPEED_RATIO = 6.0
    _REFERENCE_FAULT_OPTICAL = "optical"
    _REFERENCE_FAULT_POSE_DISCONTINUITY = "pose-discontinuity"
    # Rebuild a wedged Gen2 decoder only when raw Tracker photodiode hits have
    # returned but decoded sync/sweep output remains frozen.  Ordinary fast
    # motion and a physically hidden Tracker cannot enter this path.
    _DECODER_RESTART_AFTER_S = 2.0
    _DECODER_RESTART_COOLDOWN_S = 10.0
    _DECODER_RESTART_MAX_ATTEMPTS = 2
    _DECODER_REDISCOVERY_GRACE_S = 5.0

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        R_lh2base: np.ndarray | None = None,
        calibration_dir: str | None = None,
        command_state_provider=None,
        auto_reference: bool = False,
        device_id: str = "pika_sense",
        arm_prefix: str | None = None,
        cumulative_clutch: bool = False,
        tracker_ready_timeout_s: float = 110.0,
        tracker_min_soak_s: float = 10.0,
        tracker_stable_window_s: float = 1.0,
        tracker_stable_samples: int = 30,
        tracker_position_spread_m: float = 0.002,
        tracker_rotation_spread_deg: float = 1.0,
        require_start_confirmation: bool = True,
        tracker_recheck_window_s: float = 0.3,
        tracker_recheck_samples: int = 10,
        tracker_reference_timeout_s: float = 5.0,
        tracker_reference_position_spread_m: float = 0.005,
        tracker_reference_rotation_spread_deg: float = 2.0,
        tracker_stale_s: float = 0.1,
        tracker_reference_lost_s: float = 0.5,
        tracker_health_enabled: bool = True,
        tracker_min_optical_measurements: int = 4,
        tracker_min_optical_lighthouses: int = 1,
    ):
        # Deferred import — the Pika SDK is an optional dependency.
        # Importing here means the module loads cleanly on machines without
        # the SDK; only construction fails.
        from pika.sense import Sense

        self._device = Sense(port)
        self._command_state_provider = (
            command_state_provider
            if command_state_provider is not None
            else self._device.get_command_state
        )

        # Feature schema — shared with the follower's pose_delta mode.
        self._arm_prefix = arm_prefix
        self._act_ft_info = build_pose_delta_feature_info(prefix=arm_prefix)

        # Calibration-side rotation (persisted for compatibility; the teleop
        # path never consumes it — official composition needs no mapping).
        self._R_lh2base = R_lh2base if R_lh2base is not None else np.eye(3)

        # Gripper calibration range (updated by Calibrate RPC).
        self._gripper_min_mm = 0.0
        self._gripper_max_mm = 60.0

        # Connection / tracker state
        self._lock = threading.Lock()
        self._connected = False
        self._hardware_started = False
        self._tracker_device: str | None = None

        # auto_reference mode (see class docstring): latch + engage without a
        # client SetReference, for clients without an alignment step.
        self._auto_reference = bool(auto_reference)
        self._cumulative_clutch = bool(cumulative_clutch)
        if tracker_ready_timeout_s <= 0.0:
            raise ValueError("tracker_ready_timeout_s must be positive")
        if tracker_min_soak_s < 0.0 or tracker_stable_window_s < 0.0:
            raise ValueError("tracker soak/window durations must be non-negative")
        if tracker_stable_samples < 1 or tracker_recheck_samples < 1:
            raise ValueError("tracker stable/recheck sample counts must be positive")
        if tracker_position_spread_m < 0.0 or tracker_rotation_spread_deg < 0.0:
            raise ValueError("tracker stability spreads must be non-negative")
        self._tracker_ready_timeout_s = float(tracker_ready_timeout_s)
        self._tracker_min_soak_s = float(tracker_min_soak_s)
        self._tracker_stable_window_s = float(tracker_stable_window_s)
        self._tracker_stable_samples = int(tracker_stable_samples)
        self._tracker_position_spread_m = float(tracker_position_spread_m)
        self._tracker_rotation_spread_rad = float(
            np.radians(tracker_rotation_spread_deg)
        )
        self._require_start_confirmation = bool(require_start_confirmation)
        self._tracker_recheck_window_s = float(tracker_recheck_window_s)
        self._tracker_recheck_samples = int(tracker_recheck_samples)
        if tracker_reference_timeout_s <= 0.0:
            raise ValueError("tracker_reference_timeout_s must be positive")
        self._tracker_reference_timeout_s = float(tracker_reference_timeout_s)
        if (
            tracker_reference_position_spread_m < 0.0
            or tracker_reference_rotation_spread_deg < 0.0
        ):
            raise ValueError("tracker reference spreads must be non-negative")
        self._tracker_reference_position_spread_m = float(
            tracker_reference_position_spread_m
        )
        self._tracker_reference_rotation_spread_rad = float(
            np.radians(tracker_reference_rotation_spread_deg)
        )
        self._last_reference_failure_reason: str | None = None
        if tracker_stale_s <= 0.0 or tracker_reference_lost_s < tracker_stale_s:
            raise ValueError(
                "tracker_reference_lost_s must be >= positive tracker_stale_s"
            )
        self._tracker_stale_s = float(tracker_stale_s)
        self._tracker_reference_lost_s = float(tracker_reference_lost_s)
        self._tracker_health_enabled = bool(tracker_health_enabled)
        if (
            tracker_min_optical_measurements < 1
            or tracker_min_optical_lighthouses < 1
        ):
            raise ValueError("tracker optical support thresholds must be positive")
        self._tracker_min_optical_measurements = int(
            tracker_min_optical_measurements
        )
        self._tracker_min_optical_lighthouses = int(
            tracker_min_optical_lighthouses
        )
        self._last_seen_tracker_source_ts: float | None = None
        self._last_seen_tracker_raw_optical_ts: float | None = None
        self._last_seen_tracker_optical_ts: float | None = None
        self._last_fresh_received_monotonic: float | None = None
        self._last_valid_tracker_sample: TrackerSample | None = None
        self._pending_tracker_samples: deque[TrackerSample] = deque()
        self._bad_tracker_samples = 0
        self._reference_required = False
        self._reference_fault_kind: str | None = None
        self._tracker_recovery_samples: deque[TrackerSample] = deque()
        self._tracker_recovery_ready = False
        self._raw_optical_reacquiring = False
        self._raw_optical_reacquiring_since: float | None = None
        self._decoder_restart_after_s = self._DECODER_RESTART_AFTER_S
        self._decoder_restart_attempts = 0
        self._last_decoder_restart_monotonic: float | None = None
        self._decoder_restart_discovery_deadline: float | None = None
        self._last_tracking_state = (
            device_pb2.TrackingState.TRACKING_STATE_HELD
        )
        self._last_optical_health_log_monotonic = 0.0
        self._last_tracker_health_reason: str | None = "not sampled"

        # Latch-once state.  T_begin is locked by SetReference; the published
        # offset is the raw current displacement from that latch (frozen at
        # its last value while the clutch is disengaged).
        self._t_begin_pos: np.ndarray | None = None
        self._t_begin_rot: np.ndarray | None = None
        self._published_pos = np.zeros(3)
        self._published_rot = np.eye(3)
        self._segment_base_pos = np.zeros(3)
        self._segment_base_rot = np.eye(3)
        self._last_raw_pos: np.ndarray | None = None
        self._published_gripper_distance_mm = 60.0
        self._last_action_quality = device_pb2.FrameQuality.FRAME_QUALITY_STALE

        # Clutch state (wayfinder #10).  ``_clutched`` = following; a Pika
        # quick-squeeze command-state *change* toggles it (official /teleop_trigger
        # semantics).  While disengaged — or pending a re-engage relatch — the
        # published offset is frozen at its last value (never identity), so
        # the follower holds the stop pose.  ``_pending_relatch`` is set on
        # the engage edge and cleared by SetReference: the client sequences
        # follower.SetReference() → leader.SetReference(), so the arm never
        # sees a zero offset against a stale T_zero.
        self._clutched = False
        self._pending_relatch = False
        # Collection mode separates the operator's quick-squeeze confirmation
        # from the later follower -> leader reference commit.  Keeping this
        # state explicit prevents a single fresh sample from being mistaken
        # for a completed recovery transaction.
        self._reference_confirmation_pending = False
        self._prev_command_state: int | None = None

        # --- Calibration file persistence ---
        # Follows the lerobot convention: ~/.cache/huggingface/lerobot/
        #   calibration/teleoperators/pika_sense/<device_id>.json
        if calibration_dir is not None:
            self._calibration_dir = Path(calibration_dir)
        else:
            default = Path(
                os.environ.get(
                    "HF_LEROBOT_CALIBRATION",
                    str(Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"),
                )
            )
            self._calibration_dir = default / "teleoperators" / "pika_sense"
        self._calibration_dir.mkdir(parents=True, exist_ok=True)
        self._calibration_fpath = self._calibration_dir / f"{device_id}.json"

        # Calibration state
        self._calibrated = False
        self._calibrating = False
        self._calibrate_error = False
        # Multi-step calibration state machine (StreamCalibration protocol).
        self._calib_step = 0
        self._calib_data: dict[str, Any] = {}
        self._calibrate_done = threading.Event()
        self._calib_frame_queue: queue.Queue | None = None
        self._calibrate_thread: threading.Thread | None = None
        if R_lh2base is not None:
            # R_lh2base was explicitly provided (e.g. from --calibrate CLI
            # flag).  Persist it so subsequent runs load from the file.
            self._calibrated = True
            self._save_calibration()
        else:
            # No explicit R_lh2base — try loading from the calibration file.
            self._load_calibration()

        logger.info(
            "PikaSenseServicer created: port=%s latch_once=True raw_1to1=True "
            "calibrated=%s calibration_file=%s",
            port,
            self._calibrated,
            self._calibration_fpath,
        )

    # ------------------------------------------------------------------
    # Internal: pose reading and delta computation
    # ------------------------------------------------------------------

    def _try_discover_tracker(self) -> bool:
        """One scan for a non-lighthouse tracker; caches into ``_tracker_device``.

        Cold starts can register the tracker with pysurvive a few seconds
        after Connect's 10 s deadline (05 号议题实测: T20 registered at
        ~10–15 s; the server never retried and the pose channel stayed dead
        for the whole process lifetime).  Cheap enough to call per read
        while undiscovered; a no-op once cached.
        """
        if self._tracker_device is not None:
            return True
        devices = self._device.get_tracker_devices()
        trackers = [d for d in devices if not d.startswith("LH")]
        if trackers:
            self._tracker_device = trackers[0]
            logger.info("Vive Tracker discovered: %s", self._tracker_device)
            return True
        return False

    def _read_tracker_sample(self) -> TrackerSample | None:
        """Read a timestamped tracker sample without inventing freshness.

        The Pika SDK returns its cached fused pose after optical updates stop,
        and that fused timestamp may continue advancing from IMU prediction.
        The separate libsurvive light timestamp/counts are therefore part of
        correctness, not optional telemetry.
        """
        if not self._try_discover_tracker():
            return None
        pose = self._device.get_pose(self._tracker_device)
        if pose is None:
            return None
        pos = np.array(pose.position, dtype=float)  # [x, y, z] metres
        rot = Rot.from_quat(pose.rotation).as_matrix()  # [x,y,z,w] → 3×3
        try:
            source_timestamp_s = float(pose.timestamp)
        except (AttributeError, TypeError, ValueError):
            logger.error(
                "Tracker pose has no valid source timestamp; cached-pose "
                "freshness cannot be established."
            )
            return None
        if self._tracker_health_enabled:
            try:
                raw_optical_timestamp_s = float(pose.raw_optical_timestamp_s)
                raw_optical_age_s = float(pose.raw_optical_age_s)
                raw_optical_measurement_count = int(
                    pose.raw_optical_measurement_count
                )
                raw_optical_event_sequence = int(pose.raw_optical_event_sequence)
                optical_timestamp_s = float(pose.optical_timestamp_s)
                optical_age_s = float(pose.optical_age_s)
                optical_measurement_count = int(pose.optical_measurement_count)
                optical_lighthouse_count = int(pose.optical_lighthouse_count)
                optical_event_sequence = int(pose.optical_event_sequence)
                raw_confidence = getattr(pose, "pose_confidence", None)
                pose_confidence = (
                    None if raw_confidence is None else float(raw_confidence)
                )
            except (AttributeError, TypeError, ValueError):
                now = time.monotonic()
                if now - self._last_optical_health_log_monotonic >= 1.0:
                    self._last_optical_health_log_monotonic = now
                    logger.error(
                        "Tracker pose has unavailable libsurvive optical health "
                        "fields; refusing to treat fused IMU pose as optically "
                        "fresh. Verify that the updated pika_sdk native extension "
                        "is installed, then restart the Pika leader."
                    )
                return None
        else:
            # Explicit debug-only bypass.  Production defaults fail closed on
            # missing optical support instead of trusting fused timestamps.
            raw_optical_timestamp_s = source_timestamp_s
            raw_optical_age_s = 0.0
            raw_optical_measurement_count = self._tracker_min_optical_measurements
            raw_optical_event_sequence = 0
            optical_timestamp_s = source_timestamp_s
            optical_age_s = 0.0
            optical_measurement_count = self._tracker_min_optical_measurements
            optical_lighthouse_count = self._tracker_min_optical_lighthouses
            optical_event_sequence = 0
            pose_confidence = None
        if not (
            np.isfinite(pos).all()
            and np.isfinite(rot).all()
            and np.isfinite(source_timestamp_s)
            and np.isfinite(raw_optical_timestamp_s)
            and np.isfinite(raw_optical_age_s)
            and np.isfinite(optical_timestamp_s)
            and np.isfinite(optical_age_s)
            and (
                pose_confidence is None or np.isfinite(pose_confidence)
            )
        ):
            return None
        return TrackerSample(
            position=pos,
            rotation=rot,
            source_timestamp_s=source_timestamp_s,
            raw_optical_timestamp_s=raw_optical_timestamp_s,
            raw_optical_age_s=max(0.0, raw_optical_age_s),
            raw_optical_measurement_count=raw_optical_measurement_count,
            raw_optical_event_sequence=raw_optical_event_sequence,
            optical_timestamp_s=optical_timestamp_s,
            optical_age_s=max(0.0, optical_age_s),
            optical_measurement_count=optical_measurement_count,
            optical_lighthouse_count=optical_lighthouse_count,
            optical_event_sequence=optical_event_sequence,
            pose_confidence=pose_confidence,
            received_monotonic_s=time.monotonic(),
        )

    def _read_tracker_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Compatibility view used by calibration and the delta publisher."""
        sample = self._read_tracker_sample()
        if sample is None:
            return None
        return sample.position, sample.rotation

    @staticmethod
    def _rotation_distance_rad(a: np.ndarray, b: np.ndarray) -> float:
        relative = a.T @ b
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.arccos(cosine))

    def _samples_are_stable(
        self,
        samples: deque[TrackerSample],
        *,
        min_samples: int,
        min_window_s: float,
        position_spread_m: float | None = None,
        rotation_spread_rad: float | None = None,
    ) -> bool:
        if len(samples) < min_samples:
            return False
        if samples[-1].received_monotonic_s - samples[0].received_monotonic_s < min_window_s:
            return False
        positions = np.stack([sample.position for sample in samples])
        position_spread = float(
            np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1).max()
        )
        position_limit = (
            self._tracker_position_spread_m
            if position_spread_m is None
            else position_spread_m
        )
        if position_spread > position_limit:
            return False
        rotations = [sample.rotation for sample in samples]
        rotation_spread = max(
            self._rotation_distance_rad(a, b)
            for index, a in enumerate(rotations)
            for b in rotations[index:]
        )
        rotation_limit = (
            self._tracker_rotation_spread_rad
            if rotation_spread_rad is None
            else rotation_spread_rad
        )
        return rotation_spread <= rotation_limit

    @staticmethod
    def _sample_window_ready(
        samples: deque[TrackerSample],
        *,
        min_samples: int,
        min_window_s: float,
    ) -> bool:
        return len(samples) >= min_samples and (
            samples[-1].received_monotonic_s
            - samples[0].received_monotonic_s
            >= min_window_s
        )

    def _optical_health_reason(self, sample: TrackerSample) -> str | None:
        if sample.optical_age_s > self._tracker_stale_s:
            return f"optical age {sample.optical_age_s * 1000.0:.0f}ms"
        if sample.optical_measurement_count < self._tracker_min_optical_measurements:
            return (
                f"only {sample.optical_measurement_count} recent optical measurements"
            )
        if sample.optical_lighthouse_count < self._tracker_min_optical_lighthouses:
            return f"only {sample.optical_lighthouse_count} visible lighthouse(s)"
        return None

    def _raw_optical_health_reason(self, sample: TrackerSample) -> str | None:
        """Return why Tracker photodiode visibility is not currently proven."""
        if sample.raw_optical_age_s > self._tracker_stale_s:
            return f"raw light age {sample.raw_optical_age_s * 1000.0:.0f}ms"
        if sample.raw_optical_measurement_count < 1:
            return "no recent raw Lighthouse sensor hits"
        return None

    def _sample_has_optical_support(self, sample: TrackerSample) -> bool:
        return self._optical_health_reason(sample) is None

    def _await_reference_sample(self, context) -> TrackerSample | None:
        """Return a newly observed, briefly stable pose for explicit alignment.

        ``SetReference`` is an operator-authorized trust boundary: moving the
        Pika a long way after Connect is expected while aligning it with the
        follower.  Runtime jump checks therefore must not compare this pose
        with the last action sample.  We still require distinct optical
        timestamps and a short stable window so a cached/occluded pose cannot
        become the new reference.
        """
        deadline = time.monotonic() + self._tracker_reference_timeout_s
        samples: deque[TrackerSample] = deque()
        baseline = self._read_tracker_sample()
        last_optical_timestamp = (
            None if baseline is None else baseline.optical_timestamp_s
        )
        keep_window_s = self._tracker_recheck_window_s + 0.05
        self._last_reference_failure_reason = "no fresh tracker pose"

        while time.monotonic() < deadline and self._context_active(context):
            sample = self._read_tracker_sample()
            if (
                sample is None
                or not self._sample_has_optical_support(sample)
                or sample.optical_timestamp_s == last_optical_timestamp
            ):
                if sample is None:
                    self._last_reference_failure_reason = "no tracker pose"
                else:
                    health_reason = self._optical_health_reason(sample)
                    if health_reason is not None:
                        self._last_reference_failure_reason = health_reason
                    elif not samples:
                        self._last_reference_failure_reason = (
                            "no distinct optical timestamp"
                        )
                time.sleep(0.005)
                continue
            last_optical_timestamp = sample.optical_timestamp_s
            samples.append(sample)
            while (
                len(samples) > 1
                and samples[-1].received_monotonic_s
                - samples[0].received_monotonic_s
                > keep_window_s
            ):
                samples.popleft()
            if self._samples_are_stable(
                samples,
                min_samples=self._tracker_recheck_samples,
                min_window_s=self._tracker_recheck_window_s,
                position_spread_m=self._tracker_reference_position_spread_m,
                rotation_spread_rad=self._tracker_reference_rotation_spread_rad,
            ):
                self._last_reference_failure_reason = None
                return sample
            window_s = (
                0.0
                if len(samples) < 2
                else samples[-1].received_monotonic_s
                - samples[0].received_monotonic_s
            )
            if len(samples) < self._tracker_recheck_samples:
                self._last_reference_failure_reason = (
                    f"only {len(samples)}/{self._tracker_recheck_samples} fresh samples"
                )
            elif window_s < self._tracker_recheck_window_s:
                self._last_reference_failure_reason = (
                    f"stable window {window_s:.3f}/{self._tracker_recheck_window_s:.3f}s"
                )
            else:
                positions = np.stack([item.position for item in samples])
                position_spread = float(
                    np.linalg.norm(
                        positions[:, None, :] - positions[None, :, :], axis=-1
                    ).max()
                )
                rotations = [item.rotation for item in samples]
                rotation_spread = max(
                    self._rotation_distance_rad(a, b)
                    for index, a in enumerate(rotations)
                    for b in rotations[index:]
                )
                self._last_reference_failure_reason = (
                    f"pose spread {position_spread * 1000.0:.1f}mm/"
                    f"{np.degrees(rotation_spread):.1f}deg exceeds "
                    f"{self._tracker_reference_position_spread_m * 1000.0:.1f}mm/"
                    f"{np.degrees(self._tracker_reference_rotation_spread_rad):.1f}deg"
                )
            time.sleep(0.005)
        return None

    @staticmethod
    def _context_active(context) -> bool:
        return context is None or not hasattr(context, "is_active") or context.is_active()

    def _await_tracker_ready(self, context) -> TrackerSample:
        """Block Connect until optical samples settle and the operator confirms.

        Only distinct lighthouse timestamps with enough recent optical
        measurements count.  Fused IMU timestamps deliberately do not count.
        """
        deadline = time.monotonic() + self._tracker_ready_timeout_s
        first_fresh_monotonic: float | None = None
        last_optical_timestamp: float | None = None
        samples: deque[TrackerSample] = deque()
        phase = "settling"
        confirmation_state: int | None = None
        last_progress = 0.0

        logger.info("TRACKER READY: discovering and waiting for fresh pose samples...")
        while time.monotonic() < deadline and self._context_active(context):
            sample = self._read_tracker_sample()
            now = time.monotonic()
            if sample is None:
                time.sleep(0.01)
                continue
            health_reason = self._optical_health_reason(sample)
            if (
                health_reason is not None
                or sample.optical_timestamp_s == last_optical_timestamp
            ):
                if health_reason is not None and now - last_progress >= 1.0:
                    last_progress = now
                    logger.info("TRACKER READY: waiting for optical support (%s)", health_reason)
                time.sleep(0.01)
                continue
            last_optical_timestamp = sample.optical_timestamp_s
            if first_fresh_monotonic is None:
                first_fresh_monotonic = now
                logger.info(
                    "TRACKER READY: first fresh pose; starting %.1fs optical-"
                    "health soak (this is not a global-fit convergence proof).",
                    self._tracker_min_soak_s,
                )

            samples.append(sample)
            keep_window_s = max(
                self._tracker_stable_window_s,
                self._tracker_recheck_window_s,
            )
            while (
                len(samples) > 1
                and samples[-1].received_monotonic_s
                - samples[0].received_monotonic_s
                > keep_window_s * 1.5 + 0.1
            ):
                samples.popleft()

            if now - last_progress >= 1.0:
                last_progress = now
                soak_elapsed = now - first_fresh_monotonic
                logger.info(
                    "TRACKER READY: phase=%s fresh_samples=%d optical_soak=%.1f/%.1fs",
                    phase,
                    len(samples),
                    soak_elapsed,
                    self._tracker_min_soak_s,
                )

            if phase == "settling":
                soaked = now - first_fresh_monotonic >= self._tracker_min_soak_s
                window_ready = self._sample_window_ready(
                    samples,
                    min_samples=self._tracker_stable_samples,
                    min_window_s=self._tracker_stable_window_s,
                )
                if not soaked or not window_ready:
                    continue
                # Collection owns the later alignment/Enter flow.  At Connect
                # it only needs evidence that lighthouse tracking is healthy;
                # requiring the operator's hand to be motionless here
                # incorrectly conflates physical motion with solver health.
                if not self._auto_reference:
                    self._prime_runtime_tracker(sample)
                    logger.info(
                        "TRACKER READY: optical stream health accepted; global-fit "
                        "error is unavailable from the Pika SDK, and pose may move "
                        "until collection alignment."
                    )
                    return sample
                if not self._samples_are_stable(
                    samples,
                    min_samples=self._tracker_stable_samples,
                    min_window_s=self._tracker_stable_window_s,
                ):
                    continue
                if not (self._auto_reference and self._require_start_confirmation):
                    self._prime_runtime_tracker(sample)
                    logger.info("TRACKER READY: stable pose window accepted.")
                    return sample
                phase = "await-confirm"
                try:
                    confirmation_state = int(self._command_state_provider())
                except Exception:
                    confirmation_state = None
                logger.info(
                    "TRACKER READY: stable. Hold Pika still and quickly squeeze "
                    "the gripper to start teleoperation."
                )
                continue

            if phase == "await-confirm":
                if not self._samples_are_stable(
                    samples,
                    min_samples=self._tracker_stable_samples,
                    min_window_s=self._tracker_stable_window_s,
                ):
                    phase = "settling"
                    logger.warning(
                        "TRACKER READY: pose moved before confirmation; settling again."
                    )
                    continue
                try:
                    command_now = int(self._command_state_provider())
                except Exception:
                    command_now = None
                if not command_state_edge(command_now, confirmation_state):
                    if command_now is not None:
                        confirmation_state = command_now
                    continue
                self._prev_command_state = command_now
                phase = "recheck"
                samples.clear()
                logger.info("TRACKER READY: start confirmed; rechecking pose...")
                continue

            if phase == "recheck" and self._samples_are_stable(
                samples,
                min_samples=self._tracker_recheck_samples,
                min_window_s=self._tracker_recheck_window_s,
            ):
                self._prime_runtime_tracker(sample)
                logger.info("TRACKER READY: confirmed stable pose accepted.")
                return sample

        message = (
            f"tracker did not become stable and confirmed within "
            f"{self._tracker_ready_timeout_s:.1f}s"
        )
        if context is not None and hasattr(context, "abort"):
            import grpc

            context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
        raise RuntimeError(message)

    def _prime_runtime_tracker(self, sample: TrackerSample) -> None:
        self._last_seen_tracker_source_ts = sample.source_timestamp_s
        self._last_seen_tracker_raw_optical_ts = sample.raw_optical_timestamp_s
        self._last_seen_tracker_optical_ts = sample.optical_timestamp_s
        self._last_fresh_received_monotonic = sample.received_monotonic_s
        self._last_valid_tracker_sample = sample
        self._pending_tracker_samples.clear()
        self._bad_tracker_samples = 0
        self._reference_required = False
        self._reference_fault_kind = None
        self._reference_confirmation_pending = False
        self._tracker_recovery_samples.clear()
        self._tracker_recovery_ready = False
        self._raw_optical_reacquiring = False
        self._raw_optical_reacquiring_since = None
        self._decoder_restart_attempts = 0
        self._last_decoder_restart_monotonic = None
        self._decoder_restart_discovery_deadline = None

    def _require_new_reference(
        self,
        reason: str,
        *,
        fault_kind: str = _REFERENCE_FAULT_OPTICAL,
    ) -> None:
        if fault_kind not in (
            self._REFERENCE_FAULT_OPTICAL,
            self._REFERENCE_FAULT_POSE_DISCONTINUITY,
        ):
            raise ValueError(f"unsupported reference fault kind: {fault_kind}")
        newly_required = not self._reference_required
        # A real optical-health failure supersedes a pose-only quarantine.
        # The reverse transition is impossible while reference_required is
        # active because pose motion is no longer published or classified.
        if newly_required or fault_kind == self._REFERENCE_FAULT_OPTICAL:
            self._reference_fault_kind = fault_kind
        if newly_required:
            logger.error(
                "%s: %s — holding output; stable recovery and "
                "a new Pika quick-squeeze confirmation are required.",
                (
                    "TRACKER OPTICAL"
                    if fault_kind == self._REFERENCE_FAULT_OPTICAL
                    else "TRACKER POSE"
                ),
                reason,
            )
        self._reference_required = True
        self._reference_confirmation_pending = False
        self._clutched = False
        self._pending_relatch = True
        self._pending_tracker_samples.clear()
        self._bad_tracker_samples = 0
        # A second optical loss while waiting for operator confirmation must
        # revoke the latched ready state as well.
        self._tracker_recovery_samples.clear()
        self._tracker_recovery_ready = False

    def _update_recovery_window(self, sample: TrackerSample) -> None:
        # Once localization convergence has been established, keep the prompt
        # latched.  The operator is explicitly allowed to reposition Pika
        # before the quick squeeze; only a subsequent optical-health failure
        # calls _require_new_reference() and revokes this state.
        if self._tracker_recovery_ready:
            return
        samples = self._tracker_recovery_samples
        samples.append(sample)
        while (
            len(samples) > 1
            and samples[-1].received_monotonic_s
            - samples[0].received_monotonic_s
            > self._tracker_recheck_window_s * 1.5 + 0.1
        ):
            samples.popleft()
        ready = self._samples_are_stable(
            samples,
            min_samples=self._tracker_recheck_samples,
            min_window_s=self._tracker_recheck_window_s,
        )
        if ready and not self._tracker_recovery_ready:
            logger.warning(
                "TRACKER HEALTH: tracking recovered and converged. The "
                "operator may now align Pika, then quickly squeeze the "
                "gripper to request a new reference."
            )
        if ready:
            self._tracker_recovery_ready = True

    @staticmethod
    def _tracker_sample_dts(
        previous: TrackerSample, sample: TrackerSample
    ) -> tuple[float, float, float]:
        """Return kinematic, source and optical deltas in seconds.

        The fused pose belongs to ``source_timestamp_s``.  The optical stamp
        is the time of the latest Lighthouse sweep and is used for freshness,
        not as the denominator for fused-pose velocity.  Falling back to the
        optical delta preserves compatibility with malformed/legacy samples.
        """
        source_dt = sample.source_timestamp_s - previous.source_timestamp_s
        optical_dt = sample.optical_timestamp_s - previous.optical_timestamp_s
        if np.isfinite(source_dt) and source_dt > 0.0:
            motion_dt = source_dt
        elif np.isfinite(optical_dt) and optical_dt > 0.0:
            motion_dt = optical_dt
        else:
            motion_dt = 1e-3
        return max(float(motion_dt), 1e-3), float(source_dt), float(optical_dt)

    def _tracker_motion_metrics(
        self, previous: TrackerSample, sample: TrackerSample
    ) -> dict[str, float | bool]:
        dt, source_dt, optical_dt = self._tracker_sample_dts(previous, sample)
        translation = float(np.linalg.norm(sample.position - previous.position))
        rotation = self._rotation_distance_rad(previous.rotation, sample.rotation)
        translation_limit = min(
            self._MOTION_TRANSLATION_SOFT_CAP_M,
            self._MOTION_TRANSLATION_BASE_M
            + self._MOTION_TRANSLATION_SPEED_M_S * dt,
        )
        rotation_limit = min(
            self._MOTION_ROTATION_SOFT_CAP_RAD,
            self._MOTION_ROTATION_BASE_RAD
            + self._MOTION_ROTATION_SPEED_RAD_S * dt,
        )
        return {
            "dt": dt,
            "source_dt": source_dt,
            "optical_dt": optical_dt,
            "translation": translation,
            "rotation": rotation,
            "translation_limit": translation_limit,
            "rotation_limit": rotation_limit,
            "invalid": (
                translation > translation_limit or rotation > rotation_limit
            ),
            "absolute_jump": (
                translation > self._MOTION_TRANSLATION_ABSOLUTE_M
                or rotation > self._MOTION_ROTATION_ABSOLUTE_RAD
            ),
        }

    @staticmethod
    def _tracker_motion_vectors(
        previous: TrackerSample, sample: TrackerSample
    ) -> tuple[np.ndarray, np.ndarray, float]:
        dt, _, _ = PikaSenseServicer._tracker_sample_dts(previous, sample)
        translation_velocity = (sample.position - previous.position) / dt
        relative_rotation = previous.rotation.T @ sample.rotation
        # Express both incremental axes in the Lighthouse/world frame before
        # comparing consecutive rotations.
        rotation_velocity = (
            previous.rotation @ Rot.from_matrix(relative_rotation).as_rotvec()
        ) / dt
        return translation_velocity, rotation_velocity, dt

    @classmethod
    def _vector_motion_is_coherent(
        cls, first: np.ndarray, second: np.ndarray, *, active: bool
    ) -> bool:
        if not active:
            return True
        first_speed = float(np.linalg.norm(first))
        second_speed = float(np.linalg.norm(second))
        if first_speed <= 1e-9 or second_speed <= 1e-9:
            return False
        speed_ratio = second_speed / first_speed
        direction_cosine = float(
            np.dot(first, second) / (first_speed * second_speed)
        )
        return (
            cls._MOTION_MIN_SPEED_RATIO
            <= speed_ratio
            <= cls._MOTION_MAX_SPEED_RATIO
            and direction_cosine >= cls._MOTION_DIRECTION_COSINE
        )

    def _pending_motion_is_coherent(self, sample: TrackerSample) -> bool:
        """Whether a quarantined pose is followed by a continuous trajectory."""
        anchor = self._last_valid_tracker_sample
        if anchor is None or not self._pending_tracker_samples:
            return False
        candidate = self._pending_tracker_samples[-1]
        first_v, first_w, _ = self._tracker_motion_vectors(anchor, candidate)
        second_v, second_w, _ = self._tracker_motion_vectors(candidate, sample)
        direct = self._tracker_motion_metrics(anchor, sample)
        return self._vector_motion_is_coherent(
            first_v,
            second_v,
            active=bool(
                float(direct["translation"])
                > float(direct["translation_limit"])
            ),
        ) and self._vector_motion_is_coherent(
            first_w,
            second_w,
            active=bool(
                float(direct["rotation"]) > float(direct["rotation_limit"])
            ),
        )

    def _accept_runtime_tracker_sample(
        self, sample: TrackerSample
    ) -> tuple[np.ndarray, np.ndarray]:
        self._last_valid_tracker_sample = sample
        self._pending_tracker_samples.clear()
        self._bad_tracker_samples = 0
        self._last_tracker_health_reason = None
        return sample.position, sample.rotation

    @staticmethod
    def _held_tracker_pose(
        sample: TrackerSample | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if sample is None:
            return None
        return sample.position, sample.rotation

    def _maybe_restart_tracker_decoder(self, now: float) -> bool:
        """Rebuild a stuck decoder while the follower is already held."""
        if self._raw_optical_reacquiring_since is None:
            self._raw_optical_reacquiring_since = now
        if now - self._raw_optical_reacquiring_since < self._decoder_restart_after_s:
            return False
        if self._decoder_restart_attempts >= self._DECODER_RESTART_MAX_ATTEMPTS:
            return False
        last_restart = self._last_decoder_restart_monotonic
        if (
            last_restart is not None
            and now - last_restart < self._DECODER_RESTART_COOLDOWN_S
        ):
            return False

        restart = getattr(self._device, "restart_vive_tracker", None)
        if not callable(restart):
            logger.error(
                "TRACKER HEALTH: raw light is visible but decoded tracking is "
                "stuck; this Pika SDK cannot restart its decoder in-process."
            )
            self._decoder_restart_attempts = self._DECODER_RESTART_MAX_ATTEMPTS
            return False

        self._decoder_restart_attempts += 1
        self._last_decoder_restart_monotonic = now
        logger.warning(
            "TRACKER HEALTH: raw light remained visible without decoded "
            "sync/sweep for %.1fs; restarting the libsurvive decoder context "
            "(attempt %d/%d).",
            now - self._raw_optical_reacquiring_since,
            self._decoder_restart_attempts,
            self._DECODER_RESTART_MAX_ATTEMPTS,
        )
        try:
            restarted = bool(restart())
        except Exception:
            logger.exception("TRACKER HEALTH: decoder context restart failed")
            restarted = False
        if not restarted:
            self._last_tracker_health_reason = "tracker decoder restart failed"
            return False

        # Never reuse PoseData from the destroyed context.  Device discovery
        # and all freshness baselines restart while the published action stays
        # frozen and a new operator reference remains mandatory.
        self._tracker_device = None
        self._last_seen_tracker_source_ts = None
        self._last_seen_tracker_raw_optical_ts = None
        self._last_seen_tracker_optical_ts = None
        self._last_fresh_received_monotonic = None
        self._last_valid_tracker_sample = None
        self._pending_tracker_samples.clear()
        self._tracker_recovery_samples.clear()
        self._tracker_recovery_ready = False
        self._raw_optical_reacquiring = True
        self._raw_optical_reacquiring_since = None
        self._decoder_restart_discovery_deadline = (
            time.monotonic() + self._DECODER_REDISCOVERY_GRACE_S
        )
        self._last_tracker_health_reason = (
            "tracker decoder restarted; waiting for fresh optical poses"
        )
        return True

    def _runtime_tracker_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return only a fresh, plausible pose; otherwise freeze publication."""
        sample = self._read_tracker_sample()
        if sample is not None and not self._tracker_health_enabled:
            self._prime_runtime_tracker(sample)
            return sample.position, sample.rotation
        now = time.monotonic()
        raw_health_reason = (
            None if sample is None else self._raw_optical_health_reason(sample)
        )
        raw_optical_available = bool(
            sample is not None and raw_health_reason is None
        )
        if raw_optical_available:
            self._last_seen_tracker_raw_optical_ts = sample.raw_optical_timestamp_s
            self._decoder_restart_discovery_deadline = None
        health_reason = None if sample is None else self._optical_health_reason(sample)
        has_new_optical = bool(
            sample is not None
            and health_reason is None
            and sample.optical_timestamp_s != self._last_seen_tracker_optical_ts
        )
        if not has_new_optical:
            last_fresh = self._last_fresh_received_monotonic
            wall_age = float("inf") if last_fresh is None else now - last_fresh
            reported_age = 0.0 if sample is None else sample.optical_age_s
            age = max(wall_age, reported_age)
            self._last_tracker_health_reason = (
                health_reason
                or (
                    "no tracker sample"
                    if sample is None
                    else f"no new optical sample for {age:.3f}s"
                )
            )
            if age >= self._tracker_stale_s:
                severity = (
                    "lost"
                    if age >= self._tracker_reference_lost_s
                    else "interrupted"
                )
                self._require_new_reference(
                    f"optical tracking {severity}: {self._last_tracker_health_reason}"
                )
            if self._reference_required and raw_optical_available:
                if not self._raw_optical_reacquiring:
                    logger.warning(
                        "TRACKER HEALTH: raw Lighthouse light has returned; "
                        "waiting for decoded sync/sweep and pose convergence."
                    )
                self._raw_optical_reacquiring = True
                self._last_tracker_health_reason = (
                    "raw Lighthouse light visible; waiting for decoded tracking"
                )
                self._maybe_restart_tracker_decoder(now)
            elif not raw_optical_available:
                rediscovery_deadline = self._decoder_restart_discovery_deadline
                if rediscovery_deadline is not None and now < rediscovery_deadline:
                    self._raw_optical_reacquiring = True
                    self._last_tracker_health_reason = (
                        "tracker decoder restarted; rediscovering Tracker device"
                    )
                else:
                    self._decoder_restart_discovery_deadline = None
                    self._raw_optical_reacquiring = False
                    self._raw_optical_reacquiring_since = None
            return None if age >= self._tracker_stale_s else (
                None
                if self._last_valid_tracker_sample is None
                else (
                    self._last_valid_tracker_sample.position,
                    self._last_valid_tracker_sample.rotation,
                )
            )

        previous = self._last_valid_tracker_sample
        optical_gap_s = (
            0.0
            if self._last_fresh_received_monotonic is None
            else sample.received_monotonic_s - self._last_fresh_received_monotonic
        )
        self._last_seen_tracker_source_ts = sample.source_timestamp_s
        self._last_seen_tracker_raw_optical_ts = sample.raw_optical_timestamp_s
        self._last_seen_tracker_optical_ts = sample.optical_timestamp_s
        self._last_fresh_received_monotonic = sample.received_monotonic_s
        self._last_tracker_health_reason = None
        self._raw_optical_reacquiring = False
        self._raw_optical_reacquiring_since = None
        self._decoder_restart_discovery_deadline = None

        if optical_gap_s >= self._tracker_stale_s:
            severity = (
                "lost"
                if optical_gap_s >= self._tracker_reference_lost_s
                else "interrupted"
            )
            self._require_new_reference(
                f"optical tracking {severity}: gap lasted {optical_gap_s:.3f}s"
            )

        if self._reference_required:
            self._last_valid_tracker_sample = sample
            self._update_recovery_window(sample)
            return None

        # Clutch/re-reference repositioning is intentional.  Continue checking
        # optical freshness, but do not classify that operator motion as a
        # runtime pose discontinuity.
        motion_gate_active = self._clutched and not self._pending_relatch
        if previous is not None and motion_gate_active:
            direct = self._tracker_motion_metrics(previous, sample)
            if bool(direct["absolute_jump"]):
                logger.error(
                    "TRACKER MOTION: absolute discontinuity dpos=%.1fmm "
                    "drot=%.1fdeg source_dt=%.1fms optical_dt=%.1fms "
                    "optical_age=%.1fms events=%d channels=%d",
                    float(direct["translation"]) * 1000.0,
                    np.degrees(float(direct["rotation"])),
                    float(direct["source_dt"]) * 1000.0,
                    float(direct["optical_dt"]) * 1000.0,
                    sample.optical_age_s * 1000.0,
                    sample.optical_measurement_count,
                    sample.optical_lighthouse_count,
                )
                self._require_new_reference(
                    "implausible absolute pose jump",
                    fault_kind=self._REFERENCE_FAULT_POSE_DISCONTINUITY,
                )
                self._update_recovery_window(sample)
                return None

            if not bool(direct["invalid"]):
                if self._pending_tracker_samples:
                    logger.info(
                        "TRACKER MOTION: quarantined sample discarded/resolved; "
                        "current pose is plausible from the last accepted pose."
                    )
                return self._accept_runtime_tracker_sample(sample)

            if self._pending_motion_is_coherent(sample):
                logger.info(
                    "TRACKER MOTION: coherent fast trajectory confirmed "
                    "dpos=%.1fmm drot=%.1fdeg over %.1fms; continuing without "
                    "a client-visible tracking hold.",
                    float(direct["translation"]) * 1000.0,
                    np.degrees(float(direct["rotation"])),
                    float(direct["dt"]) * 1000.0,
                )
                return self._accept_runtime_tracker_sample(sample)

            self._pending_tracker_samples.append(sample)
            self._bad_tracker_samples = len(self._pending_tracker_samples)
            self._last_tracker_health_reason = (
                "pose motion awaiting temporal confirmation "
                f"({self._bad_tracker_samples}/{self._MOTION_CONFIRM_SAMPLES})"
            )
            logger.warning(
                "TRACKER MOTION: quarantined dpos=%.1f/%.1fmm "
                "drot=%.1f/%.1fdeg v=%.2fm/s w=%.0fdeg/s "
                "source_dt=%.1fms optical_dt=%.1fms optical_age=%.1fms "
                "events=%d channels=%d pending=%d/%d",
                float(direct["translation"]) * 1000.0,
                float(direct["translation_limit"]) * 1000.0,
                np.degrees(float(direct["rotation"])),
                np.degrees(float(direct["rotation_limit"])),
                float(direct["translation"]) / float(direct["dt"]),
                np.degrees(float(direct["rotation"]) / float(direct["dt"])),
                float(direct["source_dt"]) * 1000.0,
                float(direct["optical_dt"]) * 1000.0,
                sample.optical_age_s * 1000.0,
                sample.optical_measurement_count,
                sample.optical_lighthouse_count,
                self._bad_tracker_samples,
                self._MOTION_CONFIRM_SAMPLES,
            )
            if self._bad_tracker_samples >= self._MOTION_CONFIRM_SAMPLES:
                self._require_new_reference(
                    "pose trajectory remained discontinuous for three samples",
                    fault_kind=self._REFERENCE_FAULT_POSE_DISCONTINUITY,
                )
                self._update_recovery_window(sample)
                return None
            # Quarantine is internal: publish the last accepted pose and keep
            # tracking READY.  Collection should not tell the operator that
            # Lighthouse tracking was lost when optical data is still fresh.
            return self._held_tracker_pose(previous)

        return self._accept_runtime_tracker_sample(sample)

    # ------------------------------------------------------------------
    # Calibration file persistence
    # ------------------------------------------------------------------

    def _load_calibration(self) -> None:
        """Load R_lh2base + gripper range from the calibration JSON file.

        Sets ``_calibrated = True`` if the file exists and loads successfully.
        """
        if not self._calibration_fpath.is_file():
            return
        try:
            with open(self._calibration_fpath) as f:
                data = json.load(f)
            self._R_lh2base = np.array(data["R_lh2base"], dtype=float).reshape(3, 3)
            self._gripper_min_mm = float(data.get("gripper_min_mm", 0.0))
            self._gripper_max_mm = float(data.get("gripper_max_mm", 60.0))
            self._calibrated = True
            logger.info(
                "Loaded calibration from %s: gripper=[%.1f, %.1f]mm",
                self._calibration_fpath, self._gripper_min_mm, self._gripper_max_mm,
            )
        except Exception as e:
            logger.warning("Failed to load calibration from %s: %s", self._calibration_fpath, e)

    def _save_calibration(self) -> None:
        """Write R_lh2base + gripper range to the calibration JSON file."""
        data = {
            "R_lh2base": self._R_lh2base.tolist(),
            "gripper_min_mm": self._gripper_min_mm,
            "gripper_max_mm": self._gripper_max_mm,
        }
        with open(self._calibration_fpath, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Calibration saved to %s", self._calibration_fpath)

    # ------------------------------------------------------------------
    # Multi-step calibration (StreamCalibration protocol)
    # ------------------------------------------------------------------

    def _read_gripper(self) -> float | None:
        """Read gripper distance in mm, returning ``None`` on error."""
        try:
            return float(self._device.get_gripper_distance())
        except Exception:
            return None

    def _build_calib_frame(
        self, step_idx: int, tracker: tuple[np.ndarray, np.ndarray] | None, grip: float | None,
    ) -> device_pb2.CalibrationFrame:
        """Build a CalibrationFrame for the current step.

        Always emits exactly 5 readings (prompt + 3 position + Grip) so the
        client's ``move_cursor_up(n + 3)`` uses a constant n=5.  For movement
        steps the prompt includes **live displacement feedback** (e.g.
        "Δ: 45mm, need ≥50mm") and the position readings show delta components
        relative to the captured start position, giving the user directional
        feedback instead of meaningless absolute lighthouse coordinates.
        """
        step = self._CALIB_STEPS[step_idx]
        key = step["key"]
        tracker_pos = tracker[0] if tracker is not None else None
        grip_val = int(grip) if grip is not None else 0

        # --- Dynamic prompt with live feedback ---
        ref_key = {
            "forward_moved": "forward_start",
            "right_moved": "right_start",
            "up_moved": "up_start",
        }.get(key)
        if ref_key is not None and ref_key in self._calib_data and tracker_pos is not None:
            delta_m = tracker_pos - self._calib_data[ref_key]
            delta_mm = np.linalg.norm(delta_m) * 1000
            direction = {"forward_moved": "FORWARD", "right_moved": "RIGHT", "up_moved": "UP"}[key]
            step_num = {"forward_moved": 2, "right_moved": 4, "up_moved": 6}[key]
            prompt = f"[{step_num}/8] Move {direction} (Δ: {delta_mm:.0f}mm, need ≥50mm) → Enter"
            # Show delta components instead of absolute coordinates
            pos_mm = delta_m * 1000.0
            axis_label = "Δ"
        elif key == "gripper_close":
            prompt = f"[7/8] CLOSE gripper (now: {grip_val}mm) → Enter"
            pos_mm = tracker_pos * 1000.0 if tracker_pos is not None else np.zeros(3)
            axis_label = ""
        elif key == "gripper_open":
            prompt = f"[8/8] OPEN gripper (now: {grip_val}mm) → Enter"
            pos_mm = tracker_pos * 1000.0 if tracker_pos is not None else np.zeros(3)
            axis_label = ""
        elif key == "forward_start":
            prompt = "[1/8] Hold at START → Enter"
            pos_mm = tracker_pos * 1000.0 if tracker_pos is not None else np.zeros(3)
            axis_label = ""
        elif key == "right_start":
            prompt = "[3/8] Return to NEUTRAL → Enter"
            pos_mm = tracker_pos * 1000.0 if tracker_pos is not None else np.zeros(3)
            axis_label = ""
        elif key == "up_start":
            prompt = "[5/8] Return to NEUTRAL → Enter"
            pos_mm = tracker_pos * 1000.0 if tracker_pos is not None else np.zeros(3)
            axis_label = ""
        else:
            prompt = step["prompt"]
            pos_mm = tracker_pos * 1000.0 if tracker_pos is not None else np.zeros(3)
            axis_label = ""

        readings = [
            device_pb2.CalibrationFrame.MotorReading(name=prompt, position=0),
            device_pb2.CalibrationFrame.MotorReading(name=f"  {axis_label}X mm", position=int(pos_mm[0])),
            device_pb2.CalibrationFrame.MotorReading(name=f"  {axis_label}Y mm", position=int(pos_mm[1])),
            device_pb2.CalibrationFrame.MotorReading(name=f"  {axis_label}Z mm", position=int(pos_mm[2])),
            device_pb2.CalibrationFrame.MotorReading(name="  Grip", position=grip_val),
        ]
        return device_pb2.CalibrationFrame(readings=readings)

    def _run_calib_step(self, step_idx: int) -> None:
        """Background thread for one calibration step.

        Streams live sensor data via ``_calib_frame_queue`` until
        ``CalibrateDone`` sets the event, then captures the data point for
        this step.  After the final step, computes ``R_lh2base`` + gripper
        range and saves the calibration file.
        """
        step = self._CALIB_STEPS[step_idx]
        key = step["key"]
        try:
            while not self._calibrate_done.is_set():
                tracker = self._read_tracker_pose()
                grip = self._read_gripper()
                if self._calib_frame_queue is not None:
                    self._calib_frame_queue.put(
                        self._build_calib_frame(step_idx, tracker, grip)
                    )
                time.sleep(0.1)  # 10 Hz

            # CalibrateDone received — capture the data point.
            tracker = self._read_tracker_pose()
            if key in ("forward_start", "forward_moved", "right_start", "right_moved",
                       "up_start", "up_moved"):
                if tracker is None:
                    raise RuntimeError(f"No tracker data at step {key}")
                self._calib_data[key] = tracker[0].copy()  # position [x, y, z]
            elif key == "gripper_close":
                self._calib_data["gripper_min"] = self._read_gripper()
            elif key == "gripper_open":
                self._calib_data["gripper_max"] = self._read_gripper()
                self._finalize_calibration()

            self._calib_step += 1
        except Exception as e:
            logger.error("Calibration step %d (%s) failed: %s", step_idx, key, e)
            self._calibrate_error = True
        finally:
            self._calibrating = False
            if self._calib_frame_queue is not None:
                self._calib_frame_queue.put(None)  # end StreamCalibration

    def _finalize_calibration(self) -> None:
        """Compute ``R_lh2base`` + gripper range from captured data points."""
        forward_lh = self._calib_data["forward_moved"] - self._calib_data["forward_start"]
        norm = np.linalg.norm(forward_lh)
        if norm < 0.005:
            raise RuntimeError(
                f"Forward movement too small ({norm*1000:.1f} mm). Move at least ~10 cm."
            )
        forward_lh /= norm

        right_lh = self._calib_data["right_moved"] - self._calib_data["right_start"]
        norm = np.linalg.norm(right_lh)
        if norm < 0.005:
            raise RuntimeError(
                f"Right movement too small ({norm*1000:.1f} mm). Move at least ~10 cm."
            )
        right_lh /= norm

        # --- MEASURED up direction (replaces cross product — eliminates sign ambiguity) ---
        up_lh = self._calib_data["up_moved"] - self._calib_data["up_start"]
        norm = np.linalg.norm(up_lh)
        if norm < 0.005:
            raise RuntimeError(
                f"Up movement too small ({norm*1000:.1f} mm). Move at least ~10 cm."
            )
        up_lh /= norm

        # --- Sanity check: measured up should be close to cross(right, forward) ---
        up_cross = np.cross(right_lh, forward_lh)
        up_cross /= np.linalg.norm(up_cross)
        cos_angle = float(np.dot(up_lh, up_cross))
        if cos_angle < 0.0:
            logger.warning(
                "Calibration WARNING: measured UP is opposite to cross(RIGHT, FORWARD) "
                "(cos=%.2f). Movements may not have been orthogonal.", cos_angle,
            )
        elif cos_angle < 0.7:  # > 45° deviation
            logger.warning(
                "Calibration WARNING: measured UP deviates from cross(RIGHT, FORWARD) "
                "by %.0f°. Forward/right/up movements should be mutually perpendicular.",
                np.degrees(np.arccos(cos_angle)),
            )

        # --- Build R_lh2base from the measured forward + up directions ---
        R_lh2base = build_R_lh2base(forward_lh, up_lh)
        det = np.linalg.det(R_lh2base)
        if abs(det - 1.0) > 0.01:
            raise RuntimeError(f"Calibration produced invalid rotation (det={det:.4f}).")

        grip_min = self._calib_data.get("gripper_min")
        grip_max = self._calib_data.get("gripper_max")
        if grip_min is None or grip_max is None:
            raise RuntimeError("Gripper sensor read failed during calibration.")
        if grip_max <= grip_min:
            raise RuntimeError(
                f"Gripper range invalid: min={grip_min:.1f}mm max={grip_max:.1f}mm"
            )

        with self._lock:
            self._R_lh2base = R_lh2base
            self._gripper_min_mm = grip_min
            self._gripper_max_mm = grip_max
            self._calibrated = True
            self._calibrate_error = False
        self._save_calibration()
        logger.info(
            "Calibration complete: R_lh2base + gripper [%.1f, %.1f]mm saved.",
            grip_min, grip_max,
        )

    def _reset_latch(self) -> None:
        """Drop T_begin and the published offset (Connect)."""
        self._t_begin_pos = None
        self._t_begin_rot = None
        self._published_pos = np.zeros(3)
        self._published_rot = np.eye(3)
        self._segment_base_pos = np.zeros(3)
        self._segment_base_rot = np.eye(3)
        self._reference_confirmation_pending = False

    def _relatch_cumulative_segment(
        self, tracker: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Start a new tracker segment while preserving the published target.

        The next relative hand transform is right-composed onto the frozen
        transform.  Therefore the engage frame itself emits exactly the same
        pose, and later hand motion continues from that pose without requiring
        an out-of-band ``SetReference`` RPC.
        """
        self._segment_base_pos = self._published_pos.copy()
        self._segment_base_rot = self._published_rot.copy()
        self._t_begin_pos = tracker[0].copy()
        self._t_begin_rot = tracker[1].copy()
        self._pending_relatch = False

    def _compute_action(self) -> dict[str, float]:
        """Read hardware and publish the validated latch-once pose offset."""
        with self._lock:
            tracker = self._runtime_tracker_pose()
            now = time.time()

            # --- Clutch edge (Pika quick gripper squeeze, #10) --------------
            # Official /teleop_trigger semantics: any Command change toggles
            # follow / hold.  On the engage edge the publish stays frozen
            # until SetReference lands — the client sequences follower
            # SetReference → leader SetReference so the follower never applies
            # a zero offset against a stale reference (no crawl back to
            # Connect home).  On the disengage edge the freeze is immediate:
            # the follower keeps receiving the last offset and holds.
            try:
                command_now = int(self._command_state_provider())
            except Exception:
                command_now = None
            if command_state_edge(command_now, self._prev_command_state):
                confirmation_accepted = False
                if self._reference_required:
                    if (
                        self._tracker_recovery_ready
                        and self._last_valid_tracker_sample is not None
                    ):
                        tracker = (
                            self._last_valid_tracker_sample.position,
                            self._last_valid_tracker_sample.rotation,
                        )
                        if self._auto_reference:
                            # Direct lerobot-teleoperate has no Collection
                            # coordinator, so cumulative-clutch mode performs
                            # its local jump-free relatch as before.
                            self._reference_required = False
                            self._reference_fault_kind = None
                            self._tracker_recovery_ready = False
                            self._tracker_recovery_samples.clear()
                            self._clutched = True
                            self._pending_relatch = True
                            logger.info(
                                "TRACKER HEALTH: recovery confirmed; "
                                "re-referencing locally."
                            )
                        else:
                            # Collection mode owns the cross-endpoint commit.
                            # Publish an explicit state and stay disengaged
                            # until follower -> leader references succeed.
                            self._reference_confirmation_pending = True
                            self._clutched = False
                            self._pending_relatch = True
                            confirmation_accepted = True
                            logger.info(
                                "TRACKER HEALTH: recovery confirmation accepted; "
                                "awaiting Collection reference commit."
                            )
                    else:
                        self._clutched = False
                        logger.warning(
                            "TRACKER HEALTH: confirmation ignored until fresh "
                            "poses are stable."
                        )
                else:
                    self._clutched = not self._clutched
                    if self._clutched:
                        self._pending_relatch = True
                if confirmation_accepted:
                    pass
                elif self._clutched:
                    self._pending_relatch = True
                    logger.info(
                        "CLUTCH: engaged — pending relatch until SetReference "
                        "(publish frozen at %.1fmm).",
                        float(np.linalg.norm(self._published_pos)) * 1000.0,
                    )
                elif not self._clutched:
                    logger.info(
                        "CLUTCH: disengaged — publish frozen at %.1fmm, arm holds.",
                        float(np.linalg.norm(self._published_pos)) * 1000.0,
                    )
            if command_now is not None:
                self._prev_command_state = command_now

            if (
                self._cumulative_clutch
                and self._clutched
                and self._pending_relatch
                and tracker is not None
            ):
                self._relatch_cumulative_segment(tracker)
                logger.info(
                    "CLUTCH: cumulative relatch complete — target preserved."
                )

            # --- Validated 1:1 publish ---------------------------------------
            # While disengaged / pending relatch the published offset stays
            # frozen at its last value.  Accepted poses remain unscaled and
            # unsmoothed; follower-side geometry and IK safety remain intact.
            if (
                self._clutched
                and not self._pending_relatch
                and tracker is not None
                and self._t_begin_pos is not None
                and self._t_begin_rot is not None
            ):
                pos_now, rot_now = tracker
                desired_pos = compute_position_delta_body(
                    pos_now, self._t_begin_pos, self._t_begin_rot,
                )
                desired_rot = self._t_begin_rot.T @ rot_now
                if self._cumulative_clutch:
                    desired_pos = (
                        self._segment_base_pos
                        + self._segment_base_rot @ desired_pos
                    )
                    desired_rot = self._segment_base_rot @ desired_rot
                if np.isfinite(desired_pos).all():
                    self._published_pos = desired_pos
                if np.isfinite(desired_rot).all():
                    self._published_rot = desired_rot

            delta_pos = self._published_pos.copy()
            delta_rotvec = Rot.from_matrix(self._published_rot).as_rotvec()
            if self._t_begin_pos is None:
                delta_pos = np.zeros(3)
                delta_rotvec = np.zeros(3)

            # --- Gripper distance (calibrated range rescale, raw otherwise) --
            try:
                raw_grip = float(self._device.get_gripper_distance())
                grip_range = self._gripper_max_mm - self._gripper_min_mm
                if grip_range > 0.001:
                    # Rescale Pika [min, max] → [0, 60] mm so the follower's
                    # existing gripper_max_distance_mm mapping works correctly.
                    gripper_distance = (raw_grip - self._gripper_min_mm) / grip_range * 60.0
                    gripper_distance = max(0.0, min(60.0, gripper_distance))
                else:
                    gripper_distance = raw_grip
                if not np.isfinite(gripper_distance):
                    raise ValueError("non-finite gripper distance")
                self._published_gripper_distance_mm = float(gripper_distance)
                gripper_valid = True
            except Exception as exc:
                gripper_distance = self._published_gripper_distance_mm
                gripper_valid = False
                logger.warning(
                    "GRIPPER: invalid reading (%s); holding %.1fmm.",
                    exc,
                    gripper_distance,
                )

            # --- Diagnostic logging (1 Hz) ---
            if now - getattr(self, "_last_debug_ts", 0.0) > 1.0:
                self._last_debug_ts = now
                if tracker is None:
                    if (
                        self._reference_required
                        and self._tracker_recovery_ready
                    ):
                        held_reason = (
                            "stable pose awaiting operator reference confirmation"
                        )
                    elif (
                        self._reference_fault_kind
                        == self._REFERENCE_FAULT_POSE_DISCONTINUITY
                    ):
                        held_reason = (
                            "fused pose discontinuity quarantined; optical stream "
                            "remains under observation"
                        )
                    else:
                        held_reason = (
                            self._last_tracker_health_reason or "pose unavailable"
                        )
                    logger.warning(
                        "TRACKER: output held — %s reference_required=%s.",
                        held_reason,
                        self._reference_required,
                    )
                else:
                    off_m = (
                        0.0
                        if self._t_begin_pos is None
                        else float(np.linalg.norm(tracker[0] - self._t_begin_pos))
                    )
                    logger.info(
                        "TRACKER: pos=[%.3f,%.3f,%.3f]m off=%.1fmm pub=%.1fmm "
                        "grip=%.1fmm optical_events=%d channels=%d confidence=%s",
                        tracker[0][0], tracker[0][1], tracker[0][2],
                        off_m * 1000.0,
                        float(np.linalg.norm(self._published_pos)) * 1000.0,
                        gripper_distance,
                        self._last_valid_tracker_sample.optical_measurement_count,
                        self._last_valid_tracker_sample.optical_lighthouse_count,
                        (
                            "n/a"
                            if self._last_valid_tracker_sample.pose_confidence is None
                            else f"{self._last_valid_tracker_sample.pose_confidence:.2f}"
                        ),
                    )

            # --- Encode ---
            delta_quat = Rot.from_rotvec(delta_rotvec).as_quat()  # [x,y,z,w]
            generic_action = {
                "hand.delta_pos.x": float(delta_pos[0]),
                "hand.delta_pos.y": float(delta_pos[1]),
                "hand.delta_pos.z": float(delta_pos[2]),
                "hand.delta_rot.qx": float(delta_quat[0]),
                "hand.delta_rot.qy": float(delta_quat[1]),
                "hand.delta_rot.qz": float(delta_quat[2]),
                "hand.delta_rot.qw": float(delta_quat[3]),
                "gripper.distance": gripper_distance,
            }
            if self._reference_confirmation_pending:
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_REFERENCE_PENDING
                )
            elif (
                self._reference_required
                and self._reference_fault_kind
                == self._REFERENCE_FAULT_POSE_DISCONTINUITY
                and self._tracker_recovery_ready
            ):
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_POSE_CONFIRM_REQUIRED
                )
            elif (
                self._reference_required
                and self._reference_fault_kind
                == self._REFERENCE_FAULT_POSE_DISCONTINUITY
            ):
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_POSE_DISCONTINUITY
                )
            elif self._reference_required and self._tracker_recovery_ready:
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_CONFIRM_REQUIRED
                )
            elif self._reference_required and (
                self._raw_optical_reacquiring or self._tracker_recovery_samples
            ):
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_RECOVERING
                )
            elif self._reference_required:
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_LOST
                )
            elif tracker is None:
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_TRANSIENT_LOSS
                )
            elif not self._clutched or self._pending_relatch:
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_HELD
                )
            else:
                self._last_tracking_state = (
                    device_pb2.TrackingState.TRACKING_STATE_READY
                )

            if self._last_tracking_state in (
                device_pb2.TrackingState.TRACKING_STATE_LOST,
                device_pb2.TrackingState.TRACKING_STATE_RECOVERING,
                device_pb2.TrackingState.TRACKING_STATE_POSE_DISCONTINUITY,
                device_pb2.TrackingState.TRACKING_STATE_POSE_CONFIRM_REQUIRED,
            ):
                self._last_action_quality = (
                    device_pb2.FrameQuality.FRAME_QUALITY_STALE
                )
            elif (
                self._last_tracking_state
                != device_pb2.TrackingState.TRACKING_STATE_READY
                or not gripper_valid
            ):
                self._last_action_quality = (
                    device_pb2.FrameQuality.FRAME_QUALITY_DEGRADED
                )
            else:
                self._last_action_quality = (
                    device_pb2.FrameQuality.FRAME_QUALITY_GOOD
                )
            return {
                wire_key: generic_action[generic_key]
                for generic_key, wire_key in zip(
                    ACTION_KEYS, action_keys(self._arm_prefix), strict=True
                )
            }

    # ------------------------------------------------------------------
    # LeaderServicer RPCs
    # ------------------------------------------------------------------

    def GetInfo(self, request, context):
        ft = self._act_ft_info
        return device_pb2.GetInfoResponse(
            observation_features=[],  # leader has no observations
            action_features=list(ft.values()),
            feedback_features=[],  # prototype — no haptic feedback yet
        )

    def Connect(self, request, context):
        with self._lock:
            if self._hardware_started:
                # Hardware already alive from a previous client session.
                # Reset per-session state but keep the tracker running —
                # pysurvive/libsurvive context cannot be recreated after
                # destruction, so we never disconnect the device between
                # client sessions.
                self._reset_latch()
            else:
                self._device.connect()
                self._device.get_vive_tracker()
                self._hardware_started = True
                self._tracker_device = None
            self._reset_latch()
            self._clutched = False
            self._pending_relatch = False
            self._prev_command_state = None
            ready_sample = self._await_tracker_ready(context)
            if self._auto_reference:
                self._auto_latch(
                    (ready_sample.position, ready_sample.rotation)
                )
            self._connected = True
            if not self._calibrated:
                self._calib_step = 0
                self._calib_data = {}
            calibrated = self._calibrated
        # Return calibration status — the client auto-triggers Calibrate
        # RPC if NEED_TO_CALIBRATE (via connect(calibrate=True)).
        if calibrated:
            return device_pb2.CalibrationInfo(
                status=device_pb2.CalibrationStatus.CALIBRATED
            )
        return device_pb2.CalibrationInfo(
            status=device_pb2.CalibrationStatus.NEED_TO_CALIBRATE
        )

    def Calibrate(self, request, context):
        # Wait for any in-progress step thread to finish (CalibrateDone has
        # already set the event; the thread captures one data point and exits).
        if self._calibrate_thread is not None and self._calibrate_thread.is_alive():
            self._calibrate_thread.join(timeout=3.0)

        if self._calibrated and not request.force:
            return device_pb2.CalibrationInfo(
                status=device_pb2.CalibrationStatus.CALIBRATED
            )
        if request.force:
            self._calib_step = 0
            self._calib_data = {}
            self._calibrated = False
        if self._calib_step >= len(self._CALIB_STEPS):
            return device_pb2.CalibrationInfo(
                status=device_pb2.CalibrationStatus.CALIBRATED
            )

        # Start the current step's streaming thread.  Returning
        # NEED_TO_CALIBRATE triggers the client's _calibrate_once() which
        # opens StreamCalibration, displays live data, and waits for Enter.
        self._calibrating = True
        self._calibrate_error = False
        self._calibrate_done.clear()
        self._calib_frame_queue = queue.Queue()
        self._calibrate_thread = threading.Thread(
            target=self._run_calib_step, args=(self._calib_step,), daemon=True
        )
        self._calibrate_thread.start()
        return device_pb2.CalibrationInfo(
            status=device_pb2.CalibrationStatus.NEED_TO_CALIBRATE
        )

    def CalibrateDone(self, request, context):
        self._calibrate_done.set()
        return Empty()

    def StreamCalibration(self, request, context):
        q = self._calib_frame_queue
        if q is None:
            return
        while context.is_active():
            try:
                frame = q.get(timeout=1.0)
            except queue.Empty:
                continue
            if frame is None:
                break
            yield frame

    def Disconnect(self, request, context):
        # Unblock any running calibration step thread.
        self._calibrate_done.set()
        self._connected = False
        # Keep hardware alive — pysurvive/libsurvive context cannot be
        # recreated after destruction.  The next Connect() call reuses
        # the existing tracker session and resets per-session state.
        logger.info("Client disconnected; hardware kept alive for reuse.")
        return Empty()

    def GetStatus(self, request, context):
        # COLLECTION = clutched (following), IDLE = clutch off (holding).
        # The teleop client polls this to detect the engage/disengage edge
        # (#10) — it is the leader→client channel for the quick-squeeze state.
        if self._calibrate_error:
            return device_pb2.DeviceInfo(
                status=device_pb2.DeviceStatus.FATAL
            )
        if self._clutched:
            return device_pb2.DeviceInfo(
                status=device_pb2.DeviceStatus.COLLECTION
            )
        return device_pb2.DeviceInfo(
            status=device_pb2.DeviceStatus.IDLE
        )

    # --- alignment ---

    def _auto_latch(self, tracker) -> None:
        """Latch ``T_begin`` at the given raw tracker pose and engage.

        ``auto_reference`` mode's counterpart of ``SetReference``: the same
        state transitions minus the RPC.  Callers hold ``_lock`` and pass the
        current ``_read_tracker_pose()`` result.
        """
        self._t_begin_pos = tracker[0].copy()
        self._t_begin_rot = tracker[1].copy()
        self._published_pos = np.zeros(3)
        self._published_rot = np.eye(3)
        self._segment_base_pos = np.zeros(3)
        self._segment_base_rot = np.eye(3)
        self._clutched = True
        self._pending_relatch = False
        logger.info(
            "AUTO-REFERENCE: T_begin latched at current pose — teleop engaged."
        )

    def SetReference(self, request, context):
        """Lock T_begin at the current tracker pose, zero the offset, engage.

        Two call sites: the initial Enter alignment and every clutch
        re-engage (#10).  The client calls the follower's ``SetReference``
        *first* on a re-engage so both sides relatch their base pose before
        any fresh offset is applied — current hand = current arm.
        """
        with self._lock:
            sample = self._await_reference_sample(context)
            if sample is None:
                detail = self._last_reference_failure_reason or "unknown reason"
                message = (
                    "SetReference rejected: no fresh stable tracker pose "
                    f"({detail})"
                )
                logger.warning(message)
                # A failed Collection commit is recoverable. Keep the robot
                # side held and return to optical convergence rather than
                # leaving a stale confirmation request latched forever.
                self._reference_confirmation_pending = False
                self._reference_required = True
                self._clutched = False
                self._pending_relatch = True
                self._tracker_recovery_samples.clear()
                self._tracker_recovery_ready = False
                if context is not None and hasattr(context, "abort"):
                    import grpc

                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
                raise RuntimeError(message)
            tracker = (sample.position, sample.rotation)
            # SetReference may spend several seconds collecting distinct,
            # stable optical samples.  A quick-squeeze Command transition can
            # complete during that wait while GetAction is blocked on this
            # lock.  Consume the state that exists at the commit boundary so
            # the first identity sample cannot replay that old transition and
            # immediately disengage the freshly latched reference.
            try:
                command_now = int(self._command_state_provider())
            except Exception:
                command_now = None
            if command_now is not None:
                self._prev_command_state = command_now
            self._prime_runtime_tracker(sample)
            self._t_begin_pos = tracker[0].copy()
            self._t_begin_rot = tracker[1].copy()
            logger.info("Reference set: pos=%s", np.round(tracker[0], 4))
            self._published_pos = np.zeros(3)
            self._published_rot = np.eye(3)
            self._segment_base_pos = np.zeros(3)
            self._segment_base_rot = np.eye(3)
            # Engage / re-engage: unlock the frozen publish.  On a re-engage
            # this clears the pending state set by the command-state edge.
            self._clutched = True
            self._pending_relatch = False
        return Empty()

    # --- data flow ---

    def GetAction(self, request, context):
        """One snapshot of the 8-FLOAT32 pose-delta action per call."""
        action = self._compute_action()
        for feature in encode_feature(self._act_ft_info, action):
            feature.quality = self._last_action_quality
            feature.tracking_state = self._last_tracking_state
            yield feature

    def GetObservation(self, request, context):
        """Leader has no observations — empty stream (protocol requires the method)."""
        return
        yield  # make it a generator

    def SendFeedback(self, request_iterator, context):
        """Consume the feedback stream and optionally drive haptic output.

        The follower mirrors its action state back as feedback features.
        Left as a no-op for the prototype — enable light / vibration when
        hardware is ready.
        """
        for _ in request_iterator:
            pass
        return Empty()
