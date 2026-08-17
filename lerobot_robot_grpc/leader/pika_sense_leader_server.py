"""Pika Sense leader servicer for delta-pose teleoperation.

Wraps the Pika Sense device (Vive Tracker + gripper sensor) as a
:class:`LeaderServicer` that produces pose-delta actions.  After
:meth:`SetReference` the servicer publishes the **current offset** from
that latch (world-frame translation + body-frame rotation), not a
per-frame velocity.  The follower assigns ``T_intent = T_zero ⊕ ΔT``.

Coordinate-frame convention (wayfinder #05):

- **Translation offset** — lighthouse ``pos_now − T_begin``, rotated to the
  robot base frame via ``R_lh2base``.
- **Rotation offset** — ``R_delta = R_begin^T @ R_now`` (body-frame),
  matching the follower's ``R_zero @ R_delta``.
- **SetReference** — latch-once: locks ``T_begin`` and zeros the published
  offset.  Re-calling it rebases the Pika origin (the arm then seeks
  ``T_zero`` as the offset returns to identity).

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
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf.empty_pb2 import Empty

from lerobot.utils.rotation import Rotation as Rot

from lerobot_robot_grpc.leader.leader_server import LeaderServicer
from lerobot_robot_grpc.leader.utils import encode_feature
from lerobot_robot_grpc.pose_delta_schema import build_pose_delta_feature_info
from lerobot_robot_grpc.protos import device_pb2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure delta-computation helpers (testable without Pika hardware)
# ---------------------------------------------------------------------------


def compute_position_delta(
    pos_now: np.ndarray,
    pos_ref: np.ndarray,
    R_lh2base: np.ndarray,
) -> np.ndarray:
    """World-frame translation delta, rotated to robot base frame.

    Parameters
    ----------
    pos_now
        Current tracker position in the lighthouse world frame (metres).
    pos_ref
        Reference (or previous-frame) position in the lighthouse frame.
    R_lh2base
        3×3 rotation matrix mapping lighthouse vectors to robot-base vectors.
    """
    return R_lh2base @ (pos_now - pos_ref)


def compute_rotation_delta_rotvec(
    rot_now: np.ndarray,
    rot_ref: np.ndarray,
    R_lh2base: np.ndarray,  # unused — body-frame rotvec needs no remapping
) -> np.ndarray:
    """Body-frame rotation delta as a rotvec.

    ``R_delta = R_ref^T @ R_now`` (body-frame composition).  The rotvec
    is returned directly in the body frame — it matches the follower's
    body-frame composition (``R_target @ R_delta``) without any world-frame
    remapping, because both tracker and end-effector body frames are
    aligned when the user holds the tracker in a natural forward grip.

    Parameters
    ----------
    rot_now, rot_ref
        3×3 rotation matrices (tracker orientation in the lighthouse frame).
    R_lh2base
        Unused — kept for API compatibility.  The rotvec is in the body
        frame, not the lighthouse frame, so applying ``R_lh2base`` would
        scramble the rotation axis.
    """
    r_delta = rot_ref.T @ rot_now
    return Rot.from_matrix(r_delta).as_rotvec()


def apply_dead_zone(delta: np.ndarray, threshold: float) -> np.ndarray:
    """Zero *delta* if its Euclidean norm is below *threshold*."""
    if np.linalg.norm(delta) < threshold:
        return np.zeros_like(delta)
    return delta


def apply_soft_dead_zone(delta: np.ndarray, threshold: float) -> np.ndarray:
    """Zero below *threshold*; linearly ramp from *threshold* to ``2 * threshold``.

    A hard dead-zone is stick-slip: 1.9 mm is discarded, 2.1 mm is released in
    full.  The ramp makes the output continuous at the threshold so tracker
    noise just above the floor does not become a step.
    """
    if threshold <= 0.0:
        return delta
    norm = float(np.linalg.norm(delta))
    if norm < threshold:
        return np.zeros_like(delta)
    if norm < 2.0 * threshold:
        scale = (norm - threshold) / threshold
        return delta * scale
    return delta


def apply_ema(raw: np.ndarray, filtered: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average update.

    ``filtered_new = alpha * raw + (1 - alpha) * filtered``.
    """
    return alpha * raw + (1.0 - alpha) * filtered


def apply_rate_limit(value: float, prev: float, max_step: float) -> float:
    """Clamp ``value - prev`` to ``[-max_step, max_step]``."""
    if max_step <= 0.0:
        return value
    delta = value - prev
    if delta > max_step:
        return prev + max_step
    if delta < -max_step:
        return prev - max_step
    return value


def slerp_rotation(rot_a: np.ndarray, rot_b: np.ndarray, alpha: float) -> np.ndarray:
    """Slerp two 3×3 rotation matrices.  ``alpha=0`` → *rot_a*, ``alpha=1`` → *rot_b*."""
    r_a = Rot.from_matrix(rot_a)
    r_b = Rot.from_matrix(rot_b)
    r_delta = r_a.inv() * r_b
    return (r_a * Rot.from_rotvec(alpha * r_delta.as_rotvec())).as_matrix()


def clamp_vector(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale *vec* down so its Euclidean norm does not exceed *max_norm*."""
    if max_norm <= 0.0:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm:
        return vec
    return vec * (max_norm / norm)


def adaptive_ema_alpha(
    speed_m: float,
    alpha_still: float = 0.15,
    alpha_fast: float = 0.70,
    fast_speed_m: float = 0.008,
) -> float:
    """Lerp EMA alpha from *alpha_still* to *alpha_fast* by raw pose speed."""
    if fast_speed_m <= 0.0:
        return float(alpha_still)
    t = float(np.clip(speed_m / fast_speed_m, 0.0, 1.0))
    return float(alpha_still + t * (alpha_fast - alpha_still))


def is_pose_jump(
    pos_now: np.ndarray,
    pos_prev: np.ndarray | None,
    jump_m: float,
) -> bool:
    """True when the tracker teleported farther than *jump_m* in one sample."""
    if pos_prev is None or jump_m <= 0.0:
        return False
    return float(np.linalg.norm(pos_now - pos_prev)) > jump_m


def consume_tracker_jump(
    pos_raw: np.ndarray,
    last_raw: np.ndarray | None,
    t_begin: np.ndarray | None,
    jump_m: float,
) -> tuple[bool, np.ndarray, np.ndarray | None, np.ndarray]:
    """Compare consecutive raw samples; always advance *last_raw*.

    A jump rebases *t_begin* by the same world-frame step so the published
    offset stays continuous (lighthouse re-solve must not look like a 28 cm
    hand motion).  Returns ``(jumped, new_last_raw, new_t_begin, filt_shift)``.
    """
    pos_raw = np.asarray(pos_raw, dtype=float)
    if last_raw is None:
        return False, pos_raw.copy(), t_begin, np.zeros(3)
    step = pos_raw - np.asarray(last_raw, dtype=float)
    if jump_m <= 0.0 or float(np.linalg.norm(step)) <= jump_m:
        return False, pos_raw.copy(), t_begin, np.zeros(3)
    new_begin = None if t_begin is None else np.asarray(t_begin, dtype=float) + step
    return True, pos_raw.copy(), new_begin, step


def slew_vector(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move *current* toward *target* by at most *max_step* (metres)."""
    if max_step <= 0.0:
        return np.asarray(target, dtype=float).copy()
    return np.asarray(current, dtype=float) + clamp_vector(
        np.asarray(target, dtype=float) - np.asarray(current, dtype=float),
        max_step,
    )


def command_state_edge(command_now: int | None, command_prev: int | None) -> bool:
    """True when the Pika button's command state changed since the last sample.

    ``None`` means "no reading" and is never an edge.  A 0→1 **or** 1→0 change
    toggles the clutch (wayfinder #10) — the official ``/teleop_trigger``
    semantics: serial ``Command`` change → toggle follow/hold.
    """
    if command_now is None or command_prev is None:
        return False
    return int(command_now) != int(command_prev)


def publish_latch_offset(
    desired: np.ndarray,
    published: np.ndarray,
    *,
    dead_zone_m: float,
    max_delta_m: float,
    home_capture_m: float,
) -> np.ndarray:
    """Slew the published offset toward *desired*.

    Far from the origin a sub-threshold error is held (noise).  Inside
    *home_capture_m* the dead-zone is disabled so a return to ``T_begin``
    can actually reach zero.
    """
    desired = np.asarray(desired, dtype=float)
    published = np.asarray(published, dtype=float)
    error = desired - published
    if (
        float(np.linalg.norm(desired)) > home_capture_m
        and float(np.linalg.norm(error)) < dead_zone_m
    ):
        return published.copy()
    return slew_vector(published, desired, max_delta_m)


def publish_latch_rotation(
    desired_rot: np.ndarray,
    published_rot: np.ndarray,
    *,
    dead_zone_rad: float,
    max_delta_rad: float,
    home_capture_m: float,
    desired_pos_norm: float,
) -> np.ndarray:
    """Slew the published rotation toward *desired_rot* (body-frame ΔR)."""
    r_err = published_rot.T @ desired_rot
    err_vec = Rot.from_matrix(r_err).as_rotvec()
    err_n = float(np.linalg.norm(err_vec))
    if desired_pos_norm > home_capture_m and err_n < dead_zone_rad:
        return published_rot.copy()
    if max_delta_rad > 0.0 and err_n > max_delta_rad:
        err_vec = err_vec * (max_delta_rad / err_n)
    return published_rot @ Rot.from_rotvec(err_vec).as_matrix()


def filter_pose(
    pos_now: np.ndarray,
    rot_now: np.ndarray,
    filt_pos: np.ndarray | None,
    filt_rot: np.ndarray | None,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """EMA-filter a tracker pose.  First sample initialises the filter.

    Filter the pose *before* taking a per-frame delta — EMA on the already
    differenced signal leaves a lagging tail after the hand stops.
    ``alpha >= 1`` is a passthrough.
    """
    if filt_pos is None or filt_rot is None or alpha >= 1.0:
        return pos_now.copy(), rot_now.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return apply_ema(pos_now, filt_pos, alpha), slerp_rotation(filt_rot, rot_now, alpha)


def build_R_lh2base(forward_lh: np.ndarray, up_lh: np.ndarray) -> np.ndarray:
    """Build the lighthouse→base rotation from measured axis directions.

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
    2. Computes the current offset from ``T_begin`` (SetReference).
    3. Rotates the position offset from the lighthouse frame to the robot
       base frame via ``R_lh2base`` (#05 direction-calibration result).
    4. Applies pose EMA, jump rejection, and a slew-limited publish so
       tracker noise does not become stick-slip steps — near the origin
       the dead-zone is disabled so a return to ``T_begin`` reaches zero.
    5. Encodes the 8 FLOAT32 pose-delta action features and streams them.

    Clutch (#10): a Pika button command-state change toggles follow / hold.
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
    dead_zone_mm
        Position dead-zone threshold in millimetres.  Default ``2.0`` (#06).
        Below this the delta is zero; between this and ``2×`` it is ramped.
    dead_zone_deg
        Rotation dead-zone threshold in degrees.  Default ``0.5``.
        Same soft-ramp rule as the position dead-zone.
    pos_gain
        Multiplies the tracker offset before publish.  Default ``0.45`` so a
        ~20 cm hand move stays inside the SO-101 workspace.  Return-to-pose
        still holds: back at ``T_begin`` the offset is zero regardless of gain.
    ema_alpha
        Pose EMA coefficient applied *before* the per-frame delta.  Default
        ``0.25`` (~1.4 Hz at 30 Hz).  ``1.0`` disables filtering.
    max_delta_mm
        Max change of the *published offset* per frame in millimetres.
        Default ``20``.  Caps how fast the offset may slew, not a discard.
    gripper_ema_alpha
        Independent EMA on gripper distance.  Default matches ``ema_alpha``.
    gripper_rate_mm_s
        Max gripper distance change in mm/s.  Default ``250``.
    R_lh2base
        3×3 rotation matrix mapping lighthouse axes to robot-base axes.
        Defaults to identity (no rotation) — override with the direction-
        calibration result for real hardware.
    command_state_provider
        Callable returning the Pika button command state (0/1) or ``None``.
        Defaults to ``device.get_command_state``; injectable so the clutch
        state machine (#10) is testable without hardware.
    auto_reference
        When True, latch ``T_begin`` and engage on Connect (with a lazy
        fallback on the first action if the solver had not converged by
        then).  For clients without an alignment step — e.g.
        ``lerobot-teleoperate`` — which never call ``SetReference``; without
        a latch such a session publishes zero deltas forever.  Default False
        keeps the #10 contract: the session starts disengaged and the
        client's ``SetReference`` is what engages teleop.  In this mode the
        clutch gesture still freezes the publish, but nobody sends the
        re-engage ``SetReference`` — resume means reconnecting.
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

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        dead_zone_mm: float = 2.0,
        dead_zone_deg: float = 0.5,
        pos_gain: float = 0.45,
        ema_alpha: float = 0.15,
        ema_alpha_fast: float = 0.70,
        max_delta_mm: float = 20.0,
        home_capture_mm: float = 40.0,
        jump_mm: float = 50.0,
        gripper_ema_alpha: float | None = None,
        gripper_rate_mm_s: float = 250.0,
        R_lh2base: np.ndarray | None = None,
        calibration_dir: str | None = None,
        command_state_provider=None,
        auto_reference: bool = False,
        device_id: str = "pika_sense",
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
        self._act_ft_info = build_pose_delta_feature_info()

        # Pipeline parameters
        self._dead_zone_m = dead_zone_mm / 1000.0
        self._dead_zone_rad = np.radians(dead_zone_deg)
        self._pos_gain = pos_gain
        self._ema_alpha = float(ema_alpha)
        self._ema_alpha_fast = float(ema_alpha_fast)
        self._max_delta_m = max_delta_mm / 1000.0
        self._home_capture_m = home_capture_mm / 1000.0
        self._jump_m = jump_mm / 1000.0
        self._max_delta_rad = float(self._dead_zone_rad * 8.0) if self._dead_zone_rad > 0.0 else 0.10
        self._gripper_ema_alpha = (
            self._ema_alpha if gripper_ema_alpha is None else float(gripper_ema_alpha)
        )
        self._gripper_rate_mm_s = float(gripper_rate_mm_s)
        self._R_lh2base = R_lh2base if R_lh2base is not None else np.eye(3)

        # Gripper calibration range (updated by Calibrate RPC).
        self._gripper_min_mm = 0.0
        self._gripper_max_mm = 60.0

        # Connection / tracker state
        self._lock = threading.Lock()
        self._connected = False
        self._tracker_device: str | None = None

        # auto_reference mode (see class docstring): latch + engage without a
        # client SetReference, for clients without an alignment step.
        self._auto_reference = bool(auto_reference)

        # Latch-once state.  T_begin is locked by SetReference; the published
        # offset is the slew-limited current displacement from that latch.
        self._t_begin_pos: np.ndarray | None = None
        self._t_begin_rot: np.ndarray | None = None
        self._published_pos = np.zeros(3)
        self._published_rot = np.eye(3)
        self._last_raw_pos: np.ndarray | None = None
        self._last_jump_log_ts: float = 0.0
        self._filt_position: np.ndarray | None = None
        self._filt_rotation: np.ndarray | None = None
        self._filt_gripper: float | None = None
        self._last_action_ts: float | None = None

        # Clutch state (wayfinder #10).  ``_clutched`` = following; a Pika
        # button command-state *change* toggles it (official /teleop_trigger
        # semantics).  While disengaged — or pending a re-engage relatch — the
        # published offset is frozen at its last value (never identity), so
        # the follower holds the stop pose.  ``_pending_relatch`` is set on
        # the engage edge and cleared by SetReference: the client sequences
        # follower.SetReference() → leader.SetReference(), so the arm never
        # sees a zero offset against a stale T_zero.
        self._clutched = False
        self._pending_relatch = False
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
            "PikaSenseServicer created: port=%s "
            "latch_once=True pos_gain=%.2f dead_zone_mm=%.1f dead_zone_deg=%.1f "
            "ema_alpha=%.2f/%.2f max_delta_mm=%.1f home_capture_mm=%.0f jump_mm=%.0f "
            "calibrated=%s calibration_file=%s",
            port,
            self._pos_gain,
            dead_zone_mm,
            dead_zone_deg,
            self._ema_alpha,
            self._ema_alpha_fast,
            max_delta_mm,
            home_capture_mm,
            jump_mm,
            self._calibrated,
            self._calibration_fpath,
        )

    # ------------------------------------------------------------------
    # Internal: pose reading and delta computation
    # ------------------------------------------------------------------

    def _read_tracker_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Read current tracker pose as (position, rotation_matrix).

        Returns ``None`` if the tracker is not yet producing data.
        """
        if self._tracker_device is None:
            return None
        pose = self._device.get_pose(self._tracker_device)
        if pose is None:
            return None
        pos = np.array(pose.position, dtype=float)  # [x, y, z] metres
        rot = Rot.from_quat(pose.rotation).as_matrix()  # [x,y,z,w] → 3×3
        return pos, rot

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

    def _reset_pose_filter(self) -> None:
        """Clear per-session pose / gripper filter state (Connect / SetReference)."""
        self._filt_position = None
        self._filt_rotation = None
        self._filt_gripper = None
        self._last_action_ts = None
        self._last_raw_pos = None

    def _reset_latch(self) -> None:
        """Drop T_begin and the published offset (Connect)."""
        self._t_begin_pos = None
        self._t_begin_rot = None
        self._published_pos = np.zeros(3)
        self._published_rot = np.eye(3)

    def _compute_action(self) -> dict[str, float]:
        """Read hardware, publish the current latch-once offset, return the action dict."""
        with self._lock:
            tracker = self._read_tracker_pose()
            now = time.time()
            dt = 1.0 / 30.0 if self._last_action_ts is None else max(now - self._last_action_ts, 1e-3)
            self._last_action_ts = now

            # --- Clutch edge (Pika double-click, #10) -----------------------
            # Official /teleop_trigger semantics: any Command change toggles
            # follow / hold.  On the engage edge the publish stays frozen
            # until SetReference lands — the client sequences follower
            # SetReference → leader SetReference so the follower never applies
            # a zero offset against a stale T_zero (no crawl back to Connect
            # home).  On the disengage edge the freeze is immediate: the
            # follower keeps receiving the last offset and holds.
            try:
                command_now = int(self._command_state_provider())
            except Exception:
                command_now = None
            if command_state_edge(command_now, self._prev_command_state):
                self._clutched = not self._clutched
                if self._clutched:
                    self._pending_relatch = True
                    logger.info(
                        "CLUTCH: engaged — pending relatch until SetReference "
                        "(publish frozen at %.1fmm).",
                        float(np.linalg.norm(self._published_pos)) * 1000.0,
                    )
                else:
                    logger.info(
                        "CLUTCH: disengaged — publish frozen at %.1fmm, arm holds.",
                        float(np.linalg.norm(self._published_pos)) * 1000.0,
                    )
            if command_now is not None:
                self._prev_command_state = command_now

            if self._auto_reference and self._t_begin_pos is None and tracker is not None:
                # Lazy fallback: Connect could not latch (solver still
                # converging) — latch at the first tracker sample instead.
                self._auto_latch(tracker)

            jumped = False
            pos_now = self._filt_position if self._filt_position is not None else np.zeros(3)
            rot_now = self._filt_rotation if self._filt_rotation is not None else np.eye(3)

            if tracker is not None:
                pos_raw, rot_raw = tracker
                jumped, self._last_raw_pos, self._t_begin_pos, filt_shift = consume_tracker_jump(
                    pos_raw, self._last_raw_pos, self._t_begin_pos, self._jump_m,
                )
                if jumped:
                    if self._filt_position is not None:
                        self._filt_position = self._filt_position + filt_shift
                    if now - self._last_jump_log_ts > 1.0:
                        self._last_jump_log_ts = now
                        logger.warning(
                            "TRACKER jump rebasing T_begin: Δ=%.0fmm (threshold=%.0fmm).",
                            float(np.linalg.norm(filt_shift)) * 1000.0,
                            self._jump_m * 1000.0,
                        )
                    pos_now = (
                        self._filt_position
                        if self._filt_position is not None
                        else pos_raw
                    )
                    rot_now = (
                        self._filt_rotation
                        if self._filt_rotation is not None
                        else rot_raw
                    )
                else:
                    if self._filt_position is None:
                        alpha = self._ema_alpha
                    else:
                        speed = float(np.linalg.norm(pos_raw - self._filt_position))
                        alpha = adaptive_ema_alpha(
                            speed, self._ema_alpha, self._ema_alpha_fast,
                        )
                    pos_now, rot_now = filter_pose(
                        pos_raw, rot_raw,
                        self._filt_position, self._filt_rotation,
                        alpha,
                    )
                    self._filt_position = pos_now
                    self._filt_rotation = rot_now

            if (
                self._clutched
                and not self._pending_relatch
                and not jumped
                and self._t_begin_pos is not None
                and self._t_begin_rot is not None
                and self._filt_position is not None
            ):
                desired_pos = self._pos_gain * compute_position_delta(
                    pos_now, self._t_begin_pos, self._R_lh2base,
                )
                desired_rot = self._t_begin_rot.T @ rot_now
                if not np.isfinite(desired_pos).all():
                    desired_pos = self._published_pos.copy()
                if not np.isfinite(desired_rot).all():
                    desired_rot = self._published_rot.copy()
                self._published_pos = publish_latch_offset(
                    desired_pos,
                    self._published_pos,
                    dead_zone_m=self._dead_zone_m,
                    max_delta_m=self._max_delta_m,
                    home_capture_m=self._home_capture_m,
                )
                self._published_rot = publish_latch_rotation(
                    desired_rot,
                    self._published_rot,
                    dead_zone_rad=self._dead_zone_rad,
                    max_delta_rad=self._max_delta_rad,
                    home_capture_m=self._home_capture_m,
                    desired_pos_norm=float(np.linalg.norm(desired_pos)),
                )

            delta_pos = self._published_pos.copy()
            delta_rotvec = Rot.from_matrix(self._published_rot).as_rotvec()
            if self._t_begin_pos is None:
                delta_pos = np.zeros(3)
                delta_rotvec = np.zeros(3)

            # --- Gripper distance (mapped via calibrated range, then filtered) ---
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
            except Exception:
                gripper_distance = 0.0

            if self._filt_gripper is None:
                self._filt_gripper = gripper_distance
            else:
                if self._gripper_ema_alpha >= 1.0:
                    candidate = gripper_distance
                else:
                    candidate = (
                        self._gripper_ema_alpha * gripper_distance
                        + (1.0 - self._gripper_ema_alpha) * self._filt_gripper
                    )
                self._filt_gripper = apply_rate_limit(
                    candidate, self._filt_gripper, self._gripper_rate_mm_s * dt
                )
            gripper_distance = float(self._filt_gripper)

            # --- Diagnostic logging (1 Hz) ---
            if now - getattr(self, "_last_debug_ts", 0.0) > 1.0:
                self._last_debug_ts = now
                if tracker is None:
                    logger.warning(
                        "TRACKER: pose is None — solver not converged or tracker occluded."
                    )
                else:
                    off_m = (
                        0.0
                        if self._t_begin_pos is None
                        else float(np.linalg.norm(pos_now - self._t_begin_pos))
                    )
                    logger.info(
                        "TRACKER: pos=[%.3f,%.3f,%.3f]m off=%.1fmm pub=%.1fmm grip=%.1fmm",
                        pos_now[0], pos_now[1], pos_now[2],
                        off_m * 1000.0,
                        float(np.linalg.norm(self._published_pos)) * 1000.0,
                        gripper_distance,
                    )

            # --- Encode ---
            delta_quat = Rot.from_rotvec(delta_rotvec).as_quat()  # [x,y,z,w]
            return {
                "hand.delta_pos.x": float(delta_pos[0]),
                "hand.delta_pos.y": float(delta_pos[1]),
                "hand.delta_pos.z": float(delta_pos[2]),
                "hand.delta_rot.qx": float(delta_quat[0]),
                "hand.delta_rot.qy": float(delta_quat[1]),
                "hand.delta_rot.qz": float(delta_quat[2]),
                "hand.delta_rot.qw": float(delta_quat[3]),
                "gripper.distance": gripper_distance,
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
            if self._connected:
                # Hardware already alive from a previous client session.
                # Reset per-session state but keep the tracker running —
                # pysurvive/libsurvive context cannot be recreated after
                # destruction, so we never disconnect the device between
                # client sessions.
                self._reset_pose_filter()
                self._reset_latch()
            else:
                self._device.connect()
                # Initialise the Vive Tracker and wait for device discovery.
                self._device.get_vive_tracker()
                # Lighthouses (LH*) are discovered immediately, but the actual
                # tracker (e.g. "T20") takes ~2–5 s.  Poll until a non-LH device
                # appears — reading a lighthouse pose gives a stationary position
                # (delta永远=0), which silently breaks calibration.
                self._tracker_device = None
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    devices = self._device.get_tracker_devices()
                    trackers = [d for d in devices if not d.startswith("LH")]
                    if trackers:
                        self._tracker_device = trackers[0]
                        break
                    time.sleep(0.5)
                if self._tracker_device is None:
                    logger.warning(
                        "No tracker found (only lighthouses LH* discovered after 10s)."
                    )
                else:
                    logger.info("Vive Tracker discovered: %s", self._tracker_device)
                    # Wait for the libsurvive solver to converge before
                    # accepting connections — get_pose() returns None until
                    # MPFIT has enough optical data (30-90s depending on
                    # data quality).  Without this, deltas are zero until
                    # the solver finishes.
                    logger.info("Waiting for tracker pose data (solver converging)...")
                    pose_deadline = time.time() + 60.0
                    while time.time() < pose_deadline:
                        pose = self._device.get_pose(self._tracker_device)
                        if pose is not None:
                            logger.info("Tracker poses available (solver converged).")
                            break
                        time.sleep(1.0)
                    else:
                        logger.warning(
                            "No pose data after 60s — solver may not have "
                            "converged. Deltas will be zero until poses appear."
                        )
            self._connected = True
            self._reset_latch()
            # New client session starts with the clutch disengaged (#10):
            # the Enter alignment (SetReference) is what engages teleop.
            self._clutched = False
            self._pending_relatch = False
            self._prev_command_state = None
            if self._auto_reference:
                # Clients without an alignment step never call SetReference;
                # latch at the Connect pose so deltas flow from session start.
                # If the solver had not converged yet, _compute_action retries.
                tracker = self._read_tracker_pose()
                if tracker is not None:
                    self._auto_latch(tracker)
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
        # Keep hardware alive — pysurvive/libsurvive context cannot be
        # recreated after destruction.  The next Connect() call reuses
        # the existing tracker session and resets per-session state.
        logger.info("Client disconnected; hardware kept alive for reuse.")
        return Empty()

    def GetStatus(self, request, context):
        # COLLECTION = clutched (following), IDLE = clutch off (holding).
        # The teleop client polls this to detect the engage/disengage edge
        # (#10) — it is the leader→client channel for the button state.
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
        self._reset_pose_filter()
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
            tracker = self._read_tracker_pose()
            if tracker is not None:
                self._t_begin_pos = tracker[0].copy()
                self._t_begin_rot = tracker[1].copy()
                logger.info("Reference set: pos=%s", np.round(tracker[0], 4))
            else:
                self._t_begin_pos = None
                self._t_begin_rot = None
                logger.warning("SetReference called but no tracker data available.")
            self._published_pos = np.zeros(3)
            self._published_rot = np.eye(3)
            self._reset_pose_filter()
            # Engage / re-engage: unlock the frozen publish.  On a re-engage
            # this clears the pending state set by the command-state edge.
            self._clutched = True
            self._pending_relatch = False
        return Empty()

    # --- data flow ---

    def GetAction(self, request, context):
        """One snapshot of the 8-FLOAT32 pose-delta action per call."""
        action = self._compute_action()
        return encode_feature(self._act_ft_info, action)

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
