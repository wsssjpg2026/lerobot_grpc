"""Pika Sense leader servicer for delta-pose teleoperation.

Wraps the Pika Sense device (Vive Tracker + gripper sensor) as a
:class:`LeaderServicer` that produces pose-delta actions.  The servicer reads
the tracker's 6-DoF pose from the lighthouse frame, computes per-frame deltas
(world-frame translation + body-frame rotation per wayfinder #05), applies
EMA smoothing and dead-zone (#06), and streams the 8-FLOAT32 pose-delta
action defined by :mod:`pose_delta_schema`.

Coordinate-frame convention (wayfinder #05):

- **Translation delta** — computed in the lighthouse world frame, then rotated
  to the robot base frame via ``R_lh2base`` (determined by direction
  calibration at startup).
- **Rotation delta** — ``R_delta = R_ref^T @ R_now`` (body-frame composition,
  matching the follower's ``R_current @ R_delta``); the rotvec is rotated by
  ``R_lh2base`` (equivalent to the conjugation
  ``R_lh2base @ R_delta @ R_lh2base^T``).
- **SetReference** — supports *latch-once* (default, position control) and
  *continuous* (per-frame velocity control) modes.

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


def apply_ema(raw: np.ndarray, filtered: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average update.

    ``filtered_new = alpha * raw + (1 - alpha) * filtered``.
    """
    return alpha * raw + (1.0 - alpha) * filtered


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
    2. Computes the per-frame pose delta (current − previous frame).
    3. Rotates the position delta from the lighthouse frame to the robot
       base frame via ``R_lh2base`` (#05 direction-calibration result).
    4. Applies glitch rejection (>25 mm → zero) and a dead-zone (#06) to
       filter tracking noise.  No EMA — the follower's own ctrl EMA (50 Hz)
       provides physical smoothing.
    5. Encodes the 8 FLOAT32 pose-delta action features and streams them.

    Parameters
    ----------
    port
        Serial port path for the Pika Sense device.
    dead_zone_mm
        Position dead-zone threshold in millimetres.  Default ``2.0`` (#06).
        Per-frame deltas below this are zeroed to filter tracking noise.
    dead_zone_deg
        Rotation dead-zone threshold in degrees.  Default ``0.5``.
        Per-frame rotation deltas below this are zeroed.
    pos_gain
        Position-delta gain factor — multiplies the tracker displacement before
        sending.  Default ``1.0`` (1:1 mapping); tune with real hardware to
        match the tracker range to the SO-101 workspace (~251 mm reach).
    R_lh2base
        3×3 rotation matrix mapping lighthouse axes to robot-base axes.
        Defaults to identity (no rotation) — override with the direction-
        calibration result for real hardware.
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
        pos_gain: float = 1.0,
        R_lh2base: np.ndarray | None = None,
        calibration_dir: str | None = None,
        device_id: str = "pika_sense",
    ):
        # Deferred import — the Pika SDK is an optional dependency.
        # Importing here means the module loads cleanly on machines without
        # the SDK; only construction fails.
        from pika.sense import Sense

        self._device = Sense(port)

        # Feature schema — shared with the follower's pose_delta mode.
        self._act_ft_info = build_pose_delta_feature_info()

        # Pipeline parameters
        self._dead_zone_m = dead_zone_mm / 1000.0
        self._dead_zone_rad = np.radians(dead_zone_deg)
        self._pos_gain = pos_gain
        self._R_lh2base = R_lh2base if R_lh2base is not None else np.eye(3)

        # Gripper calibration range (updated by Calibrate RPC).
        self._gripper_min_mm = 0.0
        self._gripper_max_mm = 60.0

        # Connection / tracker state
        self._lock = threading.Lock()
        self._connected = False
        self._tracker_device: str | None = None

        # Per-frame previous pose (the delta reference)
        self._prev_position: np.ndarray | None = None
        self._prev_rotation: np.ndarray | None = None

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
            "dead_zone_mm=%.1f dead_zone_deg=%.1f calibrated=%s calibration_file=%s",
            port,
            dead_zone_mm,
            dead_zone_deg,
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

    def _compute_action(self) -> dict[str, float]:
        """Read hardware, run the full delta pipeline, return the action dict."""
        with self._lock:
            tracker = self._read_tracker_pose()

            # --- Delta computation (per-frame delta — prevents random walk) ---
            if tracker is not None:
                pos_now, rot_now = tracker
                if self._prev_position is not None:
                    delta_pos = compute_position_delta(pos_now, self._prev_position, self._R_lh2base)
                    delta_rotvec = compute_rotation_delta_rotvec(rot_now, self._prev_rotation, self._R_lh2base)
                else:
                    # First frame or just after SetReference — zero delta.
                    delta_pos = np.zeros(3)
                    delta_rotvec = np.zeros(3)
                # Advance per-frame reference — only when we have valid
                # tracker data, so a None→valid transition doesn't produce
                # a huge delta.
                self._prev_position = pos_now.copy()
                self._prev_rotation = rot_now.copy()
            else:
                # Tracker lost — emit zero deltas without touching
                # _prev_position.  When the tracker returns, deltas resume
                # from the last valid pose (a large jump will be caught by
                # glitch rejection below).
                pos_now = np.zeros(3)
                rot_now = np.eye(3)
                delta_pos = np.zeros(3)
                delta_rotvec = np.zeros(3)

            # --- Position gain (#06 scaling) ---
            delta_pos = delta_pos * self._pos_gain

            # --- Glitch rejection: reject impossibly large per-frame deltas ---
            # At 30 fps, 25 mm/frame = 750 mm/s — faster than any deliberate
            # hand movement.  Anything larger is a tracking glitch (jumps of
            # 80-130 mm observed in logs).  Zero it out instead of clamping —
            # a clamp still sends a large delta in the glitch direction.
            _MAX_DELTA_M = 0.025  # 25 mm
            if np.linalg.norm(delta_pos) > _MAX_DELTA_M:
                delta_pos = np.zeros(3)
                delta_rotvec = np.zeros(3)

            # --- NaN guard — hardware glitch can produce NaN ---
            if not np.isfinite(delta_pos).all():
                delta_pos = np.zeros(3)
            if not np.isfinite(delta_rotvec).all():
                delta_rotvec = np.zeros(3)

            # --- Dead zone: filter sub-threshold tracking noise ---
            # Applied directly to per-frame delta (no EMA).  The follower's
            # own ctrl EMA (50 Hz), workspace clamp, and position/rotation
            # deadband provide physical smoothing — the leader does not need
            # to add lag.
            if np.linalg.norm(delta_pos) < self._dead_zone_m:
                delta_pos = np.zeros(3)
            if np.linalg.norm(delta_rotvec) < self._dead_zone_rad:
                delta_rotvec = np.zeros(3)

            # --- Gripper distance (mapped via calibrated range) ---
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

            # --- Diagnostic logging (1 Hz) ---
            now = time.time()
            if now - getattr(self, "_last_debug_ts", 0.0) > 1.0:
                self._last_debug_ts = now
                if tracker is None:
                    logger.warning(
                        "TRACKER: pose is None — solver not converged or tracker occluded."
                    )
                else:
                    logger.info(
                        "TRACKER: pos=[%.3f,%.3f,%.3f]m delta=%.1fmm "
                        "dz=%s grip=%.1fmm",
                        pos_now[0], pos_now[1], pos_now[2],
                        np.linalg.norm(delta_pos) * 1000,
                        "PASS" if np.linalg.norm(delta_pos) >= self._dead_zone_m else "BLOCK",
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
                self._prev_position = None
                self._prev_rotation = None
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
        if self._calibrate_error:
            return device_pb2.DeviceInfo(
                status=device_pb2.DeviceStatus.FATAL
            )
        return device_pb2.DeviceInfo(
            status=device_pb2.DeviceStatus.COLLECTION
        )

    # --- alignment ---

    def SetReference(self, request, context):
        """Reset the delta origin to the current tracker pose.

        Clears the previous-frame state so the first post-reference frame
        emits zero delta.
        """
        with self._lock:
            tracker = self._read_tracker_pose()
            if tracker is not None:
                logger.info("Reference set: pos=%s", np.round(tracker[0], 4))
            else:
                logger.warning("SetReference called but no tracker data available.")
            self._prev_position = None
            self._prev_rotation = None
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
