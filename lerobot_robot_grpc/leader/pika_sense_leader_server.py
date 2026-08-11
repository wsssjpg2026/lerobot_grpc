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

import logging
import threading
import time

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
    R_lh2base: np.ndarray,
) -> np.ndarray:
    """Body-frame rotation delta as a rotvec, rotated to robot base frame.

    ``R_delta = R_ref^T @ R_now`` (body-frame composition).  The rotvec of
    ``R_delta`` is then rotated by ``R_lh2base`` — the rotation axis
    transforms as a vector under change of basis while the angle is unchanged
    (mathematically equivalent to the conjugation
    ``R_lh2base @ R_delta @ R_lh2base^T``).

    Parameters
    ----------
    rot_now, rot_ref
        3×3 rotation matrices (tracker orientation in the lighthouse frame).
    R_lh2base
        3×3 rotation matrix mapping lighthouse vectors to robot-base vectors.
    """
    r_delta = rot_ref.T @ rot_now
    rotvec_lh = Rot.from_matrix(r_delta).as_rotvec()
    return R_lh2base @ rotvec_lh


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


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------


class PikaSenseServicer(LeaderServicer):
    """Leader servicer wrapping Pika Sense for delta-pose teleoperation.

    In each :meth:`GetAction` call the servicer:

    1. Reads the Vive Tracker pose (position + quaternion) and gripper
       distance from the Pika Sense hardware.
    2. Computes the pose delta relative to the reference (latch) or the
       previous frame (continuous).
    3. Rotates the delta from the lighthouse frame to the robot base frame
       via ``R_lh2base`` (#05 direction-calibration result).
    4. Applies a dead-zone (#06) and EMA smoothing (#06).
    5. Encodes the 8 FLOAT32 pose-delta action features and streams them.

    Parameters
    ----------
    port
        Serial port path for the Pika Sense device.
    reference_mode
        ``"latch"`` (default) — :meth:`SetReference` captures a fixed
        reference; deltas are relative to it until the next call.  ``"continuous"``
        — each frame's delta is relative to the previous frame (velocity control).
    ema_alpha
        EMA smoothing factor (0–1).  Default ``0.25`` (#06).
    dead_zone_mm
        Position-delta dead-zone threshold in millimetres.  Default ``2.0`` (#06).
    pos_gain
        Position-delta gain factor — multiplies the tracker displacement before
        sending.  Default ``1.0`` (1:1 mapping); tune with real hardware to
        match the tracker range to the SO-101 workspace (~251 mm reach).
    R_lh2base
        3×3 rotation matrix mapping lighthouse axes to robot-base axes.
        Defaults to identity (no rotation) — override with the direction-
        calibration result for real hardware.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        reference_mode: str = "latch",
        ema_alpha: float = 0.25,
        dead_zone_mm: float = 2.0,
        pos_gain: float = 1.0,
        R_lh2base: np.ndarray | None = None,
    ):
        if reference_mode not in ("latch", "continuous"):
            raise ValueError(
                f"reference_mode must be 'latch' or 'continuous', got {reference_mode!r}"
            )

        # Deferred import — the Pika SDK is an optional dependency.
        # Importing here means the module loads cleanly on machines without
        # the SDK; only construction fails.
        from pika.sense import Sense

        self._device = Sense(port)

        # Feature schema — shared with the follower's pose_delta mode.
        self._act_ft_info = build_pose_delta_feature_info()

        # Pipeline parameters
        self._reference_mode = reference_mode
        self._ema_alpha = ema_alpha
        self._dead_zone_m = dead_zone_mm / 1000.0
        self._pos_gain = pos_gain
        self._R_lh2base = R_lh2base if R_lh2base is not None else np.eye(3)

        # Connection / tracker state
        self._lock = threading.Lock()
        self._connected = False
        self._tracker_device: str | None = None

        # Latch-mode reference pose
        self._ref_position: np.ndarray | None = None
        self._ref_rotation: np.ndarray | None = None

        # Continuous-mode previous-frame pose
        self._prev_position: np.ndarray | None = None
        self._prev_rotation: np.ndarray | None = None

        # EMA filter state
        self._ema_pos = np.zeros(3)
        self._ema_rotvec = np.zeros(3)
        self._ema_initialized = False

        logger.info(
            "PikaSenseServicer created: port=%s mode=%s ema_alpha=%.2f "
            "dead_zone_mm=%.1f",
            port,
            reference_mode,
            ema_alpha,
            dead_zone_mm,
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

    def _compute_action(self) -> dict[str, float]:
        """Read hardware, run the full delta pipeline, return the action dict."""
        with self._lock:
            tracker = self._read_tracker_pose()

            if tracker is not None:
                pos_now, rot_now = tracker
            else:
                # No data yet — emit zero deltas.
                pos_now = np.zeros(3)
                rot_now = np.eye(3)

            # --- Delta computation (#05) ---
            # Select reference source: fixed (latch) or per-frame (continuous).
            if self._reference_mode == "latch":
                ref_pos, ref_rot = self._ref_position, self._ref_rotation
            else:
                ref_pos, ref_rot = self._prev_position, self._prev_rotation

            if ref_pos is not None:
                delta_pos = compute_position_delta(pos_now, ref_pos, self._R_lh2base)
                delta_rotvec = compute_rotation_delta_rotvec(rot_now, ref_rot, self._R_lh2base)
            else:
                # No reference yet — emit zeros.
                delta_pos = np.zeros(3)
                delta_rotvec = np.zeros(3)

            # Continuous mode: advance the per-frame reference.
            if self._reference_mode == "continuous":
                self._prev_position = pos_now
                self._prev_rotation = rot_now

            # --- Position gain (#06 scaling) ---
            delta_pos = delta_pos * self._pos_gain

            # --- Dead zone (#06, position only) ---
            delta_pos = apply_dead_zone(delta_pos, self._dead_zone_m)

            # --- EMA smoothing (#06) ---
            if not self._ema_initialized:
                self._ema_pos = delta_pos.copy()
                self._ema_rotvec = delta_rotvec.copy()
                self._ema_initialized = True
            else:
                self._ema_pos = apply_ema(delta_pos, self._ema_pos, self._ema_alpha)
                self._ema_rotvec = apply_ema(
                    delta_rotvec, self._ema_rotvec, self._ema_alpha
                )

            # --- Gripper distance ---
            try:
                gripper_distance = float(self._device.get_gripper_distance())
            except Exception:
                gripper_distance = 0.0

            # --- Encode ---
            delta_quat = Rot.from_rotvec(self._ema_rotvec).as_quat()  # [x,y,z,w]
            return {
                "hand.delta_pos.x": float(self._ema_pos[0]),
                "hand.delta_pos.y": float(self._ema_pos[1]),
                "hand.delta_pos.z": float(self._ema_pos[2]),
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
            self._device.connect()
            # Initialise the Vive Tracker and wait for device discovery.
            self._device.get_vive_tracker()
            time.sleep(2.0)  # libsurvive needs ~1–2 s to discover devices
            devices = self._device.get_tracker_devices()
            if not devices:
                logger.warning("No Vive Tracker devices discovered.")
                self._tracker_device = None
            else:
                # Auto-discover — device name varies ("T20", "WM0", …).
                self._tracker_device = devices[0]
                logger.info("Vive Tracker discovered: %s", self._tracker_device)
            self._connected = True
        # Pika Sense uses absolute encoders + libsurvive auto-calibration —
        # no manual range-of-motion recording needed.
        return device_pb2.CalibrationInfo(
            status=device_pb2.CalibrationStatus.CALIBRATED
        )

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(
            status=device_pb2.CalibrationStatus.CALIBRATED
        )

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        with self._lock:
            if self._connected:
                self._device.disconnect()
                self._connected = False
        return Empty()

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(
            status=device_pb2.DeviceStatus.COLLECTION
        )

    # --- alignment ---

    def SetReference(self, request, context):
        """Snapshot the current tracker pose as the delta origin.

        Resets the EMA filter so the first post-reference frame is clean.
        """
        with self._lock:
            tracker = self._read_tracker_pose()
            if tracker is not None:
                self._ref_position, self._ref_rotation = tracker
                logger.info(
                    "Reference set: pos=%s",
                    np.round(self._ref_position, 4),
                )
            else:
                logger.warning(
                    "SetReference called but no tracker data available."
                )
            # Reset EMA and continuous-mode previous-frame state.
            self._ema_initialized = False
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
