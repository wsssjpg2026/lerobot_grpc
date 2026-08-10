"""
Shared helpers for the gRPC follower stack, in three sections:

1. Serialization: protobuf <-> Python (de)serialization (encode_feature,
   load_feature, ...) used by both the client (`GRPCFollower`) and the server
   (`FollowerServicer`).
2. H.264 inter-frame codec (`H264FrameEncoder` / `H264FrameDecoder`) for camera
   streams over `GetObservation` (Annex-B access units, see the section comment).
3. Teleop telemetry (`TeleopStats` / `TeleopMonitor`) for live bandwidth /
   latency rendering and end-of-session summaries.

Keep this module free of client/server logic so a change only ever needs to
be made in one place.
"""

from __future__ import annotations

import logging
import struct
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

try:
    import av
except ImportError:  # pragma: no cover - av is an optional runtime dep
    av = None
from google.protobuf.timestamp_pb2 import Timestamp

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _grpc_available


def _wall_now() -> float:
    """High-resolution wall clock (seconds since Unix epoch).

    protobuf's `Timestamp.GetCurrentTime()` uses GetSystemTimeAsFileTime on
    Windows, which ticks every ~15.6ms - that quantization would drown sub-ms
    latency measurements in +-15ms bias. GetSystemTimePreciseAsFileTime
    (100ns) is the Windows fix; time.time_ns() on other platforms.
    """
    if sys.platform == "win32":
        import ctypes

        class _FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", ctypes.c_uint32),
                ("dwHighDateTime", ctypes.c_uint32),
            ]

        ft = _FILETIME()
        ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime(ctypes.byref(ft))
        ticks = (ft.dwHighDateTime << 32) | ft.dwLowDateTime
        return (ticks / 1e7) - 11644473600.0  # 1601-01-01 -> Unix epoch
    return time.time_ns() / 1e9


def _ts_from_wall(now: float) -> Timestamp:
    """Builds a protobuf Timestamp from a wall-clock seconds value (Unix epoch)."""
    ts = Timestamp()
    ts.FromSeconds(int(now))
    ts.nanos = int((now % 1.0) * 1e9)
    return ts


def _now_produce_ts() -> Timestamp:
    """UTC timestamp marking when a frame was produced/sent - for latency observability."""
    return _ts_from_wall(_wall_now())

if TYPE_CHECKING or _grpc_available:
    from lerobot_robot_grpc.protos import device_pb2
else:
    device_pb2 = None

logger = logging.getLogger(__name__)


# DataType → (struct format char, numpy scalar type). Keyed by the symbolic enum value so
# the mapping tracks enum renumbering automatically (enums now follow *_UNSPECIFIED=0).
# Guarded so this module stays importable without the optional `grpcio` dependency.
_FEATURE_META_BY_PROTO: dict[int, tuple[str, type[np.generic]]] = (
    {
        # Explicit little-endian (e.g. "<f") so that encoding is deterministic across platforms.
        device_pb2.DataType.FLOAT32: ("<f", np.float32),
        device_pb2.DataType.INT32: ("<i", np.int32),
        device_pb2.DataType.UINT8: ("<B", np.uint8),
        device_pb2.DataType.UINT16: ("<H", np.uint16),
    }
    if device_pb2 is not None
    else {}
)

# DataType → Python scalar type returned by `parse_feature`. LeRobot dataset features expect
# Python scalars (see `hw_to_dataset_features`).
_PYTHON_SCALAR_BY_PROTO: dict[int, type] = (
    {
        device_pb2.DataType.FLOAT32: float,
        device_pb2.DataType.INT32: int,
        device_pb2.DataType.UINT8: int,
        device_pb2.DataType.UINT16: int,
    }
    if device_pb2 is not None
    else {}
)


def feature_meta_for(data_type: device_pb2.DataType) -> tuple[str, type[np.generic]]:
    """Maps a protobuf DataType to its struct format character and numpy scalar type."""
    if data_type not in _FEATURE_META_BY_PROTO:
        raise ValueError(f"Unsupported data type '{data_type}'.")
    return _FEATURE_META_BY_PROTO[data_type]


def python_scalar_type_for(data_type: device_pb2.DataType) -> type:
    """Maps a protobuf DataType to the Python scalar type returned by `parse_feature`."""
    if data_type not in _PYTHON_SCALAR_BY_PROTO:
        raise ValueError(f"Unsupported data type '{data_type}'.")
    return _PYTHON_SCALAR_BY_PROTO[data_type]


def encode_image(feature_info: device_pb2.OneFeatureInfo, image: np.ndarray) -> bytes:
    """Encodes an HWC numpy array (RGB(A), uint8) into the encoding declared by the feature info."""
    encoding = feature_info.encoding
    if encoding == device_pb2.Encoding.JPEG:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, feature_info.img_quality])
        elif image.ndim == 3 and image.shape[-1] == 3:
            ok, buf = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, feature_info.img_quality],
            )
        else:
            raise ValueError(f"Feature '{feature_info.key}': JPEG encoding only supports 1 or 3 channels.")
    elif encoding == device_pb2.Encoding.PNG:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            ok, buf = cv2.imencode(".png", image)
        elif image.ndim == 3 and image.shape[-1] == 3:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        elif image.ndim == 3 and image.shape[-1] == 4:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        else:
            raise ValueError(f"Feature '{feature_info.key}': PNG encoding only supports 1, 3 or 4 channels.")
    else:
        raise ValueError(f"Unsupported encoding '{encoding}' for feature '{feature_info.key}'.")
    if not ok:
        raise ValueError(f"Failed to encode feature '{feature_info.key}'.")
    return buf.tobytes()


def encode_feature(
    ft_info: dict[str, device_pb2.OneFeatureInfo],
    source: RobotObservation | RobotAction | dict[str, Any],
    ts_for_key: dict[str, Timestamp] | None = None,
) -> Iterator[device_pb2.OneFeature]:
    # When `ts_for_key` is None (e.g. SendAction), all features share one produce_ts stamped
    # here at encode time. When provided (GetObservation), each feature is stamped at its
    # true sample time (motor read vs camera peek) by the caller — see so101_follower_server.
    fallback = _now_produce_ts()
    for key in source.keys():
        if key not in ft_info:
            raise KeyError(f"Unknown action key {key}")
        feature_info = ft_info[key]
        shape = (feature_info.shape.H, feature_info.shape.W, feature_info.shape.C)
        if shape == (1, 1, 1):
            if feature_info.encoding != device_pb2.Encoding.RAW:
                raise ValueError(f"Encoding '{feature_info.encoding}' is not supported for scalar feature '{key}'.")
            fmt_char, _ = feature_meta_for(feature_info.type)
            data = struct.pack(fmt_char, source[key])
        elif feature_info.encoding == device_pb2.Encoding.RAW:
            data = source[key].tobytes()
        else:
            data = encode_image(feature_info, source[key])
        ts = ts_for_key[key] if ts_for_key is not None else fallback
        yield device_pb2.OneFeature(key=key, data=data, produce_ts=ts)


def parse_feature(feature: device_pb2.OneFeature, ft_info: dict[str, device_pb2.OneFeatureInfo]) -> Any:
    """Parses a OneFeature protobuf message into its corresponding value."""
    key = feature.key
    if key not in ft_info:
        raise KeyError(f"Feature '{key}' not found in feature info dictionary.")
    feature_info = ft_info[key]
    data_type = feature_info.type
    shape = feature_info.shape

    fmt_char, np_dtype = feature_meta_for(data_type)

    if not (shape.H == 1 and shape.W == 1 and shape.C == 1):
        if feature_info.encoding == device_pb2.Encoding.RAW:
            return np.frombuffer(feature.data, np_dtype).reshape(shape.H, shape.W, shape.C).copy()
        return decode_encoded_image(feature, feature_info, np_dtype)

    if feature_info.encoding != device_pb2.Encoding.RAW:
        raise ValueError(f"Encoding '{feature_info.encoding}' is not supported for scalar feature '{key}'.")
    expected_size = struct.calcsize(fmt_char)
    if len(feature.data) != expected_size:
        raise ValueError(
            f"Feature '{key}' payload size {len(feature.data)} does not match expected size {expected_size}."
        )
    return struct.unpack(fmt_char, feature.data)[0]


def decode_encoded_image(
    feature: device_pb2.OneFeature,
    feature_info: device_pb2.OneFeatureInfo,
    np_dtype: type[np.generic],
) -> np.ndarray:
    """Decodes a JPEG/PNG-encoded image feature into an HWC numpy array (RGB(A))."""
    encoding = feature_info.encoding
    expected_shape = (feature_info.shape.H, feature_info.shape.W, feature_info.shape.C)
    channels = expected_shape[2]
    encoded = np.frombuffer(feature.data, np.uint8)

    if encoding == device_pb2.Encoding.JPEG:
        if channels == 1:
            flags = cv2.IMREAD_GRAYSCALE
        elif channels == 3:
            flags = cv2.IMREAD_COLOR
        else:
            raise ValueError(f"Feature '{feature.key}': JPEG encoding only supports 1 or 3 channels.")
        image = cv2.imdecode(encoded, flags)
        if image is None:
            raise ValueError(f"Failed to decode JPEG feature '{feature.key}'.")
        if image.ndim == 2:
            image = image[..., None]
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding == device_pb2.Encoding.PNG:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to decode PNG feature '{feature.key}'.")
        if image.ndim == 2:
            image = image[..., None]
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unsupported encoding '{encoding}' for feature '{feature.key}'.")

    if image.shape != expected_shape:
        raise ValueError(
            f"Feature '{feature.key}' decoded shape {image.shape} does not match expected shape {expected_shape}."
        )
    return image.astype(np_dtype)


def load_feature(
    feature: device_pb2.OneFeature,
    ft_info: dict[str, device_pb2.OneFeatureInfo],
    target: RobotObservation | RobotAction | dict[str, Any],
    *,
    aux_behavior: Literal["keep_latest", "ignore"] = "keep_latest",
) -> None:
    """Loads a OneFeature into `target`.

    `aux_behavior` controls what happens when an AUXILIARY feature fails to parse:
    - "keep_latest" (client side): log and keep the previously received value.
    - "ignore" (server side): silently skip the feature.
    CRITICAL features always raise `DeviceNotConnectedError` on parse failure.
    """
    key = feature.key
    if key not in ft_info:
        raise KeyError(f"Feature '{key}' not found in feature info dictionary.")
    feature_info = ft_info[key]
    try:
        target[key] = parse_feature(feature, ft_info)
    except (struct.error, ValueError) as e:
        if feature_info.criticality == device_pb2.Criticality.CRITICALITY_CRITICAL:
            logger.error(f"Failed to retrieve critical feature {key}: {e}")
            raise DeviceNotConnectedError(f"Failed to find critical feature '{key}'") from e
        if feature_info.criticality == device_pb2.Criticality.CRITICALITY_AUXILIARY:
            if aux_behavior == "keep_latest":
                logger.info(f"AUXILIARY feature '{key}' failed to parse; keeping latest value.")
            elif aux_behavior != "ignore":
                raise ValueError(f"Unsupported aux_behavior '{aux_behavior}'.") from e
            return
        raise ValueError(f"Unsupported criticality '{feature_info.criticality}' for feature '{key}'") from e
    except Exception as e:
        logger.error(f"Failed to load feature {key}: {e}")
        raise DeviceNotConnectedError(
            f"Failed to load feature '{key}'. Likely mismatched type and/or shape, or unknown keys."
        ) from e

# =====================================================================
# H.264 (inter-frame) codec helpers
#
# Wire format: a camera's `OneFeature.data` carries exactly one H.264
# access unit (Annex-B, i.e. 00 00 00 01 start codes). Every keyframe
# embeds SPS/PPS (repeat_headers=1), so a decoder that starts fresh - or
# joins after the stream head - can sync at any keyframe; the server
# guarantees the first packet of a stream is always a keyframe. Both
# sides keep per-camera state across frames: the encoder emits P-frames
# referencing previous frames (the bandwidth win over per-frame JPEG),
# and the decoder must not be torn down between packets.
#
# Setup used for low latency: baseline profile, no B-frames, zerolatency,
# so one input RGB frame yields exactly one access unit (no flushing).
# =====================================================================
H264_AVAILABLE = av is not None

# libx264 options (kept next to the encoder): ultrafast + zerolatency keep end-to-end
# delay at one frame;
# baseline forbids B-frames (no reordering); repeat_headers embeds SPS/PPS in
# every keyframe so mid-stream sync works without side-channel negotiation.
_H264_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "ultrafast",
    "tune": "zerolatency",
    "profile": "baseline",
    "repeat_headers": "1",
}


class H264FrameEncoder:
    """Stateful H.264 encoder for one camera stream (RGB HWC uint8 in → Annex-B units out)."""

    def __init__(
        self,
        key: str,
        height: int,
        width: int,
        fps: int = 30,
        bitrate_kbps: int = 2000,
        keyint_frames: int = 60,
    ):
        if not H264_AVAILABLE:
            raise ImportError("PyAV ('av') is required for H.264 streaming. Install it with `pip install av`.")
        self.key = key
        self.height = height
        self.width = width
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(f"Camera '{key}': H.264 requires even dimensions, got {width}x{height}.")
        ctx = av.CodecContext.create(av.Codec("h264", "w"))
        ctx.width = width
        ctx.height = height
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, max(1, int(round(fps))))
        ctx.bit_rate = max(1, bitrate_kbps) * 1000
        ctx.gop_size = max(1, keyint_frames)
        ctx.options = _H264_ENCODER_OPTIONS
        self._ctx = ctx

    def close(self) -> None:
        """Releases the native PyAV encoder context. Idempotent.

        PyAV 15.x has no explicit close(): the underlying FFmpeg context is freed on GC
        (__dealloc__). We drop the reference so it can be collected promptly rather than
        accumulating across reconnects, and flush internal buffers if available.
        """
        ctx = self._ctx
        self._ctx = None
        if ctx is not None:
            flush = getattr(ctx, "flush_buffers", None)
            if flush is not None:
                try:
                    flush()
                except Exception:
                    pass

    def encode(self, image: np.ndarray) -> list[bytes]:
        """Encodes one RGB HWC uint8 frame; returns the access units to send (usually exactly one)."""
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        if frame.format is None or frame.format.name != "yuv420p":
            frame = frame.reformat(format="yuv420p")
        return [bytes(packet) for packet in self._ctx.encode(frame)]


class H264FrameDecoder:
    """Stateful H.264 decoder for one camera stream (Annex-B units in → RGB HWC uint8 out)."""

    def __init__(self, key: str, height: int, width: int):
        if not H264_AVAILABLE:
            raise ImportError("PyAV ('av') is required for H.264 streaming. Install it with `pip install av`.")
        self.key = key
        self._ctx = av.CodecContext.create(av.Codec("h264", "r"))

    def decode(self, data: bytes) -> np.ndarray | None:
        """Decodes one access unit; returns the RGB HWC uint8 frame, or None while frames are pending."""
        image = None
        for frame in self._ctx.decode(av.Packet(data)):
            image = frame.to_ndarray(format="rgb24")
        return image

    def close(self) -> None:
        """Releases the native PyAV decoder context. Idempotent.

        PyAV 15.x has no explicit close(): the underlying FFmpeg context is freed on GC
        (__dealloc__). We drop the reference so it can be collected promptly rather than
        accumulating across reconnects, and flush internal buffers if available.
        """
        ctx = self._ctx
        self._ctx = None
        if ctx is not None:
            flush = getattr(ctx, "flush_buffers", None)
            if flush is not None:
                try:
                    flush()
                except Exception:
                    pass

# =====================================================================
# Teleop telemetry: real-time bandwidth/latency in the terminal
#
# `TeleopStats`: thread-safe counters fed by the GRPCFollower observation
# stream and SendAction RPCs - bytes per feature, frame counts, per-frame
# latency samples (server produce_ts -> local receive), action RTT.
# `TeleopMonitor`: wraps a `TeleopStats` with a background renderer that
# refreshes a one-line status every `interval` seconds, and prints a full
# summary table when the session ends (`stop()`).
#
# Latency is measured from the server's `produce_ts` (wall clock, stamped at sample
# time) to the local wall clock on arrival — an absolute end-to-end delta, accurate
# when client and server share a clock (same host / NTP-synced). The client
# (`grpc_follower.py`) falls back to relative-to-first-frame jitter on cross-host skew.
# =====================================================================
# Cap on retained latency / RTT samples per bucket (ring buffer). ~5 min at 60 Hz.
_STATS_MAX_SAMPLES = 10_000


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, int(len(ordered) * p / 100.0))
    return ordered[idx]


@dataclass
class _KeyStats:
    frames: int = 0
    bytes: int = 0
    # Bounded ring buffer: a long teleop session at 30-60 Hz would otherwise grow these
    # lists without bound (and snapshot() copies them every second). Old samples drop off,
    # keeping avg/p95/max representative of the recent window.
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=_STATS_MAX_SAMPLES))


class TeleopStats:
    """Thread-safe per-feature counters + latency / action-RTT sample buckets."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, _KeyStats] = {}
        self._action_rtt_ms: deque = deque(maxlen=_STATS_MAX_SAMPLES)
        self._t0 = time.monotonic()

    def record_feature(self, key: str, nbytes: int, latency_ms: float | None = None) -> None:
        """Called once per received OneFeature (bytes on the wire + end-to-end latency)."""
        with self._lock:
            ks = self._keys.setdefault(key, _KeyStats())
            ks.frames += 1
            ks.bytes += nbytes
            if latency_ms is not None:
                ks.latencies_ms.append(latency_ms)

    def record_action(self, rtt_ms: float) -> None:
        """Called once per SendAction round trip."""
        with self._lock:
            self._action_rtt_ms.append(rtt_ms)

    def snapshot(self) -> dict[str, Any]:
        """Point-in-time copy for the live renderer / final summary."""
        with self._lock:
            elapsed = time.monotonic() - self._t0
            per_key = {k: (ks.frames, ks.bytes, list(ks.latencies_ms)) for k, ks in self._keys.items()}
            rtt = list(self._action_rtt_ms)
        total_bytes = sum(b for _, b, _ in per_key.values())
        total_frames = sum(f for f, _, _ in per_key.values())
        all_lat = [ms for _, _, lats in per_key.values() for ms in lats]
        return {
            "elapsed_s": elapsed,
            "total_bytes": total_bytes,
            "total_frames": total_frames,
            "per_key": per_key,
            "latency_ms": all_lat,
            "action_rtt_ms": rtt,
        }


class TeleopMonitor:
    """Live in-place renderer + end-of-session summary for a `TeleopStats`.

    The render thread repaints a multi-line block (header + one line per
    feature) in place every `interval` seconds, like the calibration pos table:
    the cursor is moved back up to the block start with an ANSI escape and the
    block is rewritten (each line cleared with `\\033[K` so shrinking numbers
    never leave stale tails). `stop()` erases the block and prints a full
    summary table.
    """

    def __init__(self, stats: TeleopStats, interval: float = 1.0):
        self._stats = stats
        self._interval = max(interval, 0.1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rendered_lines = 0
        # Refreshed in start(): non-TTY stdout (log redirect / CI / `tee`) gets a single-line
        # logger.info summary instead of ANSI repaints, which would dump raw escapes + whole
        # multi-line blocks every tick and flood the log.
        self._is_tty = True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._is_tty = sys.stdout.isatty()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._render_loop, daemon=True, name="teleop-monitor"
        )
        self._thread.start()

    def stop(self, print_summary: bool = True) -> None:
        """Stops the live renderer and optionally prints the session summary."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._clear_block()
        if print_summary:
            self.print_summary()

    def _clear_block(self) -> None:
        """Moves the cursor up to the live block and erases every line of it."""
        if not self._is_tty:
            return
        n = self._rendered_lines
        if n > 0:
            sys.stdout.write(f"\033[{n + 1}A\r")
            for i in range(n + 1):
                sys.stdout.write("\033[K")
                if i < n:
                    sys.stdout.write("\033[B")
            self._rendered_lines = 0
            sys.stdout.flush()

    def _render_loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._render_live()

    def _render_live(self) -> None:
        lines = self._live_lines()
        if not self._is_tty:
            # Non-interactive stdout: emit a single-line summary via the logger instead of
            # ANSI repaints (which would flood a redirected log with raw escapes + full blocks).
            logger.info(lines[0] if lines else "teleop stats")
            return
        n = len(lines)
        out = ""
        if self._rendered_lines:
            # Move up to the line ABOVE the block, clear it, then step down
            # onto the block start. This self-heals the block: if the teleop
            # loop's own per-frame print ever interleaves into this block (both
            # threads write to the same stdout), the block is one line taller
            # and the stale line sits right above it — clearing one extra line
            # on every repaint restores the fixed geometry within one tick.
            out += f"\033[{self._rendered_lines + 1}A\r\033[K\033[B"
        # Single write for the whole repaint: lerobot's teleop_loop prints its
        # own "Teleop loop time" line ~60x/s right below this block, so the
        # escape sequence must be one atomic-ish write to shrink the interleave
        # window to a minimum.
        out += "".join(line + "\033[K\n" for line in lines)
        sys.stdout.write(out)
        # The trailing newline is load-bearing: the loop pins the cursor at the
        # start of its own line below the block and rewrites it with
        # move_cursor_up(1); ending the block with a newline leaves the cursor
        # exactly there, so the loop never climbs into (or appends onto) the
        # block, and the next repaint moves up by a constant height.
        self._rendered_lines = n
        sys.stdout.flush()

    def _live_lines(self) -> list[str]:
        s = self._stats.snapshot()
        elapsed = max(s["elapsed_s"], 1e-9)
        lat = s["latency_ms"]
        rtt = s["action_rtt_ms"]
        header = f"[{elapsed:5.1f}s] obs {s['total_bytes'] * 8 / elapsed / 1e6:5.2f} Mbps"
        if lat:
            header += (
                " | latency avg/p95/max: "
                f"{sum(lat) / len(lat):5.1f}/{_percentile(lat, 95):5.1f}/{max(lat):5.1f} ms"
            )
        if rtt:
            header += f" | action RTT avg/p95: {sum(rtt) / len(rtt):4.1f}/{_percentile(rtt, 95):4.1f} ms"
        lines = [header]
        for key in sorted(s["per_key"]):
            frames, nbytes, _ = s["per_key"][key]
            lines.append(
                f"  {key:<24} {nbytes * 8 / elapsed / 1e6:8.2f} Mbps {frames / elapsed:7.1f} fps"
            )
        return lines

    def print_summary(self) -> None:
        """End-of-session aggregate: bandwidth + latency per feature and totals."""
        s = self._stats.snapshot()
        elapsed = s["elapsed_s"]
        if s["total_frames"] == 0:
            return
        print("\n=== teleop session stats (%.1f s) ===" % elapsed)
        print(f"{'feature':<24} {'frames':>8} {'MB':>8} {'Mbps':>8} {'fps':>7}   latency avg/p95/max (ms)")
        for key in sorted(s["per_key"]):
            frames, nbytes, lats = s["per_key"][key]
            lat_s = "-"
            if lats:
                lat_s = "%6.1f / %6.1f / %6.1f" % (
                    sum(lats) / len(lats),
                    _percentile(lats, 95),
                    max(lats),
                )
            print(
                f"{key:<24} {frames:>8} {nbytes / 1e6:>8.1f} "
                f"{nbytes * 8 / elapsed / 1e6:>8.2f} {frames / elapsed:>7.1f}   {lat_s}"
            )
        print(
            f"{'TOTAL':<24} {s['total_frames']:>8} {s['total_bytes'] / 1e6:>8.1f} "
            f"{s['total_bytes'] * 8 / elapsed / 1e6:>8.2f}"
        )
        rtt = s["action_rtt_ms"]
        if rtt:
            print(
                "action RTT: avg %.2f ms | p95 %.2f ms | max %.2f ms (N=%d)"
                % (sum(rtt) / len(rtt), _percentile(rtt, 95), max(rtt), len(rtt))
            )
