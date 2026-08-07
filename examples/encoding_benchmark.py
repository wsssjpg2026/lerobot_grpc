"""Encoding benchmark — per-frame JPEG vs inter-frame H.264 over gRPC, no hardware.

Serves the same synthetic camera scene through the real gRPC stack (server +
protobuf + streaming) with `camera_encoding` set to "jpeg" or "h264", and
measures, per session:

  - bandwidth: payload bytes actually put on the wire (sum of OneFeature.data),
    MB total / effective Mbps / average packet size
  - latency: server capture timestamp (produce_ts) -> client receive, including
    codec encode + gRPC transport (in-process localhost transport is negligible,
    so this mostly reflects codec pipeline delay); avg / p95 / max
  - server encode ms/frame and client decode ms/frame

Two scene modes are compared: "static" (identical frames; H.264 P-frames shrink
to ~nothing) and "motion" (a moving patch; both codecs work harder).

Run from the repo root (env with av + this package installed editable):

    python examples/encoding_benchmark.py                  # synthetic scene, defaults below
    python examples/encoding_benchmark.py --seconds 3 --width 320 --height 240
    python examples/encoding_benchmark.py --video path/to/observation.images.top/chunk-000/file-000.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import av
import cv2
import numpy as np
from google.protobuf.empty_pb2 import Empty

import grpc

from lerobot_robot_grpc.follower.follower_server import (
    FollowerServer,
    FollowerServerConfig,
    FollowerServicer,
)
from lerobot_robot_grpc.follower.utils import (
    H264FrameDecoder,
    H264FrameEncoder,
    _now_produce_ts,
    _wall_now,
)
from lerobot_robot_grpc.protos import device_pb2, device_pb2_grpc

logging.basicConfig(level=logging.WARNING)

ADDRESS = "127.0.0.1:50231"
WARMUP_FRAMES = 5  # skip stream warmup (first keyframe, decoder init) in latency stats


class Scene:
    """Synthetic camera: gradient + noise background, optional moving white patch."""

    def __init__(self, width: int, height: int, mode: str):
        self.width = width
        self.height = height
        self.mode = mode
        rng = np.random.default_rng(7)
        yy = np.arange(height, dtype=np.uint8)[:, None]
        xx = np.arange(width, dtype=np.uint8)[None, :]
        base = np.stack(
            [np.broadcast_to(xx, (height, width)),
             np.broadcast_to(yy, (height, width)),
             ((xx.astype(np.uint16) + yy.astype(np.uint16)) % 256).astype(np.uint8)],
            axis=-1,
        ).astype(np.uint8)
        self._base = (base.astype(np.uint16) + rng.integers(0, 12, base.shape)).clip(0, 255).astype(np.uint8)
        self._x = 0

    def frame(self) -> np.ndarray:
        if self.mode == "static":
            return self._base
        img = self._base.copy()
        img[60:240, self._x : self._x + 120] = 255
        img[10:40, (self._x * 2 + 40) % (self.width - 80) : (self._x * 2 + 120) % (self.width - 80)] = 0
        self._x = (self._x + 6) % (self.width - 120)
        return img


class VideoScene:
    """Replays frames from a recorded video (e.g. a lerobot dataset mp4) in order."""

    def __init__(self, path: str):
        if av is None:
            raise ImportError("PyAV ('av') is required to replay a video file.")
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        rate = self.stream.average_rate
        self.fps = float(rate) if rate else 30.0
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height
        self.frame_count = self.stream.frames or 0
        self._iter = self.container.decode(self.stream)
        self._last: np.ndarray | None = None
        self._ended = False

    def frame(self) -> np.ndarray:
        if self._ended:
            return self._last
        try:
            frame = next(self._iter)
        except StopIteration:
            self._ended = True
            return self._last
        self._last = frame.to_ndarray(format="rgb24")
        return self._last

    def close(self) -> None:
        self.container.close()


@dataclass
class ServerStats:
    frames: int = 0
    bytes_sent: int = 0
    encode_total_s: float = 0.0

    def record(self, nbytes: int, encode_s: float) -> None:
        self.frames += 1
        self.bytes_sent += nbytes
        self.encode_total_s += encode_s

    @property
    def avg_encode_ms(self) -> float:
        return self.encode_total_s * 1000.0 / max(1, self.frames)


@dataclass
class ClientStats:
    frames: int = 0          # frames counted in stats (warmup excluded)
    total_frames: int = 0    # every frame received (for the fps figure)
    bytes_received: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    decode_ms: list[float] = field(default_factory=list)

    def record(self, nbytes: int, latency_ms: float, decode_ms: float) -> None:
        self.frames += 1
        self.bytes_received += nbytes
        self.latencies_ms.append(latency_ms)
        self.decode_ms.append(decode_ms)

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return float("nan")
        return float(np.percentile(self.latencies_ms, p))

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / max(1, len(self.latencies_ms))

    @property
    def max_latency_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else float("nan")

    @property
    def avg_decode_ms(self) -> float:
        return sum(self.decode_ms) / max(1, len(self.decode_ms))


class BenchmarkServicer(FollowerServicer):
    """Serves one camera with the chosen encoding; no hardware, no joints needed."""

    def __init__(self, camera_encoding: str, scene: Scene, fps: int, quality: int):
        self.camera_encoding = camera_encoding
        self.scene = scene
        self.fps = fps
        self.quality = quality
        self.stats = ServerStats()
        self._info = device_pb2.OneFeatureInfo(
            key="cam",
            criticality=device_pb2.Criticality.CRITICALITY_CRITICAL,
            type=device_pb2.DataType.UINT8,
            shape=device_pb2.ImageShape(H=scene.height, W=scene.width, C=3),
            encoding=(
                device_pb2.Encoding.H264 if camera_encoding == "h264" else device_pb2.Encoding.JPEG
            ),
            img_quality=quality,
        )

    def GetInfo(self, request, context):
        return device_pb2.GetInfoResponse(
            observation_features=[self._info],
            action_features=[],
            feedback_features=[],
        )

    def Connect(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def Calibrate(self, request, context):
        return device_pb2.CalibrationInfo(status=device_pb2.CalibrationStatus.CALIBRATED)

    def CalibrateDone(self, request, context):
        return Empty()

    def Disconnect(self, request, context):
        return Empty()

    def SendAction(self, request, context):
        return device_pb2.Action(features=[])

    def GetFeedback(self, request, context):
        return iter(())

    def GetStatus(self, request, context):
        return device_pb2.DeviceInfo(status=device_pb2.DeviceStatus.COLLECTION)

    def GetObservation(self, request, context):
        # Mirrors the real server: per-stream H.264 encoder (keyframe first, keyint=2s),
        # produce_ts stamped BEFORE encoding so latency includes codec delay.
        encoders: dict[str, H264FrameEncoder] = {}
        period = 1.0 / self.fps
        bitrate_kbps = max(500, 2000 * (self.scene.height * self.scene.width) // (640 * 480))
        while context.is_active():
            frame = self.scene.frame()
            ts = _now_produce_ts()
            t0 = time.perf_counter()
            if self.camera_encoding == "h264":
                encoder = encoders.get("cam")
                if encoder is None:
                    encoder = H264FrameEncoder(
                        "cam", self.scene.height, self.scene.width,
                        fps=self.fps, bitrate_kbps=bitrate_kbps, keyint_frames=self.fps * 2,
                    )
                    encoders["cam"] = encoder
                data = encoder.encode(frame)[0]
            else:
                ok, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality],
                )
                if not ok:
                    raise RuntimeError("JPEG encode failed")
                data = buf.tobytes()
            encode_s = time.perf_counter() - t0
            self.stats.record(len(data), encode_s)
            yield device_pb2.OneFeature(key="cam", data=data, produce_ts=ts)
            time.sleep(period)


def run_session(
    camera_encoding: str, scene: Scene | VideoScene, fps: float, seconds: float, quality: int
) -> tuple[ServerStats, ClientStats]:
    width, height = scene.width, scene.height
    servicer = BenchmarkServicer(camera_encoding, scene, fps, quality)
    server = FollowerServer(FollowerServerConfig(address=ADDRESS, server_grace_period_s=1.0), servicer)
    server.start()
    channel = grpc.insecure_channel(ADDRESS)
    stub = device_pb2_grpc.RobotStub(channel)
    info = stub.GetInfo(device_pb2.GetInfoRequest())

    # Same process: produce_ts (wall clock) <-> perf_counter mapping.
    # perf_counter (QPC) is used instead of monotonic: monotonic has ~15.6ms
    # granularity on Windows, which would quantize sub-ms latencies to zero.
    offset = time.perf_counter() - _wall_now()
    decoder = H264FrameDecoder("cam", height, width) if camera_encoding == "h264" else None
    stop = threading.Event()
    stats = ClientStats()

    def consume() -> None:
        idx = 0
        try:
            for feat in stub.GetObservation(Empty()):
                if stop.is_set():
                    break
                recv = time.perf_counter()
                stamp_wall = feat.produce_ts.seconds + feat.produce_ts.nanos / 1e9
                latency_ms = (recv - (stamp_wall + offset)) * 1000.0
                t0 = recv
                if decoder is not None:
                    decoder.decode(feat.data)
                else:
                    cv2.imdecode(np.frombuffer(feat.data, np.uint8), cv2.IMREAD_COLOR)
                decode_ms = (time.perf_counter() - t0) * 1000.0
                stats.total_frames += 1
                if idx >= WARMUP_FRAMES:
                    stats.record(len(feat.data), latency_ms, decode_ms)
                idx += 1
        except grpc.RpcError:
            pass

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    time.sleep(seconds)
    stop.set()
    server.stop()
    thread.join(timeout=5.0)
    channel.close()
    return servicer.stats, stats


def format_session(camera_encoding: str, mode: str, seconds: float, s: ServerStats, c: ClientStats) -> None:
    mb = c.bytes_received / 1e6
    mbps = mb * 8 / seconds
    print(f"\n--- encoding={camera_encoding} scene={mode} ({seconds:.0f}s) ---")
    print(f"  server  : {s.frames} frames, {s.bytes_sent/1e6:.1f} MB sent, encode {s.avg_encode_ms:.2f} ms/frame")
    print(f"  client  : {c.frames} frames, {mb:.1f} MB received ({mbps:.1f} Mbps), "
          f"avg packet {c.bytes_received / max(1, c.frames) / 1e3:.1f} KB, fps {c.total_frames / seconds:.0f}")
    print(f"  latency : avg {c.avg_latency_ms:.2f} ms | p95 {c.percentile(95):.2f} ms | max {c.max_latency_ms:.2f} ms")
    print(f"  decode  : {c.avg_decode_ms:.2f} ms/frame")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0, help="duration per session (2 encodings x 2 scenes)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality, mirrors the real server's img_quality")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="replay frames from a recorded video (e.g. a lerobot dataset mp4) instead of the synthetic scene; "
             "resolution/fps are taken from the file and the static/motion modes are ignored",
    )
    args = parser.parse_args()

    if args.video:
        scenes: list[tuple[str, Scene | VideoScene]] = [("replay", VideoScene(args.video))]
        width, height, fps = scenes[0][1].width, scenes[0][1].height, scenes[0][1].fps
        size_mb = os.path.getsize(args.video) / 1e6
        src_mbps = size_mb * 8 / max(1.0, scenes[0][1].frame_count / fps)
        print(f"reference: recorded video {args.video} = {size_mb:.1f} MB, "
              f"{scenes[0][1].frame_count} frames @ {fps:.0f} fps = {src_mbps:.1f} Mbps (file encoding)")
    else:
        scenes = [(mode, Scene(args.width, args.height, mode)) for mode in ("static", "motion")]
        width, height, fps = args.width, args.height, float(args.fps)

    if width % 2 != 0 or height % 2 != 0:
        parser.error("width and height must be even (H.264 yuv420p requirement)")
    if args.seconds < 2:
        parser.error("--seconds should be >= 2 (warmup excluded from stats)")

    print(f"benchmark: {width}x{height}@{fps:.0f} fps, {args.seconds:.0f}s per session, JPEG q{args.quality} vs H.264")
    results: dict[tuple[str, str], tuple[ServerStats, ClientStats]] = {}
    for camera_encoding in ("jpeg", "h264"):
        for mode, scene in scenes:
            s, c = run_session(camera_encoding, scene, fps, args.seconds, args.quality)
            results[(camera_encoding, mode)] = (s, c)
            format_session(camera_encoding, mode, args.seconds, s, c)

    print("\n=== summary: H.264 vs JPEG (per scene mode) ===")
    for mode, _ in scenes:
        jpeg = results[("jpeg", mode)][1]
        h264 = results[("h264", mode)][1]
        print(
            f"  scene={mode:<7} bandwidth JPEG {jpeg.bytes_received*8/args.seconds/1e6:.1f} Mbps "
            f"-> H.264 {h264.bytes_received*8/args.seconds/1e6:.1f} Mbps "
            f"({h264.bytes_received / jpeg.bytes_received * 100:.1f}%), "
            f"latency avg {jpeg.avg_latency_ms:.2f} -> {h264.avg_latency_ms:.2f} ms "
            f"| p95 {jpeg.percentile(95):.2f} -> {h264.percentile(95):.2f} ms"
        )


if __name__ == "__main__":
    main()
