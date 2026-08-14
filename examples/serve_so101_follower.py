"""Launch a gRPC follower server wrapping a real SO-101 follower arm.

Cameras are passed as an inline YAML/JSON dict string on ``--robot.cameras``,
exactly like ``lerobot-record`` / ``lerobot-teleoperate`` (draccus does not
support dot-syntax for ``dict[str, CameraConfig]`` fields). The per-camera
``type`` discriminator selects the backend; ``index_or_path`` is the device
index (webcam) or file path.

Linux / bash:

    python serve_so101_follower.py \\
        --robot.port=/dev/ttyUSB0 --robot.id=follower \\
        --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --address=0.0.0.0:5555

Windows / PowerShell (double-quote the whole value; the brace is a single token):

    python serve_so101_follower.py `
        --robot.port=COM6 --robot.id=follower `
        --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
        --address=0.0.0.0:5555

Multiple cameras — add more keys inside the outer braces:

    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"

For RealSense, set ``type: realsense`` (requires ``lerobot[realsense]`` installed).

The arm connects on first client Connect RPC. Calibration via
Calibrate/CalibrateDone RPCs (use lerobot-calibrate --robot.type=grpc_follower ...).

`--camera_encoding` selects the camera wire format:
  - h264  (default): inter-frame H.264; the server keeps a per-stream encoder and
    the client keeps a per-camera decoder, so P-frames compress across snapshots.
    Requires `av` (PyAV) on both sides.
  - jpeg: per-frame JPEG, backward compatible with the original protocol behavior.
Depth maps ('<cam>_depth') are always RAW regardless of this flag.

`--action_mode=pose_delta` switches the follower to end-effector pose-delta
actions (Pika Sense teleop): the servicer loads the shared MuJoCo kinematics
oracle and drives the sim/real-shared PoseDeltaLaw.  Requires a calibrated
arm; pair with a latch-once leader.
"""
import logging
import time
from dataclasses import dataclass

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 -- register for draccus
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot_robot_grpc.follower.follower_server import FollowerServer, FollowerServerConfig
from lerobot_robot_grpc.follower.so101_follower_server import (
    SO101FollowerAdapted,
    SO101FollowerServicer,
)


@dataclass
class ServeFollowerConfig:
    robot: SOFollowerRobotConfig
    address: str = "0.0.0.0:5555"
    camera_encoding: str = "h264"  # "h264" (inter-frame) or "jpeg" (per-frame)
    # "joint" (default, A-class) or "pose_delta" (B-class: 8-feature EE pose
    # deltas solved by the shared PoseDeltaLaw with the real-arm safety
    # posture -- base safety sphere / residual-hold / stale-hold / ~5mm slew).
    action_mode: str = "joint"


@parser.wrap()
def main(cfg: ServeFollowerConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    robot = SO101FollowerAdapted(cfg.robot)
    servicer = SO101FollowerServicer(
        robot, camera_encoding=cfg.camera_encoding, action_mode=cfg.action_mode
    )
    server = FollowerServer(FollowerServerConfig(address=cfg.address), servicer)
    server.start()
    cam_names = list(cfg.robot.cameras.keys())
    logging.info(
        "SO-101 follower server ready: address=%s serial=%s id=%s cameras=%s camera_encoding=%s action_mode=%s",
        cfg.address, cfg.robot.port, cfg.robot.id, cam_names, cfg.camera_encoding,
        cfg.action_mode,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping follower server...")
        server.stop()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
