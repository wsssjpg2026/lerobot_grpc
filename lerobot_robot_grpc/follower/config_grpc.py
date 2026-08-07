"""
@author: Yiming Yang
@date: 2026-07-29
@brief: GRPCFollowerConfig - configuration class for GRPCFollower
"""

from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("grpc_follower")
@dataclass
class GRPCFollowerConfig(RobotConfig):
    # Network Configuration
    address: str = "localhost:5555"

    need_warmup: bool = True
    warmup_timeout_s: float = 10.0
    connect_timeout_s: float = 5.0
    data_timeout_s: float = 5.0
    max_calibration_attempts: int = 10

    force_recalibrate: bool = False

    # Telemetry: when True, GRPCFollower auto-creates a TeleopStats/TeleopMonitor
    # (live bandwidth+latency line + end-of-session summary). Zero code changes:
    # just add `--robot.teleop_stats=true` to any lerobot CLI invocation.
    teleop_stats: bool = False
    stats_interval_s: float = 1.0

    use_ssl: bool = False
    ssl_cert_path: Path | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.use_ssl and not self.ssl_cert_path:
            raise ValueError("'ssl_cert_path' is required when 'use_ssl' is True.")
        if self.stats_interval_s < 0.1:
            raise ValueError(f"'stats_interval_s' must be >= 0.1, got {self.stats_interval_s}.")
