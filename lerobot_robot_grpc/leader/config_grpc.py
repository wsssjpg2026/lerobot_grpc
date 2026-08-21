"""
@author: Yiming Yang
@date: 2026-07-29
@brief: GRPCLeaderConfig - configuration class for GRPCLeader
"""

from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("grpc_leader")
@dataclass
class GRPCLeaderConfig(TeleoperatorConfig):
    # Network Configuration
    address: str = "localhost:5555"

    need_warmup: bool = True
    warmup_timeout_s: float = 10.0
    connect_timeout_s: float = 5.0
    data_timeout_s: float = 5.0
    # SetReference may intentionally wait up to 5s for a stable Pika pose.
    # Keep its RPC deadline separate from the hot-path data deadline and
    # leave enough margin for the server to return FAILED_PRECONDITION.
    reference_timeout_s: float = 7.0
    max_calibration_attempts: int = 10

    force_recalibrate: bool = False

    use_ssl: bool = False
    ssl_cert_path: Path | None = None

    def __post_init__(self):
        if self.reference_timeout_s <= 0.0:
            raise ValueError("'reference_timeout_s' must be positive.")
        if self.use_ssl and not self.ssl_cert_path:
            raise ValueError("'ssl_cert_path' is required when 'use_ssl' is True.")
