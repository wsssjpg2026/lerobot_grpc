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
    max_calibration_attempts: int = 10

    use_ssl: bool = False
    ssl_cert_path: Path | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.use_ssl and not self.ssl_cert_path:
            raise ValueError("'ssl_cert_path' is required when 'use_ssl' is True.")