from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import grpc
from concurrent import futures
from lerobot_robot_grpc.protos import device_pb2, device_pb2_grpc

logger = logging.getLogger(__name__)

@dataclass
class LeaderServerConfig:
    """Configuration for the LeaderServer."""
    address: str = "localhost:5555"  # Default gRPC server address

    server_grace_period_s: float = 5.0

    use_ssl: bool = False
    ssl_cert_path: Path | None = None
    ssl_key_path: Path | None = None

    def __post_init__(self):
        if self.use_ssl and not self.ssl_cert_path:
            raise ValueError("'ssl_cert_path' is required when 'use_ssl' is True.")
        if self.use_ssl and not self.ssl_key_path:
            raise ValueError("'ssl_key_path' is required when 'use_ssl' is True.")

class LeaderServicer(device_pb2_grpc.TeleoperatorServicer, ABC):
    """Abstract base class for the Leader gRPC service."""

    @abstractmethod
    def GetInfo(self, request, context):
        """Gets all feature schemas (observation + action + feedback) in one call."""
        pass

    @abstractmethod
    def Connect(self, request, context):
        """Connects to the teleoperator."""
        pass

    @abstractmethod
    def Calibrate(self, request, context):
        """Calibrates the teleoperator."""
        pass

    @abstractmethod
    def CalibrateDone(self, request, context):
        """Indicates that calibration is done."""
        pass

    @abstractmethod
    def Disconnect(self, request, context):
        """Disconnects from the teleoperator."""
        pass

    @abstractmethod
    def GetObservation(self, request, context):
        """Gets the observation from the teleoperator."""
        pass

    @abstractmethod
    def GetAction(self, request, context):
        """Gets an action from the teleoperator."""
        pass

    @abstractmethod
    def SendFeedback(self, request_iterator, context):
        """Sends feedback to the teleoperator."""
        pass

    @abstractmethod
    def GetStatus(self, request, context):
        """Gets the status of the teleoperator."""
        pass

    def GetTrackingReadiness(self, request, context):
        """Non-tracking leaders are ready without a Tracker lease."""
        return device_pb2.TrackingReadiness(
            state=(
                device_pb2.TrackingReadinessState.
                TRACKING_READINESS_STATE_NOT_APPLICABLE
            ),
            reason="teleoperator does not use optical tracking",
        )

    @abstractmethod
    def SetReference(self, request, context):
        """Sets the reference for the teleoperator."""
        pass

class LeaderServer:
    """gRPC server for the Leader teleoperator."""

    def __init__(self, config: LeaderServerConfig, servicer: LeaderServicer):
        self.config = config
        self.address = config.address
        self.use_ssl = config.use_ssl
        self.ssl_cert_path = config.ssl_cert_path
        self.ssl_key_path = config.ssl_key_path
        self.servicer = servicer
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        device_pb2_grpc.add_TeleoperatorServicer_to_server(self.servicer, self.server)
        if self.use_ssl:
            if not self.ssl_key_path or not self.ssl_cert_path:
                raise ValueError("Both `ssl_key_path` and `ssl_cert_path` must be provided when `use_ssl` is True.")
            with open(self.ssl_key_path, "rb") as f:
                private_key = f.read()
            with open(self.ssl_cert_path, "rb") as f:
                certificate = f.read()
            self.server.add_secure_port(self.address, grpc.ssl_server_credentials([(private_key, certificate)]))
        else:
            self.server.add_insecure_port(self.address)

        self._lock = Lock()
        self._is_running = False
        self.grace_period = config.server_grace_period_s

    def start(self):
        """Starts the gRPC server."""
        with self._lock:
            if self._is_running:
                logger.warning("LeaderServer is already running.")
                return
            self.server.start()
            self._is_running = True
            logger.info(f"LeaderServer started on {self.address}.")

    def stop(self):
        """Stops the gRPC server."""
        with self._lock:
            if not self._is_running:
                logger.warning("LeaderServer is not running.")
                return
            logger.info("Stopping LeaderServer...")
            self.server.stop(self.grace_period)
            self._is_running = False
            logger.info("LeaderServer stopped successfully.")
