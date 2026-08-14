from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import grpc
from concurrent import futures
from lerobot_robot_grpc.protos import device_pb2_grpc

logger = logging.getLogger(__name__)

@dataclass
class FollowerServerConfig:
    """Configuration for the FollowerServer."""
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

class FollowerServicer(device_pb2_grpc.RobotServicer, ABC):
    """Abstract base class for the Follower gRPC service."""

    @abstractmethod
    def GetInfo(self, request, context):
        """Gets all feature schemas (observation + action + feedback) in one call."""
        pass

    @abstractmethod
    def Connect(self, request, context):
        """Connects to the robot."""
        pass

    @abstractmethod
    def Calibrate(self, request, context):
        """Calibrates the robot."""
        pass

    @abstractmethod
    def CalibrateDone(self, request, context):
        """Indicates that calibration is done."""
        pass

    @abstractmethod
    def Disconnect(self, request, context):
        """Disconnects from the robot."""
        pass

    @abstractmethod
    def GetObservation(self, request, context):
        """Gets the observation from the robot."""
        pass

    @abstractmethod
    def SendAction(self, request, context):
        """Sends an action to the robot (unary: request=Action, returns executed Action)."""
        pass

    @abstractmethod
    def GetFeedback(self, request, context):
        """Gets feedback from the robot."""
        pass

    @abstractmethod
    def GetStatus(self, request, context):
        """Gets the status of the robot."""
        pass

    def SetReference(self, request, context):
        """Re-latch the pose_delta base pose (T_zero) at the current FK.

        Default implementation: not supported.  Joint-space followers have no
        Cartesian reference to latch, so they must fail loudly rather than
        pretend success.  ``MuJoCoSO101Servicer`` (pose_delta mode) overrides
        this to re-lock ``T_zero`` from the current gripperframe FK — the
        clutch re-engage contract (wayfinder #10): current hand = current arm.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(
            "SetReference is not supported by this follower (no pose_delta base pose)"
        )
        from google.protobuf.empty_pb2 import Empty

        return Empty()

class FollowerServer:
    """gRPC server for the Follower robot."""

    def __init__(self, config: FollowerServerConfig, servicer: FollowerServicer):
        self.config = config
        self.address = config.address
        self.use_ssl = config.use_ssl
        self.ssl_cert_path = config.ssl_cert_path
        self.ssl_key_path = config.ssl_key_path
        self.servicer = servicer
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        device_pb2_grpc.add_RobotServicer_to_server(self.servicer, self.server)
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
                logger.warning("FollowerServer is already running.")
                return
            self.server.start()
            self._is_running = True
            logger.info(f"FollowerServer started on {self.address}.")

    def stop(self):
        """Stops the gRPC server."""
        with self._lock:
            if not self._is_running:
                logger.warning("FollowerServer is not running.")
                return
            logger.info("Stopping FollowerServer...")
            self.server.stop(self.grace_period)
            self._is_running = False
            logger.info("FollowerServer stopped successfully.")