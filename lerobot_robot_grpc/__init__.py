"""lerobot_robot_grpc — gRPC follower/leader devices for lerobot.

This package is auto-discovered by lerobot's CLI because the distribution name
matches the ``lerobot_robot_`` prefix (see
``lerobot.utils.import_utils.register_third_party_plugins``). Importing this
top-level package registers two device types with lerobot's ChoiceRegistry:

- ``grpc_follower``  — a :class:`lerobot.robots.robot.Robot` (remote follower)
- ``grpc_leader``    — a :class:`lerobot.teleoperators.teleoperator.Teleoperator`

Only the lightweight config modules are imported here so that registering the
choices does not pull grpcio / opencv / the device implementations into every
CLI process. The device classes themselves are resolved lazily by lerobot's
``make_device_from_device_class`` reflective fallback when a config is
instantiated.
"""

from .follower.config_grpc import GRPCFollowerConfig
from .leader.config_grpc import GRPCLeaderConfig

__all__ = ["GRPCFollowerConfig", "GRPCLeaderConfig"]
