# lerobot_robot_grpc

gRPC **follower** + **leader** devices for [lerobot](https://github.com/huggingface/lerobot), packaged as a standalone plugin so it no longer needs to live inside the lerobot source tree.

The repository is named `lerobot_grpc`; the installable distribution / import package is `lerobot_robot_grpc` so that lerobot's plugin auto-discovery (which keys on the `lerobot_robot_` name prefix) picks it up with **zero CLI flags**.

## What it provides

| Device type | Base class | Role |
|---|---|---|
| `grpc_follower` | `lerobot.robots.robot.Robot` | gRPC **client** on the recording/training machine, talking to a remote follower |
| `grpc_leader`   | `lerobot.teleoperators.teleoperator.Teleoperator` | gRPC **client** for a remote leader (teleop) |

Plus the **server** side that wraps real SO101 hardware and serves it over gRPC (`so101_follower_server`, `so101_leader_server`).

## Install

```bash
# Client only (recording/training machine)
pip install "lerobot[grpcio-dep]>=0.6.1,<0.7"
pip install git+https://github.com/<your-org>/lerobot_grpc

# Robot-side machine that drives real SO101 hardware
pip install "lerobot_robot_grpc[server]" git+https://github.com/<your-org>/lerobot_grpc
```

## Usage

Because the distribution name starts with `lerobot_robot_`, lerobot auto-imports this package at every CLI startup, registering both `grpc_follower` and `grpc_leader`. No `--discover_packages_path` flag needed:

```bash
lerobot-record \
    --robot.type=grpc_follower --robot.address=<follower-host>:5555 \
    --teleop.type=grpc_leader  --teleop.address=<leader-host>:5555 \
    ...
```

## Extending — adding new hardware

The client (`grpc_follower` / `grpc_leader`) is hardware-agnostic; the feature schema
is negotiated with the server at runtime. So supporting a new robot or input device
means writing **one new server file**, not touching the client or the proto.

See **[docs/extending.md](docs/extending.md)** for the full guide, including a worked
Quest 3 → Unitree G1 example (hand-pose retargeting, dual-arm, companion-PC leader).

## Why the package isn't named `lerobot_grpc`

lerobot discovers third-party plugins by scanning installed distributions whose name starts with one of `lerobot_robot_` / `lerobot_teleoperator_` / `lerobot_camera_` / `lerobot_policy_` / `lerobot_env_`, then doing `importlib.import_module(dist_name)`. There is no neutral prefix, so a distribution named `lerobot_grpc` would **not** be auto-imported and would require `--robot.discover_packages_path=lerobot_grpc` on every invocation. Naming the dist `lerobot_robot_grpc` buys zero-flag UX; the import name is internal plumbing that end users never type (the git URL, the `lerobot-record` CLI, and the device `--type` values are all they see). A single import registers both the follower and the leader because registration is a side-effect of import, keyed only on the package being imported — not on which kind it registers.
