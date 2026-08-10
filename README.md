# lerobot_robot_grpc

gRPC **follower** + **leader** devices for [lerobot](https://github.com/huggingface/lerobot), packaged as a standalone plugin so it no longer needs to live inside the lerobot source tree.

The repository is named `lerobot_grpc`; the installable distribution / import package is `lerobot_robot_grpc` so that lerobot's plugin auto-discovery (which keys on the `lerobot_robot_` name prefix) picks it up with **zero CLI flags**.

## What it provides

| Device type | Base class | Role |
|---|---|---|
| `grpc_follower` | `lerobot.robots.robot.Robot` | gRPC **client** on the recording/training machine, talking to a remote follower |
| `grpc_leader`   | `lerobot.teleoperators.teleoperator.Teleoperator` | gRPC **client** for a remote leader (teleop) |

Plus the **server** side that wraps real SO101 hardware and serves it over gRPC (`follower/so101_follower_server.py`, `leader/so101_leader_server.py`).

## Install

### Prerequisites

- Python >= 3.12
- lerobot >= 0.6.1, < 0.7
- A conda/venv environment with lerobot already installed

### Option A: Full install (client + server on same machine)

Typical for a development machine that both drives hardware and runs recording:

```bash
pip install -e ".[all]"
```

This pulls in:
- `lerobot[grpcio-dep]` — gRPC core (grpcio, protobuf)
- `lerobot[dataset]` — recording (datasets, pyarrow, av, torchcodec)
- `lerobot[hardware]` — keyboard controls (**pynput**), pyserial, deepdiff
- `lerobot[viz]` — real-time visualization (rerun-sdk, foxglove-sdk)
- `lerobot[feetech]` — Feetech motor SDK for SO101 hardware

### Option B: Client only (recording/training machine)

```bash
pip install -e ".[client]"
```

### Option C: Server only (robot-side machine)

```bash
pip install -e ".[server]"
```

### Windows notes

- **Keyboard controls**: `pynput` (from `lerobot[hardware]`) is required for interactive recording controls (Right=next episode, Left=re-record, Esc/q=quit, n=next). Without it, `TerminalKeyListener` cannot work on Windows because it needs POSIX `termios`.
- **Video encoding**: `torchcodec` may show a DLL loading warning on Windows if FFmpeg shared libraries are missing. This is harmless — lerobot automatically falls back to `pyav` for video encoding.
- **rerun visualization**: `rerun-sdk` (from `lerobot[viz]`) provides real-time data visualization during recording (`--display_data=true`).

## Development

### Running tests

Install the dev extra on top of your normal environment, then run pytest:

```bash
pip install -e ".[dev]"
pytest tests/
```

Current tests cover the leader server's bus-lock safety mechanism (`_acquire_bus` / `_release_bus` / `_bus_watchdog` / `_reset_bus_lock_state`).

## Video streaming (H.264)

Cameras are streamed over the existing `GetObservation` RPC (no extra RPC needed).
The server keeps a per-stream H.264 encoder and the client a per-camera decoder,
so P-frames compress across snapshots instead of shipping a standalone JPEG per
frame (order-of-magnitude bandwidth reduction for static scenes). Wire details:

- One `OneFeature` message = one H.264 access unit (Annex-B, SPS/PPS embedded in
  every keyframe); a fresh decoder can sync at any keyframe.
- The server opens every stream with a keyframe and inserts one every ~2 s
  (`keyint`); if the connection drops, the client reconnects and resyncs.
- The client keeps a **persistent** `GetObservation` stream open in a background
  thread; `get_observation()` returns the latest frame. Servers from this repo
  behave the same way (streams until the client disconnects).
- `GetObservation` is now one continuous call per client — per-call "poll" style
  clients (from older versions of this repo) will time out against new servers.
  Upgrade client + server together.
- Depth maps (`<cam>_depth`) stay RAW; `--camera_encoding=jpeg` on
  `serve_so101_follower.py` restores per-frame JPEG for all cameras.
- Requires PyAV (`av`, from `lerobot[dataset]` / `lerobot[feetech]`+`av`); if it's
  missing the server falls back to JPEG automatically, and the client errors out
  clearly when an H.264 feature arrives.

## Quick reference — gRPC device parameters

Both `grpc_follower` and `grpc_leader` accept these config parameters via CLI flags:

| Parameter | Default | Description |
|---|---|---|
| `--robot.address` / `--teleop.address` | `localhost:5555` | gRPC server `host:port` |
| `--robot.id` / `--teleop.id` | — | Device ID (used for calibration file naming) |
| `--robot.force_recalibrate` | `false` | Bypass cached calibration and force recalibration |
| `--robot.need_warmup` | `true` | Verify feature schema on connect |
| `--robot.connect_timeout_s` | `5.0` | gRPC connect timeout |
| `--robot.data_timeout_s` | `5.0` | gRPC data (observation/action) timeout |
| `--robot.teleop_stats` | `false` | Live bandwidth/latency monitor + end-of-session summary (follower only) |
| `--robot.stats_interval_s` | `1.0` | Monitor refresh interval (s) |

## Usage

> **Line continuation**: Examples below use PowerShell syntax (`` ` ``). On bash/Linux, replace `` ` `` with `\`.

### 1. Start the servers (robot-side)

Open **two terminals** — one for the follower, one for the leader:

```powershell
# Terminal 1 — Follower server (SO101 arm on COM6, USB camera at index 0)
python examples/serve_so101_follower.py `
    --robot.port=COM6 `
    --robot.id=follower `
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --address=0.0.0.0:5555
```

> 相机参数是 draccus 的整体字典字符串（内联 YAML/JSON 均可，`JSON ⊆ YAML`），不能拆成 `--robot.cameras.top.type=...` 扁平键。
> Windows PowerShell 5.1 会把调用外部程序时参数内嵌的 `"` 剥掉，但 YAML 写法不需要字面引号，双引号包裹整段值即可（`{...}` 在引号内是普通字符）。
> 相机名可任取（示例用 `top`）；配置必须有 `type`（如 `opencv`）及 lerobot 要求的 `width`/`height`/`fps`。

```powershell
# Terminal 2 — Leader server (SO101 leader arm on COM4)
python examples/serve_so101_leader.py `
    --robot.port=COM4 `
    --robot.id=leader `
    --address=0.0.0.0:5556
```

The follower server listens on `0.0.0.0:5555`; the leader server on `0.0.0.0:5556`. Pass `--address=0.0.0.0:<port>` so clients on other machines can reach it (the default binds all interfaces; a specific IP also works but `0.0.0.0` is simplest).

To find the correct serial port: `lerobot-find-port`. To find camera indices: `lerobot-find-cameras`.

### 2. Calibrate — `lerobot-calibrate`

Calibration is needed **once** per device (first use, after hardware change, or after motor replacement). The result is cached as a JSON file so subsequent sessions skip calibration automatically.

```powershell
# Calibrate the follower arm (6 motors: shoulder_pan, shoulder_lift, elbow_flex, wrist_pitch, wrist_roll, gripper)
lerobot-calibrate `
    --robot.type=grpc_follower `
    --robot.address=127.0.0.1:5555 `
    --robot.id=follower
```

```powershell
# Calibrate the leader arm
lerobot-calibrate `
    --teleop.type=grpc_leader `
    --teleop.address=127.0.0.1:5556 `
    --teleop.id=leader
```

Force recalibration (ignore the cached file):

```powershell
lerobot-calibrate `
    --robot.type=grpc_follower `
    --robot.address=127.0.0.1:5555 `
    --robot.id=follower `
    --robot.force_recalibrate=true
```

Calibration files are stored at:
- Follower: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json`
- Leader: `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader.json`

### 3. Teleoperate — `lerobot-teleoperate`

Drive the follower arm in real time using the leader arm:

```powershell
lerobot-teleoperate `
    --robot.type=grpc_follower --robot.address=127.0.0.1:5555 --robot.id=follower `
    --teleop.type=grpc_leader  --teleop.address=127.0.0.1:5556 --teleop.id=leader `
    --teleop_time_s=60 `
    --display_data=true
```

- `--teleop_time_s=60` — run for 60 seconds (omit for unlimited until Ctrl+C).
- `--display_data=true` — open a **rerun** viewer showing joint positions in real time.

To watch bandwidth/latency live and get a session summary when it ends, add
`--robot.teleop_stats=true` (no code changes — the flag works with any lerobot
CLI that instantiates the follower):

```powershell
lerobot-teleoperate `
    --robot.type=grpc_follower --robot.address=127.0.0.1:5555 --robot.id=follower `
    --teleop.type=grpc_leader  --teleop.address=127.0.0.1:5556 --teleop.id=leader `
    --teleop_time_s=60 `
    --robot.teleop_stats=true --robot.stats_interval_s=1.0
```

Live output is a multi-line block repainted in place (header + per-feature
Mbps/fps, latency avg/p95/max, action RTT) — same mechanism as the calibration
pos table, so nothing scrolls; on disconnect it prints a full per-feature
summary table. It composes with lerobot's own in-place displays: with
`--display_data=true` the stats block stacks above lerobot's NORM table and
both repaint in place (any one-off interleave from the concurrent writers
self-heals on the next refresh). A standalone demo without hardware:

```powershell
python examples/teleop_monitor_demo.py --seconds 10 --interval 1.0
```

> **Latency accuracy (cross-host):** the `latency` metric is computed as an absolute
> wall-clock delta from the server's `produce_ts` (stamped at sample time) to local
> receive time. It is **accurate only when client and server share a clock** — i.e. the
> same host, or hosts with synced clocks (NTP/PTP). On **cross-host, unsynced** setups the
> delta can be negative or wildly off; the monitor detects this and **falls back to a
> relative-to-first-frame jitter** (still labeled `latency` in the table) so the trend
> stays meaningful, and logs a one-time warning. True absolute latency for cross-host
> deployments requires RTT-based clock-offset estimation (planned, not yet implemented).

#### One-shot dual-arm teleop — `examples/teleop_so101.py`

No lerobot CLI needed: this script starts **both** gRPC servers in-process
(follower arm on `--follower-port`, leader arm on `--leader-port`), connects the
`GRPCFollower` / `GRPCLeader` clients, and streams leader joint actions to the
follower at `--rate` Hz until Ctrl+C (or `--time` seconds):

```powershell
python examples/teleop_so101.py `
    --follower-port=COM5 --leader-port=COM7 `
    --cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --teleop-stats --time 60
```

`--cameras` is the same inline-YAML dict as `serve_so101_follower.py`
(omit for a bare joint-only teleop); `--teleop-stats` adds the live
bandwidth/latency monitor + end-of-session summary; arms auto-calibrate on
connect (move them through their full range when prompted, or pass `--no-calibrate`).

### 4. Record a dataset — `lerobot-record`

Collect teleoperation episodes for training:

```powershell
lerobot-record `
    --robot.type=grpc_follower --robot.address=127.0.0.1:5555 --robot.id=follower `
    --teleop.type=grpc_leader  --teleop.address=127.0.0.1:5556 --teleop.id=leader `
    --dataset.repo_id=wsss/so101_grpc_test `
    --dataset.single_task="Pick up the cup" `
    --dataset.num_episodes=5 `
    --dataset.episode_time_s=30 `
    --dataset.push_to_hub=false `
    --display_data=true
```

Key parameters:

| Parameter | Description |
|---|---|
| `--dataset.repo_id` | HuggingFace repo ID (`user/dataset_name`); dataset is saved locally under `~/.cache/huggingface/lerobot/` |
| `--dataset.single_task` | Text label for every frame in this dataset |
| `--dataset.num_episodes` | Number of episodes to record before stopping |
| `--dataset.episode_time_s` | Duration of each episode in seconds (30 fps default) |
| `--dataset.push_to_hub` | `false` = local only; `true` = upload to HuggingFace Hub |
| `--display_data=true` | Open a **rerun** viewer (joint state + camera + action streams) |

**Keyboard controls during recording** (requires `pynput` installed via `lerobot[hardware]`):

| Key | Action |
|---|---|
| Right arrow / `n` | End current episode early, move to next |
| Left arrow / `r` | Re-record current episode |
| Esc / `q` | Stop recording |

Datasets are saved to: `~/.cache/huggingface/lerobot/<repo_id>_<timestamp>/`

### 5. Replay a dataset — `lerobot-replay`

Play back a recorded dataset on the follower arm:

```powershell
lerobot-replay `
    --robot.type=grpc_follower --robot.address=127.0.0.1:5555 --robot.id=follower `
    --dataset.repo_id=wsss/so101_grpc_test_20260806_001214 `
    --dataset.episode=0
```

- `--dataset.repo_id` — use the **stamped** dataset name (with `_<timestamp>` suffix; check `~/.cache/huggingface/lerobot/wsss/` for exact names).
- `--dataset.episode=0` — which episode to replay (0-indexed).

## Extending — adding new hardware

The client (`grpc_follower` / `grpc_leader`) is hardware-agnostic; the feature schema
is negotiated with the server at runtime. So supporting a new robot or input device
means writing **one new server file**, not touching the client or the proto.

See **[docs/extending.md](docs/extending.md)** for the full guide, including a worked
Quest 3 → Unitree G1 example (hand-pose retargeting, dual-arm, companion-PC leader).

## Why the package isn't named `lerobot_grpc`

lerobot discovers third-party plugins by scanning installed distributions whose name starts with one of `lerobot_robot_` / `lerobot_teleoperator_` / `lerobot_camera_` / `lerobot_policy_` / `lerobot_env_`, then doing `importlib.import_module(dist_name)`. There is no neutral prefix, so a distribution named `lerobot_grpc` would **not** be auto-imported and would require `--robot.discover_packages_path=lerobot_grpc` on every invocation. Naming the dist `lerobot_robot_grpc` buys zero-flag UX; the import name is internal plumbing that end users never type (the git URL, the `lerobot-record` CLI, and the device `--type` values are all they see). A single import registers both the follower and the leader because registration is a side-effect of import, keyed only on the package being imported — not on which kind it registers.
