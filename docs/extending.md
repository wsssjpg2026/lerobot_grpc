# Extending `lerobot_robot_grpc` — adding a new follower or leader

This guide is for anyone who wants to drive **new hardware** through this package's
gRPC layer. Maybe you have a different arm, a VR controller, a humanoid — the steps
are the same.

> Read the [README](../README.md) first for the overall package layout and the
> auto-discovery mechanism.

---

## 1. The mental model (read this first)

The architecture has a deliberate asymmetry. Understand it once and the rest is
mechanical:

```
                ┌─────────────── recording / training machine ───────────────┐
                │  lerobot-record                                             │
                │    --robot.type=grpc_follower   --robot.address=<host>:5555 │
                │    --teleop.type=grpc_leader   --teleop.address=<host>:5556 │
                │          │ GENERIC                            │ GENERIC    │
                └──────────┼────────────────────────────────────┼────────────┘
                           │ gRPC (device.proto)                 │ gRPC
              ┌────────────▼─────────────┐         ┌─────────────▼──────────────┐
              │  YOUR follower server    │         │  YOUR leader server        │
              │  (wraps real hardware)   │         │  (wraps input device)      │
              └──────────────────────────┘         └────────────────────────────┘
```

- **The client is generic.** `GRPCFollower` / `GRPCLeader` discover the remote
  device's feature schema at runtime via the `Get*FeatureInfo` RPCs. They do **not**
  know — or need to know — whether the other end is a SO-101, a Unitree G1, or a
  Meta Quest 3. You will **never** touch the client to support new hardware.
- **Each piece of hardware gets exactly one server-side *servicer*** — a subclass of
  `FollowerServicer` (for a robot you drive) or `LeaderServicer` (for an input device
  you read). That servicer wraps the hardware and exposes it over `device.proto`.

So: **supporting new hardware = writing one new server file.** The proto, the base
classes, the client, and lerobot's plugin registration do not change.

### The one semantic difference to internalise

| | Follower (`Robot` service) | Leader (`Teleoperator` service) |
|---|---|---|
| Produces | observations **and** feedback | **actions** (human input) |
| Consumes | **actions** (joint commands) | feedback (mirrored state) |
| Extra RPC | — | `SetReference` (zero / origin pose) |

A follower *receives* `SendAction`; a leader *answers* `GetAction`. Both advertise
`action_features`, but the direction of flow is opposite.

---

## 2. What you do NOT touch

- `device.proto` and the generated `device_pb2*.py` — the wire contract is generic.
- `follower/follower_server.py`, `leader/leader_server.py` — the abstract bases and the `*Server` runners.
- `follower/grpc_follower.py`, `leader/grpc_leader.py` — the generic clients.
- `lerobot_robot_grpc/__init__.py` — plugin registration. It already registers
  `grpc_follower` / `grpc_leader`; new hardware does **not** add new client types.
- The client config (`GRPCFollowerConfig` / `GRPCLeaderConfig`).

If you find yourself editing any of these, stop — you're probably trying to put
hardware-specific logic in the wrong layer.

---

## 3. Adding a new follower (robot you drive)

**Reference implementation:** `follower/so101_follower_server.py` (`SO101FollowerServicer`
wrapping lerobot's `SO101Follower`). Skim it alongside this section.

### 3.1 Subclass `FollowerServicer`

Create `follower/<your_robot>_follower_server.py`. Your servicer wraps a lerobot
robot instance and implements the 11 RPCs of the `Robot` service:

```python
from google.protobuf.empty_pb2 import Empty

from lerobot_robot_grpc.follower.follower_server import FollowerServicer
from lerobot_robot_grpc.follower.utils import encode_feature, load_feature
from lerobot_robot_grpc.protos import device_pb2

# The lerobot device you wrap:
from lerobot.robots.<your_robot>.config_<your_robot> import YourRobotConfig
from lerobot.robots.<your_robot>.<your_robot> import YourRobot


class YourRobotFollowerServicer(FollowerServicer):
    def __init__(self, robot: YourRobot):
        self.robot = robot

    # --- feature introspection: tell the client what this device speaks --------
    def GetObservationFeatureInfo(self, request, context):
        return self._encode_feature_info(self.robot.observation_features)

    def GetActionFeatureInfo(self, request, context):
        return self._encode_feature_info(self.robot.action_features)

    def GetFeedbackFeatureInfo(self, request, context):
        return self._encode_feature_info(self.robot.action_features)  # or a distinct set

    # --- lifecycle -------------------------------------------------------------
    def Connect(self, request, context): ...
    def Calibrate(self, request, context): ...      # see §6 on calibration
    def CalibrateDone(self, request, context): ...
    def Disconnect(self, request, context): ...

    # --- data flow -------------------------------------------------------------
    def GetObservation(self, request, context): ...  # encode obs -> stream OneFeature
    def SendAction(self, request_iterator, context): ...  # decode stream -> robot.send_action
    def GetFeedback(self, request, context): ...

    def GetStatus(self, request, context): ...
```

The easiest path: **copy `follower/so101_follower_server.py` verbatim**, then swap the
wrapped class and the bits that are SO-101-specific (the calibration thread,
`_calibrate_done` event). The `_encode_feature_info` helper and the
`encode_feature` / `load_feature` round-trip are reusable as-is.

### 3.2 Implement the feature-info encoder

`device.proto` describes every feature as a `OneFeatureInfo { key, type, shape, encoding, ... }`.
The SO-101 server's `_encode_feature_info` is the template — copy it. The rule:

- **Scalar** (e.g. `shoulder.pos`, a joint angle) → `shape=(1,1,1)`, `encoding=RAW`,
  `type` = the matching `DataType` (`float`→`FLOAT32`, `int`→`INT32`).
- **Image** (RGB camera) → `shape=(H,W,3)`, `encoding=JPEG`, `type=UINT8`.
- **Depth / raw array** → `shape=(H,W,C)`, `encoding=RAW` (sent as raw little-endian bytes;
  use `UINT16` for mm depth maps so JPEG doesn't truncate to 8-bit).

### 3.3 Wire `SendAction` and `GetObservation`

```python
def GetObservation(self, request, context):
    obs = self.robot.get_observation()                              # dict[str, scalar|ndarray]
    return encode_feature(self._feature_info_for(obs), obs)         # -> stream OneFeature

def SendAction(self, request_iterator, context):
    action = {}
    for feat in request_iterator:
        load_feature(feat, self._action_feature_info, action)
    self.robot.send_action(action)                                  # drive the hardware
    return Empty()
```

`encode_feature` / `load_feature` (in `follower/utils.py`) handle the scalar/array/image
bytes for you. You only supply the feature-info dict and the raw Python values.

---

## 4. Adding a new leader (input device you read)

**Reference implementation:** `leader/so101_leader_server.py` (`SO101LeaderServicer`
wrapping lerobot's `SO101Leader`).

It mirrors the follower, with the direction flipped and one extra RPC:

- Implement `GetAction` (the human's current input) instead of `SendAction`.
- Implement `SendFeedback(request_iterator)` (consume mirrored state) instead of `GetFeedback`.
- Implement **`SetReference`** — define the zero/origin pose, for relative control.
  The SO-101 leader uses it to subtract a reference so the operator can re-center
  without the robot jumping. VR controllers (e.g. a Meta Quest 3) will want the same.

Everything else (feature-info encoder, lifecycle, status) is the same shape as the follower.

```python
from lerobot_robot_grpc.leader.leader_server import LeaderServicer, LeaderServer, LeaderServerConfig

class YourLeaderServicer(LeaderServicer):
    def GetAction(self, request, context):
        action = self.device.read_current_input()        # e.g. hand poses, joint angles
        return encode_feature(self._action_feature_info, action)

    def SetReference(self, request, context):
        self.device.set_origin_to_current_state()
        return Empty()
```

---

## 5. The feature-schema contract (the one thing that must line up)

On `Connect`, the client asks the server for three schemas: **observation**,
**action**, and **feedback**. There is no static type shared across machines —
the schema is whatever each server reports at runtime.

This means: **the action schema the leader produces must match the action schema the
follower consumes** (same keys, same types). `lerobot-record` pumps the leader's
`get_action()` dict straight into the follower's `send_action()`. Mismatched keys →
`KeyError` at runtime.

- When both sides wrap the **same** kinematics (e.g. SO-101 arm → SO-101 arm), each
  server just exposes its own `robot.action_features` and they naturally agree.
- When the two sides are **kinematically different** (e.g. a VR hand pose driving a
  humanoid arm), factor the shared schema into a small module both servers import,
  so it can't drift:

  ```
  lerobot_robot_grpc/hand_pose_schema.py   # ACTION_KEYS, build_action_feature_info(), ...
  ```

See §7 (the virtual-action-space pattern) for why this case needs extra care.

---

## 6. Calibration

`Calibrate` / `CalibrateDone` are intentionally open-ended — each hardware owns its
own procedure. Three common shapes:

- **Manual range-of-motion recording** (SO-101, feeble servos): `Calibrate` starts a
  background thread that records min/max per joint until the client sends
  `CalibrateDone`. See `follower/so101_follower_server.py` — including the `_calibrate_done`
  `threading.Event` and the non-blocking `_calibration_lock` that rejects
  observation/action access while moving.

> **Bus-lock robustness** (both SO-101 servers): a `bus_call_timeout_s` watchdog
> force-releases the lock (and dumps the stuck thread's stack) when a non-calibration
> bus call appears wedged — e.g. a dead serial port, or a feetech SDK packet timeout
> suppressed by a wall-clock step (the servers patch the port handler to monotonic
> time, and bound `readPort` so a garbage serial stream can't hold the lock forever).
> The gRPC clients likewise retry `SendAction`/`GetAction` briefly while the robot is
> busy instead of failing the teleop loop. New servers may copy these patterns.
- **Absolute encoders / no calibration needed** (most humanoids): `Calibrate` just
  returns `CalibrationStatus.CALIBRATED`.
- **Define an origin** (VR leaders): not joint calibration at all — `SetReference`
  captures the current pose as zero for relative control.

`GetStatus` should report `FATAL` if a calibration thread failed (the client raises
`DeviceNotConnectedError` on seeing it), so the operator is never silently driving
against a half-calibrated robot.

---

## 7. Pattern: virtual action space (retargeting on the server)

Sometimes your follower's *native* action space is **not** what the operator produces.
Example: a Meta Quest 3 yields **hand poses** (translation + quaternion per hand); a
Unitree G1 expects **arm joint angles**. You chose (per the project plan) to do the
retargeting **on the follower server**, reusing lerobot's `g1_kinematics` IK.

In that case the follower server advertises a **virtual** action space — the hand-pose
schema — that differs from `self.robot.action_features` (which is joint angles):

```python
class G1FollowerServicer(FollowerServicer):
    def GetActionFeatureInfo(self, request, context):
        return build_hand_pose_feature_info()           # ← NOT self.robot.action_features

    def SendAction(self, request_iterator, context):
        hand_poses = {}
        for feat in request_iterator:
            load_feature(feat, HAND_POSE_FEATURE_INFO, hand_poses)
        joint_targets = self.retarget(hand_poses)       # pose-stream -> joint angles (IK)
        self.robot.send_action(joint_targets)            # drive G1 in its native space
        return Empty()
```

The contract from §5 still holds: the leader server imports the **same**
`hand_pose_schema` module, so both sides agree on keys even though only the follower
actually understands what the poses mean.

> When to use this pattern: the follower's native action differs from the operator's
> input, and you want the leader to stay hardware-agnostic. The cost is a retargeting
> solver living on the follower server (a real research item — streaming VR
> retargeting wants warm-starting / smoothing, which single-config IK may not give you).

---

## 8. Packaging and dependencies

Put your SDK behind a dedicated optional dependency so client-only users don't pay for
it. In `pyproject.toml`:

```toml
[project.optional-dependencies]
server  = ["lerobot[feetech]>=0.6.1,<0.7"]   # SO-101 server (already present)
unitree = ["lerobot[unitree_g1]>=0.6.1,<0.7"] # your G1 server
quest3  = ["pyopenxr>=..."]                    # your VR leader server
```

Guard the SDK import at the top of your server module (exactly how the SO-101 server
leans on the `feetech` extra) so that importing the package for plugin registration
doesn't drag the SDK into every CLI process. The client side is unaffected — its base
dependency stays `lerobot[grpcio-dep]`.

---

## 9. Launching your server

`FollowerServer` / `LeaderServer` give you a runner that takes a config + your servicer.
There is no built-in console script per hardware (yet); launch from a small script:

```python
import logging, time
from lerobot_robot_grpc.follower.follower_server import FollowerServer, FollowerServerConfig
# your servicer:
from lerobot_robot_grpc.follower.your_robot_follower_server import YourRobotFollowerServicer
# the lerobot device you wrap:
from lerobot.robots.your_robot.config_your_robot import YourRobotConfig
from lerobot.robots.your_robot.your_robot import YourRobot

logging.basicConfig(level=logging.INFO)

robot = YourRobot(YourRobotConfig(...))
robot.connect()
servicer = YourRobotFollowerServicer(robot)
server = FollowerServer(FollowerServerConfig(address="0.0.0.0:5555"), servicer)
server.start()
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    server.stop()
    robot.disconnect()
```

Run this on the robot-side machine. The recording machine then just does:

```bash
lerobot-record --robot.type=grpc_follower --robot.address=<robot-host>:5555 ...
```

---

## 10. Worked example: Meta Quest 3 → Unitree G1 (dual arm)

A concrete instance of everything above, mapped onto a real target:

| Side | New file | Wraps | Action space advertised | Notes |
|---|---|---|---|---|
| Follower | `follower/g1_follower_server.py` | lerobot `UnitreeG1` (→ ZMQ → its DDS bridge, untouched) | **hand poses** (virtual) | `SendAction` retargets pose→joints via `g1_kinematics`, then calls `UnitreeG1.send_action` |
| Leader | `leader/quest3_leader_server.py` | OpenXR / Meta Movement SDK on a companion PC | **hand poses** (produced) | `GetAction` streams current hand poses; `SetReference` zeros the origin |

Shared: `hand_pose_schema.py` (keys for `left_hand.position.{x,y,z}`,
`left_hand.rotation.{qx,qy,qz,qw}`, same for right). Both servers import it.

Gotchas specific to this example:
- lerobot's `UnitreeG1` only takes **joint** actions natively (`{joint}.q`); its
  `G1_29_ArmIK` is currently used for gravity-comp torques, not pose→joint retargeting
  in the action path. So the G1 server must own the retargeting step — confirm the IK
  solver is suitable for streaming VR input, or bring in a retargeting library.
- The Quest runs Android; the gRPC leader server runs on a **companion PC** tethered
  via USB / Air Link, not on the headset.
- G1 has absolute encoders → `Calibrate` can be a no-op returning `CALIBRATED`.

---

## 11. Checklist

- [ ] One new file: `follower/<robot>_follower_server.py` **or** `leader/<device>_leader_server.py`.
- [ ] Subclass `FollowerServicer` / `LeaderServicer`; implement every abstract RPC.
- [ ] Copy `_encode_feature_info` from the SO-101 server; adapt the feature set.
- [ ] Reuse `encode_feature` / `load_feature` for the byte round-trip — don't hand-roll.
- [ ] If action spaces differ across sides, factor a shared schema module.
- [ ] Add an optional dependency in `pyproject.toml`; guard the SDK import.
- [ ] Add a small launch script (or a `[project.scripts]` entry) on the device machine.
- [ ] Client side: **unchanged**. `--robot.type=grpc_follower` / `--teleop.type=grpc_leader` just works.
