"""Response metrics over one recorded sequence run.

The observation stream carries joint angles only — there is no EE-pose
feature — so ground truth comes from test-side forward kinematics on a
SEPARATE MuJoCo model built from ``assets/so101/scene.xml`` (the same
independent-model pattern as ``tests/test_real_pose_delta.py``; no placo
dependency).

Metrics (all recorded, none gated — thresholds are B-3 placeholders):

- tracking error vs the composed target ``T_ref @ Δ(t)``: per-axis and
  norm RMSE / P95 / max in mm, rotation error in deg;
- end-to-end lag via normalized cross-correlation on the dominant motion
  axis (ms);
- smoothness: EE jerk RMS (m/s^3) on a uniform 100 Hz resample;
- stream health: observation frame rate, drop rate vs nominal 50 Hz,
  feature latency and SendAction RTT (mean / P95) from ``TeleopStats``;
- gripper step response time (63.2 % crossing, ms).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lerobot.utils.rotation import Rotation

from .backends import XML_PATH

# Nominal server observation rate (GetObservation loops at ~50 Hz) and the
# action rate the harness sends at.
OBS_NOMINAL_HZ = 50.0
ACTION_HZ = 30.0

BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
# MuJoCo gripper actuator range (so101_new_calib.xml), mirrors
# mujoco_follower_server.norm_value_to_rad.
_GRIP_RAD_MIN, _GRIP_RAD_MAX = -0.17453, 1.74533
_GRIP_MAX_MM = 60.0


class TestFK:
    """Independent FK oracle on assets/so101/scene.xml (joints -> EE pose)."""

    def __init__(self, xml_path: Path | str = XML_PATH):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )

    def qpos_from_obs(self, obs: dict) -> np.ndarray:
        """lerobot-normalised observation -> full model qpos (6, radians)."""
        qpos = np.zeros(self.model.nq)
        for i, joint in enumerate(BODY_JOINTS):
            qpos[i] = math.radians(float(obs[f"{joint}.pos"]))
        grip = float(obs["gripper.pos"])
        qpos[5] = (grip / 100.0) * (_GRIP_RAD_MAX - _GRIP_RAD_MIN) + _GRIP_RAD_MIN
        return qpos

    def pose(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """(position (3,), rotation matrix (3,3)) of gripperframe."""
        qpos = self.qpos_from_obs(obs)
        self.data.qpos[:] = qpos
        self._mj.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, mat

    def joint_limits_deg(self) -> np.ndarray:
        """Model joint ranges for the 5 body joints, degrees."""
        return np.degrees(self.model.jnt_range[: len(BODY_JOINTS)].copy())


# ---------------------------------------------------------------------------
# Per-sample record helpers (shared shapes; the harness fills them)
# ---------------------------------------------------------------------------


class Sample:
    __slots__ = ("t", "obs", "ee", "mat", "qpos")

    def __init__(self, t: float, obs: dict, ee: np.ndarray, mat: np.ndarray, qpos: np.ndarray):
        self.t = t
        self.obs = obs
        self.ee = ee
        self.mat = mat
        self.qpos = qpos


class SentRecord:
    __slots__ = ("t", "action", "rtt_ms")

    def __init__(self, t: float, action: dict, rtt_ms: float):
        self.t = t
        self.action = action
        self.rtt_ms = rtt_ms


def expected_target(ref: np.ndarray, action: dict) -> tuple[np.ndarray, np.ndarray]:
    """The follower's composition T_target = T_arm_ref @ Δ, test-side.

    ref is the 4x4 latched reference; action the 8-feature dict.  Returns
    (target_pos (3,), target_rot (3,3)) — the same numbers
    pose_delta_law.PoseDeltaLaw.solve composes.
    """
    dp = np.array(
        [action["hand.delta_pos.x"], action["hand.delta_pos.y"], action["hand.delta_pos.z"]]
    )
    quat = np.array(
        [action["hand.delta_rot.qx"], action["hand.delta_rot.qy"],
         action["hand.delta_rot.qz"], action["hand.delta_rot.qw"]]
    )
    r_delta = Rotation.from_quat(quat).as_matrix()
    return ref[:3, 3] + ref[:3, :3] @ dp, ref[:3, :3] @ r_delta


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------


def _p95(x: np.ndarray) -> float:
    return float(np.percentile(x, 95)) if x.size else float("nan")


def tracking_metrics(
    samples: list[Sample],
    frame_fn,
    ref: np.ndarray,
    *,
    lead_s: float,
) -> dict:
    """Position/rotation tracking error vs the composed target.

    Only samples after lead_s (the identity settle window) and while a frame
    is being commanded count; the lead window yields the handshake-offset
    baseline instead (identity tracking error).
    """
    errs: list[np.ndarray] = []
    rot_errs: list[float] = []
    lead_errs: list[np.ndarray] = []
    for s in samples:
        action = frame_fn(s.t)
        if action is None:
            continue
        tp, tr = expected_target(ref, action)
        if s.t < lead_s:
            lead_errs.append(s.ee - tp)
            continue
        errs.append(s.ee - tp)
        r_rel = tr.T @ s.mat
        rot_errs.append(float(np.degrees(np.arccos(
            np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
        ))))

    out: dict = {"samples": len(samples), "tracked_samples": len(errs)}
    if lead_errs:
        lead = np.array(lead_errs)
        out["lead_offset_mm"] = float(np.linalg.norm(lead.mean(axis=0)) * 1000.0)
    if not errs:
        return out
    e = np.array(errs) * 1000.0  # mm
    tail_n = max(1, len(rot_errs) // 10) if rot_errs else 1
    out.update(
        {
            "rmse_mm": float(np.sqrt((e ** 2).sum(axis=1).mean())),
            "rmse_xyz_mm": [float(np.sqrt((e[:, i] ** 2).mean())) for i in range(3)],
            "p95_mm": _p95(np.linalg.norm(e, axis=1)),
            "max_mm": float(np.linalg.norm(e, axis=1).max()),
            "final_err_mm": float(np.linalg.norm(e[-max(1, len(e) // 10):].mean(axis=0))),
            "rot_rmse_deg": float(np.sqrt(np.mean(np.square(rot_errs)))) if rot_errs else float("nan"),
            "rot_final_deg": float(np.mean(rot_errs[-tail_n:])) if rot_errs else float("nan"),
        }
    )
    return out


def step_metrics(samples: list[Sample], frame_fn, ref, *, lead_s: float) -> dict:
    """Displacement sanity numbers: where the EE ended up vs where the step
    asked it to go (direction cosine + fraction of amplitude reached).

    The excursion bound is always reported; displacement direction/fraction
    only when the sequence actually asks for net motion (reject-recovery
    probes end where they started, by design).
    """
    out: dict = {
        "max_ee_mm_from_ref": float(max(
            np.linalg.norm(s.ee - ref[:3, 3]) for s in samples
        ) * 1000.0),
    }
    lead_idx = [i for i, s in enumerate(samples) if s.t < lead_s]
    tail_idx = [i for i, s in enumerate(samples) if s.t >= lead_s]
    if not lead_idx or not tail_idx:
        return out
    ee0 = np.mean([samples[i].ee for i in lead_idx], axis=0)
    ee1 = np.mean([samples[i].ee for i in tail_idx[-max(1, len(tail_idx) // 5):]], axis=0)
    # expected displacement: target at a late tail time vs target at lead.
    t_tail = samples[tail_idx[-1]].t
    a0, a1 = frame_fn(0.0), frame_fn(t_tail)
    if a0 is None or a1 is None:
        return out
    p0, _ = expected_target(ref, a0)
    p1, _ = expected_target(ref, a1)
    expected = p1 - p0
    disp = ee1 - ee0
    norm_e = float(np.linalg.norm(expected))
    if norm_e < 1e-9:
        return out
    out.update(
        {
            "expected_disp_mm": norm_e * 1000.0,
            "actual_disp_mm": float(np.linalg.norm(disp)) * 1000.0,
            "direction_cosine": float(np.dot(disp, expected) / (np.linalg.norm(disp) * norm_e)),
            "amplitude_fraction": float(np.dot(disp, expected) / (norm_e ** 2)),
        }
    )
    return out


def _uniform_grid(samples: list[Sample], values: np.ndarray, hz: float = 100.0):
    t = np.array([s.t for s in samples])
    t0, t1 = t.min(), t.max()
    n = max(2, int((t1 - t0) * hz))
    grid = np.linspace(t0, t1, n)
    return grid, np.stack([np.interp(grid, t, values[:, i]) for i in range(3)], axis=1), 1.0 / hz


def lag_metrics(samples: list[Sample], frame_fn, ref, *, lead_s: float,
                motion_axis: int | None = None) -> dict:
    """Cross-correlation lag between commanded target and actual EE position
    on the dominant-motion axis (ms, positive = actual lags target)."""
    tracked = [
        (s, expected_target(ref, frame_fn(s.t))[0])
        for s in samples
        if s.t >= lead_s and frame_fn(s.t) is not None
    ]
    if len(tracked) < 20:
        return {"lag_ms": None, "note": "insufficient motion for lag estimation"}
    actual = np.array([s.ee for s, _ in tracked])
    target = np.array([tp for _, tp in tracked])
    t = np.array([s.t for s, _ in tracked])
    if motion_axis is None:
        motion_axis = int(np.argmax(np.ptp(target, axis=0)))
    if np.ptp(target[:, motion_axis]) < 0.005:
        return {"lag_ms": None, "note": "motion amplitude below 5 mm on all axes"}
    grid, act_g, dt = _uniform_grid(
        [s for s, _ in tracked], actual, hz=100.0
    )
    _, tgt_g, _ = _uniform_grid([s for s, _ in tracked], target, hz=100.0)
    a = act_g[:, motion_axis] - act_g[:, motion_axis].mean()
    b = tgt_g[:, motion_axis] - tgt_g[:, motion_axis].mean()
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if denom < 1e-12:
        return {"lag_ms": None, "note": "degenerate signals"}
    corr = np.correlate(a, b, mode="full") / denom
    lag_idx = int(np.argmax(corr)) - (len(b) - 1)
    corr_peak = float(corr.max())
    return {
        "lag_ms": float(lag_idx * dt * 1000.0),
        "lag_axis": "xyz"[motion_axis],
        "corr_peak": corr_peak,
    }


def smoothness_metrics(samples: list[Sample]) -> dict:
    """EE jerk RMS on a uniform 100 Hz resample (m/s^3) + max joint speed."""
    if len(samples) < 8:
        return {}
    ee = np.array([s.ee for s in samples])
    grid, g, dt = _uniform_grid(samples, ee, hz=100.0)
    jerk = np.diff(g, n=3, axis=0) / dt ** 3
    qpos = np.array([s.qpos[: len(BODY_JOINTS)] for s in samples])
    dq = np.diff(np.degrees(qpos), axis=0)
    dt_q = np.diff(np.array([s.t for s in samples]))
    safe = np.where(dt_q > 1e-6, dt_q, 1e-6)
    speed = np.abs(dq / safe[:, None]).max()
    return {
        "jerk_rms_m_s3": float(np.sqrt((jerk ** 2).sum(axis=1).mean())),
        "max_joint_speed_deg_s": float(speed),
    }


def stream_metrics(stats_diff: dict, window_s: float) -> dict:
    """Observation stream + action RTT stats from a TeleopStats window diff."""
    per_key = stats_diff.get("per_key", {})
    # One scalar feature key stands in for the frame count (6 keys, 1 frame
    # each per tick).
    ref_key = "shoulder_pan.pos"
    frames = per_key.get(ref_key, (0, 0, []))[0]
    lat = np.array(per_key.get(ref_key, (0, 0, []))[2], dtype=float)
    rtt = np.array(stats_diff.get("action_rtt_ms", []), dtype=float)
    out = {
        "obs_frames": int(frames),
        "window_s": float(window_s),
        "obs_hz_measured": float(frames / window_s) if window_s > 0 else float("nan"),
        "drop_rate_vs_nominal": float(
            max(0.0, 1.0 - frames / (OBS_NOMINAL_HZ * window_s))
        ) if window_s > 0 else float("nan"),
        "actions_sent": int(len(rtt)),
    }
    if lat.size:
        out.update(
            {
                "feat_latency_ms_mean": float(lat.mean()),
                "feat_latency_ms_p95": _p95(lat),
            }
        )
    if rtt.size:
        out.update(
            {
                "action_rtt_ms_mean": float(rtt.mean()),
                "action_rtt_ms_p95": _p95(rtt),
            }
        )
    return out


def gripper_metrics(samples: list[Sample], sent: list[SentRecord],
                    *, from_mm: float, to_mm: float, norm_max: float = 100.0) -> dict:
    """Gripper step response: 63.2 % crossing time and final tracking.

    Gripper obs is lerobot 0-100; commands are mm (0-60 -> 0-100).
    """
    if not sent or not samples:
        return {}
    t_step = sent[0].t  # first commanded sample of the new aperture
    target_norm = to_mm / _GRIP_MAX_MM * norm_max
    start_norm = from_mm / _GRIP_MAX_MM * norm_max
    after = [s for s in samples if s.t >= t_step]
    if not after:
        return {}
    y0 = after[0].obs["gripper.pos"]
    final = float(np.mean([s.obs["gripper.pos"] for s in after[-10:]]))
    cross = target_norm - (target_norm - start_norm) * math.exp(-1.0)
    t63 = None
    for s in after:
        y = s.obs["gripper.pos"]
        if (target_norm >= start_norm and y >= cross) or (
            target_norm < start_norm and y <= cross
        ):
            t63 = s.t - t_step
            break
    return {
        "gripper_t63_ms": float(t63 * 1000.0) if t63 is not None else None,
        "gripper_start_norm": float(y0),
        "gripper_final_norm": final,
        "gripper_target_norm": float(target_norm),
        "gripper_final_err_norm": float(final - target_norm),
    }


def solutions_summary(solutions: list, body_joints=BODY_JOINTS) -> dict:
    """Law-side diagnostics over the run's JointSolution records (sim only)."""
    if not solutions:
        return {}
    pos_err = [s.pos_err_m for s in solutions]
    max_step_deg = 0.0
    for prev, cur in zip(solutions, solutions[1:]):
        steps = [
            abs(cur.joint_action[f"{j}.pos"] - prev.joint_action[f"{j}.pos"])
            for j in body_joints
        ]
        max_step_deg = max(max_step_deg, max(steps))
    return {
        "solves": len(solutions),
        "rejected": sum(1 for s in solutions if s.rejected),
        "stale": sum(1 for s in solutions if s.stale),
        "jumped": sum(1 for s in solutions if s.jumped),
        "collided": sum(1 for s in solutions if s.collided),
        "final_pos_err_mm": float(pos_err[-1] * 1000.0),
        "mean_pos_err_mm": float(np.mean(pos_err) * 1000.0),
        "max_commanded_step_deg": float(max_step_deg),
    }
