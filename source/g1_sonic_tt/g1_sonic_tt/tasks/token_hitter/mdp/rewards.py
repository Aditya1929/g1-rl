"""Rewards: HITTER's goal terms with SMASH-style adaptive widths, imitation, and regularization.

Residual recipe (default):
  imitation   pose-space tracking of the reference clip (joints, joint velocities, racket path),
              with the right arm / racket path relaxed in a window around impact, + decoder_match
  prior       ||dz||^2 and its rate (GRAIL: keep the token near the encoder's z_ref)
  goal        racket pos / vel / face at impact (adaptive sigma), success bonus, base target
  reg         termination, joint limits, orientation, foot slip, table penetration
Absolute recipe (G1-SonicTT-TokenAbs-v0) uses token_match instead of the pose terms.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from g1_sonic_tt.sonic.constants import SONIC_JOINTS

from .racket import racket_state_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .actions import SonicTokenAction
    from .commands import StrikeCommand


def _cmd(env, name="strike") -> "StrikeCommand":
    return env.command_manager.get_term(name)


def _act(env, name="token") -> "SonicTokenAction":
    return env.action_manager.get_term(name)


# ----------------------------------------------------------------------------- imitation
def token_match(env, std: float = 0.1, slack: int = 3, command_name="strike", action_name="token"):
    """exp(-min_k mean((z_raw - z_ref[t+k])^2) / std^2), k in [-slack, slack].
    Uses the continuous (pre-quantization) token so the signal is smooth."""
    z = _act(env, action_name).z_raw[:, None]                    # (N, 1, 64)
    win = _cmd(env, command_name).z_ref_window(slack)            # (N, 2k+1, 64)
    err = ((z - win) ** 2).mean(-1).min(dim=1).values
    return torch.exp(-err / std**2)


class decoder_match(ManagerTermBase):
    """Decoder-space imitation: compare what the policy token and z_ref *mean* given the
    live proprio history: exp(-||W (a_pol - a_ref)||^2 / (n * std^2))."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        w = torch.ones(len(SONIC_JOINTS), device=env.device)
        for i, n in enumerate(SONIC_JOINTS):
            if n.startswith("right_") and any(k in n for k in ("shoulder", "elbow", "wrist")) or "waist" in n:
                w[i] = cfg.params.get("arm_weight", 2.0)
        self.w = w / w.mean()

    def __call__(self, env, std: float = 0.3, arm_weight: float = 2.0, action_name: str = "token"):
        a = _act(env, action_name)
        err = (self.w * (a.a_pol - a.a_ref) ** 2).mean(-1)
        return torch.exp(-err / std**2)


def _arm_waist_mask(device):
    arm = torch.tensor([n.startswith("right_") and any(k in n for k in ("shoulder", "elbow", "wrist"))
                        for n in SONIC_JOINTS], device=device)
    waist = torch.tensor(["waist" in n for n in SONIC_JOINTS], device=device)
    return arm, waist


class _PoseTracking(ManagerTermBase):
    """exp(-min_k wmean((x - x_ref[t+1+k])^2) / std^2), k in [-slack, slack].

    Weights: right arm and waist x arm_weight. Near impact the right-arm weights are
    multiplied down to impact_floor over a +-impact_half_width-frame window, so the
    racket goal terms (not the clip) decide the hitting arm's pose at contact."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        arm, waist = _arm_waist_mask(env.device)
        self.arm = arm.float()
        self.w0 = torch.where(arm | waist, cfg.params.get("arm_weight", 2.0), 1.0)

    def _state(self, env, act) -> torch.Tensor:
        raise NotImplementedError

    def _ref(self, cmd, k) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, env, std: float, slack: int = 2, arm_weight: float = 2.0, impact_floor: float = 0.2,
                 impact_half_width: float = 10.0, command_name: str = "strike", action_name: str = "token"):
        cmd, act = _cmd(env, command_name), _act(env, action_name)
        g = cmd.impact_gate(impact_half_width)[:, None]
        w = self.w0 * (1.0 - (1.0 - impact_floor) * g * self.arm)              # (N, 29)
        x = self._state(env, act)[:, None]                                     # (N, 1, 29)
        err = (w[:, None] * (x - self._ref(cmd, slack)) ** 2).sum(-1) / w.sum(-1, keepdim=True)
        return torch.exp(-err.min(dim=1).values / std**2)


class pose_joint_tracking(_PoseTracking):
    def _state(self, env, act):
        return act.robot.data.joint_pos[:, act.sim_ids]

    def _ref(self, cmd, k):
        return cmd.ref_q_window(k)


class pose_joint_vel_tracking(_PoseTracking):
    def _state(self, env, act):
        return act.robot.data.joint_vel[:, act.sim_ids]

    def _ref(self, cmd, k):
        return cmd.ref_dq_window(k)


def racket_path_tracking(env, std: float = 0.08, slack: int = 2, impact_floor: float = 0.2,
                         impact_half_width: float = 10.0, command_name: str = "strike"):
    """Racket contact point (root-relative, yaw frame) vs the clip's contact path; relaxed near impact."""
    cmd, robot = _cmd(env, command_name), env.scene["robot"]
    p, _, _ = racket_state_w(env)
    p_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), p - robot.data.root_pos_w)
    err = ((p_b[:, None] - cmd.ref_racket_window(slack)) ** 2).sum(-1).min(dim=1).values
    g = cmd.impact_gate(impact_half_width)
    return torch.exp(-err / std**2) * (1.0 - (1.0 - impact_floor) * g)


def residual_l2(env, action_name: str = "token"):
    """mean(dz^2): GRAIL's latent-residual penalty (keeps tokens near the encoder's z_ref)."""
    return (_act(env, action_name).delta ** 2).mean(-1)


def residual_rate_l2(env, action_name: str = "token"):
    a = _act(env, action_name)
    return ((a.delta - a.prev_delta) ** 2).mean(-1)


# ----------------------------------------------------------------------------- goal
class _AdaptiveGatedTracking(ManagerTermBase):
    """SMASH Eq. 12: r = exp(-e / sigma) * 1[gate], sigma adapted to a running mean of e
    (clamped to [sigma_min, sigma_init]). Without the adaptation SMASH's success fell 86% -> 23%."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.sigma = torch.tensor(cfg.params["sigma_init"], device=env.device)

    def _error(self, env, cmd: "StrikeCommand") -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, env, sigma_init: float, sigma_min: float, half_window: int = 0,
                 ema: float = 0.99, command_name: str = "strike"):
        cmd = _cmd(env, command_name)
        gate = cmd.near_hit(half_window) if half_window > 0 else cmd.at_hit
        e = self._error(env, cmd)
        if gate.any():
            self.sigma = torch.clamp(ema * self.sigma + (1 - ema) * e[gate].mean(), sigma_min, sigma_init)
        return torch.exp(-e / self.sigma) * gate.float()


class racket_pos_tracking(_AdaptiveGatedTracking):
    def _error(self, env, cmd):
        p, _, _ = racket_state_w(env)
        tgt, _ = cmd.racket_target_world()
        return torch.norm(p - tgt, dim=-1)


class racket_vel_tracking(_AdaptiveGatedTracking):
    def _error(self, env, cmd):
        _, v, _ = racket_state_w(env)
        _, tgt = cmd.racket_target_world()
        return torch.norm(v - tgt, dim=-1)


def _face_angle(n, n_tgt):
    """Angle between face normal and target normal, symmetric in the face side (SMASH Eq. 11)."""
    return torch.acos(torch.abs((n * n_tgt).sum(-1)).clamp(max=1.0))


class racket_face_tracking(_AdaptiveGatedTracking):
    """Face normal vs the commanded normal (training: library normal; planner: velocity direction)."""

    def _error(self, env, cmd):
        _, _, n = racket_state_w(env)
        return _face_angle(n, cmd.racket_target_normal_world())


def strike_success(env, pos_tol: float = 0.05, vel_tol: float = 0.5, face_tol: float = 0.35,
                   command_name: str = "strike"):
    """Sparse bonus at the impact step when position, velocity and face are all within tolerance."""
    cmd = _cmd(env, command_name)
    p, v, n = racket_state_w(env)
    tp, tv = cmd.racket_target_world()
    face = _face_angle(n, cmd.racket_target_normal_world())
    ok = (torch.norm(p - tp, dim=-1) < pos_tol) & (torch.norm(v - tv, dim=-1) < vel_tol) & (face < face_tol)
    return (ok & cmd.at_hit).float()


def base_pos_tracking(env, std: float = 0.1, command_name: str = "strike"):
    """Track the base target until impact."""
    cmd, robot = _cmd(env, command_name), env.scene["robot"]
    tgt = cmd.base_target_xy + env.scene.env_origins[:, :2]
    err = torch.norm(tgt - robot.data.root_pos_w[:, :2], dim=-1)
    return torch.exp(-err / std) * cmd.before_hit.float()


# ----------------------------------------------------------------------------- regularization
def token_rate_l2(env, action_name: str = "token"):
    a = _act(env, action_name)
    return ((a.z_raw - a.prev_z_raw) ** 2).sum(-1)


def table_penetration(env, near_edge_x: float, length: float = 2.74, half_width: float = 0.7625,
                      top_z: float = 0.76, margin: float = 0.02):
    """Penalize the racket contact point and the wrist going below the tabletop over the table."""
    robot = env.scene["robot"]
    p, _, _ = racket_state_w(env)
    pts = [p - env.scene.env_origins]
    wrist = robot.find_bodies("right_wrist_yaw_link")[0][0]
    pts.append(robot.data.body_pos_w[:, wrist] - env.scene.env_origins)
    pen = torch.zeros(env.num_envs, device=env.device)
    for q in pts:
        over = (q[:, 0] > near_edge_x) & (q[:, 0] < near_edge_x + length) & (q[:, 1].abs() < half_width)
        pen += (top_z + margin - q[:, 2]).clamp(min=0.0) * over
    return pen


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    contact: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    vel = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(vel.norm(dim=-1) * contacts, dim=1)
