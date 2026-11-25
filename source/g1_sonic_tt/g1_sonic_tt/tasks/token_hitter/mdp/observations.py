"""Observation terms. Actor gets HITTER's task commands + proprio, the current reference
token z_ref and the previous residual (residual mode) or previous token (absolute mode);
the critic additionally gets future reference tokens/joints and racket errors."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .racket import racket_state_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .actions import SonicTokenAction
    from .commands import StrikeCommand


def _cmd(env, name="strike") -> "StrikeCommand":
    return env.command_manager.get_term(name)


def _act(env, name="token") -> "SonicTokenAction":
    return env.action_manager.get_term(name)


# ----- actor -----
def forward_vec_xy(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    x = torch.zeros_like(robot.data.root_pos_w)
    x[:, 0] = 1.0
    return quat_apply(yaw_quat(robot.data.root_quat_w), x)[:, :2]


def base_target_err_b(env, command_name: str = "strike") -> torch.Tensor:
    """Base target minus current base position, xy, in the yaw frame."""
    cmd, robot = _cmd(env, command_name), env.scene["robot"]
    d = torch.zeros_like(robot.data.root_pos_w)
    d[:, :2] = cmd.base_target_xy + env.scene.env_origins[:, :2] - robot.data.root_pos_w[:, :2]
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), d)[:, :2]


def racket_target_pos_b(env, command_name: str = "strike") -> torch.Tensor:
    return _cmd(env, command_name).racket_target_pos_b


def racket_target_vel_b(env, command_name: str = "strike") -> torch.Tensor:
    return _cmd(env, command_name).racket_target_vel_b


def racket_target_normal_b(env, command_name: str = "strike") -> torch.Tensor:
    return _cmd(env, command_name).racket_target_normal_b


def ref_token(env, command_name: str = "strike") -> torch.Tensor:
    """z_ref the residual is added to (the deploy bridge has the same library)."""
    return _cmd(env, command_name).z_ref


def prev_residual(env, action_name: str = "token") -> torch.Tensor:
    """Residual dz applied at the last step, in action units."""
    return _act(env, action_name).delta


def t_strike(env, command_name: str = "strike") -> torch.Tensor:
    return _cmd(env, command_name).t_strike[:, None]


def swing_type(env, command_name: str = "strike") -> torch.Tensor:
    return _cmd(env, command_name).swing_onehot


def prev_token(env, action_name: str = "token") -> torch.Tensor:
    """Quantized token applied at the last step."""
    return _act(env, action_name).z


# ----- critic (privileged) -----
def ref_tokens(env, command_name: str = "strike", offsets: tuple[int, ...] = (0, 5, 10)) -> torch.Tensor:
    return _cmd(env, command_name).z_ref_future(offsets)


def ref_joint_pos_rel(env, command_name: str = "strike", action_name: str = "token") -> torch.Tensor:
    return _cmd(env, command_name).ref_q - _act(env, action_name).default_q


def ref_joint_vel(env, command_name: str = "strike") -> torch.Tensor:
    return _cmd(env, command_name).ref_dq


def racket_errors_b(env, command_name: str = "strike") -> torch.Tensor:
    """Racket position/velocity minus target (yaw frame) and face normal (yaw frame): 9 dims."""
    cmd, robot = _cmd(env, command_name), env.scene["robot"]
    p, v, n = racket_state_w(env)
    tgt_p, tgt_v = cmd.racket_target_world()
    q = yaw_quat(robot.data.root_quat_w)
    return torch.cat([quat_apply_inverse(q, p - tgt_p), quat_apply_inverse(q, v - tgt_v),
                      quat_apply_inverse(q, n)], dim=1)


def time_left(env) -> torch.Tensor:
    return (env.max_episode_length - env.episode_length_buf).float()[:, None] * env.step_dt
