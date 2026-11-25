"""Racket pose from the wrist link (the racket is merged into right_wrist_yaw_link)."""
from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply

from g1_sonic_tt.sonic.constants import RACKET_CONTACT_OFFSET, RACKET_FACE_AXIS, RACKET_PARENT_LINK

_CACHE: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}


def racket_state_w(env, asset_name: str = "robot") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World-frame racket contact point, its linear velocity, and the face normal. Each (N, 3)."""
    robot: Articulation = env.scene[asset_name]
    key = id(robot)
    if key not in _CACHE:
        body = robot.find_bodies(RACKET_PARENT_LINK)[0][0]
        off = torch.tensor(RACKET_CONTACT_OFFSET, device=env.device).expand(env.num_envs, 3).clone()
        axis = torch.tensor(RACKET_FACE_AXIS, device=env.device).expand(env.num_envs, 3).clone()
        _CACHE[key] = (body, off, axis)
    body, off, axis = _CACHE[key]
    q = robot.data.body_quat_w[:, body]
    r = quat_apply(q, off)
    pos = robot.data.body_pos_w[:, body] + r
    vel = robot.data.body_lin_vel_w[:, body] + torch.cross(robot.data.body_ang_vel_w[:, body], r, dim=-1)
    normal = quat_apply(q, axis)
    return pos, vel, normal
