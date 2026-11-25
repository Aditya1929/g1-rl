"""Frozen SONIC decoder as a batched torch module, plus the proprio history it needs.

Reproduces what gear_sonic_deploy does in C++ for policy/release/observation_config.yaml:

    obs(994) = [token(64),
                ang_vel_b   (10 x 3),     oldest first
                q - default (10 x 29),
                dq          (10 x 29),
                last_action (10 x 29),    raw decoder output of previous steps
                gravity_b   (10 x 3)]
    q_target = default + action_scale * decoder(obs)

All joint tensors here are in SONIC order (constants.SONIC_JOINTS); callers map
to/from the simulator's joint order by name.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .constants import DECODER_OBS_DIM, HISTORY_LEN, MUJOCO_TO_ISAACLAB, NUM_JOINTS, TOKEN_DIM


class SonicDecoder(nn.Module):
    """994 -> [Linear, SiLU] x6 -> Linear -> 29. Weights from scripts/convert_decoder.py."""

    def __init__(self, weights_npz: str | Path):
        super().__init__()
        w = np.load(weights_npz)
        n = len([k for k in w.files if k.startswith("W")])
        layers: list[nn.Module] = []
        for i in range(n):
            W, b = w[f"W{i}"], w[f"b{i}"]
            lin = nn.Linear(W.shape[0], W.shape[1])
            lin.weight.data = torch.from_numpy(W.T.copy()).float()
            lin.bias.data = torch.from_numpy(b.copy()).float()
            layers.append(lin)
            if i < n - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        assert self.net[0].in_features == DECODER_OBS_DIM
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def load_decoder_params(json_path: str | Path, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """default_angles, action_scale in SONIC order (json is MuJoCo order)."""
    p = json.loads(Path(json_path).read_text())
    idx = torch.tensor(MUJOCO_TO_ISAACLAB)
    default = torch.tensor(p["default_angles"], dtype=torch.float32)[idx]
    scale = torch.tensor(p["action_scale"], dtype=torch.float32)[idx]
    return default.to(device), scale.to(device)


class DecoderHistory:
    """Per-env 10-frame proprio history in the decoder's layout."""

    def __init__(self, num_envs: int, default_q: torch.Tensor, action_scale: torch.Tensor, device):
        self.n, self.device = num_envs, device
        self.default_q, self.scale = default_q.to(device), action_scale.to(device)
        H = HISTORY_LEN
        self.ang_vel = torch.zeros(num_envs, H, 3, device=device)
        self.q_rel = torch.zeros(num_envs, H, NUM_JOINTS, device=device)
        self.dq = torch.zeros(num_envs, H, NUM_JOINTS, device=device)
        self.last_action = torch.zeros(num_envs, H, NUM_JOINTS, device=device)
        self.gravity = torch.zeros(num_envs, H, 3, device=device)
        self.gravity[..., 2] = -1.0
        self.needs_fill = torch.ones(num_envs, dtype=torch.bool, device=device)
        # the action applied at the previous step, written into the history at the next push
        self._pending_action = torch.zeros(num_envs, NUM_JOINTS, device=device)

    def reset(self, env_ids):
        self.needs_fill[env_ids] = True

    def push(self, ang_vel_b, q, dq, gravity_b):
        """Append the current frame (q, dq in SONIC order). Freshly reset envs are filled with it
        (the C++ clamps history indices at the start, i.e. repeats the first frame)."""
        q_rel = q - self.default_q
        # after a reset, assume the robot was holding its pose: last_action == current pose
        fill = self.needs_fill
        self._pending_action[fill] = q_rel[fill] / self.scale
        frame = (ang_vel_b, q_rel, dq, self._pending_action, gravity_b)
        for buf, x in zip((self.ang_vel, self.q_rel, self.dq, self.last_action, self.gravity), frame):
            buf[:] = torch.roll(buf, -1, dims=1)
            buf[:, -1] = x
            if fill.any():
                buf[fill] = x[fill].unsqueeze(1).expand(-1, HISTORY_LEN, -1)
        self.needs_fill[:] = False

    def record_action(self, action):
        """Raw decoder output applied this step; enters the history at the next push."""
        self._pending_action = action.clone()

    def proprio(self) -> torch.Tensor:
        """(N, 930) history block, feature-major, oldest first."""
        return torch.cat([b.reshape(self.n, -1) for b in
                          (self.ang_vel, self.q_rel, self.dq, self.last_action, self.gravity)], dim=1)

    @staticmethod
    def build_obs(token: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        assert token.shape[-1] == TOKEN_DIM
        return torch.cat([token, proprio], dim=-1)
