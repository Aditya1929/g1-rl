"""SONIC token action: policy output -> FSQ token -> frozen SONIC decoder -> PD targets.

Per policy step (50 Hz, once in process_actions):
    mode "residual" (default, GRAIL-style):  dz = clip(a, -1, 1);  z_raw = z_ref + lambda * dz   (lambda 0.1)
    mode "absolute":                         z_raw = tanh(a)
    mode "replay":                           z_raw = z_ref           (a = 0 in residual mode is the same)
    z     = quantize(z_raw)            (32-level FSQ grid [-1, 15/16], after the sum; what the deploy controller receives)
    push current proprio into the 10-frame decoder history
    a_pol = decoder([z,     history])
    a_ref = decoder([z_ref, history])  (same batch; used by the decoder-space imitation reward)
    q_tgt = default + scale * a_pol    -> set as joint position targets for the 4 physics substeps

The decoder is frozen and part of the environment dynamics; PPO never trains it.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from g1_sonic_tt.sonic.constants import FSQ_MAX, FSQ_MIN, SONIC_JOINTS, TOKEN_DIM
from g1_sonic_tt.sonic.decoder import DecoderHistory, SonicDecoder, load_decoder_params
from g1_sonic_tt.sonic.fsq import quantize, squash

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands import StrikeCommand


class SonicTokenAction(ActionTerm):
    cfg: "SonicTokenActionCfg"

    def __init__(self, cfg: "SonicTokenActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # simulator joint indices listed in SONIC order
        self.sim_ids, names = self.robot.find_joints(SONIC_JOINTS, preserve_order=True)
        assert names == SONIC_JOINTS, f"joint mismatch: {names}"
        self.decoder = SonicDecoder(cfg.decoder_weights).to(self.device)
        self.default_q, self.scale = load_decoder_params(cfg.decoder_params, self.device)
        # the robot's own default pose must equal SONIC's, or q - default is shifted
        sim_default = self.robot.data.default_joint_pos[0, self.sim_ids]
        assert torch.allclose(sim_default, self.default_q, atol=1e-4), \
            f"robot default pose != SONIC default\n{sim_default}\n{self.default_q}"
        n = self.num_envs
        self.history = DecoderHistory(n, self.default_q, self.scale, self.device)
        self._raw = torch.zeros(n, TOKEN_DIM, device=self.device)
        self.z_raw = torch.zeros(n, TOKEN_DIM, device=self.device)
        self.prev_z_raw = torch.zeros_like(self.z_raw)
        self.delta = torch.zeros_like(self.z_raw)       # residual in action units, [-1, 1]
        self.prev_delta = torch.zeros_like(self.z_raw)
        self.z = torch.zeros_like(self.z_raw)          # token sent to the decoder this step
        self.a_pol = torch.zeros(n, len(SONIC_JOINTS), device=self.device)
        self.a_ref = torch.zeros_like(self.a_pol)
        self.q_target = self.default_q.expand(n, -1).clone()
        self._just_reset = torch.ones(n, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return TOKEN_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.z

    def _command(self) -> "StrikeCommand":
        return self._env.command_manager.get_term(self.cfg.command_name)

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        z_ref = self._command().z_ref
        lam = self.cfg.residual_scale
        if self.cfg.mode == "residual":
            delta = actions.clamp(-1.0, 1.0)
            z_raw = torch.clamp(z_ref + lam * delta, FSQ_MIN, FSQ_MAX)
        elif self.cfg.mode == "absolute":
            z_raw = squash(actions)
            delta = (z_raw - z_ref) / lam          # implied residual (for logging / the l2 term)
        elif self.cfg.mode == "replay":
            z_raw = z_ref.clone()
            delta = torch.zeros_like(z_raw)
        else:
            raise ValueError(self.cfg.mode)
        # first step after a reset: no previous token -> treat this one as previous (no rate penalty)
        fresh = self._just_reset[:, None]
        self.prev_z_raw = torch.where(fresh, z_raw, self.z_raw)
        self.prev_delta = torch.where(fresh, delta, self.delta)
        self._just_reset[:] = False
        self.z_raw = z_raw
        self.delta = delta
        self.z = quantize(z_raw)

        d = self.robot.data
        self.history.push(d.root_ang_vel_b, d.joint_pos[:, self.sim_ids], d.joint_vel[:, self.sim_ids],
                          d.projected_gravity_b)
        prop = self.history.proprio()
        obs = torch.cat([DecoderHistory.build_obs(self.z, prop), DecoderHistory.build_obs(z_ref, prop)])
        out = self.decoder(obs)
        self.a_pol, self.a_ref = out[: self.num_envs], out[self.num_envs:]
        self.history.record_action(self.a_pol)
        self.q_target = self.default_q + self.scale * self.a_pol

    def apply_actions(self):
        self.robot.set_joint_position_target(self.q_target, joint_ids=self.sim_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.history.reset(env_ids)
        self._just_reset[env_ids] = True
        self._raw[env_ids] = 0.0


@configclass
class SonicTokenActionCfg(ActionTermCfg):
    class_type: type = SonicTokenAction
    asset_name: str = "robot"
    command_name: str = "strike"
    decoder_weights: str = MISSING
    decoder_params: str = MISSING
    mode: str = "residual"          # residual | absolute | replay (replay = library tokens, ignores actions)
    residual_scale: float = 0.1     # lambda: |dz| <= 1 moves each token dim by <= 1.6 FSQ steps (GRAIL: 0.1)
