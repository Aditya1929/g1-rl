"""Curriculum: decay the imitation weights so the task reward takes over."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def linear_reward_weight(env: "ManagerBasedRLEnv", env_ids, term_name: str, start_weight: float,
                         end_weight: float, start_step: int, end_step: int):
    """Linearly move a reward weight from start to end between two global env steps
    (env.common_step_counter counts policy steps; 24 steps = 1 PPO iteration by default)."""
    s = env.common_step_counter
    a = min(max((s - start_step) / max(end_step - start_step, 1), 0.0), 1.0)
    cfg = env.reward_manager.get_term_cfg(term_name)
    cfg.weight = start_weight + a * (end_weight - start_weight)
    env.reward_manager.set_term_cfg(term_name, cfg)
    return cfg.weight
