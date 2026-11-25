"""Shared helpers for the Isaac Lab scripts (import AFTER AppLauncher has started)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/g1_sonic_tt"))

LOG_ROOT = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl"


def make_env(task: str, num_envs: int | None, device: str | None = None, render: bool = False, cfg_hook=None):
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import g1_sonic_tt.tasks  # noqa: F401  registers tasks

    kw = {"num_envs": num_envs} if num_envs else {}
    env_cfg = parse_env_cfg(task, device=device or "cuda:0", **kw)
    if cfg_hook is not None:
        cfg_hook(env_cfg)
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if render else None)
    return env, RslRlVecEnvWrapper(env), env_cfg


def agent_cfg(task: str):
    from isaaclab_tasks.utils import load_cfg_from_registry

    import g1_sonic_tt.tasks  # noqa: F401  registers tasks (agent_cfg may be called before make_env)

    return load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")


def latest_checkpoint(experiment: str, run: str | None = None) -> str:
    root = LOG_ROOT / experiment
    runs = sorted([d for d in root.iterdir() if d.is_dir()]) if root.exists() else []
    if run:
        runs = [d for d in runs if d.name == run]
    for d in reversed(runs):
        ckpts = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            return str(ckpts[-1])
    raise FileNotFoundError(f"no checkpoint under {root}")


def env_unwrapped(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


os.environ.setdefault("HYDRA_FULL_ERROR", "1")
