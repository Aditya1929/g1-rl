import gymnasium as gym

_AGENT = f"{__name__}.agents.rsl_rl_ppo_cfg:TokenHitterPPORunnerCfg"
_AGENT_ABS = f"{__name__}.agents.rsl_rl_ppo_cfg:TokenHitterAbsPPORunnerCfg"

gym.register(
    id="G1-SonicTT-Token-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:TokenHitterEnvCfg", "rsl_rl_cfg_entry_point": _AGENT},
)
gym.register(
    id="G1-SonicTT-Token-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:TokenHitterEnvCfg_PLAY", "rsl_rl_cfg_entry_point": _AGENT},
)
gym.register(
    id="G1-SonicTT-Token-Replay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:TokenHitterEnvCfg_REPLAY", "rsl_rl_cfg_entry_point": _AGENT},
)
gym.register(
    id="G1-SonicTT-TokenAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:TokenHitterEnvCfg_ABS", "rsl_rl_cfg_entry_point": _AGENT_ABS},
)
gym.register(
    id="G1-SonicTT-TokenAbs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:TokenHitterEnvCfg_ABS_PLAY",
            "rsl_rl_cfg_entry_point": _AGENT_ABS},
)
