from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TokenHitterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 250
    experiment_name = "g1_sonic_token_hitter"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        # residual units: token = quantize(z_ref + 0.2 a). One FSQ step needs |a| > 0.16, so
        # std 0.3 flips ~60% of dims by one step; much smaller std explores almost nothing.
        init_noise_std=0.3,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],   # HITTER Sec. V-B3
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,  # run 1 (0.003): noise std drifted 0.5 -> 1.0 once the task saturated
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TokenHitterAbsPPORunnerCfg(TokenHitterPPORunnerCfg):
    """Absolute-token recipe: pre-tanh units, token = tanh(a); std 0.15 ~ 2.4 FSQ steps near 0."""

    experiment_name = "g1_sonic_token_hitter_abs"

    def __post_init__(self):  # (the base cfg defines no __post_init__)
        self.policy.init_noise_std = 0.15
