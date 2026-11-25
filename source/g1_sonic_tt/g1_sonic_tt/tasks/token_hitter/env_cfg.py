"""HITTER-style whole-body strike training where the policy outputs SONIC tokens.

No ball in training: the command term samples racket targets (pos/vel/face normal at
impact), a base target and the strike time, swing after swing (HITTER Sec. V-B).

Default (residual, GRAIL-style): token = quantize(z_ref + 0.2 * clip(a)), pose-space
imitation of the face-corrected reference clip, ||dz||^2 prior penalty. At a = 0 the
env replays the library, so PPO starts from a working swing (no BC needed).
TokenHitterEnvCfg_ABS keeps the absolute-token + token-matching recipe for comparison.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from g1_sonic_tt import paths
from g1_sonic_tt.assets.g1_racket import G1_RACKET_SONIC_CFG

from . import mdp

# Table: robot starts at the env origin facing +X; the library strike plane is ~0.43 m
# in front of the pelvis, so the near table edge sits just beyond it.
TABLE_NEAR_EDGE_X = 0.55
TABLE_LENGTH, TABLE_WIDTH, TABLE_TOP_Z = 2.74, 1.525, 0.76


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0)),
    )
    robot: ArticulationCfg = G1_RACKET_SONIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_LENGTH, TABLE_WIDTH, TABLE_TOP_Z),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.22, 0.48)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_NEAR_EDGE_X + TABLE_LENGTH / 2, 0.0, TABLE_TOP_Z / 2)),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))


@configclass
class CommandsCfg:
    # v2 library: wider / lower forehands + side-steps, calibrated against SONIC (scripts/build_library_v2.py,
    # scripts/calibrate_library.py). Serves cover the whole forehand half; strikes within reach of a clip.
    strike = mdp.StrikeCommandCfg(asset_name="robot", library_file=str(paths.FOREHAND_LIBRARY_V2), debug_vis=True,
                                  target_source="planner", table_near_edge_x=TABLE_NEAR_EDGE_X, table_top_z=TABLE_TOP_Z,
                                  serve_y_bounce=(-0.8, 0.0), reach_radius=0.08)


@configclass
class ActionsCfg:
    token = mdp.SonicTokenActionCfg(
        asset_name="robot", command_name="strike",
        decoder_weights=str(paths.DECODER_WEIGHTS), decoder_params=str(paths.DECODER_PARAMS_JSON),
        mode="residual", residual_scale=0.2,  # 0.1 removed the bracket offset; bending the swing needs more
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # HITTER Table I (actor) + face-normal target + z_ref; previous action = previous residual
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        forward_vec = ObsTerm(func=mdp.forward_vec_xy, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_target_err = ObsTerm(func=mdp.base_target_err_b, noise=Unoise(n_min=-0.02, n_max=0.02))
        racket_target_pos = ObsTerm(func=mdp.racket_target_pos_b)
        racket_target_vel = ObsTerm(func=mdp.racket_target_vel_b)
        racket_target_normal = ObsTerm(func=mdp.racket_target_normal_b)
        t_strike = ObsTerm(func=mdp.t_strike)
        swing_type = ObsTerm(func=mdp.swing_type)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        ref_token = ObsTerm(func=mdp.ref_token)
        prev_residual = ObsTerm(func=mdp.prev_residual)
        prev_token = None  # absolute mode uses this instead of ref_token / prev_residual

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_tokens = ObsTerm(func=mdp.ref_tokens, params={"offsets": (0, 5, 10)})
        ref_joint_pos = ObsTerm(func=mdp.ref_joint_pos_rel)
        ref_joint_vel = ObsTerm(func=mdp.ref_joint_vel)
        racket_errors = ObsTerm(func=mdp.racket_errors_b)
        time_left = ObsTerm(func=mdp.time_left)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.4, 1.2),
                "dynamic_friction_range": (0.4, 1.0), "restitution_range": (0.0, 0.1), "num_buckets": 64},
    )
    racket_mass = EventTerm(  # racket (0.20 kg nominal) is merged into the wrist link; vary its load
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
                "mass_distribution_params": (-0.04, 0.06), "operation": "add"},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.1, 0.1)},
                "velocity_range": {}},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.03, 0.03), "velocity_range": (0.0, 0.0)},
    )
    # push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(5.0, 10.0),
    #                        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}})


@configclass
class RewardsCfg:
    # imitation in pose space (HITTER-style), right arm relaxed +-10 frames around impact.
    # Weights decay via curriculum.
    pose_joint = RewTerm(func=mdp.pose_joint_tracking, weight=1.0, params={
        "std": 0.3, "slack": 2, "arm_weight": 2.0, "impact_floor": 0.2, "impact_half_width": 10.0})
    pose_joint_vel = RewTerm(func=mdp.pose_joint_vel_tracking, weight=0.2, params={
        "std": 2.5, "slack": 2, "arm_weight": 2.0, "impact_floor": 0.2, "impact_half_width": 10.0})
    racket_path = RewTerm(func=mdp.racket_path_tracking, weight=0.5, params={
        "std": 0.08, "slack": 2, "impact_floor": 0.2, "impact_half_width": 10.0})
    decoder_match = RewTerm(func=mdp.decoder_match, weight=0.3, params={"std": 0.3, "arm_weight": 2.0})
    token_match = None  # absolute recipe only
    # stay near the prior (GRAIL): residual magnitude and smoothness
    residual_l2 = RewTerm(func=mdp.residual_l2, weight=-0.5)
    residual_rate = RewTerm(func=mdp.residual_rate_l2, weight=-0.5)
    # goal (sparse, high weight, gated around impact)
    racket_pos = RewTerm(func=mdp.racket_pos_tracking, weight=100.0,
                         params={"sigma_init": 0.10, "sigma_min": 0.02, "half_window": 0})
    racket_vel = RewTerm(func=mdp.racket_vel_tracking, weight=30.0,
                         params={"sigma_init": 0.5, "sigma_min": 0.1, "half_window": 2})
    racket_face = RewTerm(func=mdp.racket_face_tracking, weight=30.0,
                          params={"sigma_init": 0.5, "sigma_min": 0.1, "half_window": 2})
    strike_success = RewTerm(func=mdp.strike_success, weight=50.0)
    base_pos = RewTerm(func=mdp.base_pos_tracking, weight=1.0, params={"std": 0.1})
    # regularization
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)
    token_rate = None   # absolute recipe only (residual_rate covers it)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.25, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")})
    table_penetration = RewTerm(func=mdp.table_penetration, weight=-50.0,
                                params={"near_edge_x": TABLE_NEAR_EDGE_X, "top_z": TABLE_TOP_Z})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_contact = DoneTerm(func=mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link", "pelvis"]), "threshold": 1.0})
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class CurriculumCfg:
    # ~2000 PPO iterations x 24 steps: imitation weights fade so the racket goal terms take over
    pose_joint_decay = CurrTerm(func=mdp.linear_reward_weight, params={
        "term_name": "pose_joint", "start_weight": 1.0, "end_weight": 0.5, "start_step": 0, "end_step": 48_000})
    pose_joint_vel_decay = CurrTerm(func=mdp.linear_reward_weight, params={
        "term_name": "pose_joint_vel", "start_weight": 0.2, "end_weight": 0.1, "start_step": 0, "end_step": 48_000})
    racket_path_decay = CurrTerm(func=mdp.linear_reward_weight, params={
        "term_name": "racket_path", "start_weight": 0.5, "end_weight": 0.2, "start_step": 0, "end_step": 48_000})
    decoder_match_decay = CurrTerm(func=mdp.linear_reward_weight, params={
        "term_name": "decoder_match", "start_weight": 0.3, "end_weight": 0.1, "start_step": 0, "end_step": 48_000})
    token_match_decay = None


@configclass
class TokenHitterEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=4.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4               # 0.005 s physics, 50 Hz policy == SONIC rate
        self.sim.dt = 0.005
        self.episode_length_s = 10.0      # ~4 consecutive 2.5 s swings (HITTER)
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (2.0, -2.5, 1.8)
        self.viewer.lookat = (0.4, 0.0, 0.9)


@configclass
class TokenHitterEnvCfg_ABS(TokenHitterEnvCfg):
    """Previous recipe: absolute token tanh(a), BC init, token-space imitation."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.token.mode = "absolute"
        for g in (self.observations.policy, self.observations.critic):
            g.ref_token = None
            g.prev_residual = None
            g.prev_token = ObsTerm(func=mdp.prev_token)
        r, c = self.rewards, self.curriculum
        r.pose_joint = r.pose_joint_vel = r.racket_path = None
        r.residual_l2 = r.residual_rate = None
        r.token_match = RewTerm(func=mdp.token_match, weight=0.3, params={"std": 0.1, "slack": 3})
        r.decoder_match.weight = 0.7
        r.token_rate = RewTerm(func=mdp.token_rate_l2, weight=-0.5)
        c.pose_joint_decay = c.pose_joint_vel_decay = c.racket_path_decay = None
        c.token_match_decay = CurrTerm(func=mdp.linear_reward_weight, params={
            "term_name": "token_match", "start_weight": 0.3, "end_weight": 0.1, "start_step": 0, "end_step": 48_000})
        c.decoder_match_decay.params["start_weight"], c.decoder_match_decay.params["end_weight"] = 0.7, 0.2


@configclass
class TokenHitterEnvCfg_PLAY(TokenHitterEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.curriculum = None


@configclass
class TokenHitterEnvCfg_ABS_PLAY(TokenHitterEnvCfg_ABS):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.curriculum = None


@configclass
class TokenHitterEnvCfg_REPLAY(TokenHitterEnvCfg_PLAY):
    """Library tokens straight into the decoder (no policy): the closed-loop sanity check."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.token.mode = "replay"
        self.events.reset_joints = None
        self.events.reset_base = None
