"""HITTER-style strike command: consecutive swings, each with a racket target
(position + velocity at impact), a base target and a strike time.

In training there is no ball in the robot's scene, but the targets come from the
same planner used at deployment (target_source="planner", default): analytic serves
aimed at the robot's forehand are run through planner.strike.plan, and its strike
point, racket velocity, face normal (= velocity direction) and strike time become
the swing's targets. Only strikes inside the library window (+- target_margin) are
kept; the pool (~70k strikes) is built once at startup. target_source="library" samples around the clips' own contacts instead.
At evaluation / deployment the planner drives the same fields via `set_external_targets`.

Frames: each swing has an *anchor* frame, fixed for the whole swing: the root's
xy and yaw at swing start, z at the clips' nominal pelvis height (the deploy bridge
uses the same thing: a static nominal robot pose in the table frame). The racket
target (position, velocity, face normal) is stored in the anchor frame, so it is a
fixed point in the world: rewards score the racket against it even when the pelvis
sways during the swing (the ball does not move with the pelvis). Observations show
the target in the anchor frame (+X forward, +Y left). Base target is a world-xy point.

Refinement noise (planner source, training only): the *observed* target starts off
by a random error that shrinks linearly to zero at `refine_commit_s` before impact,
like a planner whose prediction sharpens as the ball approaches; rewards use the
true target. Eval / deploy feed real refinements through update_external_targets_w.

Face normal target: in training it is the library's (face-corrected) normal at
impact plus a small random tilt; from the planner it defaults to the racket
velocity direction (HITTER: face perpendicular to the racket velocity).

Timing: each swing replays one library clip (T frames, impact at `hit_frame`).
`frame` is the clip frame whose token is commanded at this step; the decision
at `frame` is executed over the following 20 ms, so the racket state seen by the
rewards after the step corresponds to frame + 1. Impact rewards are therefore
gated on `frame + 1 == hit_frame`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from g1_sonic_tt.ball.model import BallParams, sample_serves
from g1_sonic_tt.planner.strike import PlannerParams, plan
from g1_sonic_tt.sonic.library import SwingLibrary

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StrikeCommand(CommandTerm):
    cfg: "StrikeCommandCfg"

    def __init__(self, cfg: "StrikeCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.library = SwingLibrary(cfg.library_file, device=self.device)
        self.hit_frame = self.library.hit_frame
        self.num_frames = self.library.num_frames
        n = self.num_envs
        self.motion = torch.zeros(n, dtype=torch.long, device=self.device)
        self.frame = torch.zeros(n, dtype=torch.long, device=self.device)
        self.racket_target_pos_b = torch.zeros(n, 3, device=self.device)
        self.racket_target_vel_b = torch.zeros(n, 3, device=self.device)
        self.racket_target_normal_b = torch.zeros(n, 3, device=self.device)
        self.racket_target_normal_b[:, 0] = 1.0
        # true targets (anchor frame); racket_target_*_b above are the observed (possibly noisy) ones
        self._true_pos_b = torch.zeros(n, 3, device=self.device)
        self._true_vel_b = torch.zeros(n, 3, device=self.device)
        self._true_normal_b = self.racket_target_normal_b.clone()
        self._noise_pos = torch.zeros(n, 3, device=self.device)
        self._noise_vel = torch.zeros(n, 3, device=self.device)
        self._t0 = torch.zeros(n, device=self.device)
        self.anchor_pos_w = torch.zeros(n, 3, device=self.device)
        self.anchor_quat_w = torch.zeros(n, 4, device=self.device)
        self.anchor_quat_w[:, 0] = 1.0
        self.base_target_xy = torch.zeros(n, 2, device=self.device)  # env-origin relative
        self.is_forehand = torch.ones(n, dtype=torch.bool, device=self.device)
        # external (planner-driven) targets: when set, sampling is skipped
        self.external = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ball = BallParams(near_edge_x=cfg.table_near_edge_x, top_z=cfg.table_top_z)
        self._planner = PlannerParams(ball=self._ball)
        self.metrics["racket_pos_err_at_hit"] = torch.zeros(n, device=self.device)
        if cfg.target_source == "planner":
            self._build_planner_pool()
        self.metrics["swings"] = torch.zeros(n, device=self.device)

    # ----- quantities used by obs / rewards / the action term -----
    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.racket_target_pos_b, self.racket_target_vel_b, self.racket_target_normal_b,
                          self.t_strike[:, None]], dim=1)

    @property
    def t_strike(self) -> torch.Tensor:
        """Seconds from now until impact (negative after impact, during follow-through/return)."""
        return (self.hit_frame - self.frame).float() * self._env.step_dt

    @property
    def z_ref(self) -> torch.Tensor:
        return self.library.token(self.motion, self.frame)

    def z_ref_window(self, k: int) -> torch.Tensor:
        return self.library.token_window(self.motion, self.frame, k)

    def z_ref_future(self, offsets: Sequence[int]) -> torch.Tensor:
        return torch.cat([self.library.token(self.motion, self.frame + o) for o in offsets], dim=1)

    @property
    def ref_q(self) -> torch.Tensor:
        return self.library.ref_q[self.motion, self.frame.clamp(max=self.num_frames - 1)]

    @property
    def ref_dq(self) -> torch.Tensor:
        return self.library.ref_dq[self.motion, self.frame.clamp(max=self.num_frames - 1)]

    # ----- reference state the robot should be in *after* this step (frame + 1) -----
    def _next_window(self, table: torch.Tensor, k: int) -> torch.Tensor:
        """(N, 2k+1, D): table[motion, frame + 1 + j] for j in [-k, k] (clamped): timing slack."""
        offs = torch.arange(-k, k + 1, device=self.device)
        f = (self.frame[:, None] + 1 + offs[None]).clamp(0, self.num_frames - 1)
        return table[self.motion[:, None], f]

    def ref_q_window(self, k: int = 0) -> torch.Tensor:
        return self._next_window(self.library.ref_q, k)

    def ref_dq_window(self, k: int = 0) -> torch.Tensor:
        return self._next_window(self.library.ref_dq, k)

    def ref_racket_window(self, k: int = 0) -> torch.Tensor:
        """Reference racket contact path (pelvis frame of the clip == root yaw frame, root-relative)."""
        return self._next_window(self.library.contact_traj, k)

    def impact_gate(self, half_width: float) -> torch.Tensor:
        """Smooth window around impact: ~1 within +-half_width frames, ~0 outside (super-Gaussian)."""
        d = ((self.frame + 1) - self.hit_frame).float() / half_width
        return torch.exp(-0.5 * d**4)

    @property
    def at_hit(self) -> torch.Tensor:
        """True on the step whose outcome is the impact instant."""
        return (self.frame + 1) == self.hit_frame

    def near_hit(self, half_window: int) -> torch.Tensor:
        return ((self.frame + 1) - self.hit_frame).abs() <= half_window

    @property
    def before_hit(self) -> torch.Tensor:
        return (self.frame + 1) <= self.hit_frame

    @property
    def swing_onehot(self) -> torch.Tensor:
        return torch.stack([self.is_forehand, ~self.is_forehand], dim=1).float()

    def racket_target_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        """True racket target (world): fixed in space for the whole swing."""
        pos = self.anchor_pos_w + quat_apply(self.anchor_quat_w, self._true_pos_b)
        return pos, quat_apply(self.anchor_quat_w, self._true_vel_b)

    def racket_target_normal_world(self) -> torch.Tensor:
        return quat_apply(self.anchor_quat_w, self._true_normal_b)

    # ----- anchor frame -----
    def _set_anchor(self, ids):
        self.anchor_quat_w[ids] = yaw_quat(self.robot.data.root_quat_w[ids])
        a = self.robot.data.root_pos_w[ids].clone()
        a[:, 2] = self._env.scene.env_origins[ids, 2] + self.library.ref_root_height
        self.anchor_pos_w[ids] = a

    def world_to_anchor(self, ids, pos_w, vel_w):
        q = self.anchor_quat_w[ids]
        return quat_apply_inverse(q, pos_w - self.anchor_pos_w[ids]), quat_apply_inverse(q, vel_w)

    def _set_true(self, ids, pos_b, vel_b, normal_b, noisy: bool = False):
        self._true_pos_b[ids] = pos_b
        self._true_vel_b[ids] = vel_b
        self._true_normal_b[ids] = torch.nn.functional.normalize(normal_b, dim=-1)
        k = len(self._true_pos_b[ids])
        sp, sv = (self.cfg.refine_noise_pos, self.cfg.refine_noise_vel) if noisy else (0.0, 0.0)
        self._noise_pos[ids] = torch.randn(k, 3, device=self.device) * sp
        self._noise_vel[ids] = torch.randn(k, 3, device=self.device) * sv
        self._t0[ids] = self.t_strike[ids]
        self._refresh_observed(ids)

    def _refresh_observed(self, ids=slice(None)):
        c = self.cfg.refine_commit_s
        d = ((self.t_strike[ids] - c) / (self._t0[ids] - c).clamp(min=1e-6)).clamp(0, 1)[:, None]
        self.racket_target_pos_b[ids] = self._true_pos_b[ids] + d * self._noise_pos[ids]
        v = self._true_vel_b[ids] + d * self._noise_vel[ids]
        self.racket_target_vel_b[ids] = v
        noisy = (self._noise_vel[ids].abs().sum(-1, keepdim=True) > 0) & (d > 0)
        self.racket_target_normal_b[ids] = torch.where(
            noisy, torch.nn.functional.normalize(v, dim=-1), self._true_normal_b[ids])

    # ----- external targets (ball planner at eval / deploy) -----
    def set_external_targets(self, env_ids, pos_b, vel_b, frames_to_hit, motion=None, normal_b=None):
        """Drive swings from a planner, targets given in the new anchor frame (set to the robot's
        current xy / yaw, nominal pelvis height). frames_to_hit: steps until impact (sets the
        clip frame). normal_b defaults to the racket velocity direction."""
        self.external[env_ids] = True
        self._set_anchor(env_ids)
        self.frame[env_ids] = (self.hit_frame - frames_to_hit).clamp(0, self.num_frames - 1)
        self._set_true(env_ids, pos_b, vel_b, vel_b if normal_b is None else normal_b)
        self.motion[env_ids] = self.library.nearest(pos_b) if motion is None else motion
        self._set_base_target(env_ids)

    def _set_base_target(self, ids, offset_xy=None):
        """Base target = swing-start anchor + the clip's pelvis displacement at impact (step clips)."""
        d = torch.zeros(len(self.motion[ids]), 3, device=self.device)
        d[:, :2] = self.library.root_disp[self.motion[ids], :2]
        if offset_xy is not None:
            d[:, :2] += offset_xy
        xy = self.anchor_pos_w[ids, :2] + quat_apply(self.anchor_quat_w[ids], d)[:, :2]
        self.base_target_xy[ids] = xy - self._env.scene.env_origins[ids, :2]

    def set_external_targets_w(self, env_ids, pos_w, vel_w, frames_to_hit, motion=None):
        """As set_external_targets, with the planner's world-frame strike point / racket velocity."""
        self._set_anchor(env_ids)
        pos_b, vel_b = self.world_to_anchor(env_ids, pos_w, vel_w)
        self.set_external_targets(env_ids, pos_b, vel_b, frames_to_hit, motion)

    def update_external_targets_w(self, env_ids, pos_w, vel_w):
        """Planner refinement during a swing (keeps the anchor, clock and clip)."""
        pos_b, vel_b = self.world_to_anchor(env_ids, pos_w, vel_w)
        self._set_true(env_ids, pos_b, vel_b, vel_b)

    # ----- CommandTerm interface -----
    def reset(self, env_ids: Sequence[int] | None = None):
        # an episode reset always drops planner-driven targets and starts a fresh sampled swing
        self.external[slice(None) if env_ids is None else env_ids] = False
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids[~self.external[env_ids]]
        if len(env_ids) == 0:
            return
        n, cfg, lib = len(env_ids), self.cfg, self.library
        self.is_forehand[env_ids] = True  # TODO: backhand once a backhand token library exists
        u = lambda lo, hi: torch.empty(n, device=self.device).uniform_(lo, hi)  # noqa: E731
        pos, vel, normal = self._library_targets(n)
        frame = torch.zeros(n, dtype=torch.long, device=self.device)
        if cfg.target_source == "planner":
            pos, vel, t_p = self._planner_targets(n)
            normal = torch.nn.functional.normalize(vel, dim=-1)
            # start the clip so its hit frame lands at the planner's strike time
            frame = (self.hit_frame - torch.round(t_p / self._env.step_dt).long()).clamp(0, self.hit_frame - 1)
        elif cfg.target_source != "library":
            raise ValueError(cfg.target_source)
        self.frame[env_ids] = frame
        self.motion[env_ids] = lib.nearest(pos)
        self._set_anchor(env_ids)
        self._set_true(env_ids, pos, vel, normal, noisy=cfg.target_source == "planner")
        self._set_base_target(env_ids, torch.stack([u(*cfg.base_offset_x), u(*cfg.base_offset_y)], 1))
        self.metrics["swings"][env_ids] += 1

    def _library_targets(self, n: int):
        """Targets around the clips' own contacts (contact velocity / face normal + jitter)."""
        cfg, lib = self.cfg, self.library
        u = lambda lo, hi: torch.empty(n, device=self.device).uniform_(lo, hi)  # noqa: E731
        m = cfg.target_margin
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = lib.contact_pos[:, 0].mean() + u(*cfg.strike_x_jitter)
        pos[:, 1] = u(float(lib.pos_min[1]) - m, float(lib.pos_max[1]) + m)
        pos[:, 2] = u(float(lib.pos_min[2]) - m, float(lib.pos_max[2]) + m)
        motion = lib.nearest(pos)
        vel = lib.contact_vel[motion] + torch.empty(n, 3, device=self.device).uniform_(-1, 1) * torch.tensor(
            cfg.vel_jitter, device=self.device)
        tilt = torch.randn(n, 3, device=self.device) * cfg.normal_jitter
        return pos, vel, torch.nn.functional.normalize(lib.face_normal[motion] + tilt, dim=-1)

    @torch.no_grad()
    def _build_planner_pool(self):
        """Serve analytic balls, plan each strike, keep those inside the library window.
        Frame: robot at its nominal pose (env origin, facing +X, pelvis at the clips' height),
        the same assumption the deploy bridge makes. Built once; swings draw from it."""
        cfg, lib = self.cfg, self.library
        root = torch.tensor([0.0, 0.0, lib.ref_root_height], device=self.device)
        pos, vel, t = [], [], []
        for _ in range(cfg.planner_pool_batches):
            p0, v0 = sample_serves(cfg.planner_pool_batch, self._ball, self.device, y_bounce=cfg.serve_y_bounce)
            valid, t_hit, p_hit, v_r, _ = plan(p0, v0, self._planner)
            pb = p_hit - root
            good = valid & (lib.reach(pb) <= cfg.reach_radius)  # within reach of some clip
            pos.append(pb[good]), vel.append(v_r[good]), t.append(t_hit[good])
        self._pool_pos, self._pool_vel, self._pool_t = torch.cat(pos), torch.cat(vel), torch.cat(t)
        n_srv = cfg.planner_pool_batches * cfg.planner_pool_batch
        assert len(self._pool_pos) > 0, "no planner strikes fall within reach of the library"
        print(f"[StrikeCommand] planner pool: {len(self._pool_pos)} strikes in window from {n_srv} serves; "
              f"|v_r| mean {self._pool_vel.norm(dim=-1).mean():.2f} m/s, t_hit mean {self._pool_t.mean():.2f} s")

    def _planner_targets(self, n: int):
        i = torch.randint(len(self._pool_pos), (n,), device=self.device)
        return self._pool_pos[i], self._pool_vel[i], self._pool_t[i]

    def _update_command(self):
        self.frame += 1
        done = torch.nonzero(self.frame >= self.num_frames).squeeze(-1)
        if len(done) > 0:
            self.external[done] = False
            self._resample_command(done)
        self._refresh_observed()

    def _update_metrics(self):
        hit = self.at_hit
        if hit.any():
            from .racket import racket_state_w  # local import to avoid a cycle
            p, _, _ = racket_state_w(self._env, self.cfg.asset_name)
            tgt, _ = self.racket_target_world()
            err = torch.norm(p - tgt, dim=-1)
            self.metrics["racket_pos_err_at_hit"][hit] = err[hit]

    # ----- debug visualization -----
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_target_vis"):
                self._target_vis = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                self._racket_vis = VisualizationMarkers(self.cfg.racket_visualizer_cfg)
            self._target_vis.set_visibility(True)
            self._racket_vis.set_visibility(True)
        elif hasattr(self, "_target_vis"):
            self._target_vis.set_visibility(False)
            self._racket_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        from .racket import racket_state_w
        tgt, _ = self.racket_target_world()
        self._target_vis.visualize(tgt)
        p, _, _ = racket_state_w(self._env, self.cfg.asset_name)
        self._racket_vis.visualize(p)


def _sphere(path, color, r=0.03):
    return VisualizationMarkersCfg(prim_path=path, markers={"s": sim_utils.SphereCfg(
        radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))})


@configclass
class StrikeCommandCfg(CommandTermCfg):
    class_type: type = StrikeCommand
    asset_name: str = "robot"
    library_file: str = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # swings resample themselves
    # racket target sampling around the library contact grid (pelvis yaw frame)
    target_source: str = "planner"         # planner (deployment distribution) | library (around clip contacts)
    target_margin: float = 0.03            # library source: (y, z) contact box widened by this much
    reach_radius: float = 0.08             # planner source: keep strikes within this of some clip contact
    serve_y_bounce: tuple[float, float] = (-0.55, -0.3)  # serve bounce lateral range (aimed at the forehand)
    planner_pool_batch: int = 50_000       # serves per planner batch; ~1/3 land in the window
    planner_pool_batches: int = 4
    table_near_edge_x: float = 0.55        # must match the env's table
    table_top_z: float = 0.76
    strike_x_jitter: tuple[float, float] = (-0.02, 0.02)
    vel_jitter: tuple[float, float, float] = (0.15, 0.15, 0.1)
    normal_jitter: float = 0.1             # ~0.1 rad random tilt of the face-normal target
    # observed-target refinement noise (planner source): std at swing start, 0 at refine_commit_s
    refine_noise_pos: float = 0.03
    refine_noise_vel: float = 0.1
    refine_commit_s: float = 0.3
    # base target offsets from the swing-start position (library pelvis never moves -> start at 0)
    base_offset_x: tuple[float, float] = (0.0, 0.0)
    base_offset_y: tuple[float, float] = (0.0, 0.0)
    target_visualizer_cfg: VisualizationMarkersCfg = _sphere("/Visuals/Command/racket_target", (0.0, 1.0, 0.0))
    racket_visualizer_cfg: VisualizationMarkersCfg = _sphere("/Visuals/Command/racket_actual", (1.0, 0.2, 0.2), 0.02)
