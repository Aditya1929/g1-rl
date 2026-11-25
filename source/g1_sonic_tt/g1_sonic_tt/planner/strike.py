"""Model-based strike planner (HITTER Sec. IV): ball state -> (strike point, strike time,
racket velocity). Batched torch; used in Isaac Lab eval and in deploy/bridge.py.

  1. Integrate the ball (drag + table bounce) until it crosses the strike plane
     x = plane_x moving toward the robot (-X), after its bounce on the robot's half.
  2. Outgoing velocity to land at p_land after dt_out (HITTER Eq. 5):
        v_o = (p_land - p_hit) / dt_out - 0.5 * g * dt_out
  3. Racket velocity (HITTER Eq. 6), face normal along u:
        u = (v_o - v_i) / |v_o - v_i|,  v_r = (v_o.u + C_r v_i.u) / (1 + C_r) * u
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from g1_sonic_tt.ball.model import G, BallParams, flight_step


@dataclass
class PlannerParams:
    plane_x: float = 0.43            # strike plane (world x); library contacts are 0.43 m ahead of the pelvis
    horizon_s: float = 1.5
    dt: float = 0.005
    dt_out: float = 0.55             # flight time of the return
    land_offset: tuple[float, float] = (0.68, 0.0)  # landing point: net_x + offset (opponent half centre)
    require_bounce: bool = True
    ball: BallParams = field(default_factory=BallParams)


@torch.no_grad()
def predict_crossing(p, v, prm: PlannerParams):
    """Returns (valid, t_hit, p_hit, v_in). Shapes (N,), (N,), (N,3), (N,3)."""
    n = p.shape[0]
    valid = torch.zeros(n, dtype=torch.bool, device=p.device)
    t_hit = torch.full((n,), float("nan"), device=p.device)
    p_hit = torch.zeros_like(p)
    v_in = torch.zeros_like(v)
    bounced = torch.zeros_like(valid)
    p, v = p.clone(), v.clone()
    for k in range(int(prm.horizon_s / prm.dt)):
        p_prev = p
        p, v, b = flight_step(p, v, prm.dt, prm.ball)
        bounced |= b
        ok = bounced if prm.require_bounce else torch.ones_like(bounced)
        cross = (~valid) & ok & (p_prev[:, 0] >= prm.plane_x) & (p[:, 0] < prm.plane_x) & (v[:, 0] < 0)
        if cross.any():
            s = ((p_prev[:, 0] - prm.plane_x) / (p_prev[:, 0] - p[:, 0]).clamp(min=1e-9))[:, None]
            p_hit = torch.where(cross[:, None], p_prev + s * (p - p_prev), p_hit)
            v_in = torch.where(cross[:, None], v, v_in)
            t_hit = torch.where(cross, (k + s[:, 0]) * prm.dt, t_hit)
            valid |= cross
        if valid.all():
            break
    return valid, t_hit, p_hit, v_in


def racket_velocity(p_hit, v_in, prm: PlannerParams):
    g = torch.tensor(G, device=p_hit.device)
    land = torch.tensor([prm.ball.net_x + prm.land_offset[0], prm.land_offset[1], prm.ball.top_z],
                        device=p_hit.device).expand_as(p_hit)
    v_o = (land - p_hit) / prm.dt_out - 0.5 * g * prm.dt_out
    u = torch.nn.functional.normalize(v_o - v_in, dim=-1)
    cr = prm.ball.racket_cr
    mag = ((v_o * u).sum(-1) + cr * (v_in * u).sum(-1)) / (1 + cr)
    return mag[:, None] * u, v_o


@torch.no_grad()
def plan(p, v, prm: PlannerParams):
    """Full plan: (valid, t_hit, p_hit, racket_vel, v_out)."""
    valid, t_hit, p_hit, v_in = predict_crossing(p, v, prm)
    v_r, v_o = racket_velocity(p_hit, v_in, prm)
    return valid, t_hit, p_hit, v_r, v_o
