"""Analytic ping-pong ball (HITTER Eq. 1-2): quadratic drag, table bounce, racket impact.

Batched torch, no simulator dependency. Used by
  * planner/strike.py  (prediction)
  * scripts/eval_ball.py (ground-truth ball in Isaac Lab eval)
PhysX is deliberately not used for the ball: a 2.7 g sphere at 5-10 m/s against a
thin racket tunnels / bounces wrongly.

Frame: the sim world frame of an env (robot at origin facing +X, table spans
x in [near_edge, near_edge + 2.74], tabletop at z = top_z, net at the midpoint).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

G = (0.0, 0.0, -9.81)
BALL_RADIUS = 0.02


@dataclass
class BallParams:
    drag_k: float = 0.13          # 0.5 rho Cd A / m for a 40 mm, 2.7 g ball  [1/m]
    table_cv: float = 0.90        # vertical restitution (fit on real bounces!)
    table_ch: float = 0.75        # horizontal velocity retention
    racket_cr: float = 0.85       # racket normal restitution (HITTER C_r)
    racket_mu: float = 0.3        # tangential velocity loss on the racket
    near_edge_x: float = 0.55
    table_length: float = 2.74
    table_half_width: float = 0.7625
    top_z: float = 0.76

    @property
    def net_x(self) -> float:
        return self.near_edge_x + self.table_length / 2


def _over_table(p, prm: BallParams):
    return (p[..., 0] >= prm.near_edge_x) & (p[..., 0] <= prm.near_edge_x + prm.table_length) & \
           (p[..., 1].abs() <= prm.table_half_width)


def flight_step(p, v, dt: float, prm: BallParams):
    """One semi-implicit Euler step with drag + table bounce. Returns p, v, bounced mask."""
    g = torch.tensor(G, device=p.device, dtype=p.dtype)
    a = -prm.drag_k * v.norm(dim=-1, keepdim=True) * v + g
    v = v + a * dt
    p = p + v * dt
    z_c = prm.top_z + BALL_RADIUS
    bounce = (p[..., 2] < z_c) & (v[..., 2] < 0) & _over_table(p, prm)
    if bounce.any():
        c = torch.tensor([prm.table_ch, prm.table_ch, -prm.table_cv], device=p.device, dtype=p.dtype)
        v = torch.where(bounce[..., None], v * c, v)
        p = p.clone()
        p[..., 2] = torch.where(bounce, 2 * z_c - p[..., 2], p[..., 2])
    return p, v, bounce


def racket_impact(p0, p1, v, racket_p0, racket_p1, racket_v, normal, radius: float, prm: BallParams):
    """Swept ball-vs-racket-disk test over one substep and the impact response.

    p0/p1: ball positions at the start/end of the substep; racket_p0/p1 likewise.
    Returns (hit mask, post-impact velocity)."""
    d0 = ((p0 - racket_p0) * normal).sum(-1)
    d1 = ((p1 - racket_p1) * normal).sum(-1)
    crossed = (d0 * d1 <= 0) | (d1.abs() < BALL_RADIUS)
    s = (d0 / (d0 - d1).where((d0 - d1).abs() > 1e-9, torch.ones_like(d0))).clamp(0, 1)[..., None]
    pc = p0 + s * (p1 - p0)
    rc = racket_p0 + s * (racket_p1 - racket_p0)
    rel = pc - rc
    radial = (rel - (rel * normal).sum(-1, keepdim=True) * normal).norm(dim=-1)
    approaching = (((v - racket_v) * normal).sum(-1) * torch.sign(d0 + 1e-9)) < 0
    hit = crossed & (radial <= radius) & approaching
    vrel = v - racket_v
    vn = (vrel * normal).sum(-1, keepdim=True) * normal
    vt = vrel - vn
    v_out = racket_v - prm.racket_cr * vn + (1 - prm.racket_mu) * vt
    return hit, torch.where(hit[..., None], v_out, v)


def sample_serves(n: int, prm: BallParams, device, y_bounce=(-0.55, -0.2), x_bounce=(0.85, 1.45),
                  flight_time=(0.40, 0.60), gen: torch.Generator | None = None):
    """Launch balls from the opponent end so they bounce once on the robot's half.
    Ballistic (drag-free) solve; drag makes them land slightly short, which is fine for eval."""
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=device, generator=gen)  # noqa: E731
    p0 = torch.stack([u(prm.near_edge_x + prm.table_length - 0.05, prm.near_edge_x + prm.table_length + 0.25),
                      u(-0.3, 0.3), u(prm.top_z + 0.15, prm.top_z + 0.35)], -1)
    pb = torch.stack([u(*x_bounce), u(*y_bounce), torch.full((n,), prm.top_z + BALL_RADIUS, device=device)], -1)
    T = u(*flight_time)[:, None]
    g = torch.tensor(G, device=device)
    v0 = (pb - p0) / T - 0.5 * g * T
    return p0, v0
