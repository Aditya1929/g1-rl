"""Simulator-free checks: decoder, history layout, FSQ, library, ball + planner.

  ~/miniconda3/envs/isaaclab/bin/python -m pytest -q tests      (or run this file directly)
Requires the generated data/ files (scripts/build_library.py, convert_decoder.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/g1_sonic_tt"))
from g1_sonic_tt import paths  # noqa: E402
from g1_sonic_tt.ball.model import BallParams, flight_step, racket_impact, sample_serves  # noqa: E402
from g1_sonic_tt.planner.strike import PlannerParams, plan  # noqa: E402
from g1_sonic_tt.sonic.decoder import DecoderHistory, SonicDecoder, load_decoder_params  # noqa: E402
from g1_sonic_tt.sonic.fsq import quantize, squash, unsquash  # noqa: E402
from g1_sonic_tt.sonic.library import SwingLibrary  # noqa: E402


def _np_forward(x):
    w = np.load(paths.DECODER_WEIGHTS)
    h = x.astype(np.float64)
    for i in range(7):
        h = h @ w[f"W{i}"] + w[f"b{i}"]
        if i < 6:
            h = h / (1 + np.exp(-h))
    return h


def test_decoder_matches_numpy():
    dec = SonicDecoder(paths.DECODER_WEIGHTS)
    x = np.random.default_rng(1).normal(0, 0.5, (8, 994)).astype(np.float32)
    out = dec(torch.from_numpy(x)).numpy()
    assert np.abs(out - _np_forward(x)).max() < 1e-4


def test_history_layout_matches_verify_decoder():
    """Same layout as sonic-dataset/verify_decoder.py::decoder_input (oldest first, feature-major)."""
    default, scale = load_decoder_params(paths.DECODER_PARAMS_JSON)
    h = DecoderHistory(1, default, scale, "cpu")
    rng = np.random.default_rng(2)
    frames = [dict(w=rng.normal(size=3), q=default.numpy() + rng.normal(0, .1, 29), dq=rng.normal(size=29),
                   g=rng.normal(size=3)) for _ in range(12)]
    acts = [rng.normal(size=29) for _ in range(12)]
    for i, f in enumerate(frames):
        h.push(*(torch.tensor(f[k][None], dtype=torch.float32) for k in ("w", "q", "dq", "g")))
        h.record_action(torch.tensor(acts[i][None], dtype=torch.float32))
    last = frames[-10:]
    # last_action entry for frame i is the action recorded after frame i-1
    la = [acts[i - 1] for i in range(2, 12)]
    expect = np.concatenate([np.concatenate([f["w"] for f in last]),
                             np.concatenate([f["q"] - default.numpy() for f in last]),
                             np.concatenate([f["dq"] for f in last]),
                             np.concatenate(la),
                             np.concatenate([f["g"] for f in last])])
    assert np.allclose(h.proprio()[0].numpy(), expect, atol=1e-5)


def test_fsq_roundtrip_library_tokens():
    lib = SwingLibrary(paths.FOREHAND_LIBRARY)
    z = lib.tokens.reshape(-1, 64)
    assert torch.equal(quantize(z), z), "library tokens should already lie on the FSQ grid"
    assert torch.equal(quantize(squash(unsquash(z))), z), "BC targets must survive squash/quantize"


def test_library_nearest():
    lib = SwingLibrary(paths.FOREHAND_LIBRARY)
    idx = lib.nearest(lib.contact_pos)
    assert torch.equal(idx, torch.arange(lib.num_motions))
    assert lib.hit_frame == 50 and lib.num_frames == 126


def test_planner_on_serves():
    prm = PlannerParams()
    p0, v0 = sample_serves(256, prm.ball, "cpu", gen=torch.Generator().manual_seed(0))
    valid, t_hit, p_hit, v_r, v_o = plan(p0, v0, prm)
    assert valid.float().mean() > 0.9, valid.float().mean()
    assert torch.allclose(p_hit[valid, 0], torch.full_like(p_hit[valid, 0], prm.plane_x), atol=1e-3)
    print(f"\n  serves: t_hit {t_hit[valid].mean():.2f}s, p_hit z {p_hit[valid, 2].mean():.2f} "
          f"y [{p_hit[valid, 1].min():.2f}, {p_hit[valid, 1].max():.2f}], |v_racket| {v_r[valid].norm(dim=-1).mean():.2f} m/s")


def test_racket_impact_returns_ball():
    """A racket moving at the planned velocity, face along it, sends the ball roughly toward v_out."""
    prm = PlannerParams()
    p0, v0 = sample_serves(64, prm.ball, "cpu", gen=torch.Generator().manual_seed(1))
    valid, t_hit, p_hit, v_r, v_o = plan(p0, v0, prm)
    p, v = p0.clone(), v0.clone()
    n = torch.nn.functional.normalize(v_r, dim=-1)
    hit_any = torch.zeros(64, dtype=torch.bool)
    for k in range(int(1.5 / prm.dt)):
        t = k * prm.dt
        rp0 = p_hit + v_r * (t - t_hit)[:, None]
        rp1 = p_hit + v_r * (t + prm.dt - t_hit)[:, None]
        p1, v1, _ = flight_step(p, v, prm.dt, prm.ball)
        hit, v1 = racket_impact(p, p1, v1, rp0, rp1, v_r, n, 0.075, prm.ball)
        hit = hit & ~hit_any
        hit_any |= hit
        if hit.any():
            cos = torch.nn.functional.cosine_similarity(v1[hit], v_o[hit], dim=-1)
            assert (cos > 0.8).all(), cos
        p, v = p1, v1
    assert hit_any[valid].float().mean() > 0.9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
