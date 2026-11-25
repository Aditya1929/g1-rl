"""Simulator-free checks for deploy/: wire format, frames, obs builder."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "source/g1_sonic_tt"))

from frames import TrackerToSim  # noqa: E402
from token_stream import HEADER_SIZE, pack  # noqa: E402


def test_pack_layout():
    tok = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    b = pack(tok, 7)
    assert len(b) == len(b"pose") + HEADER_SIZE + 64 * 4 + 8
    body = b[len(b"pose") + HEADER_SIZE:]
    assert np.allclose(np.frombuffer(body[:256], "<f4"), tok)
    assert np.frombuffer(body[256:], "<i8")[0] == 7


def test_tracker_frame():
    c = TrackerToSim(near_edge_x=0.55, top_z=0.76)
    # tag 5 (near-edge centre, on the tabletop) -> sim near edge, table height
    assert np.allclose(c.pos([0, 0, 0]), [0.55, 0, 0.76])
    # tracker +Y (toward net) -> sim +X;  tracker +X (robot's right) -> sim -Y
    assert np.allclose(c.vel([0, 1, 0]), [1, 0, 0])
    assert np.allclose(c.vel([1, 0, 0]), [0, -1, 0])
    assert np.isclose(c.yaw(np.pi / 2), 0.0)


RESIDUAL_TERMS = [("base_ang_vel", [3]), ("projected_gravity", [3]), ("forward_vec", [2]),
                  ("base_target_err", [2]), ("racket_target_pos", [3]), ("racket_target_vel", [3]),
                  ("racket_target_normal", [3]), ("t_strike", [1]), ("swing_type", [2]), ("joint_pos", [29]),
                  ("joint_vel", [29]), ("ref_token", [64]), ("prev_residual", [64])]


def _builder(d, terms, action):
    from bridge import ObsBuilder
    from g1_sonic_tt.sonic.constants import SONIC_JOINTS

    sim_joints = sorted(SONIC_JOINTS)  # any permutation; the builder maps by name
    default = np.arange(29) * 0.01
    p = Path(d) / "obs_spec.json"
    p.write_text(json.dumps({"terms": terms, "joint_names": sim_joints, "default_joint_pos": default.tolist(),
                             "action": action}))
    q_sonic = np.array([default[sim_joints.index(n)] for n in SONIC_JOINTS])  # robot at default pose
    return ObsBuilder(p), q_sonic


def test_obs_builder_dims():
    from bridge import NominalRobotSource

    with tempfile.TemporaryDirectory() as d:
        ob, q_sonic = _builder(d, RESIDUAL_TERMS, {"mode": "residual", "residual_scale": 0.1})
        rs = NominalRobotSource(q_sonic).read()
        z_ref = np.full(64, 0.125, np.float32)
        obs = ob.build(rs, np.zeros(3), np.array([0.0, 2.0, 0.0]), 0.5, z_ref, np.zeros(64), np.zeros(64))
    assert obs.shape == (1, 208)
    assert np.allclose(obs[0, 3:6].numpy(), [0, 0, -1])            # gravity, upright
    assert np.allclose(obs[0, 16:19].numpy(), [0, 1, 0])           # normal target = velocity direction
    assert np.allclose(obs[0, 22:51].numpy(), 0, atol=1e-6)        # joint_pos_rel == 0 at default
    assert np.allclose(obs[0, 80:144].numpy(), 0.125)              # z_ref


def test_residual_token_matches_env_rule():
    import torch
    from g1_sonic_tt.sonic.fsq import quantize

    with tempfile.TemporaryDirectory() as d:
        ob, _ = _builder(d, RESIDUAL_TERMS, {"mode": "residual", "residual_scale": 0.1})
    z_ref = (np.arange(64) % 17 - 8).astype(np.float32) / 16       # on the FSQ grid
    a = torch.linspace(-3, 3, 64)[None]
    tok, dz = ob.token(a, z_ref)
    want = quantize(torch.from_numpy(z_ref) + 0.1 * a[0].clamp(-1, 1)).numpy()
    assert np.allclose(tok, want) and np.allclose(dz, a[0].clamp(-1, 1).numpy())
    tok0, _ = ob.token(torch.zeros(1, 64), z_ref)                   # a = 0 replays the library
    assert np.allclose(tok0, z_ref)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
