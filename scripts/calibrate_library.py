"""Calibrate a library against SONIC: replay every clip (no policy), measure where the racket
and the pelvis really are at impact, drop clips SONIC cannot track, write the result.

Measured per clip, in the swing-start anchor frame (start pelvis xy / yaw, nominal height):
  contact_pos  racket contact point at impact      -> replaces the kinematic one (clip selection
                                                      and the planner's reach filter use it)
  root_disp    pelvis displacement at impact        -> base target of step clips
Kept if repeatable: no falls / torso contacts, spread of the measured racket contact <= --max_spread,
lateral pelvis-displacement spread <= --max_disp_sd, and SONIC's bias vs the clip's own contact
(pelvis frame; mostly a consistent upward offset, ~10-19 cm) <= --max_err. The bias is what the
calibration absorbs: the measured contact replaces the kinematic one.

  python scripts/calibrate_library.py --library data/forehand_library_v2.npz --num_envs 96 --headless --seconds 60
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--library", required=True)
ap.add_argument("--out", default=None, help="default: overwrite --library")
ap.add_argument("--num_envs", type=int, default=96)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--min_n", type=int, default=4)
ap.add_argument("--max_err", type=float, default=0.25, help="sanity cap on SONIC's bias (m)")
ap.add_argument("--max_spread", type=float, default=0.04, help="max spread of the measured contact (m)")
ap.add_argument("--max_disp_sd", type=float, default=0.05)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import env_unwrapped  # noqa: E402

TASK = "G1-SonicTT-Token-Replay-v0"


def main():
    import gymnasium as gym
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import g1_sonic_tt.tasks  # noqa: F401
    from g1_sonic_tt.tasks.token_hitter.mdp.racket import racket_state_w

    cfg = parse_env_cfg(TASK, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.commands.strike.library_file = args.library
    cfg.commands.strike.target_source = "library"
    env = gym.make(TASK, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env)
    base = env_unwrapped(env)
    dev, n = base.device, base.num_envs
    cmd = base.command_manager.get_term("strike")
    lib = cmd.library
    M = lib.num_motions
    # uniform clip draw, least-sampled first so every clip gets measured
    counts = torch.zeros(M, device=dev)

    def draw(pos):
        k = len(pos)
        w = 1.0 / (1.0 + counts)
        idx = torch.multinomial(w, k, replacement=True)
        counts.index_add_(0, idx, torch.ones(k, device=dev))
        return idx
    lib.nearest = draw
    robot = base.scene["robot"]
    origins = base.scene.env_origins
    tm = base.termination_manager
    wrapped.reset()
    start_p = robot.data.root_pos_w.clone()
    start_q = yaw_quat(robot.data.root_quat_w).clone()
    started = torch.zeros(n, dtype=torch.bool, device=dev)
    S = defaultdict(lambda: defaultdict(list))
    bad = torch.zeros(M, device=dev)
    act = torch.zeros(n, 64, device=dev)
    for _ in range(int(args.seconds / base.step_dt)):
        at_hit = cmd.at_hit.clone()
        motion = cmd.motion.clone()
        with torch.inference_mode():
            _, _, dones, extras = wrapped.step(act)
        d = dones.bool()
        fail = d & ~extras["time_outs"].bool()
        if fail.any():
            bad += torch.bincount(motion[fail], minlength=M).float()
        r = robot.data
        if at_hit.any():
            p, _, nrm = racket_state_w(base)
            yq = yaw_quat(r.root_quat_w)
            lib_w = r.root_pos_w + quat_apply(yq, lib.contact_traj[motion, lib.hit_frame])
            err = torch.norm(p - lib_w, dim=-1)
            anchor = start_p.clone()
            anchor[:, 2] = origins[:, 2] + lib.ref_root_height
            c_a = quat_apply_inverse(start_q, p - anchor)
            disp = quat_apply_inverse(start_q, r.root_pos_w - anchor)
            for i in (at_hit & ~d & started).nonzero().squeeze(-1).tolist():
                m = int(motion[i])
                S[m]["err"].append(float(err[i]))
                S[m]["c"].append(c_a[i].tolist())
                S[m]["disp"].append(disp[i].tolist())
        new = (cmd.frame == 0) & ~d
        start_p[new] = r.root_pos_w[new]
        start_q[new] = yaw_quat(r.root_quat_w[new])
        started = (started | new) & ~d

    data = dict(np.load(args.library))
    keys = [str(k) for k in data["motion_keys"]]
    keep = np.zeros(M, bool)
    cpos = data["contact_pos_kin"].copy() if "contact_pos_kin" in data else data["contact_pos"].copy()
    disp_m = data["root_disp"].copy() if "root_disp" in data else np.zeros((M, 3), np.float32)
    print(f"\n{'clip':30s} {'n':>3s} {'fail':>4s} {'err cm':>7s} {'contact kin -> meas (y,z) cm':>32s} {'disp y cm':>12s}")
    for m in range(M):
        s = S[m]
        nn = len(s["err"])
        if nn < args.min_n:
            print(f"{keys[m]:30s} {nn:3d} {int(bad[m]):4d}   too few samples")
            continue
        e = float(np.mean(s["err"]))
        c = np.mean(s["c"], 0)
        dsp = np.array(s["disp"])
        cs = np.array(s["c"])
        spread = float(np.linalg.norm(cs - c, axis=1).mean())
        ok = bad[m] == 0 and e <= args.max_err and spread <= args.max_spread and dsp[:, 1].std() <= args.max_disp_sd
        keep[m] = ok
        print(f"{keys[m]:30s} {nn:3d} {int(bad[m]):4d} {e * 100:7.1f} "
              f"   ({cpos[m, 1] * 100:+5.0f},{cpos[m, 2] * 100:+4.0f}) -> ({c[1] * 100:+5.0f},{c[2] * 100:+4.0f}) "
              f"{dsp[:, 1].mean() * 100:+6.1f}±{dsp[:, 1].std() * 100:3.1f}  spread {spread * 100:3.1f}"
              f"{'' if ok else '  DROP'}")
        cpos[m], disp_m[m] = c, dsp.mean(0)
    data["contact_pos"] = cpos.astype(np.float32)
    data["root_disp"] = disp_m.astype(np.float32)
    data["calibrated"] = np.bool_(True)
    per_clip = [k for k, v in data.items() if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == M]
    for k in per_clip:
        data[k] = data[k][keep]
    out = args.out or args.library
    np.savez_compressed(out, **data)
    kind = data.get("kind", np.zeros(keep.sum(), int))
    print(f"\nkept {keep.sum()}/{M} clips -> {out}  (by kind: "
          f"{', '.join(f'{k}: {(kind == k).sum()}' for k in np.unique(kind))}); failures {int(bad.sum())}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
