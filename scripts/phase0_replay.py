"""Phase 0: replay a test library (phase0_build.py) through the frozen SONIC decoder, no policy.

Per clip: falls, racket error at impact vs the clip's own contact point (pelvis yaw frame,
so it isolates SONIC tracking from where the robot stands), face-normal error, and the
pelvis displacement (swing start -> impact and -> clip end) vs the clip's commanded
root_disp (side-step clips). Clips are drawn uniformly at random.

  python scripts/phase0_replay.py --library data/phase0_sidestep.npz --num_envs 48 --headless --seconds 30
  python scripts/phase0_replay.py --library data/phase0_backhand.npz --num_envs 4     # GUI
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--library", required=True)
ap.add_argument("--num_envs", type=int, default=48)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--match", default=None, help="only replay clips whose key contains this (e.g. step)")
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
    M, T = lib.num_motions, lib.num_frames
    pool = torch.tensor([m for m, k in enumerate(lib.motion_keys) if args.match is None or args.match in k],
                        device=dev)
    print(f"[phase0_replay] {len(pool)} of {M} clips in the draw")
    cmd.library.nearest = lambda pos: pool[torch.randint(len(pool), (len(pos),), device=dev)]  # uniform draw
    disp_cmd = torch.as_tensor(np.load(args.library)["root_disp"], device=dev) \
        if "root_disp" in np.load(args.library).files else torch.zeros(M, 3, device=dev)
    robot = base.scene["robot"]
    feet_ids = [robot.find_bodies(f"{s}_ankle_roll_link")[0][0] for s in ("left", "right")]
    wrapped.reset()
    start_p = robot.data.root_pos_w.clone()
    start_q = yaw_quat(robot.data.root_quat_w).clone()
    started = torch.zeros(n, dtype=torch.bool, device=dev)  # a clean swing start was seen
    feet0 = robot.data.body_pos_w[:, feet_ids].clone()      # (n, 2, 3) at swing start
    lift = torch.zeros(n, 2, device=dev)                    # max foot lift during the swing
    stats = defaultdict(lambda: defaultdict(list))
    falls = torch.zeros(M, device=dev)
    tm = base.termination_manager
    term_counts = {k: 0 for k in tm.active_terms if k != "time_out"}
    act = torch.zeros(n, 64, device=dev)
    for _ in range(int(args.seconds / base.step_dt)):
        at_hit = cmd.at_hit.clone()
        motion = cmd.motion.clone()
        at_end = (cmd.frame == T - 2)
        with torch.inference_mode():
            _, _, dones, extras = wrapped.step(act)
        d = dones.bool()
        fell = d & ~extras["time_outs"].bool()
        for k in term_counts:
            term_counts[k] += int(tm.get_term(k).sum())
        if fell.any():
            falls += torch.bincount(motion[fell], minlength=M)
        r = robot.data
        rel = quat_apply_inverse(start_q, r.root_pos_w - start_p)
        if at_hit.any():
            p, _, nrm = racket_state_w(base)
            yq = yaw_quat(r.root_quat_w)
            lib_w = r.root_pos_w + quat_apply(yq, lib.contact_pos[motion])
            n_lib = quat_apply(yq, lib.face_normal[motion])
            err = torch.norm(p - lib_w, dim=-1)
            face = torch.rad2deg(torch.acos((nrm * n_lib).sum(-1).clamp(-1, 1)))
            for i in (at_hit & ~d).nonzero().squeeze(-1).tolist():
                m = int(motion[i])
                stats[m]["err"].append(float(err[i]))
                stats[m]["face"].append(float(face[i]))
                if started[i]:
                    stats[m]["disp_hit"].append(rel[i, :2].tolist())
        fp = r.body_pos_w[:, feet_ids]
        lift = torch.maximum(lift, fp[..., 2] - feet0[..., 2])
        fmove = torch.stack([quat_apply_inverse(start_q, fp[:, k] - feet0[:, k]) for k in range(2)], 1)
        for i in (at_end & ~d & started).nonzero().squeeze(-1).tolist():
            stats[int(motion[i])]["disp_end"].append(rel[i, :2].tolist())
            stats[int(motion[i])]["feet"].append([lift[i, 0].item(), lift[i, 1].item(),
                                                 fmove[i, 0, 1].item(), fmove[i, 1, 1].item()])
        # new swing begins (clip restarted at frame 0 this step): reset the start pose
        new = (cmd.frame == 0) & ~d
        start_p[new] = r.root_pos_w[new]
        start_q[new] = yaw_quat(r.root_quat_w[new])
        feet0[new] = r.body_pos_w[new][:, feet_ids]
        lift[new] = 0.0
        started |= new
        started &= ~d

    print("\n===== phase0_replay:", args.library, "=====")
    print(f"{'clip':28s} {'n':>3s} {'falls':>5s} {'racket err cm':>14s} {'face deg':>9s} "
          f"{'cmd y cm':>8s} {'pelvis @hit x,y cm':>20s} {'@end x,y cm (sd y)':>22s}")
    for m in range(M):
        s = stats[m]
        e = np.array(s["err"]) * 100
        dh = np.array(s["disp_hit"]) * 100 if s["disp_hit"] else np.zeros((0, 2))
        de = np.array(s["disp_end"]) * 100 if s["disp_end"] else np.zeros((0, 2))
        fmt = lambda a: f"{a[:, 0].mean():+5.1f},{a[:, 1].mean():+6.1f}" if len(a) else "   -"  # noqa: E731
        sd = f" ({de[:, 1].std():.1f})" if len(de) > 1 else ""
        es = f"{e.mean():5.1f} (p90 {np.percentile(e, 90):4.1f})" if len(e) else "-"
        fs = f"{np.mean(s['face']):5.1f}" if len(e) else "-"
        print(f"{lib.motion_keys[m]:28s} {len(e):3d} {int(falls[m]):5d} {es:>14s} {fs:>9s} "
              f"{disp_cmd[m, 1].item() * 100:+8.0f} {fmt(dh):>20s} {fmt(de) + sd:>22s}")
    print("\nfeet per clip (mean over swings): max lift L/R cm | lateral foot move at clip end L/R cm")
    for m in range(M):
        f = np.array(stats[m]["feet"]) * 100
        if len(f):
            print(f"  {lib.motion_keys[m]:28s} lift {f[:, 0].mean():4.1f}/{f[:, 1].mean():4.1f}   "
                  f"move {f[:, 2].mean():+6.1f}/{f[:, 3].mean():+6.1f}")
    allerr = np.concatenate([np.array(stats[m]["err"]) for m in range(M)]) * 100
    print("terminations by cause:", term_counts)
    print(f"overall: {len(allerr)} impacts, falls {int(falls.sum())}, racket err mean {allerr.mean():.1f} cm "
          f"(p90 {np.percentile(allerr, 90):.1f})")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
