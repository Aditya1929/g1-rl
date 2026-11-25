"""Live viewer: Isaac Lab GUI that follows a training run and hot-reloads new checkpoints.

  python scripts/view.py                              # latest run of the residual experiment
  python scripts/view.py --run 2026-11-24_13-05-10_bracket_residual --num_envs 8
  python scripts/view.py --checkpoint logs/rsl_rl/.../model_250.pt --num_envs 2   # one fixed checkpoint
  python scripts/view.py --replay                     # library tokens only (a = 0), for comparison

Green sphere = commanded racket target at impact, red sphere = racket contact point.
Every --reload_s seconds it loads the newest model_*.pt of the run (saved every 250
iterations) and prints the racket position / face errors at impact since the last load.
Keep --num_envs small while training runs: both share the 8 GB GPU.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="G1-SonicTT-Token-Play-v0")
ap.add_argument("--experiment", default="g1_sonic_token_hitter")
ap.add_argument("--run", default=None, help="run directory name (default: newest)")
ap.add_argument("--checkpoint", default=None, help="show this model_*.pt only (no hot reload)")
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--reload_s", type=float, default=60.0)
ap.add_argument("--replay", action="store_true", help="zero actions (= library replay in residual mode)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import time  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from common import agent_cfg, env_unwrapped, latest_checkpoint, make_env  # noqa: E402


def main():
    env, wrapped, _ = make_env(args.task, args.num_envs, args.device)
    base = env_unwrapped(env)
    from g1_sonic_tt.tasks.token_hitter.mdp.racket import racket_state_w
    from g1_sonic_tt.tasks.token_hitter.mdp.rewards import _face_angle

    cmd = base.command_manager.get_term("strike")
    runner = OnPolicyRunner(wrapped, agent_cfg(args.task).to_dict(), log_dir=None, device=base.device)
    loaded, policy = None, None

    def refresh():
        nonlocal loaded, policy
        if args.replay:
            return
        try:
            path = args.checkpoint or latest_checkpoint(args.experiment, args.run)
        except FileNotFoundError:
            return
        if path != loaded:
            runner.load(path)
            policy = runner.get_inference_policy(device=base.device)
            loaded = path
            print(f"[view] loaded {path}", flush=True)

    refresh()
    if policy is None and not args.replay:
        print("[view] no checkpoint yet; showing library replay (a = 0) until one appears", flush=True)
    obs = wrapped.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    zeros = torch.zeros(base.num_envs, 64, device=base.device)
    pos_err, face_err, last = [], [], time.monotonic()
    while app.is_running():
        hit = cmd.at_hit.clone()
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(zeros if policy is None else policy(obs))
            if hit.any():
                p, _, n = racket_state_w(base)
                tgt, _ = cmd.racket_target_world()
                pos_err.append(torch.norm(p - tgt, dim=-1)[hit])
                face_err.append(torch.rad2deg(_face_angle(n, cmd.racket_target_normal_world()))[hit])
        if time.monotonic() - last > args.reload_s:
            if pos_err:
                e, f = torch.cat(pos_err), torch.cat(face_err)
                print(f"[view] {len(e)} impacts: racket pos error {e.mean() * 100:.1f} cm, "
                      f"face error {f.mean():.1f} deg", flush=True)
            pos_err, face_err, last = [], [], time.monotonic()
            refresh()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
