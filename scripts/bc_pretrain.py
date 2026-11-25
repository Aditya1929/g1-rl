"""Behavior-clone the actor onto the library tokens (DAgger) before PPO.

The policy architecture/normalizer are exactly rsl_rl's ActorCritic, so the result
is saved as an rsl_rl checkpoint that train.py --init_checkpoint can start from.

Loop: roll out the training env with actions mixed between the expert
(a* = atanh(2 z_ref), which quantizes back to z_ref exactly) and the student
(prob 1-beta, beta decays 1 -> beta_min), record (actor obs, z_ref), and regress
    loss = MSE(tanh(mu(obs)), z_ref)
Rolling out on the env's own observation pipeline means obs match training exactly,
and DAgger covers the states the student itself visits.

  python scripts/bc_pretrain.py --task G1-SonicTT-TokenAbs-v0 --num_envs 1024 --iterations 60 --headless

Only for the absolute-token recipe. In residual mode (the default task) the expert
action is 0, which train.py gets for free by zeroing the actor's output layer.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="G1-SonicTT-TokenAbs-v0")
ap.add_argument("--num_envs", type=int, default=1024)
ap.add_argument("--iterations", type=int, default=60)
ap.add_argument("--steps_per_iter", type=int, default=48)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--batch", type=int, default=8192)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--beta_min", type=float, default=0.2)
ap.add_argument("--buffer", type=int, default=400_000, help="max stored samples (GPU memory!)")
ap.add_argument("--out", default=None, help="checkpoint path (default logs/rsl_rl/<exp>/<time>_bc/model_0.pt)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402
from datetime import datetime  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from common import LOG_ROOT, agent_cfg, env_unwrapped, make_env  # noqa: E402


def main():
    env, wrapped, _ = make_env(args.task, args.num_envs, args.device)
    base = env_unwrapped(env)
    if base.action_manager.get_term("token").cfg.mode != "absolute":
        raise SystemExit("[bc] BC is only needed for the absolute-token task (G1-SonicTT-TokenAbs-v0); "
                         "residual mode starts from library replay via train.py's zero init.")
    cfg = agent_cfg(args.task)
    out = args.out or str(LOG_ROOT / cfg.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_bc") / "model_0.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=None, device=base.device)
    policy = runner.alg.policy
    opt = torch.optim.Adam(policy.actor.parameters(), lr=args.lr)

    from g1_sonic_tt.sonic.fsq import squash, unsquash

    cmd = base.command_manager.get_term("strike")
    obs = wrapped.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    buf_obs, buf_z = [], []
    for it in range(args.iterations):
        beta = max(args.beta_min, 1.0 - it / max(1, 0.6 * args.iterations))
        policy.eval()
        for _ in range(args.steps_per_iter):
            z_ref = cmd.z_ref.clone()
            with torch.no_grad():
                policy.update_normalization(obs)
                a_student = policy.act_inference(obs)
            a_expert = unsquash(z_ref)
            use_expert = (torch.rand(base.num_envs, 1, device=base.device) < beta).float()
            buf_obs.append(policy.get_actor_obs(obs).clone())
            buf_z.append(z_ref)
            obs, _, _, _ = wrapped.step(use_expert * a_expert + (1 - use_expert) * a_student)
        X, Z = torch.cat(buf_obs), torch.cat(buf_z)
        if len(X) > args.buffer:
            X, Z = X[-args.buffer:], Z[-args.buffer:]
        buf_obs, buf_z = [X], [Z]
        policy.train()
        losses = []
        for _ in range(args.epochs):
            perm = torch.randperm(len(X), device=X.device)
            for i in range(0, len(X), args.batch):
                idx = perm[i:i + args.batch]
                with torch.no_grad():
                    xn = policy.actor_obs_normalizer(X[idx])
                loss = ((squash(policy.actor(xn)) - Z[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        rms_steps = (sum(losses[-20:]) / len(losses[-20:])) ** 0.5 * 16
        print(f"[bc] iter {it:3d}  beta {beta:.2f}  samples {len(X):7d}  loss {losses[-1]:.5f}  "
              f"(~{rms_steps:.2f} FSQ steps RMS)")
    runner.save(out)
    print(f"[bc] saved {out}\n  -> python scripts/train.py --init_checkpoint {out} --headless")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
