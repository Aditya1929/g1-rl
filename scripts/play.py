"""Roll out a trained policy (sampled targets) and export it for deployment.

  python scripts/play.py --checkpoint logs/rsl_rl/g1_sonic_token_hitter/<run>/model_5000.pt
Exports <run>/exported/policy.pt (TorchScript, includes the obs normalizer) and
policy.onnx. The exported net maps actor obs -> action a. Deploy turns it into the token
exactly as the env does (obs_spec.json "action"): residual mode
z = quantize(z_ref + lambda * clip(a, -1, 1)); absolute mode z = quantize(tanh(a)).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="G1-SonicTT-Token-Play-v0")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--seconds", type=float, default=30.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from common import agent_cfg, env_unwrapped, make_env  # noqa: E402


def main():
    env, wrapped, _ = make_env(args.task, args.num_envs, args.device)
    base = env_unwrapped(env)
    runner = OnPolicyRunner(wrapped, agent_cfg(args.task).to_dict(), log_dir=None, device=base.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=base.device)
    nn_ = runner.alg.policy
    export_dir = os.path.join(os.path.dirname(args.checkpoint), "exported")
    export_policy_as_jit(nn_, nn_.actor_obs_normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(nn_, normalizer=nn_.actor_obs_normalizer, path=export_dir, filename="policy.onnx")
    # observation spec for deploy/bridge.py: term order/dims and the sim joint order
    om = base.observation_manager
    spec = {
        "terms": list(zip(om.active_terms["policy"], [list(d) for d in om.group_obs_term_dim["policy"]])),
        "joint_names": list(base.scene["robot"].joint_names),
        "default_joint_pos": base.scene["robot"].data.default_joint_pos[0].tolist(),
        "step_dt": base.step_dt,
        "hit_frame": base.command_manager.get_term("strike").hit_frame,
        "action": {"mode": base.action_manager.get_term("token").cfg.mode,
                   "residual_scale": base.action_manager.get_term("token").cfg.residual_scale},
    }
    with open(os.path.join(export_dir, "obs_spec.json"), "w") as f:
        json.dump(spec, f, indent=1)
    print(f"[play] exported policy.pt / policy.onnx / obs_spec.json to {export_dir}")

    cmd = base.command_manager.get_term("strike")
    obs = wrapped.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    errs = []
    for _ in range(int(args.seconds / base.step_dt)):
        hit = cmd.at_hit.clone()  # the metric is written during this step, before the frame advances
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
        if hit.any():
            errs.append(cmd.metrics["racket_pos_err_at_hit"][hit].clone())
    if errs:
        e = torch.cat(errs)
        print(f"[play] racket error at impact (sampled targets): mean {e.mean()*100:.1f} cm over {len(e)} swings")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
