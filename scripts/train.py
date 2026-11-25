"""PPO training (rsl_rl) for the SONIC-token hitter.

  python scripts/train.py --headless --num_envs 2048                       # residual (default task)
  python scripts/train.py --task G1-SonicTT-TokenAbs-v0 --headless --init_checkpoint <bc model_0.pt>
  python scripts/train.py --headless --resume <run dir name>              # continue a run

Residual mode starts with the actor's output layer zeroed, so the mean action is 0
and the env replays the library swing (z = z_ref) until PPO learns a correction.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="G1-SonicTT-Token-v0")
ap.add_argument("--num_envs", type=int, default=None)
ap.add_argument("--max_iterations", type=int, default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--init_checkpoint", default=None, help="start from this checkpoint (e.g. the BC actor)")
ap.add_argument("--resume", default=None, help="run directory name under logs/rsl_rl/<experiment>/ to resume")
ap.add_argument("--run_name", default="")
ap.add_argument("--curriculum_done", action="store_true",
                help="start reward-weight curricula at their end weights (fine-tuning a trained policy)")
ap.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb", "neptune"])
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402
from datetime import datetime  # noqa: E402

import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from common import LOG_ROOT, agent_cfg, env_unwrapped, latest_checkpoint, make_env  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    cfg = agent_cfg(args.task)
    if args.max_iterations:
        cfg.max_iterations = args.max_iterations
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.logger = args.logger
    env, wrapped, env_cfg = make_env(args.task, args.num_envs, args.device)
    env_unwrapped(env).cfg.seed = cfg.seed
    if args.curriculum_done:  # edit the manager's own term cfgs (the manager deep-copies env.cfg)
        cm = env_unwrapped(env).curriculum_manager
        for name, term in zip(cm.active_terms, cm._term_cfgs):
            if "end_weight" in term.params:
                term.params["start_weight"] = term.params["end_weight"]
                print(f"[train] curriculum {name}: held at end weight {term.params['end_weight']}")
    log_dir = str(LOG_ROOT / cfg.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                                    + (f"_{args.run_name}" if args.run_name else "")))
    runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=log_dir, device=cfg.device)
    if args.resume:
        path = latest_checkpoint(cfg.experiment_name, args.resume)
        print(f"[train] resuming {path}")
        runner.load(path)
    elif args.init_checkpoint:
        print(f"[train] initializing from {args.init_checkpoint} (weights + normalizer, fresh optimizer)")
        runner.load(args.init_checkpoint, load_optimizer=False)
        runner.current_learning_iteration = 0
    elif env_unwrapped(env).action_manager.get_term("token").cfg.mode == "residual":
        last = [m for m in runner.alg.policy.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        print("[train] residual mode: actor output layer zeroed (mean action 0 == library replay)")
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), cfg)
    runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
