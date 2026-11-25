"""Closed-loop sanity check: stream library tokens through the frozen decoder in Isaac Lab.

No policy. If SONIC + our sim + racket are consistent, the robot should stay up and
the racket should pass near each sampled target at impact. Run this before training.

  python scripts/replay_reference.py --num_envs 64 --headless --seconds 20
Reports fall rate and racket error at impact (vs. the sampled target AND vs. the
library contact point of the replayed clip — the latter isolates tracking error).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--seconds", type=float, default=20.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from common import env_unwrapped, make_env  # noqa: E402

TASK = "G1-SonicTT-Token-Replay-v0"


def main():
    env, wrapped, _ = make_env(TASK, args.num_envs, args.device)
    base = env_unwrapped(env)
    from g1_sonic_tt.tasks.token_hitter.mdp.racket import racket_state_w
    from isaaclab.utils.math import quat_apply, yaw_quat

    cmd = base.command_manager.get_term("strike")
    wrapped.reset()
    steps = int(args.seconds / base.step_dt)
    falls, err_tgt, err_lib, face, vec = 0, [], [], [], []
    actions = torch.zeros(base.num_envs, 64, device=base.device)
    for _ in range(steps):
        at_hit = cmd.at_hit.clone()
        lib_pt_b = cmd.library.contact_pos[cmd.motion].clone()
        lib_n_b = cmd.library.face_normal[cmd.motion].clone()
        tgt_w, _ = cmd.racket_target_world()
        _, _, dones, extras = wrapped.step(actions)
        if at_hit.any():
            p, _, n = racket_state_w(base)
            r = base.scene["robot"].data
            lib_w = r.root_pos_w + quat_apply(yaw_quat(r.root_quat_w), lib_pt_b)
            ok = at_hit & ~dones.bool()
            err_tgt.append(torch.norm(p - tgt_w, dim=-1)[ok])
            err_lib.append(torch.norm(p - lib_w, dim=-1)[ok])
            from isaaclab.utils.math import quat_apply_inverse
            vec.append(quat_apply_inverse(yaw_quat(r.root_quat_w), p - lib_w)[ok])
            n_lib = quat_apply(yaw_quat(r.root_quat_w), lib_n_b)
            face.append(torch.rad2deg(torch.acos((n * n_lib).sum(-1).clamp(-1, 1)))[ok])
        falls += int((dones.bool() & ~extras["time_outs"].bool()).sum())
    e_t, e_l = torch.cat(err_tgt), torch.cat(err_lib)
    swings = len(e_t)
    print("\n===== replay_reference =====")
    print(f"envs {base.num_envs}, {args.seconds:.0f}s each, impacts measured: {swings}, falls: {falls}")
    if swings:
        print(f"racket error vs library contact point: mean {e_l.mean()*100:.1f} cm, p90 {e_l.quantile(.9)*100:.1f} cm")
        print(f"racket error vs sampled target:        mean {e_t.mean()*100:.1f} cm, p90 {e_t.quantile(.9)*100:.1f} cm")
        v = torch.cat(vec)
        print(f"  mean error vector (yaw frame, x fwd / y left / z up): {(v.mean(0)*100).cpu().numpy().round(1)} cm, "
              f"spread around it: {((v - v.mean(0)).norm(dim=-1).mean()*100):.1f} cm")
        f = torch.cat(face)
        print(f"face normal vs library normal at impact: mean {f.mean():.1f} deg, p90 {f.quantile(.9):.1f} deg")
    print("Pass criteria (suggested): no falls; library-contact error well under the 7.5 cm racket radius.")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
