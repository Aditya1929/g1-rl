"""Ball-in-the-loop evaluation in Isaac Lab (no training).

Serve analytic balls -> planner predicts the strike (point, time, racket velocity)
-> overrides the strike command -> trained policy -> tokens -> SONIC decoder -> robot.
The analytic ball is stepped alongside physics and collides with the racket disk
(swept test, racket pose interpolated across the 20 ms step). Scores contact and
legal returns (first bounce on the opponent half).

  python scripts/eval_ball.py --checkpoint <model.pt> --num_envs 64 --headless --seconds 60
  python scripts/eval_ball.py --replay ...   # library tokens instead of a policy (baseline)
  python scripts/eval_ball.py --checkpoint <model.pt> --num_envs 1 --headless --video logs/hit.mp4 --seconds 20
  python scripts/eval_ball.py ... --all_serves --y_bounce -0.75 0.75   # full-plane heatmap, unfiltered
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--task", default="G1-SonicTT-Token-Play-v0", help="G1-SonicTT-TokenAbs-Play-v0 for absolute runs")
ap.add_argument("--replay", action="store_true", help="use library tokens (no policy) as a baseline")
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--commit_s", type=float, default=0.3, help="freeze target updates this close to impact")
ap.add_argument("--y_bounce", type=float, nargs=2, default=None,
                help="lateral range of the serve's bounce point (default: the training cfg's serve_y_bounce)")
ap.add_argument("--all_serves", action="store_true",
                help="serve every ball in --y_bounce, including strikes no clip can reach (full-plane heatmap). "
                     "Default: like training, keep only serves whose planned strike (robot at its nominal "
                     "pose) is within the training reach_radius of some clip contact")
ap.add_argument("--diag_out", default="logs/eval_ball_diag.pt", help="per-swing rows (for plot_heatmap.py)")
ap.add_argument("--window_margin", type=float, default=0.08,
                help="in reach = planned strike within this of some library clip contact (y, z)")
ap.add_argument("--video", default=None, help="record env 0 from the viewer camera to this .mp4 (needs --headless "
                "or a GUI; turns on cameras); stops at --seconds")
ap.add_argument("--cam_eye", type=float, nargs=3, default=(1.6, -2.2, 1.5), help="camera, relative to env 0's origin")
ap.add_argument("--cam_lookat", type=float, nargs=3, default=(0.7, 0.0, 0.9))
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from common import agent_cfg, env_unwrapped, make_env  # noqa: E402

FLYING, DONE = 0, 1


def main():
    task = "G1-SonicTT-Token-Replay-v0" if args.replay else args.task
    def cam(c):  # follow env 0
        c.viewer.origin_type, c.viewer.env_index = "env", 0
        c.viewer.eye, c.viewer.lookat = tuple(args.cam_eye), tuple(args.cam_lookat)
    env, wrapped, env_cfg = make_env(task, args.num_envs, args.device, render=bool(args.video),
                                     cfg_hook=cam if args.video else None)
    frames = []
    base = env_unwrapped(env)
    dev, n, dt = base.device, base.num_envs, base.step_dt

    from g1_sonic_tt.ball.model import BallParams, flight_step, racket_impact, sample_serves
    from g1_sonic_tt.planner.strike import PlannerParams, plan
    from g1_sonic_tt.sonic.constants import RACKET_FACE_RADIUS
    from g1_sonic_tt.tasks.token_hitter.env_cfg import TABLE_NEAR_EDGE_X, TABLE_TOP_Z
    from g1_sonic_tt.tasks.token_hitter.mdp.racket import racket_state_w

    ball_prm = BallParams(near_edge_x=TABLE_NEAR_EDGE_X, top_z=TABLE_TOP_Z)
    prm = PlannerParams(ball=ball_prm)
    if args.replay:
        policy = lambda obs: torch.zeros(n, 64, device=dev)  # noqa: E731  (replay mode ignores actions)
    else:
        assert args.checkpoint, "--checkpoint required unless --replay"
        runner = OnPolicyRunner(wrapped, agent_cfg(task).to_dict(), log_dir=None, device=dev)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=dev)

    cmd = base.command_manager.get_term("strike")
    robot = base.scene["robot"]
    origins = base.scene.env_origins
    vis = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/ball", markers={
        "b": sim_utils.SphereCfg(radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)))}))

    p = torch.zeros(n, 3, device=dev)
    v = torch.zeros(n, 3, device=dev)
    state = torch.full((n,), DONE, dtype=torch.long, device=dev)
    hit = torch.zeros(n, dtype=torch.bool, device=dev)
    planned = torch.zeros_like(hit)
    in_win = torch.zeros_like(hit)
    lib = cmd.library
    scfg = env_cfg.commands.strike
    y_bounce = tuple(args.y_bounce) if args.y_bounce is not None else tuple(scfg.serve_y_bounce)
    nominal = torch.tensor([0.0, 0.0, lib.ref_root_height], device=dev)

    def serve(k):
        """Training's serve distribution: sample, plan from the nominal pose, keep strikes in reach."""
        if args.all_serves:
            return sample_serves(k, ball_prm, dev, y_bounce=y_bounce)
        P, V, got = [], [], 0
        for _ in range(200):
            p0, v0 = sample_serves(8 * k + 64, ball_prm, dev, y_bounce=y_bounce)
            ok, _, ph, _, _ = plan(p0, v0, prm)
            good = ok & (lib.reach(ph - nominal) <= scfg.reach_radius)
            P.append(p0[good]), V.append(v0[good])
            got += int(good.sum())
            if got >= k:
                break
        assert got >= k, "no serves reach the library"
        return torch.cat(P)[:k], torch.cat(V)[:k]
    print(f"[eval_ball] serves: y_bounce {y_bounce}, "
          f"{'unfiltered' if args.all_serves else f'filtered to reach <= {scfg.reach_radius} m (as training)'}")
    stats = dict(serves=0, hits=0, returns=0, planned=0, planned_in=0, hits_in=0, returns_in=0)
    # miss diagnostics, recorded at the commanded impact step of each planned swing:
    # [policy err |racket - target|, planner err |target - ball|, along-track (ball - racket).v_hat,
    #  across-track |(ball - racket) perp|, ball speed, flag]
    rec = torch.zeros(n, 6, device=dev)
    rec_ok = torch.zeros_like(hit)
    returned = torch.zeros_like(hit)
    plan_b = torch.zeros(n, 3, device=dev)  # planned strike point, swing anchor frame
    clip = torch.zeros(n, 3, device=dev)    # chosen clip index, its kind (2 = side-step), swing-start pelvis y
    rows = []
    # row columns: 0-5 rec, 6 hit, 7 in_win, 8 returned, 9-11 planned strike point (anchor frame),
    # 12 clip index, 13 clip kind, 14 swing-start pelvis y relative to the env origin
    COLS = ["racket_target", "target_ball", "along", "across", "ball_speed", "hit_at_rec", "hit", "in_win",
            "returned", "plan_x", "plan_y", "plan_z", "motion", "kind", "start_y"]

    def flush(ids):
        m = rec_ok[ids]
        if m.any():
            j = ids[m]
            rows.append(torch.cat([rec[j], hit[j, None].float(), in_win[j, None].float(),
                                   returned[j, None].float(), plan_b[j], clip[j]], 1).cpu())
        rec_ok[ids] = False

    def save():
        if rows:
            torch.save({"rows": torch.cat(rows), "cols": COLS, "checkpoint": args.checkpoint,
                        "contacts": lib.contact_pos.cpu(), "kind": lib.kind.cpu(),
                        "ref_root_height": lib.ref_root_height, "reach": args.window_margin}, args.diag_out)

    obs = wrapped.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    for step in range(int(args.seconds / dt)):
        if not app.is_running():  # GUI window closed
            break
        if step % 500 == 0:
            save()
        # --- serve new balls once the previous ball is finished and the planner-driven clip has
        # ended (robot back at the ready pose), like the deploy bridge (`not swinging`). Serving
        # during the follow-through made the next swing start from a pose training never sees.
        idle = (state == DONE) & ~cmd.external
        if idle.any():
            ids = idle.nonzero().squeeze(-1)
            flush(ids)
            p0, v0 = serve(len(ids))
            p[ids], v[ids] = p0, v0
            state[ids], hit[ids], planned[ids], returned[ids] = FLYING, False, False, False
            stats["serves"] += len(ids)

        # --- planner -> strike command (before impact; frozen within commit_s)
        live = (state == FLYING) & ~hit
        if live.any():
            ok, t_hit, p_hit, v_r, _ = plan(p, v, prm)
            fth = torch.round(t_hit / dt).nan_to_num(1e6).long()
            p_w = p_hit + origins
            start = live & ok & ~planned & (fth <= cmd.hit_frame)
            if start.any():
                ids = start.nonzero().squeeze(-1)
                cmd.set_external_targets_w(ids, p_w[ids], v_r[ids], fth[ids])
                planned[ids] = True
                pb = cmd.racket_target_pos_b[ids]
                plan_b[ids] = pb
                mo = cmd.motion[ids]
                clip[ids] = torch.stack([mo.float(), lib.kind[mo].float(),
                                         cmd.anchor_pos_w[ids, 1] - origins[ids, 1]], 1)
                in_win[ids] = lib.reach(pb) <= args.window_margin  # "in reach": near some clip contact
                stats["planned"] += len(ids)
                stats["planned_in"] += int(in_win[ids].sum())
            update = live & ok & planned & ~start & (t_hit > args.commit_s)
            if update.any():  # refine targets in the swing's anchor frame, keep the swing clock and clip
                ids = update.nonzero().squeeze(-1)
                cmd.update_external_targets_w(ids, p_w[ids], v_r[ids])

        # --- physics step
        rp0, _, _ = racket_state_w(base)
        at_hit = cmd.at_hit.clone() & planned & ~rec_ok
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        rp1, rv1, rn1 = racket_state_w(base)
        rp0, rp1 = rp0 - origins, rp1 - origins

        # --- ball substeps with racket interpolation
        sub = 4
        for k in range(sub):
            a0, a1 = k / sub, (k + 1) / sub
            p_next, v_next, bounce = flight_step(p, v, dt / sub, ball_prm)
            r0, r1 = rp0 + a0 * (rp1 - rp0), rp0 + a1 * (rp1 - rp0)
            contact, v_next = racket_impact(p, p_next, v_next, r0, r1, rv1, rn1, RACKET_FACE_RADIUS, ball_prm)
            new_hit = contact & (state == FLYING) & ~hit
            hit |= new_hit
            stats["hits"] += int(new_hit.sum())
            stats["hits_in"] += int((new_hit & in_win).sum())
            ret = hit & bounce & (p_next[:, 0] > ball_prm.net_x) & (state == FLYING)
            stats["returns"] += int(ret.sum())
            stats["returns_in"] += int((ret & in_win).sum())
            returned |= ret
            state[ret] = DONE
            # misses / out of play
            gone = (state == FLYING) & ((p_next[:, 2] < 0.3) | (p_next[:, 0] < -0.8) | (p_next[:, 0] > 4.5)
                                        | (hit & bounce & (p_next[:, 0] <= ball_prm.net_x)))
            state[gone] = DONE
            p, v = p_next, v_next
        if at_hit.any():
            tgt, _ = cmd.racket_target_world()
            tgt = tgt - origins
            d = p - rp1
            vh = torch.nn.functional.normalize(v, dim=-1)
            along = (d * vh).sum(-1)
            across = torch.norm(d - along[:, None] * vh, dim=-1)
            r = torch.stack([torch.norm(rp1 - tgt, dim=-1), torch.norm(tgt - p, dim=-1), along, across,
                             torch.norm(v, dim=-1), hit.float()], 1)
            rec[at_hit] = r[at_hit]
            rec_ok |= at_hit
        state[dones.bool()] = DONE
        vis.visualize(p + origins)
        if args.video:
            frames.append(env.render())

    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=round(1 / dt), macro_block_size=1)
        print(f"[eval_ball] saved {len(frames)} frames ({len(frames) * dt:.1f} s) -> {args.video}")
    s = stats
    print("\n===== eval_ball =====")
    pct = lambda a, b: f"{100 * a / max(1, b):.1f}%"  # noqa: E731
    out_p = s["planned"] - s["planned_in"]
    print(f"serves {s['serves']}, planned swings {s['planned']} "
          f"(in library window {s['planned_in']}, outside {out_p})")
    print(f"  all swings:       hits {s['hits']} ({pct(s['hits'], s['planned'])}), "
          f"returns {s['returns']} ({pct(s['returns'], s['planned'])})")
    print(f"  in window:        hits {s['hits_in']} ({pct(s['hits_in'], s['planned_in'])}), "
          f"returns {s['returns_in']} ({pct(s['returns_in'], s['planned_in'])})")
    print(f"  outside window:   hits {s['hits'] - s['hits_in']} ({pct(s['hits'] - s['hits_in'], out_p)}), "
          f"returns {s['returns'] - s['returns_in']} ({pct(s['returns'] - s['returns_in'], out_p)})")
    flush(torch.arange(n, device=dev))
    save()
    if rows:
        R = torch.cat(rows)
        cm = lambda x: f"{100 * x.median():5.1f}"  # noqa: E731
        print("\n  by clip type (in-window swings)")
        for name, m in (("no step  ", (R[:, 7] > 0) & (R[:, 13] != 2)), ("side-step", (R[:, 7] > 0) & (R[:, 13] == 2))):
            k = int(m.sum())
            print(f"    {name} n={k:4d}  hits {pct(int(R[m, 6].sum()), k)}  returns {pct(int(R[m, 8].sum()), k)}"
                  + (f"  racket-target {cm(R[m, 0])} cm" if k else ""))
        print(f"  swing-start pelvis |y| from home: median {cm(R[:, 14].abs())} cm, "
              f"p90 {100 * R[:, 14].abs().quantile(0.9):.1f} cm")
        print("\n  diagnostics at commanded impact (median, cm; along < 0 = ball not yet arrived, racket early)")
        for name, m in (("in-window hit ", (R[:, 7] > 0) & (R[:, 6] > 0)),
                        ("in-window miss", (R[:, 7] > 0) & (R[:, 6] == 0))):
            Q = R[m]
            if len(Q):
                print(f"    {name} n={len(Q):4d}: racket-target {cm(Q[:, 0])}  target-ball {cm(Q[:, 1])}  "
                      f"along {cm(Q[:, 2])} (|.| {cm(Q[:, 2].abs())}, = {1000 * (Q[:, 2].abs() / Q[:, 4]).median():.0f} ms)  "
                      f"across {cm(Q[:, 3])}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
