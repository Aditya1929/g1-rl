"""Build data/forehand_library_v2.npz: wider / lower forehands and step-and-swing clips.

Starts from data/forehand_library.npz (v1: face-corrected forehands, fixed pelvis,
straight legs) and writes a library in the same format plus:
    root_disp  (M, 3)  pelvis displacement at impact vs the swing-start anchor
    step_cost  (M,)    selection penalty (m) for clips that step
    kind       (M,)    0 = v1 forehand, 1 = grid forehand, 2 = step-and-swing
    contact_pos_kin    kinematic contact at impact, anchor frame (contact_pos too until
                       scripts/calibrate_library.py replaces it with SONIC's measured one)

Frames: the anchor is the swing-start pelvis xy / yaw at the athletic-stance height
(ref_root_height = v1 height - --stance). contact_traj stays pelvis-relative (the path
reward compares it to the live root); contact_pos / contact_vel / root_disp are
anchor-relative (where the racket / pelvis end up in the world at impact).

Clip families (all share the stance and the 126-frame / impact-50 timing):
  0  v1 forehands with stance legs (upper body unchanged)
  1  grid forehands: cells every --cell m on the hitting plane, y in --grid_y, z in --grid_z
     (anchor frame), skipping cells already covered by a v1 contact. Each is solved from the
     nearest v1 clip: its contact path is shifted by the cell offset with a smooth bump
     (0 at the ready pose, 1 from the backswing through the follow-through) and waist +
     right arm (10 joints) are re-solved per frame to follow it, keeping the clip's face
     orientation, regularised to the clip, with torso / pelvis / left-arm clearance. Low
     cells also crouch (knees) up to --max_crouch with the same bump.
  2  step-and-swing: the upper body of the central forehands over a side-step of d in
     --steps (lead foot first, --lift foot lift; the variant that SONIC tracks, phase 0).

  ~/sonic-dataset/.venv/bin/python scripts/build_library_v2.py [--jobs 16]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_library import ENCODER_ONNX, RIGHT_ARM, Encoder  # noqa: E402
from phase0_build import LEGS, Body, clearance, minjerk  # noqa: E402

from g1_sonic_tt import paths  # noqa: E402
from g1_sonic_tt.sonic.constants import SONIC_JOINTS  # noqa: E402
from g1_sonic_tt.sonic.urdf_fk import UrdfChain  # noqa: E402

DT = 0.02
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
UPPER = WAIST + RIGHT_ARM
OUT = paths.DATA_DIR / "forehand_library_v2.npz"
_G = {}  # per-process globals (FK chains are not cheap to rebuild per call)


def _init():
    _G["body"] = Body()
    _G["legs"] = {s: UrdfChain(paths.SONIC_URDF, "pelvis", f"{s}_ankle_roll_link") for s in ("left", "right")}


def bump(T, hit):
    t = np.arange(T, dtype=float)
    return minjerk(t, 5, 30) * (1 - minjerk(t, hit + 15, 110))


def solve_legs(q, pel, feet):
    """Leg IK: pelvis offset pel (T,3) from the zero-leg standing pelvis, feet (dict side -> (T,3))
    world-fixed targets in that same frame; foot orientation kept flat. Edits q (IL order) in place."""
    legs, names = _G["legs"], SONIC_JOINTS
    q0 = dict(zip(names, q[0]))
    for s in ("left", "right"):
        idx = [names.index(n) for n in LEGS[s]]
        lim = legs[s].limits
        lo = np.array([lim[n][0] for n in LEGS[s]]) + 0.01
        hi = np.array([lim[n][1] for n in LEGS[s]]) - 0.01
        zero = dict(q0)
        zero.update({n: 0.0 for n in LEGS[s]})
        R_goal = legs[s].pose(zero)[0]
        prev = np.clip(np.array([-0.2, 0, 0, 0.4, -0.2, 0]), lo, hi)
        worst = 0.0
        for k in range(len(q)):
            tgt = feet[s][k] - pel[k]

            def res(a, tgt=tgt, prev=prev):
                qd = dict(q0)
                qd.update(zip(LEGS[s], a))
                R, p = legs[s].pose(qd)
                return np.concatenate([50 * (p - tgt), 5 * (R - R_goal).ravel(), 0.05 * (a - prev)])
            sol = least_squares(res, prev, bounds=(lo, hi), max_nfev=200)
            q[k, idx] = prev = sol.x
            worst = max(worst, np.linalg.norm(res(sol.x)[:3]) / 50)
    return worst


def feet_zero(q):
    qd = dict(zip(SONIC_JOINTS, q[0]))
    qd.update({n: 0.0 for s in LEGS for n in LEGS[s]})
    return {s: _G["legs"][s].pose(qd)[1] for s in ("left", "right")}


def stance_clip(job):
    """Family 0 and the leg part of every clip: stance crouch (+ extra crouch c * bump), optional step."""
    q, stance, crouch, step, lift = job["q"].copy(), job["stance"], job.get("crouch", 0.0), job.get("step", 0.0), \
        job.get("lift", 0.10)
    T, hit = len(q), job["hit"]
    t = np.arange(T, dtype=float)
    f0 = feet_zero(q)
    pel = np.zeros((T, 3))
    pel[:, 2] = -stance - crouch * bump(T, hit)
    feet = {s: np.repeat(f0[s][None], T, 0) for s in f0}
    if step:
        slow = 1.3
        pel[:, 1] = step * minjerk(t, 4, 4 + 30 * slow)
        lead, trail = ("left", "right") if step > 0 else ("right", "left")
        f = lambda x: 4 + (x - 4) * slow  # noqa: E731
        for s, (a, b) in ((lead, (f(6), f(18))), (trail, (f(20), f(32)))):
            ph = np.clip((t - a) / (b - a), 0, 1)
            feet[s][:, 1] += step * minjerk(t, a, b)
            feet[s][:, 2] += lift * np.sin(np.pi * ph)
    err = solve_legs(q, pel, feet)
    return q, pel, err


def upper_clip(job):
    """Family 1: re-solve waist + right arm so the racket follows base path + bump * delta (anchor frame)."""
    body, names = _G["body"], SONIC_JOINTS
    q, hit, delta = job["q"].copy(), job["hit"], np.asarray(job["delta"])
    T = len(q)
    w = bump(T, hit)
    # legs first (stance + crouch); pelvis offset relative to the anchor (stance pelvis)
    q, pel, leg_err = stance_clip(dict(job, q=q))
    pel_anchor = pel - pel[0]
    idx = [names.index(n) for n in UPPER]
    lim = body.wrist.limits
    lo = np.array([lim[n][0] for n in UPPER]) + 0.02
    hi = np.array([lim[n][1] for n in UPPER]) - 0.02
    lo[0], hi[0] = -0.8, 0.8                                   # natural waist yaw range
    base_q = job["q"]
    base_traj = np.array([body.racket(dict(zip(names, base_q[k])))[0] for k in range(T)])
    base_n = np.array([body.racket(dict(zip(names, base_q[k])))[1] for k in range(T)])
    target = base_traj + w[:, None] * delta[None] - pel_anchor  # pelvis-relative goal
    b = np.exp(-0.5 * ((np.arange(T) - hit) * DT / 0.33) ** 4)
    reg = np.r_[0.2, 0.2, 0.2, [0.07] * 7]
    A = np.zeros((T, len(UPPER)))
    prev = base_q[0, idx]
    for k in range(T):
        a_base = base_q[k, idx]

        def res(a, k=k, a_base=a_base, prev=prev):
            qq = q[k].copy()
            qq[idx] = a
            qd = dict(zip(names, qq))
            c, n, _, _ = body.racket(qd)
            return np.concatenate([12 * (c - target[k]), (0.5 + 2.5 * b[k]) * (n - base_n[k]),
                                   30 * np.minimum(clearance(body, qd), 0.0), reg * (a - a_base),
                                   0.2 * (a - prev)])
        A[k] = prev = least_squares(res, np.clip(prev, lo, hi), bounds=(lo, hi), max_nfev=150).x
    A = gaussian_filter1d(A, 1.0, axis=0, mode="nearest")
    A[-1] = A[0]
    q[:, idx] = A
    traj = np.array([body.racket(dict(zip(names, q[k])))[0] for k in range(T)])
    path_err = np.linalg.norm(traj - target, axis=1)
    n_hit = body.racket(dict(zip(names, q[hit])))[1]
    clear = min(clearance(body, dict(zip(names, q[k]))).min() for k in range(0, T, 3))
    return dict(q=q, pel=pel_anchor, leg_err=leg_err, path_max=path_err.max(), hit_err=path_err[hit],
                face_err=np.degrees(np.arccos(np.clip(n_hit @ base_n[hit], -1, 1))), clear=clear)


def run(job):
    if job["kind"] == 1:
        r = upper_clip(job)
    else:
        q, pel, leg_err = stance_clip(job)
        r = dict(q=q, pel=pel - pel[0], leg_err=leg_err, path_max=0.0, hit_err=0.0, face_err=0.0, clear=0.0)
    body = _G["body"]
    T = len(r["q"])
    r["traj"] = np.array([body.racket(dict(zip(SONIC_JOINTS, r["q"][k])))[0] for k in range(T)])
    r["n"] = body.racket(dict(zip(SONIC_JOINTS, r["q"][job["hit"]])))[1]
    r.update(key=job["key"], kind=job["kind"], step=job.get("step", 0.0), crouch=job.get("crouch", 0.0))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--stance", type=float, default=0.06, help="athletic-stance crouch of every clip (m)")
    ap.add_argument("--cell", type=float, default=0.10)
    ap.add_argument("--grid_y", type=float, nargs=2, default=(-0.85, -0.25))
    ap.add_argument("--grid_z", type=float, nargs=2, default=(-0.25, 0.35))
    ap.add_argument("--skip_radius", type=float, default=0.05, help="skip grid cells this close to a v1 contact")
    ap.add_argument("--max_crouch", type=float, default=0.15)
    ap.add_argument("--steps", type=float, nargs="+", default=[-0.35, -0.25, 0.25, 0.35])
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--wide_y", type=float, nargs="+", default=[-0.65, -0.75], help="grid columns stepped further out")
    ap.add_argument("--wide_z", type=float, nargs="+", default=[-0.05, 0.05, 0.15])
    ap.add_argument("--wide_steps", type=float, nargs="+", default=[-0.25, -0.35])
    ap.add_argument("--limit", type=int, default=0, help="debug: build only the first N jobs")
    args = ap.parse_args()

    v1 = np.load(paths.FOREHAND_LIBRARY)
    assert list(v1["joint_names"]) == SONIC_JOINTS
    Q, hit, T = v1["ref_q"].astype(np.float64), int(v1["hit_frame"]), v1["ref_q"].shape[1]
    keys = [str(k) for k in v1["motion_keys"]]
    c1 = v1["contact_pos"].astype(np.float64)               # v1 pelvis frame == v2 anchor frame (stance-shifted)
    jobs = [dict(kind=0, key=f"fh_{k[-9:]}", q=Q[m], hit=hit, stance=args.stance) for m, k in enumerate(keys)]
    ys = np.arange(args.grid_y[0], args.grid_y[1] + 1e-6, args.cell)
    zs = np.arange(args.grid_z[0], args.grid_z[1] + 1e-6, args.cell)
    for y in ys:
        for z in zs:
            cell = np.array([c1[:, 0].mean(), y, z])
            d = np.linalg.norm(c1[:, 1:] - cell[1:], axis=1)
            if d.min() < args.skip_radius:
                continue
            m = int(d.argmin())
            crouch = float(np.clip(-(z - c1[:, 2].min()) * 0.6, 0.0, args.max_crouch))
            delta = cell - c1[m]                             # anchor frame; the pelvis offset handles the crouch
            jobs.append(dict(kind=1, key=f"grid_y{y:+.2f}_z{z:+.2f}", q=Q[m], hit=hit, stance=args.stance,
                             delta=delta, crouch=crouch))
    centre = [int(np.linalg.norm(c1[:, 1:] - np.array([y, z]), axis=1).argmin())
              for y in (-0.35, -0.45, -0.55) for z in (0.05, 0.12, 0.2)]
    for m in sorted(set(centre)):
        for s in args.steps:
            jobs.append(dict(kind=2, key=f"step{s * 100:+.0f}_{keys[m][-9:]}", q=Q[m], hit=hit, stance=args.stance,
                             step=s, lift=args.lift))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} clips: {sum(j['kind'] == 0 for j in jobs)} v1, {sum(j['kind'] == 1 for j in jobs)} grid, "
          f"{sum(j['kind'] == 2 for j in jobs)} step; {args.jobs} workers", flush=True)
    def report(r):
        print(f"  {r['key']:28s} leg IK {r['leg_err'] * 100:4.1f} cm | path max {r['path_max'] * 100:4.1f} cm, "
              f"hit {r['hit_err'] * 100:4.1f} cm, face {r['face_err']:4.1f} deg, clearance {r['clear'] * 100:+5.1f} cm",
              flush=True)

    with Pool(args.jobs, initializer=_init) as pool:
        res = []
        for r in pool.imap(run, jobs):
            res.append(r)
            report(r)
        # stage 2: wide reach = the widest grid swings (upper body) over a right step
        wide = [r for r in res if r["kind"] == 1 and any(f"_y{y:+.2f}_z{z:+.2f}" in r["key"]
                                                         for y in args.wide_y for z in args.wide_z)]
        jobs2 = [dict(kind=2, key=f"step{s * 100:+.0f}_{r['key'][5:]}", q=r["q"], hit=hit, stance=args.stance,
                      step=s, lift=args.lift) for r in wide for s in args.wide_steps]
        print(f"stage 2: {len(jobs2)} wide step clips", flush=True)
        for r in pool.imap(run, jobs2):
            res.append(r)
            report(r)

    enc = Encoder(ENCODER_ONNX)
    toks, dqs = [], []
    for r in res:
        tok, dq = enc(r["q"], np.tile([1.0, 0, 0, 0], (T, 1)))
        toks.append(tok), dqs.append(dq)
    traj = np.array([r["traj"] for r in res])                 # pelvis-relative
    pel = np.array([r["pel"] for r in res])                   # pelvis vs anchor
    anchor_path = traj + pel
    disp = pel[:, hit]
    step_cost = np.array([0.02 + 0.05 * abs(r["step"]) if r["kind"] == 2 else 0.0 for r in res])
    np.savez_compressed(
        OUT, tokens=np.array(toks, np.float32), ref_q=np.array([r["q"] for r in res], np.float32),
        ref_dq=np.array(dqs, np.float32), contact_traj=traj.astype(np.float32),
        contact_pos=anchor_path[:, hit].astype(np.float32), contact_pos_kin=anchor_path[:, hit].astype(np.float32),
        contact_vel=np.gradient(anchor_path, DT, axis=1)[:, hit].astype(np.float32),
        face_normal=np.array([r["n"] for r in res], np.float32), hit_frame=np.int32(hit), fps=np.int32(50),
        motion_keys=np.array([r["key"] for r in res]), joint_names=np.array(SONIC_JOINTS),
        ref_root_height=np.float32(float(v1["ref_root_height"]) - args.stance),
        root_disp=disp.astype(np.float32), step_cost=step_cost.astype(np.float32),
        kind=np.array([r["kind"] for r in res], np.int32), calibrated=np.bool_(False),
        face_corrected=np.bool_(True), face_normal_goal=v1["face_normal_goal"])
    cp = anchor_path[:, hit]
    print(f"saved {OUT}: {len(res)} clips; contact y {cp[:, 1].min():+.2f}..{cp[:, 1].max():+.2f}, "
          f"z {cp[:, 2].min():+.2f}..{cp[:, 2].max():+.2f} (anchor frame); "
          f"worst leg IK {max(r['leg_err'] for r in res) * 100:.1f} cm, "
          f"grid hit err max {max(r['hit_err'] for r in res) * 100:.1f} cm")


if __name__ == "__main__":
    main()
