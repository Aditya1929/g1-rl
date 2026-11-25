"""Phase 0 feasibility clips (no training): a backhand set and a side-step set, in the
same library format as data/forehand_library.npz, so they can be replayed through the
frozen SONIC decoder with scripts/phase0_replay.py.

backhand  (data/phase0_backhand.npz)
  Seed: each forehand clip mirrored left<->right (legs, waist twist, arms), with the
  original idle left arm kept. The right arm (7 joints) is re-solved per frame (warm
  started, smoothness-regularised) so that the racket contact point follows the mirrored
  forehand contact path shifted by --y_shift toward the body centre, the face normal
  blends to the mirrored forehand goal around impact (either blade side), and the
  elbow / wrist / racket keep clear of the torso, the pelvis and the left arm.

sidestep  (data/phase0_sidestep.npz)
  Upper body of the nominal forehand clip; legs replaced by a solved side-step: the lead
  foot (in the step direction) moves first, then the trail foot, the pelvis follows a
  min-jerk shift of d in y, at a crouch of --crouch below the clips' standing height (a
  straight leg cannot move its foot sideways). Encoder mode 0 has no pelvis translation:
  the step exists only through the leg joints, so the replay measures how far SONIC
  actually moves. Stored per clip as root_disp (commanded).

  ~/sonic-dataset/.venv/bin/python scripts/phase0_build.py [--only backhand|sidestep]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_library import ENCODER_ONNX, OFF, RIGHT_ARM, Encoder  # noqa: E402

from g1_sonic_tt import paths  # noqa: E402  (build_library put the package on sys.path)
from g1_sonic_tt.sonic.constants import MUJOCO_TO_ISAACLAB, RACKET_FACE_AXIS, RACKET_PARENT_LINK  # noqa: E402
from g1_sonic_tt.sonic.urdf_fk import UrdfChain  # noqa: E402

AXIS = np.array(RACKET_FACE_AXIS)
LEFT_ARM = [n.replace("right_", "left_") for n in RIGHT_ARM]
LEGS = {s: [f"{s}_hip_pitch_joint", f"{s}_hip_roll_joint", f"{s}_hip_yaw_joint", f"{s}_knee_joint",
            f"{s}_ankle_pitch_joint", f"{s}_ankle_roll_joint"] for s in ("left", "right")}
FPS, DT = 50, 0.02


def load_clips():
    keys = sorted(d.name for d in paths.FOREHAND_GRID.iterdir() if (d / "reference.npz").exists())
    clips = []
    for k in keys:
        import json
        d = paths.FOREHAND_GRID / k
        ref = np.load(d / "reference.npz")
        meta = json.loads((d / "metadata.json").read_text())
        clips.append(dict(key=k, names=[str(n) for n in ref["joint_names"]], q=ref["dof"].astype(np.float64),
                          contact=ref["actual_contact_pos"] - ref["root_pos"], hit=int(meta["hit_idx"]),
                          root=ref["root_pos"], time=ref["time"]))
    return clips


def mirror(q, names):
    """Left<->right mirror of a kit-order joint vector (roll / yaw joints change sign)."""
    out = np.empty_like(q)
    for i, n in enumerate(names):
        p = n.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")
        s = -1.0 if ("roll" in n or "yaw" in n) else 1.0
        out[..., i] = s * q[..., names.index(p)]
    return out


class Body:
    """FK points in the pelvis frame for collision checks and the racket."""

    def __init__(self):
        u = paths.SONIC_URDF
        self.wrist = UrdfChain(u, "pelvis", RACKET_PARENT_LINK)
        self.elbow = UrdfChain(u, "pelvis", "right_elbow_link")
        self.torso = UrdfChain(u, "pelvis", "torso_link")
        self.lwrist = UrdfChain(u, "pelvis", "left_wrist_yaw_link")
        self.lelbow = UrdfChain(u, "pelvis", "left_elbow_link")

    def racket(self, qd):
        R, p = self.wrist.pose(qd)
        return p + R @ OFF, R @ AXIS, p + R @ np.array([0.2365, 0, 0]), p

    def obstacles(self, qd):
        """Capsules (a, b, radius): torso (vertical), pelvis (lateral); spheres at the left arm."""
        R, p = self.torso.pose(qd)
        caps = [(p + R @ [0, 0, -0.05], p + R @ [0, 0, 0.40], 0.13),
                (np.array([0, -0.11, 0.0]), np.array([0, 0.11, 0.0]), 0.12)]
        sph = [(self.lwrist.pose(qd)[1], 0.07), (self.lelbow.pose(qd)[1], 0.07)]
        return caps, sph


def _seg_dist(x, a, b):
    ab = b - a
    t = np.clip((x - a) @ ab / (ab @ ab), 0, 1)
    return np.linalg.norm(x - (a + t * ab))


def clearance(body, qd):
    """Min (distance - radius - margin) over right-arm points vs obstacles (negative = collision)."""
    _, _, blade, wrist = body.racket(qd)
    elbow = body.elbow.pose(qd)[1]
    caps, sph = body.obstacles(qd)
    pts = [(elbow, 0.05), (wrist, 0.05), (blade, 0.08)]
    out = []
    for x, r in pts:
        out += [_seg_dist(x, a, b) - rc - r for a, b, rc in caps]
        out += [np.linalg.norm(x - c) - rc - r for c, rc in sph]
    return np.array(out)


# ----------------------------------------------------------------------------- backhand
def build_mirror_only(clips, body):
    """Diagnostic: the mirrored body (left-handed forehand motion, idle right arm), no IK."""
    names = clips[0]["names"]
    out = []
    for c in clips[:4]:
        q = mirror(c["q"], names)
        tgt = np.array([body.racket(dict(zip(names, q[t])))[0] for t in range(len(q))])
        out.append(finish(c, q, body, names, f"mirror_{c['key'][-9:]}", tgt, np.zeros(3), extra={}))
    return out


def build_backhand(clips, body, y_shift, n_goal_fh):
    names = clips[0]["names"]
    idx = [names.index(n) for n in RIGHT_ARM]
    lidx = [names.index(n) for n in LEFT_ARM]
    lim = body.wrist.limits
    lo = np.array([lim[n][0] for n in RIGHT_ARM]) + 0.02
    hi = np.array([lim[n][1] for n in RIGHT_ARM]) - 0.02
    n_goal = n_goal_fh * np.array([1, -1, 1])
    n_goal /= np.linalg.norm(n_goal)
    out = []
    for c in clips:
        q = mirror(c["q"], names)
        q[:, lidx] = c["q"][:, lidx]                       # keep the idle (non-racket) left arm
        target = c["contact"] * np.array([1, -1, 1]) + np.array([0, -y_shift, 0])
        T, hit = len(q), c["hit"]
        b = np.exp(-0.5 * ((np.arange(T) - hit) * DT / 0.33) ** 4)

        def solve(t, a_init, a_prev, sign, w_smooth, a_home=None, w_home=0.0):
            def res(a):
                qq = q[t].copy()
                qq[idx] = a
                qd = dict(zip(names, qq))
                cpt, n, _, _ = body.racket(qd)
                col = np.minimum(clearance(body, qd), 0.0)
                r = [12 * (cpt - target[t]), 30 * col, w_smooth * (a - a_prev)]
                if w_home > 0:
                    r.append(w_home * (a - a_home))
                if b[t] > 1e-3:
                    r.append(2 * b[t] * (sign * n - n_goal))
                return np.concatenate(r)
            s = least_squares(res, np.clip(a_init, lo, hi), bounds=(lo, hi), max_nfev=200)
            return s.x, s.cost

        # impact pose first (multi-start, both blade sides), then sweep outward from it
        rng = np.random.default_rng(0)
        best = None
        for sign in (1.0, -1.0):
            for k in range(10):
                a0 = q[hit, idx] if k == 0 else rng.uniform(lo, hi)
                a, cost = solve(hit, a0, a0, sign, 0.0)
                if best is None or cost < best[1]:
                    best = (a, cost, sign)
        a_hit, _, sign = best
        A = np.zeros((T, 7))
        A[hit] = a_hit
        prev = a_hit                                        # back from impact to the ready pose
        for t in range(hit - 1, -1, -1):
            A[t], _ = solve(t, prev, prev, sign, 0.3)
            prev = A[t]
        prev = a_hit                                        # follow-through, pulled home to A[0]
        for t in range(hit + 1, T):
            w = 4.0 * np.clip((t - hit - 15) / (T - 1 - hit - 15), 0, 1) ** 2
            A[t], _ = solve(t, prev, prev, sign, 0.3, A[0], w)
            prev = A[t]
        A = gaussian_filter1d(A, 1.0, axis=0, mode="nearest")
        loop_jump = np.abs(A[-1] - A[0]).max()
        A[-1] = A[0]
        q[:, idx] = A
        out.append(finish(c, q, body, names, f"backhand_{c['key'][-9:]}", target, np.zeros(3),
                          extra=dict(loop_jump=loop_jump, sign=sign)))
        r = out[-1]["report"]
        print(f"{r['key']}: path err max {r['path_max']:.1f} cm, at hit {r['hit_err']:.1f} cm, "
              f"face vs goal {np.degrees(np.arccos(np.clip(abs(r['n_hit'] @ n_goal), -1, 1))):.1f} deg, "
              f"min clearance {r['min_clear']:.1f} cm, loop jump {np.degrees(loop_jump):.0f} deg, "
              f"blade side {'+' if sign > 0 else '-'}", flush=True)
    return out


# ----------------------------------------------------------------------------- side-step
def minjerk(t, t0, t1):
    s = np.clip((t - t0) / (t1 - t0), 0, 1)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def build_sidestep(clips, body, dists, crouch, lift=0.04, slow=1.0):
    c = next(x for x in clips if x["key"].endswith("y+000_z000"))
    names = c["names"]
    T = len(c["q"])
    legs = {s: UrdfChain(paths.SONIC_URDF, "pelvis", f"{s}_ankle_roll_link") for s in ("left", "right")}
    q0 = dict(zip(names, c["q"][0]))
    feet0 = {s: legs[s].pose(q0) for s in legs}            # zero-leg stance (pelvis frame)
    t = np.arange(T, dtype=float)
    out = []
    for d in dists:
        lead, trail = ("left", "right") if d > 0 else ("right", "left")
        pel = np.zeros((T, 3))
        pel[:, 1] = d * minjerk(t, 4, 4 + 30 * slow)
        pel[:, 2] = -crouch                                 # relative to the standing height
        foot = {}
        f = lambda x: 4 + (x - 4) * slow  # noqa: E731  (stretch the step timing)
        for s, (f0, f1) in ((lead, (f(6), f(18))), (trail, (f(20), f(32)))):
            p = np.repeat(feet0[s][1][None], T, 0)
            ph = np.clip((t - f0) / (f1 - f0), 0, 1)
            p[:, 1] += d * minjerk(t, f0, f1)
            p[:, 2] += lift * np.sin(np.pi * ph)
            foot[s] = p
        q = c["q"].copy()
        for s in ("left", "right"):
            idx = [names.index(n) for n in LEGS[s]]
            lim = legs[s].limits
            lo = np.array([lim[n][0] for n in LEGS[s]]) + 0.01
            hi = np.array([lim[n][1] for n in LEGS[s]]) - 0.01
            R_goal = feet0[s][0]
            prev = np.clip(np.array([-0.2, 0, 0, 0.4, -0.2, 0]), lo, hi)
            for k in range(T):
                tgt = foot[s][k] - pel[k]

                def res(a, k=k, tgt=tgt):
                    qd = dict(q0)
                    qd.update(zip(LEGS[s], a))
                    R, p = legs[s].pose(qd)
                    return np.concatenate([50 * (p - tgt), 5 * (R - R_goal).ravel(), 0.05 * (a - prev)])
                sol = least_squares(res, prev, bounds=(lo, hi), max_nfev=200)
                q[k, idx] = prev = sol.x
                if k in (0, 12, 26, T - 1):
                    err = np.linalg.norm(res(sol.x)[:3]) / 50
                    if err > 0.005:
                        print(f"  warn: {s} foot IK error {err * 100:.1f} cm at frame {k} (d={d})")
        q[-1] = q[0]
        out.append(finish(c, q, body, names, f"sidestep_{'L' if d > 0 else 'R'}{abs(d) * 100:.0f}cm",
                          c["contact"], np.array([0.0, d, 0.0]), extra=dict(d=d)))
        r = out[-1]["report"]
        print(f"{r['key']}: commanded step {d * 100:+.0f} cm, racket path vs forehand {r['path_max']:.1f} cm", flush=True)
    return out


# ----------------------------------------------------------------------------- shared
def finish(c, q, body, names, key, target, root_disp, extra):
    traj = np.array([body.racket(dict(zip(names, q[t])))[0] for t in range(len(q))])
    hit = c["hit"]
    n_hit = body.racket(dict(zip(names, q[hit])))[1]
    clear = min(clearance(body, dict(zip(names, q[t]))).min() for t in range(0, len(q), 2))
    rep = dict(key=key, path_max=100 * np.linalg.norm(traj - target, axis=1).max(),
               hit_err=100 * np.linalg.norm(traj[hit] - target[hit]), n_hit=n_hit, min_clear=100 * clear, **extra)
    return dict(key=key, q=q, traj=traj, hit=hit, n_hit=n_hit, root_disp=root_disp, report=rep)


def save(items, enc, dst, root_height):
    toks, qs, dqs = [], [], []
    for it in items:
        q_il = it["q"][:, MUJOCO_TO_ISAACLAB]
        tok, dq = enc(q_il, np.tile([1.0, 0, 0, 0], (len(q_il), 1)))
        toks.append(tok), qs.append(q_il), dqs.append(dq)
    hit = items[0]["hit"]
    traj = np.array([it["traj"] for it in items], np.float32)
    np.savez_compressed(
        dst, tokens=np.array(toks, np.float32), ref_q=np.array(qs, np.float32), ref_dq=np.array(dqs, np.float32),
        contact_traj=traj, contact_pos=traj[:, hit], contact_vel=np.gradient(traj, DT, axis=1)[:, hit],
        face_normal=np.array([it["n_hit"] for it in items], np.float32), hit_frame=np.int32(hit), fps=np.int32(FPS),
        motion_keys=np.array([it["key"] for it in items]), ref_root_height=np.float32(root_height),
        root_disp=np.array([it["root_disp"] for it in items], np.float32), face_corrected=np.bool_(True),
        joint_names=np.array([n for n in np.load(paths.FOREHAND_LIBRARY)["joint_names"]]),
        face_normal_goal=np.zeros(3, np.float32))
    print(f"saved {dst}: {len(items)} clips, contact y range "
          f"{traj[:, hit, 1].min():+.2f}..{traj[:, hit, 1].max():+.2f} m, z {traj[:, hit, 2].min():+.2f}..{traj[:, hit, 2].max():+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["backhand", "sidestep", "mirror"])
    ap.add_argument("--y_shift", type=float, default=0.2, help="backhand contact path shift toward the body (m)")
    ap.add_argument("--crouch", type=float, default=0.06)
    ap.add_argument("--lift", type=float, default=0.04, help="side-step foot lift (m)")
    ap.add_argument("--slow", type=float, default=1.0, help="side-step timing stretch (1 = 0.6 s)")
    ap.add_argument("--out_suffix", default="")
    ap.add_argument("--dists", type=float, nargs="+", default=[-0.15, -0.25, -0.35, 0.15, 0.25, 0.35])
    args = ap.parse_args()
    clips, body, enc = load_clips(), Body(), Encoder(ENCODER_ONNX)
    lib = np.load(paths.FOREHAND_LIBRARY)
    h = float(lib["ref_root_height"])
    if args.only == "mirror":
        save(build_mirror_only(clips, body), enc, paths.DATA_DIR / "phase0_mirror.npz", h)
        return
    if args.only in (None, "backhand"):
        print("== backhand")
        save(build_backhand(clips, body, args.y_shift, np.array(lib["face_normal_goal"], float)), enc,
             paths.DATA_DIR / "phase0_backhand.npz", h)
    if args.only in (None, "sidestep"):
        print("== sidestep")
        save(build_sidestep(clips, body, args.dists, args.crouch, args.lift, args.slow), enc,
             paths.DATA_DIR / f"phase0_sidestep{args.out_suffix}.npz",
             h - args.crouch)


if __name__ == "__main__":
    main()
