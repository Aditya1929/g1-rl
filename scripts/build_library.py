"""Build data/forehand_library.npz: the reference library (priors) the env uses.

Per motion of table-tennis-motion-kit/examples/forehand_grid:
  1. Face correction (default on; --no-face-fix to skip). Around impact the
     raw clips hit with the racket face at ~45 deg (normal ~ (0.71, 0.71, 0)).
     Re-solve the 7 right-arm joints per frame with least squares so that
       - the racket contact point stays on the clip's contact path,
       - the face normal blends to --face-normal (default (1, 0, 0.04), forward)
         with weight b(t) = exp(-0.5 ((t - t_hit) / 0.33)^4),
       - the joints stay close to the clip (regularizer),
     within the kit's optimized joint bounds. Then smooth (Gaussian, 1 frame) and
     close the loop (last frame = first). Same recipe as
     vision_robotic_table_tennis hitting_plane/train_model.py.
  2. Re-encode every frame with SONIC's frozen release encoder (mode 0), with the
     exact input layout of ~/sonic-dataset/build_dataset.py (checked: encoding
     the uncorrected clip reproduces the published token dataset).
  3. Store tokens, joint refs (pos, vel), the racket contact path, and contact
     position / velocity / face normal at impact.

Positions are in the reference pelvis frame (pelvis fixed, unrotated in the clips).
Needs numpy, scipy, onnxruntime (the sonic-dataset venv):

  ~/sonic-dataset/.venv/bin/python scripts/build_library.py [--no-face-fix]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/g1_sonic_tt"))
from g1_sonic_tt import paths  # noqa: E402
from g1_sonic_tt.sonic.constants import (  # noqa: E402
    MUJOCO_TO_ISAACLAB, RACKET_CONTACT_OFFSET, RACKET_FACE_AXIS, RACKET_PARENT_LINK, SONIC_JOINTS,
)
from g1_sonic_tt.sonic.urdf_fk import UrdfChain  # noqa: E402

sys.path.insert(0, str(paths.SONIC_DATASET))
from build_dataset import encoder_inputs  # noqa: E402  (the dataset's verified encoder layout)

ENCODER_ONNX = paths.SONIC_ROOT / f"gear_sonic_deploy/policy/{paths.SONIC_POLICY}/model_encoder.onnx"
RIGHT_ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
OFF, AXIS = np.array(RACKET_CONTACT_OFFSET), np.array(RACKET_FACE_AXIS)


class Encoder:
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name

    def __call__(self, joint_pos_il, root_wxyz, fps=50):
        jv = np.gradient(joint_pos_il, 1.0 / fps, axis=0)
        x = encoder_inputs(joint_pos_il, jv, root_wxyz)
        return np.concatenate([self.s.run(None, {self.name: x[t:t + 1]})[0] for t in range(len(x))]), jv


def racket(chain, names, q):
    """Contact point and face normal (pelvis frame) for one frame of kit-order joints."""
    R, p = chain.pose(dict(zip(names, q)))
    return p + R @ OFF, R @ AXIS


def face_fix(chain, names, q, time, hit_t, contact_target, n_goal, bounds, margin=0.015):
    """Per-frame IK on the right arm (see module docstring). Returns corrected (T, 29) kit-order q."""
    idx = [names.index(n) for n in RIGHT_ARM]
    lo = np.array([bounds[n][0] for n in RIGHT_ARM]) + margin
    hi = np.array([bounds[n][1] for n in RIGHT_ARM]) - margin
    out = q.copy()
    for t in range(len(time)):
        b = np.exp(-0.5 * ((time[t] - hit_t) / 0.33) ** 4)
        if b < 1e-4:
            continue
        _, n0 = racket(chain, names, q[t])
        n_des = (1 - b) * n0 + b * n_goal
        n_des /= np.linalg.norm(n_des)
        a0 = q[t, idx]

        def res(a, t=t):
            qq = q[t].copy()
            qq[idx] = a
            c, n = racket(chain, names, qq)
            return np.r_[12 * (c - contact_target[t]), 2 * (n - n_des), 0.07 * (a - a0)]

        out[t, idx] = least_squares(res, np.clip(a0, lo, hi), bounds=(lo, hi), max_nfev=60).x
    out[:, idx] = gaussian_filter1d(out[:, idx], 1, axis=0, mode="nearest")
    out[-1] = out[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-face-fix", action="store_true", help="keep the raw clips (45 deg face at impact)")
    ap.add_argument("--face-normal", type=float, nargs=3, default=[1.0, 0.0, 0.04])
    args = ap.parse_args()
    n_goal = np.array(args.face_normal) / np.linalg.norm(args.face_normal)

    chain = UrdfChain(paths.SONIC_URDF, "pelvis", RACKET_PARENT_LINK)
    enc = Encoder(ENCODER_ONNX)
    ds = np.load(paths.TOKEN_DATASET)
    ds_keys = [str(k) for k in ds["motion_keys"]]
    keys = sorted(d.name for d in paths.FOREHAND_GRID.iterdir() if (d / "reference.npz").exists())
    assert keys == ds_keys, "motion set differs from the token dataset"
    M = len(keys)

    out = {k: [] for k in ("tokens", "ref_q", "ref_dq", "contact_pos", "contact_vel", "face_normal",
                           "contact_traj")}
    hit_frames, root_heights, report = set(), [], []
    for m, key in enumerate(keys):
        d = paths.FOREHAND_GRID / key
        meta = json.loads((d / "metadata.json").read_text())
        ref = np.load(d / "reference.npz")
        names = [str(n) for n in ref["joint_names"]]
        assert [names[i] for i in MUJOCO_TO_ISAACLAB] == SONIC_JOINTS
        assert meta["fps"] == 50 and np.allclose(ref["root_wxyz"], [1, 0, 0, 0]), "fixed, unrotated pelvis expected"
        hit, time, root = int(meta["hit_idx"]), ref["time"], ref["root_pos"]
        hit_frames.add(hit)
        q = ref["dof"].astype(np.float64)
        target_c = ref["actual_contact_pos"] - root

        # sanity: our FK + racket offset reproduce the kit's contact point
        c_hit, n_hit_raw = racket(chain, names, q[hit])
        assert np.linalg.norm(c_hit - target_c[hit]) < 2e-3, f"{key}: FK/racket offset mismatch"
        if m == 0:  # sanity: our encoding of the raw clip == the published token dataset
            tok_raw, _ = enc(q[:, MUJOCO_TO_ISAACLAB], ref["root_wxyz"].astype(np.float64))
            ds_tok = ds["token"][ds["motion_id"] == 0][np.argsort(ds["frame"][ds["motion_id"] == 0])]
            assert np.abs(tok_raw - ds_tok).max() < 1e-5, "encoder layout drifted from the dataset"

        if not args.no_face_fix:
            q = face_fix(chain, names, q, time, time[hit], target_c, n_goal, meta["optimized_joint_bounds_rad"])
        q_il = q[:, MUJOCO_TO_ISAACLAB]
        tokens, dq_il = enc(q_il, ref["root_wxyz"].astype(np.float64))
        traj = np.array([racket(chain, names, q[t])[0] for t in range(len(time))])
        n_hit = racket(chain, names, q[hit])[1]
        out["tokens"].append(tokens)
        out["ref_q"].append(q_il)
        out["ref_dq"].append(dq_il)
        out["contact_traj"].append(traj)
        out["contact_pos"].append(traj[hit])
        out["contact_vel"].append(np.gradient(traj, 0.02, axis=0)[hit])
        out["face_normal"].append(n_hit)
        root_heights.append(float(root[0, 2]))
        report.append((key, 100 * np.linalg.norm(traj - target_c, axis=1).max(),
                       100 * np.linalg.norm(traj[hit] - target_c[hit]),
                       np.degrees(np.arccos(np.clip(n_hit @ n_goal, -1, 1))),
                       np.degrees(np.arccos(np.clip(n_hit_raw @ n_goal, -1, 1)))))
        print(f"{m:2d} {key}: path err max {report[-1][1]:.1f} cm, at hit {report[-1][2]:.1f} cm, "
              f"face vs goal {report[-1][4]:.0f} -> {report[-1][3]:.1f} deg", flush=True)

    assert len(hit_frames) == 1
    lib = {k: np.asarray(v, np.float32) for k, v in out.items()}
    dst = paths.FOREHAND_LIBRARY
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst, **lib, hit_frame=np.int32(hit_frames.pop()), fps=np.int32(50), motion_keys=np.array(keys),
        joint_names=np.array(SONIC_JOINTS), ref_root_height=np.float32(np.mean(root_heights)),
        face_corrected=np.bool_(not args.no_face_fix), face_normal_goal=n_goal.astype(np.float32),
    )
    r = np.array([x[1:] for x in report])
    print(f"saved {dst}: {M} motions x {lib['tokens'].shape[1]} frames, face_corrected={not args.no_face_fix}")
    print(f"contact path error max {r[:, 0].max():.1f} cm | at hit max {r[:, 1].max():.1f} cm | "
          f"face vs goal at hit: mean {r[:, 2].mean():.1f} deg (raw clips {r[:, 3].mean():.0f} deg)")
    print("contact_pos range", lib["contact_pos"].min(0).round(3), lib["contact_pos"].max(0).round(3))
    print("contact_vel mean ", lib["contact_vel"].mean(0).round(3), "| face normal mean",
          lib["face_normal"].mean(0).round(3))


if __name__ == "__main__":
    main()
