"""Reference swing library (data/forehand_library*.npz, scripts/build_library.py / build_library_v2.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SwingLibrary:
    """M reference swings x T frames: SONIC tokens, joint refs, racket contact targets.

    Positions/velocities are in the (fixed) reference pelvis frame, +X forward, +Y left, +Z up.
    """

    def __init__(self, path: str | Path, device="cpu"):
        d = np.load(path)
        t = lambda k: torch.as_tensor(d[k], dtype=torch.float32, device=device)  # noqa: E731
        self.tokens = t("tokens")              # (M, T, 64)
        self.ref_q = t("ref_q")                # (M, T, 29) SONIC order
        self.ref_dq = t("ref_dq")
        self.contact_pos = t("contact_pos")    # (M, 3) racket contact point at impact
        self.contact_vel = t("contact_vel")    # (M, 3)
        self.face_normal = t("face_normal")    # (M, 3)
        self.contact_traj = t("contact_traj")  # (M, T, 3)
        self.hit_frame = int(d["hit_frame"])
        self.fps = int(d["fps"])
        self.motion_keys = [str(k) for k in d["motion_keys"]]
        self.ref_root_height = float(d["ref_root_height"])
        M = self.tokens.shape[0]
        # v2: pelvis displacement at impact (anchor frame), selection cost of stepping clips
        self.root_disp = t("root_disp") if "root_disp" in d.files else torch.zeros(M, 3, device=device)
        self.step_cost = t("step_cost") if "step_cost" in d.files else torch.zeros(M, device=device)
        self.kind = torch.as_tensor(d["kind"], device=device) if "kind" in d.files else \
            torch.zeros(M, dtype=torch.long, device=device)
        self.num_motions, self.num_frames = self.tokens.shape[:2]
        self.pos_min = self.contact_pos.min(0).values
        self.pos_max = self.contact_pos.max(0).values

    def contact_dist(self, target_pos: torch.Tensor) -> torch.Tensor:
        """(N, M) distance in the strike plane (y, z) from each target to each clip's contact."""
        return torch.cdist(target_pos[:, 1:], self.contact_pos[:, 1:])

    def nearest(self, target_pos: torch.Tensor) -> torch.Tensor:
        """Clip for each target (N, 3): nearest contact in (y, z) (all clips share one strike
        plane, x ~ 0.43 m), plus each clip's step_cost so a step is used only when it pays."""
        return (self.contact_dist(target_pos) + self.step_cost[None]).argmin(dim=1)

    def reach(self, target_pos: torch.Tensor) -> torch.Tensor:
        """(N,) distance from each target to the closest clip contact."""
        return self.contact_dist(target_pos).min(dim=1).values

    def token(self, motion: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        f = frame.clamp(0, self.num_frames - 1)
        return self.tokens[motion, f]

    def token_window(self, motion: torch.Tensor, frame: torch.Tensor, k: int) -> torch.Tensor:
        """(N, 2k+1, 64) tokens at frames frame-k .. frame+k (clamped), for timing-slack matching."""
        offs = torch.arange(-k, k + 1, device=frame.device)
        f = (frame[:, None] + offs[None]).clamp(0, self.num_frames - 1)
        return self.tokens[motion[:, None], f]
