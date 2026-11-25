"""Return-rate heatmap over the hitting plane from eval_ball.py's per-swing rows.

Each planned swing is binned by its planned strike point (swing anchor frame: y = lateral,
+ left; z = height above the floor at the nominal pelvis height + z offset). Cell colour =
return rate (sequential blue, light = 0 %, dark = 100 %); cell text = rate and swing count.
Cells with fewer than --min_n swings are left blank. The dashed box is the library's
contact window (+- the eval's window margin); dots are the library clips' contacts.
Viewed from behind the robot: its left on the left.

  python scripts/plot_heatmap.py logs/eval_ball_diag.pt --out logs/heatmap.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/g1_sonic_tt"))
from g1_sonic_tt import paths  # noqa: E402

# sequential single-hue ramp (light -> dark blue), text tokens
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diag", nargs="?", default="logs/eval_ball_diag.pt")
    ap.add_argument("--out", default="logs/heatmap.png")
    ap.add_argument("--cell", type=float, default=0.05, help="cell size, m")
    ap.add_argument("--min_n", type=int, default=5)
    ap.add_argument("--title", default=None)
    ap.add_argument("--yrange", type=float, nargs=2, default=(-1.2, 0.8), help="lateral extent shown, m")
    ap.add_argument("--zrange", type=float, nargs=2, default=(0.45, 1.15), help="height extent shown, m")
    args = ap.parse_args()

    d = torch.load(args.diag)
    R, cols = d["rows"].numpy(), d["cols"]
    c = {k: i for i, k in enumerate(cols)}
    if "contacts" in d:  # library used by the eval (v2: anchor frame, measured)
        cp, kind, z0 = d["contacts"].numpy(), d["kind"].numpy(), float(d["ref_root_height"])
    else:
        lib = np.load(paths.FOREHAND_LIBRARY)
        cp, kind, z0 = lib["contact_pos"], np.zeros(len(lib["contact_pos"]), int), float(lib["ref_root_height"])
    y, z, ret = R[:, c["plan_y"]], R[:, c["plan_z"]] + z0, R[:, c["returned"]] > 0

    s = args.cell
    ye = np.arange(args.yrange[0], args.yrange[1] + 1e-6, s)
    ze = np.arange(args.zrange[0], args.zrange[1] + 1e-6, s)
    n, _, _ = np.histogram2d(y, z, [ye, ze])
    k, _, _ = np.histogram2d(y[ret], z[ret], [ye, ze])
    rate = np.where(n >= args.min_n, k / np.maximum(n, 1), np.nan)

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(ye) + 2), max(4, 0.55 * len(ze) + 1.5)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(SURFACE)
    m = ax.pcolormesh(ye, ze, rate.T * 100, cmap=cmap, vmin=0, vmax=100, edgecolors=SURFACE, linewidth=2)
    for i in range(len(ye) - 1):
        for j in range(len(ze) - 1):
            if not np.isnan(rate[i, j]):
                v = rate[i, j] * 100
                ax.text((ye[i] + ye[i + 1]) / 2, (ze[j] + ze[j + 1]) / 2, f"{v:.0f}%\nn={int(n[i, j])}",
                        ha="center", va="center", fontsize=6.5, color="white" if v > 55 else INK)
    st = kind == 2
    ax.scatter(cp[~st, 1], cp[~st, 2] + z0, s=14, c=SURFACE, edgecolors=INK, linewidths=1, zorder=3,
               label="clip contacts (no step)")
    if st.any():
        ax.scatter(cp[st, 1], cp[st, 2] + z0, s=16, marker="^", c=SURFACE, edgecolors=INK, linewidths=1, zorder=3,
                   label="clip contacts (side-step)")
    ax.invert_xaxis()  # robot's left (+y) on the left, viewed from behind
    ax.set_xlabel("lateral y (m)   ← robot's left · robot's right →", color=INK2)
    ax.set_ylabel("strike height z (m above floor)", color=INK2)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2)
    tot = f"{ret.mean() * 100:.0f}% of {len(ret)} swings returned (all serves, incl. off-plot)"
    ax.set_title(args.title or f"Return rate over the hitting plane: {tot}", color=INK, fontsize=10, loc="left")
    cb = fig.colorbar(m, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("returns (%)", color=INK2)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(colors=INK2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"saved {args.out}: {tot}; cells shown {int((n >= args.min_n).sum())}")


if __name__ == "__main__":
    main()
