"""Figure 3: decoder predictions, stacked-DOF style (one row per DOF, offset).

One panel per feature, same decoder configuration, whole test set stitched
together (gaps between trials removed, as in the lab's kinematics plots).
Target = grey, prediction = feature colour. The number at the right of each
row is the pipeline's intended-trial RMSE for that DOF (current_eval json),
i.e. the poster's metric.

Usage (project env, from repo root):
    python plot_sfn_fig_003_preds.py ~/sfn_session36 reports/stats_sfn2026/final
Options:
    --config 8DOF     decoder configuration to show (3DOF, 6DOF, 8DOF)
    --dofs 1 2 10     subset of DOFs to show (default: all in the config)
    --offset 1.75     vertical spacing between DOF rows (config.yaml vis value)
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.family": "Arial", "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})

FEATS = ["NFR", "SBP-RAW", "DWT-DB4", "MAV"]
COLS = {"NFR": "#E69F00", "SBP-RAW": "#56B4E9", "DWT-DB4": "#009E73", "MAV": "#CC79A7"}
NAMES = {1: "Thumb F/E", 2: "Index F/E", 3: "Middle F/E", 4: "Ring F/E",
         5: "Little F/E", 6: "Thumb Intrinsic", 10: "Wrist Pitch", 12: "Wrist Roll"}

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--config", default="8DOF")
ap.add_argument("--dofs", type=int, nargs="*")
ap.add_argument("--offset", type=float, default=1.75)
ap.add_argument("--variant", default="",
                help="'' = original predictions; e.g. mkf_gt reads mkf_gt/<cfg>/decode "
                     "and current_eval_mkf_gt/ (from run_mkf_36.sh)")
a = ap.parse_args()
root = a.root.expanduser()
a.out.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 4, figsize=(18, 9), sharey=True)
for ax, feat in zip(axes, FEATS):
    pred_root = root / a.variant if a.variant else root
    with h5py.File(pred_root / a.config / "decode" / f"{feat}_predictions.h5", "r") as f:
        ks = [int(k) for k in f["kinematic_indices"][()]]
        p, y = f["predictions"][()], f["true_kinematics"][()]
        t = f["test_nip_time"][()]
    eval_dir = root / (f"current_eval_{a.variant}" if a.variant else "current_eval")
    m = json.load(open(eval_dir / f"{a.config}_{feat}_metrics.json"))
    per_dof = m["aggregate_per_dof"]
    show = a.dofs or ks
    x = np.arange(len(t)) * m["summary"]["dt"]          # stitched time, gaps removed
    jumps = np.where(np.diff(t) > 3 * np.median(np.diff(t)))[0]   # trial joins

    for row, k in enumerate(reversed(show)):          # first DOF on top
        j = ks.index(k)
        base = row * a.offset
        ax.axhline(base, color="0.85", lw=0.8, zorder=0)
        ax.plot(x, y[:, j] + base, color="0.55", lw=1.8)
        ax.plot(x, p[:, j] + base, color=COLS[feat], lw=1.3)
        r = per_dof.get(str(k), {}).get("intended", {}).get("rmse_mean", np.nan)
        ax.text(x[-1] * 1.01, base + 0.35, f"{r:.2f}", fontsize=11, va="center")
        if ax is axes[0]:
            ax.text(-x[-1] * 0.02, base + 0.35, NAMES.get(k, f"DOF {k}"),
                    ha="right", va="center", fontsize=12)
    for jj in jumps:
        ax.axvline(x[jj], color="0.92", lw=0.5, zorder=0)
    total = m["summary"]["intended"]["rmse_mean"]
    ax.set_title(f"{feat}\nintended RMSE {total:.2f}", color=COLS[feat],
                 fontweight="bold", fontsize=15)
    ax.set_xlabel("Test-set time, trials stitched (s)")
    ax.set_yticks([])

handles = [plt.Line2D([], [], color="0.55", lw=1.8, label="Target"),
           plt.Line2D([], [], color="0.2", lw=1.3, label="Prediction")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
fig.tight_layout(rect=(0.04, 0, 0.98, 0.95))

stem = a.out / f"fig3_predictions_{a.config}{'_' + a.variant if a.variant else ''}"
for ext in ("svg", "pdf", "png"):
    fig.savefig(f"{stem}.{ext}", dpi=150)
print("Saved", f"{stem}.svg/.pdf/.png")

