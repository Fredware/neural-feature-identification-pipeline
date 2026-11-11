"""Figure 3 (poster size): decoder predictions, stacked-DOF style.

One panel per feature, one row per DOF (offset), same decoder configuration.
Only flexion trials are shown: the USEAs are in the median and ulnar nerves, so
extension (radial-nerve) movements are left out of this figure. Test trials are
stitched together (gaps removed); thin grey lines mark the joins.
Target = grey, prediction = feature colour.

Direction per trial comes from the movement matrix (amplitude sign, as in
select_features._get_signature_from_amplitudes). Flexion is +1 for every DOF
except Thumb Intrinsic (DOF 6), where flexion is -1. Override with --flexion-sign.

RMSE shown = mean over the SHOWN (flexion) test trials of the per-trial RMSE of the
moved DOF, computed here from the plotted predictions, with the evaluate step's
burn-in (first 15 test samples) dropped. As a check, the same computation over ALL
test trials is logged next to the pipeline's own number from the metrics json.
With --variant mkf_gt these are MKF (visualisation) predictions, not the raw Kalman
output used for the statistics.

Usage (project env, repo root):
    python scripts/python/plot_sfn_fig_003_preds.py ~/sfn_session144 reports/stats_sfn2026/final \
        --config 6DOF --variant mkf_gt
Options:
    --size 23 13.5          printed size in inches (poster slot)
    --font-size 34          every text element, Arial regular
    --line-width 2.5        data lines; guides/axes use 2.0 (minimum 2 pt)
    --gap 0.35              space between DOF rows (each row is as tall as its target range)
    --flexion-sign 12=-1    change which amplitude sign counts as flexion for a DOF
    --flip-display 6        DOFs drawn upside down (default: 6, Thumb Intrinsic, so flexion points up);
                            pass the flag alone to draw every DOF with its real sign
    --min-coverage 0.5      drop "sliver" trials: a few test samples inside a neighbouring trial's window
    --title "S7, day 485"   default: looked up from reports/aggregated_decoder_metrics.csv
    --all-directions        keep extension trials (previous behaviour)
    --trial-lines dof       join lines between DOF blocks (default), every trial ('all'), or 'none'
Only trials that move a shown DOF are drawn (an 8-DOF session decoded at 6 DOF also has
ring/little trials in its test set; they are left out).
"""
import argparse
import json
import logging
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.dev.ConsoleRenderer(),
])
log = structlog.get_logger(__name__)

FEATS = ["NFR", "SBP-RAW", "DWT-DB4", "MAV"]
LABELS = {"NFR": "NFR", "SBP-RAW": "SBP", "DWT-DB4": "DWT-DB4", "MAV": "MAV"}
COLS = {"NFR": "#E69F00", "SBP-RAW": "#56B4E9", "DWT-DB4": "#009E73", "MAV": "#CC79A7"}
NAMES = {1: "Thumb", 2: "Index", 3: "Middle", 4: "Ring", 5: "Little",
         6: "Thumb\nIntrinsic", 10: "Wrist\nFlexion", 12: "Wrist\nRotation"}
SUBJ = {"P201501": "S5", "P201601": "S6", "P201701": "S7", "P202201": "S8"}
FLEXION_SIGN = {k: 1 for k in range(1, 13)}
FLEXION_SIGN[6] = -1          # Thumb Intrinsic: flexion -1, extension +1
GUIDE_LW = 2.0
TARGET_GREY, JOIN_GREY = "0.55", "0.88"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", default="6DOF")
    ap.add_argument("--dofs", type=int, nargs="*")
    ap.add_argument("--gap", type=float, default=0.35,
                    help="space between DOF rows (kinematic units); each row is as tall as its target range")
    ap.add_argument("--variant", default="",
                    help="'' = raw predictions; mkf_gt reads mkf_gt/<cfg>/decode and current_eval_mkf_gt/")
    ap.add_argument("--size", type=float, nargs=2, default=[23.0, 13.5], metavar=("W", "H"))
    ap.add_argument("--font-size", type=float, default=34)
    ap.add_argument("--line-width", type=float, default=2.5)
    ap.add_argument("--burn-in", type=int, default=15)
    ap.add_argument("--flexion-sign", nargs="*", default=[], metavar="DOF=SIGN")
    ap.add_argument("--flip-display", type=int, nargs="*", default=[6],
                    help="DOFs drawn upside down; default 6 so Thumb Intrinsic flexion points up. "
                         "Pass the flag with no values to draw every DOF with its real sign.")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="a trial is drawn only if at least this fraction of its samples are test samples")
    ap.add_argument("--title", default=None,
                    help="figure title; default looks up participant and day for the session folder")
    ap.add_argument("--results-csv", type=Path, default=Path("reports/aggregated_decoder_metrics.csv"))
    ap.add_argument("--all-directions", action="store_true")
    ap.add_argument("--trial-lines", choices=["dof", "all", "none"], default="dof",
                    help="vertical join lines: between DOF blocks (default), every trial, or none")
    a = ap.parse_args()
    if a.line_width < GUIDE_LW:
        ap.error(f"--line-width must be >= {GUIDE_LW} pt")
    return a


def style(fs):
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # one warning below, not hundreds
    try:
        font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
    except ValueError:
        log.warning("arial_not_installed",
                    note="SVG keeps the name 'Arial' (renders as Arial on the poster machine); "
                         "PDF/PNG use Liberation Sans (same metrics as Arial) if installed")
    plt.rcParams.update({
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.weight": "normal", "font.size": fs,
        "axes.titlesize": fs, "axes.labelsize": fs, "axes.titleweight": "normal",
        "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs,
        "axes.linewidth": GUIDE_LW, "xtick.major.width": GUIDE_LW, "xtick.major.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    })


def load_events(path):
    """Same reading as decode.load_events: trial start/stop NIP times, (n_trials, 12, 4) matrices."""
    with h5py.File(path, "r") as f:
        starts = f["trial_start_idxs"][:].ravel()
        stops = f["trial_stop_idxs"][:].ravel()
        mm = f["movement_matrices"][:]
    if mm.shape[-1] != 4:
        mm = np.transpose(mm, (2, 1, 0))
    return starts, stops, mm


def test_trials(test_nip, all_nip, starts, stops, mm):
    """Per trial with any test samples: (positions in the test array, moved DOF, amplitude sign,
    coverage). Any overlap counts, as in decode.build_test_trial_boundaries; coverage = test samples
    in the trial window / all samples in it, so slivers (a few test samples spilling into a
    neighbouring trial's window) can be told apart from real test trials. Rest/compound skipped."""
    out = []
    for t0, t1, m in zip(starts, stops, mm):
        pos = np.where((test_nip >= t0) & (test_nip <= t1))[0]
        active = np.nonzero(m[:, 0])[0]
        if len(pos) == 0 or len(active) != 1:
            continue
        n_all = int(np.sum((all_nip >= t0) & (all_nip <= t1)))
        k = int(active[0]) + 1
        out.append((pos, k, int(np.sign(m[active[0], 0])), len(pos) / max(n_all, 1)))
    return out


def session_title(root, results_csv):
    """'S7 · day 485' from the aggregated metrics (job id = digits in the session folder name)."""
    import re
    import pandas as pd
    m = re.search(r"(\d+)$", root.name.rstrip("/"))
    if m is None or not results_csv.exists():
        return None
    df = pd.read_csv(results_csv, usecols=["job_id", "participant_id", "days_since_implant"])
    row = df[df["job_id"].astype(str) == m.group(1)].head(1)
    if row.empty:
        return None
    subj = SUBJ.get(row["participant_id"].iat[0], row["participant_id"].iat[0])
    # days_since_implant holds days since the first session (implant dates are placeholders, #6)
    return f"{subj}, day {int(row['days_since_implant'].iat[0])} since first session"


def trial_rmse(trials, p, y, ks, burn_in):
    """Mean over trials of per-trial RMSE of the moved DOF, per DOF; burn-in samples excluded."""
    per = {}
    for pos, k, *_ in trials:
        pos = pos[pos >= burn_in]
        if len(pos) == 0 or k not in ks:
            continue
        j = ks.index(k)
        per.setdefault(k, []).append(np.sqrt(np.mean((p[pos, j] - y[pos, j]) ** 2)))
    return {k: float(np.mean(v)) for k, v in per.items()}


def main():
    a = parse_args()
    style(a.font_size)
    root = a.root.expanduser()
    a.out.mkdir(parents=True, exist_ok=True)
    flex = dict(FLEXION_SIGN)
    for item in a.flexion_sign:
        k, s = item.split("=")
        flex[int(k)] = int(s)
    starts, stops, mm = load_events(root / "events.h5")
    with h5py.File(root / "kinematics.h5", "r") as f:
        all_nip = f["nip_time"][()].ravel()
    title = a.title or session_title(root, a.results_csv)
    if title is None:
        log.warning("no_title", note="session not found in results csv; pass --title")
    pred_root = root / a.variant if a.variant else root
    eval_dir = root / (f"current_eval_{a.variant}" if a.variant else "current_eval")

    # ---- pass 1: load every feature, pick the shown trials, check RMSE against the pipeline
    data = {}
    for feat in FEATS:
        with h5py.File(pred_root / a.config / "decode" / f"{feat}_predictions.h5", "r") as f:
            ks = [int(k) for k in f["kinematic_indices"][()]]
            p, y = f["predictions"][()], f["true_kinematics"][()]
            test_nip = f["test_nip_time"][()].ravel()
        show = a.dofs or ks
        trials = test_trials(test_nip, all_nip, starts, stops, mm)
        slivers = [(t[1], len(t[0]), round(t[3], 2)) for t in trials if t[3] < a.min_coverage]
        if slivers:
            log.warning("sliver_trials", feature=feat, n=len(slivers), dof_ntest_coverage=slivers)
        # shown DOFs only (6DOF config on an 8-DOF session also has ring/little trials in its test set),
        # flexion only, real test trials only (no slivers)
        kept = [t for t in trials if t[1] in show and t[3] >= a.min_coverage
                and (a.all_directions or t[2] == flex[t[1]])]
        r_all, r_shown = trial_rmse(trials, p, y, ks, a.burn_in), trial_rmse(kept, p, y, ks, a.burn_in)
        jpath = eval_dir / f"{a.config}_{feat}_metrics.json"
        pipe = {}
        if jpath.exists():
            per_dof = json.load(open(jpath)).get("aggregate_per_dof", {})
            pipe = {k: per_dof.get(str(k), {}).get("intended", {}).get("rmse_mean") for k in ks}
        for k in show:
            log.info("rmse_per_dof", feature=feat, dof=k, shown_trials=round(r_shown.get(k, np.nan), 3),
                     all_trials_here=round(r_all.get(k, np.nan), 3),
                     all_trials_pipeline=None if pipe.get(k) is None else round(pipe[k], 3))
        log.info("trials", feature=feat, kept=len(kept), dropped=len(trials) - len(kept),
                 kept_by_dof={k: sum(t[1] == k for t in kept) for k in show})
        data[feat] = dict(ks=ks, p=p, y=y, test_nip=test_nip, show=show, kept=kept, r_shown=r_shown,
                          idx=np.concatenate([t[0] for t in kept]))

    # ---- row layout, shared by all panels: stack DOFs bottom-up; each row spans the target and the
    # 1st-99th percentile of every feature's prediction (always including 0), so negative-flexion
    # rows (Thumb Intrinsic) and noisy predictions don't run into the neighbouring row.
    show = data[FEATS[0]]["show"]
    sgn = {k: -1 if k in a.flip_display else 1 for k in show}
    bases, cursor = {}, 0.0
    for k in reversed(show):
        vals = []
        for d in data.values():
            j = d["ks"].index(k)
            vals += [sgn[k] * d["y"][d["idx"], j], np.percentile(sgn[k] * d["p"][d["idx"], j], [1, 99])]
        vals = np.concatenate([np.ravel(v) for v in vals])
        lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
        bases[k] = (cursor - lo, lo, hi)
        cursor = cursor - lo + hi + a.gap

    # ---- pass 2: draw
    W, H = a.size
    fig, axes = plt.subplots(1, 4, figsize=(W, H), sharey=True)
    top_in = 2.6 + (0.85 if title else 0.0)
    fig.subplots_adjust(left=2.3 / W, right=1 - 1.2 / W, bottom=1.75 / H, top=1 - top_in / H, wspace=0.42)
    if title:
        fig.suptitle(title, y=1 - 0.15 / H, va="top", fontweight="normal")
    for ax, feat in zip(axes, FEATS):
        d = data[feat]
        idx, ks = d["idx"], d["ks"]
        x = np.arange(len(idx)) * np.median(np.diff(d["test_nip"])) / 30e3
        ends = np.cumsum([len(t[0]) for t in d["kept"]])[:-1]
        dofs = [t[1] for t in d["kept"]]
        if a.trial_lines == "dof":      # only where the moved DOF changes
            ends = [e for e, k0, k1 in zip(ends, dofs[:-1], dofs[1:]) if k0 != k1]
        if a.trial_lines != "none":
            for jn in ends:
                ax.axvline(x[jn], color=JOIN_GREY, lw=GUIDE_LW, zorder=0)
        for k in show:
            j = ks.index(k)
            base, lo, hi = bases[k]
            mid = base + (lo + hi) / 2
            ax.axhline(base, color=JOIN_GREY, lw=GUIDE_LW, zorder=0)
            ax.plot(x, sgn[k] * d["y"][idx, j] + base, color=TARGET_GREY, lw=a.line_width + 1)
            ax.plot(x, sgn[k] * d["p"][idx, j] + base, color=COLS[feat], lw=a.line_width)
            ax.text(x[-1] * 1.03, mid, f"{d['r_shown'].get(k, np.nan):.2f}", va="center")
            if ax is axes[0]:
                ax.text(-x[-1] * 0.06, mid, NAMES.get(k, f"DOF {k}"), ha="right", va="center",
                        linespacing=0.95)
        shown_mean = np.mean([d["r_shown"][k] for k in show if k in d["r_shown"]])
        ax.set_title(f"{LABELS[feat]}\nRMSE {shown_mean:.2f}", color=COLS[feat], pad=18)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.set_xlim(0, x[-1])
    ax.set_ylim(min(b[0] + b[1] for b in bases.values()) - 0.05, cursor - a.gap + 0.05)
    axes[0].set_xlabel("Time (s)")

    handles = [plt.Line2D([], [], color=TARGET_GREY, lw=a.line_width + 1, label="Target"),
               plt.Line2D([], [], color="0.15", lw=a.line_width, label="Prediction")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1 - (0.85 if title else 0.0) / H), handlelength=1.6)

    tag = "_" + a.variant if a.variant else ""
    tag += "_alldirections" if a.all_directions else "_flexion"
    stem = a.out / f"fig3_predictions_{a.config}{tag}_poster"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    log.info("saved", stem=str(stem), size_in=f"{W} x {H}", font_pt=a.font_size, line_pt=a.line_width)


if __name__ == "__main__":
    main()

