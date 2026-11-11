#!/usr/bin/env python3
"""Plot decoded kinematic predictions vs ground truth for a single session.

Loads a predictions HDF5 and optionally events HDF5 to annotate trial
boundaries and intended/unintended DOFs per trial.

Usage:
    python plot_traces.py \
        --scratch-root /scratch/.../neural-feature-identification-pipeline \
        --job-id 1 --feature-set NFR --dof-config 8DOF \
        --output reports/figures/traces/1_NFR_8DOF.png
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DOF_NAMES = {
    1: "Thumb Flex",
    2: "Index Flex",
    3: "Middle Flex",
    4: "Ring Flex",
    5: "Pinky Flex",
    6: "Thumb Rot",
    10: "Wrist Flex",
    12: "Wrist Pro",
}


def load_predictions(filepath: Path) -> dict:
    with h5py.File(filepath, "r") as f:
        return {
            "predictions": f["predictions"][:],
            "true_kinematics": f["true_kinematics"][:],
            "kinematic_indices": f["kinematic_indices"][:].ravel().astype(int),
            "test_nip_time": f["test_nip_time"][:].ravel() if "test_nip_time" in f else None,
            "alignment_lag": int(f.attrs.get("alignment_lag", 0)),
            "null_result": bool(f.attrs.get("null_result", False)),
        }


def load_events(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(filepath, "r") as f:
        trial_starts = f["trial_start_idxs"][:].ravel()
        trial_stops = f["trial_stop_idxs"][:].ravel()
        movement_matrices = f["movement_matrices"][:]
    if movement_matrices.shape[-1] != 4:
        movement_matrices = np.transpose(movement_matrices, (2, 1, 0))
    return trial_starts, trial_stops, movement_matrices


def plot_traces(
    predictions: np.ndarray,
    true_kinematics: np.ndarray,
    kinematic_indices: np.ndarray,
    test_nip_time: np.ndarray | None = None,
    trial_starts: np.ndarray | None = None,
    trial_stops: np.ndarray | None = None,
    movement_matrices: np.ndarray | None = None,
    alignment_lag: int = 0,
    title: str = "",
    output_path: Path | None = None,
    frame_rate: float = 30.0,
):
    n_samples, n_dofs = predictions.shape
    time = np.arange(n_samples) / frame_rate

    # Extra width on the right for annotations
    fig, axes = plt.subplots(
        n_dofs, 1,
        figsize=(16, 2.8 * n_dofs + 1.0),
        sharex=True,
        squeeze=False,
    )

    # Build trial boundaries in test-array coordinates
    trial_info = []
    if (test_nip_time is not None and trial_starts is not None
            and trial_stops is not None and movement_matrices is not None):
        nip_dt = np.median(np.diff(test_nip_time)) if len(test_nip_time) > 1 else 1.0
        shifted_starts = trial_starts - alignment_lag * nip_dt
        shifted_stops = trial_stops - alignment_lag * nip_dt

        for t_idx, (ts, te) in enumerate(zip(shifted_starts, shifted_stops)):
            in_trial = (test_nip_time >= ts) & (test_nip_time <= te)
            positions = np.where(in_trial)[0]
            if len(positions) > 0:
                trial_info.append({
                    "start": positions[0],
                    "end": positions[-1] + 1,
                    "original_idx": t_idx,
                    "mvmt": movement_matrices[t_idx],
                })

    for i, dof_idx in enumerate(kinematic_indices):
        ax = axes[i, 0]
        true = true_kinematics[:, i]
        pred = predictions[:, i]

        # Shade intended vs unintended trial regions
        for tinfo in trial_info:
            t_start_s = tinfo["start"] / frame_rate
            t_end_s = tinfo["end"] / frame_rate
            is_intended = tinfo["mvmt"][dof_idx - 1, 0] != 0
            color = "#E3F2FD" if is_intended else "#FFF3E0"
            ax.axvspan(t_start_s, t_end_s, alpha=0.3, color=color, linewidth=0)

        ax.plot(time, true, color="#2196F3", alpha=0.8, linewidth=1, label="True")
        ax.plot(time, pred, color="#FF5722", alpha=0.8, linewidth=1, label="Predicted")

        # Per-trial RMSE annotations at the top, outside the data range
        for tinfo in trial_info:
            ts, te = tinfo["start"], tinfo["end"]
            is_intended = tinfo["mvmt"][dof_idx - 1, 0] != 0
            seg_true = true[ts:te]
            seg_pred = pred[ts:te]

            if len(seg_true) < 2:
                continue
            tau=0.10
            seg_residuals = seg_true - seg_pred
            penalties = np.maximum(0, np.abs(seg_residuals)-tau)
            rmse = np.sqrt(np.mean(penalties ** 2))
            mid_t = (ts + te) / 2 / frame_rate
            label = "I" if is_intended else "U"

            ax.annotate(
                f"{label}:{rmse:.2f}",
                xy=(mid_t, 1.0), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=5.5, alpha=0.7,
                annotation_clip=False,
            )

        # Global metrics outside the axes on the right
        residuals = true - pred
        rmse = np.sqrt(np.mean(residuals ** 2))

        tau = 0.10
        penalties = np.maximum(0, np.abs(residuals)-tau)
        d_rmse = np.sqrt(np.mean(penalties ** 2))

        if np.std(true) > 1e-8 and np.std(pred) > 1e-8:
            r = np.corrcoef(true, pred)[0, 1]
        else:
            r = 0.0

        dof_name = DOF_NAMES.get(dof_idx, f"DOF {dof_idx}")
        ax.set_ylabel(dof_name, fontsize=10)

        # Annotation outside axes, right side
        ax.annotate(
            f"RMSE={rmse:.3f}\nd_RMSE={d_rmse:.3f}\nr={r:.3f}",
            xy=(1.01, 0.5), xycoords="axes fraction",
            ha="left", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#CCCCCC"),
            annotation_clip=False,
        )

        ax.set_xlim(time[0], time[-1])

    # Legend outside, above the top axes
    if trial_info:
        intended_patch = mpatches.Patch(color="#E3F2FD", alpha=0.5, label="Intended")
        unintended_patch = mpatches.Patch(color="#FFF3E0", alpha=0.5, label="Unintended")
        axes[0, 0].legend(
            handles=[
                plt.Line2D([0], [0], color="#2196F3", label="True"),
                plt.Line2D([0], [0], color="#FF5722", label="Predicted"),
                intended_patch,
                unintended_patch,
            ],
            loc="lower left", bbox_to_anchor=(0, 1.15),
            fontsize=7, ncol=4, frameon=True,
        )
    else:
        axes[0, 0].legend(
            loc="lower left", bbox_to_anchor=(0, 1.05), fontsize=8,
        )

    axes[-1, 0].set_xlabel("Time (s)", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    fig.subplots_adjust(right=0.88, hspace=0.35)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot decoded kinematic traces vs ground truth."
    )
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument("--feature-set", type=str, default=None)
    parser.add_argument("--dof-config", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.predictions:
        pred_path = args.predictions
    elif args.scratch_root and args.job_id and args.feature_set and args.dof_config:
        pred_path = (
            args.scratch_root / str(args.job_id) / args.dof_config
            / "decode" / f"{args.feature_set}_predictions.h5"
        )
    else:
        parser.error(
            "Provide either --predictions or all of "
            "--scratch-root, --job-id, --feature-set, --dof-config"
        )

    if not pred_path.exists():
        print(f"Error: {pred_path} not found")
        return

    data = load_predictions(pred_path)

    if data["null_result"] or data["predictions"].size == 0:
        print("Null result (no features selected). Nothing to plot.")
        return

    events_path = pred_path.parent.parent.parent / "events.h5"
    trial_starts, trial_stops, movement_matrices = None, None, None
    if events_path.exists():
        trial_starts, trial_stops, movement_matrices = load_events(events_path)

    parts = pred_path.parts
    decode_idx = parts.index("decode")
    title = (
        f"Job {parts[decode_idx-2]} | {parts[decode_idx-1]} | "
        f"{pred_path.stem.replace('_predictions', '')}"
    )

    plot_traces(
        predictions=data["predictions"],
        true_kinematics=data["true_kinematics"],
        kinematic_indices=data["kinematic_indices"],
        test_nip_time=data["test_nip_time"],
        trial_starts=trial_starts,
        trial_stops=trial_stops,
        movement_matrices=movement_matrices,
        alignment_lag=data["alignment_lag"],
        title=title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

