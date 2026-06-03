#!/usr/bin/env python3
"""Plot decoded kinematic predictions vs ground truth for a single session.

Loads a predictions HDF5 from the decode step and produces a multi-panel
figure showing each DOF's predicted vs actual trace, with per-trial RMSE
and correlation annotated.

Usage:
    python plot_traces.py \
        --predictions /scratch/.../1/8DOF/decode/NFR_predictions.h5 \
        --output reports/figures/traces/1_NFR_8DOF.png

    # Or find by job_id/feature_set/dof_config:
    python plot_traces.py \
        --scratch-root /scratch/.../neural-feature-identification-pipeline \
        --job-id 1 --feature-set NFR --dof-config 8DOF \
        --output reports/figures/traces/1_NFR_8DOF.png
"""

import argparse
from pathlib import Path

import h5py
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
    """Load predictions and ground truth from decode output.

    Args:
        filepath: Path to {feature_set}_predictions.h5.

    Returns:
        Dict with predictions, true_kinematics, kinematic_indices arrays.
    """
    with h5py.File(filepath, "r") as f:
        return {
            "predictions": f["predictions"][:],
            "true_kinematics": f["true_kinematics"][:],
            "kinematic_indices": f["kinematic_indices"][:].ravel().astype(int),
        }


def plot_traces(
    predictions: np.ndarray,
    true_kinematics: np.ndarray,
    kinematic_indices: np.ndarray,
    title: str = "",
    output_path: Path | None = None,
    frame_rate: float = 30.0,
):
    """Plot predicted vs true kinematics for each DOF.

    Args:
        predictions: ndarray of shape (n_samples, n_dofs).
        true_kinematics: ndarray of shape (n_samples, n_dofs).
        kinematic_indices: 1-based DOF indices.
        title: Figure title.
        output_path: If provided, save figure to this path.
        frame_rate: Sampling rate for time axis (Hz).
    """
    n_samples, n_dofs = predictions.shape
    time = np.arange(n_samples) / frame_rate

    fig, axes = plt.subplots(
        n_dofs, 1,
        figsize=(14, 2.5 * n_dofs),
        sharex=True,
        squeeze=False,
    )

    for i, dof_idx in enumerate(kinematic_indices):
        ax = axes[i, 0]
        true = true_kinematics[:, i]
        pred = predictions[:, i]

        ax.plot(time, true, color="#2196F3", alpha=0.7, linewidth=1, label="True")
        ax.plot(time, pred, color="#FF5722", alpha=0.7, linewidth=1, label="Predicted")

        # Compute metrics
        residuals = true - pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        if np.std(true) > 1e-8 and np.std(pred) > 1e-8:
            r = np.corrcoef(true, pred)[0, 1]
        else:
            r = 0.0

        dof_name = DOF_NAMES.get(dof_idx, f"DOF {dof_idx}")
        ax.set_ylabel(dof_name, fontsize=10)
        ax.text(
            0.98, 0.95,
            f"RMSE={rmse:.3f}  r={r:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

        ax.set_xlim(time[0], time[-1])

    axes[-1, 0].set_xlabel("Time (s)", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    fig.tight_layout()

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

    # Direct path
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Direct path to predictions HDF5.")

    # Or lookup by components
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument("--feature-set", type=str, default=None)
    parser.add_argument("--dof-config", type=str, default=None)

    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path. If not set, displays interactively.")
    args = parser.parse_args()

    # Resolve predictions path
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

    # Skip empty predictions (null results)
    if data["predictions"].size == 0:
        print(f"Null result (no features selected). Nothing to plot.")
        return

    # Build title from path
    parts = pred_path.parts
    decode_idx = parts.index("decode")
    title = f"Job {parts[decode_idx-2]} | {parts[decode_idx-1]} | {pred_path.stem.replace('_predictions', '')}"

    plot_traces(
        predictions=data["predictions"],
        true_kinematics=data["true_kinematics"],
        kinematic_indices=data["kinematic_indices"],
        title=title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

