#!/usr/bin/env python3
"""Decode kinematic predictions from neural features using a Modified Kalman Filter.

Pipeline rule: decode
Inputs:  features HDF5, kinematics HDF5, events HDF5, selected channels JSON
Outputs: metrics JSON, predictions HDF5
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from modified_kalman_filter import ModifiedKalmanFilter, compute_decode_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_selection_result(filepath: Path) -> dict:
    """Load the output of select_features.py.

    Args:
        filepath: Path to selected.json.

    Returns:
        Dict with selected channels, split info, and alignment metadata.
    """
    with open(filepath) as f:
        return json.load(f)


def load_events(filepath: Path) -> tuple[NDArray, NDArray, NDArray]:
    """Load trial timestamps and movement matrices from events HDF5.

    Args:
        filepath: Path to events.h5.

    Returns:
        Tuple of (trial_starts, trial_stops, movement_matrices).
    """
    with h5py.File(filepath, "r") as f:
        trial_starts = f["trial_start_idxs"][:].ravel()
        trial_stops = f["trial_stop_idxs"][:].ravel()
        movement_matrices = f["movement_matrices"][:]

    if movement_matrices.shape[-1] != 4:
        movement_matrices = np.transpose(movement_matrices, (2, 1, 0))

    return trial_starts, trial_stops, movement_matrices


def load_features(filepath: Path) -> NDArray:
    """Load features from extract_features output.

    Args:
        filepath: Path to {feature_set}.h5.

    Returns:
        ndarray as stored (caller resolves orientation).
    """
    with h5py.File(filepath, "r") as f:
        return f["features"][:]


def load_kinematics(filepath: Path) -> tuple[NDArray, NDArray]:
    """Load kinematics and NIP timestamps.

    Args:
        filepath: Path to kinematics.h5.

    Returns:
        Tuple of (kinematics, nip_time) with kinematics as
        (n_samples, n_dofs).
    """
    with h5py.File(filepath, "r") as f:
        kinematics = f["kinematics"][:]
        nip_time = f["nip_time"][:]
    nip_time = nip_time.ravel()

    if kinematics.ndim == 2 and kinematics.shape[0] < kinematics.shape[1]:
        kinematics = kinematics.T

    return kinematics, nip_time


def resolve_feature_orientation(
    features: NDArray,
    n_kin_samples: int,
    max_mismatch: int = 5,
) -> tuple[NDArray, int]:
    """Resolve feature array orientation and sample mismatch.

    Args:
        features: Raw features array from HDF5.
        n_kin_samples: Number of kinematic samples to match against.
        max_mismatch: Maximum allowed sample difference (NFR drops 1-2).

    Returns:
        Tuple of (features_oriented, n_trim) where n_trim is the number
        of leading kinematic samples to discard for alignment.

    Raises:
        ValueError: If shape cannot be resolved.
    """
    n_trim = 0

    if features.shape[0] == n_kin_samples:
        pass
    elif features.shape[1] == n_kin_samples:
        features = features.T
    elif abs(features.shape[0] - n_kin_samples) <= max_mismatch:
        n_trim = n_kin_samples - features.shape[0]
    elif abs(features.shape[1] - n_kin_samples) <= max_mismatch:
        features = features.T
        n_trim = n_kin_samples - features.shape[0]
    else:
        raise ValueError(
            f"Features {features.shape} vs kinematics {n_kin_samples} mismatch"
        )

    return features, n_trim


# =============================================================================
# Trial Boundary Mapping
# =============================================================================

def build_test_trial_boundaries(
    test_mask: NDArray,
    nip_time: NDArray,
    trial_starts: NDArray,
    trial_stops: NDArray,
) -> list[tuple[int, int]]:
    """Map trial timestamps to index ranges within the contiguous test array.

    When test_mask extracts test samples, the resulting array is contiguous
    but the sample indices no longer correspond to the full recording.
    This function finds which test-array indices belong to each trial.

    Args:
        test_mask: Boolean mask of shape (n_samples,) over the full recording.
        nip_time: NIP timestamps of shape (n_samples,) for the full recording.
        trial_starts: NIP timestamps for trial onsets.
        trial_stops: NIP timestamps for trial completions.

    Returns:
        List of (start_idx, end_idx) tuples, 0-based and end-exclusive,
        indexing into the test-masked array. Only trials with at least
        one test sample are included.
    """
    test_nip = nip_time[test_mask]

    boundaries = []
    for t_start_nip, t_stop_nip in zip(trial_starts, trial_stops):
        in_trial = (test_nip >= t_start_nip) & (test_nip <= t_stop_nip)
        trial_positions = np.where(in_trial)[0]

        if len(trial_positions) == 0:
            continue

        boundaries.append((int(trial_positions[0]), int(trial_positions[-1]) + 1))

    return boundaries


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode kinematics from neural features via Modified Kalman Filter."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True,
                        help="Path to selected.json from select_features.")
    parser.add_argument("--burn-in", type=int, default=36,
                        help="Burn-in samples to exclude from metrics (~1.2s at 30Hz).")
    parser.add_argument("--apply-gain-threshold", action="store_true",
                        help="Apply MKF gain-threshold post-processing.")
    parser.add_argument("--apply-latching", action="store_true",
                        help="Apply latching (smoothing) filter.")
    parser.add_argument("--latch-constant", type=float, default=1.0,
                        help="Latching filter constant.")
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the decode pipeline step."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Load selection results ---
    logger.info("Loading selection results from %s", args.selected)
    selection = load_selection_result(args.selected)

    kinematic_indices = selection["kinematic_indices"]
    selected_joint = selection["selected_channels_joint"]
    train_mask = np.array(selection["split"]["train_mask"])
    test_mask = np.array(selection["split"]["test_mask"])
    alignment_lag = selection["alignment"]["lag_samples"]

    if len(selected_joint) == 0:
        logger.warning(
            "No features were selected (null selection result). "
            "Skipping decode — writing null metrics."
        )
        metrics = {
            "per_trial": [],
            "aggregate_rmse_per_dof": {},
            "aggregate_correlation_per_dof": {},
            "summary": {
                "rmse_mean": float("nan"),
                "correlation_mean": float("nan"),
                "n_trials": 0,
                "n_burn_in_excluded": args.burn_in,
                "null_result": True,
                "reason": "no_features_passed_preselection",
            },
            "feature_set": args.features.stem,
            "dof_config": args.output_metrics.parent.parent.name,
        }
        args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        # Write empty predictions HDF5
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output_predictions, "w") as f:
            f.create_dataset("predictions", data=np.array([]))
            f.create_dataset("true_kinematics", data=np.array([]))
            f.create_dataset("kinematic_indices", data=np.array(kinematic_indices))
        logger.info("Null metrics written to %s", args.output_metrics)
        return

    # --- Load data ---
    logger.info("Loading features from %s", args.features)
    features = load_features(args.features)

    logger.info("Loading kinematics from %s", args.kinematics)
    kinematics, nip_time = load_kinematics(args.kinematics)

    logger.info("Loading events from %s", args.events)
    trial_starts, trial_stops, _ = load_events(args.events)

    # --- Resolve feature orientation ---
    features, n_trim = resolve_feature_orientation(features, kinematics.shape[0])
    if n_trim > 0:
        logger.warning("Trimming %d leading kinematic samples", n_trim)
        kinematics = kinematics[n_trim:, :]
        nip_time = nip_time[n_trim:]

    logger.info("Data: features %s, kinematics %s", features.shape, kinematics.shape)

    # --- Apply alignment (shift kinematics and event timestamps) ---
    if alignment_lag != 0:
        kinematics = np.roll(kinematics, -alignment_lag, axis=0)
        if alignment_lag > 0:
            kinematics[-alignment_lag:, :] = 0.0
        else:
            kinematics[:-alignment_lag, :] = 0.0

        # Shift event timestamps to match
        nip_dt = np.median(np.diff(nip_time))
        nip_shift = alignment_lag * nip_dt
        trial_starts = trial_starts - nip_shift
        trial_stops = trial_stops - nip_shift

        # Snap to nearest existing NIP timestamp
        start_idxs = np.clip(np.searchsorted(nip_time, trial_starts), 0, len(nip_time) - 1)
        stop_idxs = np.clip(np.searchsorted(nip_time, trial_stops), 0, len(nip_time) - 1)
        trial_starts = nip_time[start_idxs]
        trial_stops = nip_time[stop_idxs]

        logger.info("Applied alignment lag=%d to kinematics and events", alignment_lag)

    # --- Subset to active DOFs and selected features ---
    active_kin = kinematics[:, [idx - 1 for idx in kinematic_indices]]
    selected_feat = features[:, selected_joint]

    logger.info(
        "Active DOFs: %s (%d), selected features: %d",
        kinematic_indices, active_kin.shape[1], selected_feat.shape[1],
    )

    # --- Fit ---
    train_kin = active_kin[train_mask, :]
    train_feat = selected_feat[train_mask, :]
    logger.info("Training on %d samples", train_kin.shape[0])

    decoder = ModifiedKalmanFilter(
        apply_gain_threshold=args.apply_gain_threshold,
        apply_latching=args.apply_latching,
        latch_constant=args.latch_constant,
    )
    decoder.fit(train_feat, train_kin)

    # --- Predict on test data ---
    test_feat = selected_feat[test_mask, :]
    test_kin = active_kin[test_mask, :]
    logger.info("Predicting on %d test samples", test_feat.shape[0])

    predictions = decoder.predict(test_feat)

    # --- Build trial boundaries within the test array ---
    trial_boundaries = build_test_trial_boundaries(
        test_mask, nip_time, trial_starts, trial_stops
    )
    logger.info("Found %d test trials", len(trial_boundaries))

    # --- Compute per-trial metrics ---
    metrics = compute_decode_metrics(
        test_kin, predictions,
        dof_indices=kinematic_indices,
        trial_boundaries=trial_boundaries,
        burn_in=args.burn_in,
    )

    # Add identifiers
    feature_set = args.features.stem
    dof_config = args.output_metrics.parent.parent.name
    metrics["feature_set"] = feature_set
    metrics["dof_config"] = dof_config

    # --- Write outputs ---
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", args.output_metrics)

    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_predictions, "w") as f:
        f.create_dataset("predictions", data=predictions)
        f.create_dataset("true_kinematics", data=test_kin)
        f.create_dataset("kinematic_indices", data=np.array(kinematic_indices))
    logger.info("Predictions written to %s", args.output_predictions)

    logger.info(
        "Results: mean RMSE=%.4f (±%.4f), mean r=%.4f (±%.4f) across %d trials",
        metrics["summary"]["rmse_mean"], metrics["summary"]["rmse_std"],
        metrics["summary"]["correlation_mean"], metrics["summary"]["correlation_std"],
        metrics["summary"]["n_trials"],
    )


if __name__ == "__main__":
    main()

