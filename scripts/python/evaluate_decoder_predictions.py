"""Evaluate decoder predictions against ground truth kinematics.

Pipeline rule: evaluate_decoder_predictions.smk
Inputs: predictions HDF5 (from decode_features), events HDF5 (from preprocess)
Outputs: metrics JSON

Computes per-trial, per-DOF RMSE and Pearson correlation with intended vs unintended
classification based on movement matrices.
"""
import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EPS = np.finfo(np.float64).eps

# --- Data Loading ---
def load_predictions(filepath: Path) -> dict:
    """Load decoder outputs from predictions.h5

    Returns:
        Dict with predictions, true_kinematics, kinematic_indices,
        test_nip_time, alignment_lag, and null_result_flag
    """
    with h5py.File(filepath, "r") as f:
        result = {
            "predictions": f["predictions"][:],
            "true_kinematics": f["true_kinematics"][:],
            "kinematic_indices": f["kinematic_indices"][:].ravel().astype(int),
            "test_nip_time": f["test_nip_time"][:].ravel(),
            "alignment_lag": int(f.attrs.get("alignment_lag", 0)),
            "null_result": bool(f.attrs.get("null_result", False)),
        }
    return result


def load_events(filepath: Path) -> tuple[NDArray, NDArray, NDArray]:
    """Load trial timestamps and movement matrices from events.h5

    Returns:
        Tuple of (trial_starts, trial_stops, movement_matrices)
        movement_matrices shape: (n_trials, 12, 4)
    """
    with h5py.File(filepath, "r") as f:
        trial_starts = f["trial_start_idxs"][:].ravel()
        trial_stops = f["trial_stop_idxs"][:].ravel()
        movement_matrices = f["movement_matrices"][:]
    
    if movement_matrices.shape[-1] != 4:
        movement_matrices = np.transpose(movement_matrices, (2,1,0))
    return trial_starts, trial_stops, movement_matrices

# --- Trial Boundary Mapping ---
def build_test_trial_boundaries(
    test_nip_time: NDArray,
    trial_starts: NDArray,
    trial_stops: NDArray,
) -> list[tuple[int, int, int]]:
    """
    Map trial timestamps to index ranges within the test_array.

    Args:
        test_nip_time: NIP timestamps for test samples
        trial_starts: NIP timestamps for trial onsets
        trial_stops: NIP timestamps for trial completions
    
    Returns:
        List of (start_idx, end_idx, original_trial_idx) tuples.
        Indices are 0-based into the test array, end-exclusive.
        Only trials with at least one test sample are included.
    """

    boundaries = []

    for trial_idx, (t_start, t_stop) in enumerate(zip(trial_starts, trial_stops)):
        in_trial = (test_nip_time >= t_start) & (test_nip_time <= t_stop)
        positions = np.where(in_trial)[0]

        if len(positions) == 0:
            continue

        boundaries.append((
            int(positions[0]),
            int(positions[-1]) + 1,
            trial_idx,
        ))
    return boundaries

def shift_trial_timestamps(
    trial_starts: NDArray,
    trial_stops: NDArray,
    test_nip_time: NDArray,
    alignment_lag: int,
) -> tuple[NDArray, NDArray]:
    """Apply alignment shift to trial timestamps.
    
    Args:
        trial_starts: Original trial start NIP timestamps.
        trial_stops: Original trial stop NIP timestamps.
        test_nip_time: NIP timestamps for the test samples.
        alignment_lag: Lag in samples applied during decoding.

    Returns:
        Tuple of (shifted_starts, shifted_stops)
    """
    if alignment_lag == 0:
        return trial_starts.copy(), trial_stops.copy()
    
    # Shift event timestamps to match
    nip_dt = np.median(np.diff(test_nip_time))
    nip_shift = alignment_lag * nip_dt
    shifted_starts = trial_starts - nip_shift
    shifted_stops = trial_stops - nip_shift

    # Snap to nearest existing NIP timestamp
    start_idxs = np.clip(np.searchsorted(test_nip_time, shifted_starts), 0, len(test_nip_time) - 1)
    stop_idxs = np.clip(np.searchsorted(test_nip_time, shifted_stops), 0, len(test_nip_time) - 1)
    trial_starts = test_nip_time[start_idxs]
    trial_stops = test_nip_time[stop_idxs]

    return trial_starts, trial_stops

# --- Metrics Computation ---
def compute_trial_dof_metrics(
    y_true: NDArray,
    y_pred: NDArray,
) -> tuple[float, float]:
    """Compute RMSE and Pearson Correlation for a single trial x DOF.

    Args:
        y_true: True kinematics, shape (n_samples,)
        y_pred: Predicted kinematics, shape (n_samples,)

    Returns:
        Tuple of (rmse, correlation)
    """
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    std_thresh = np.sqrt(_EPS)
    if np.std(y_true) < std_thresh or np.std(y_pred) < std_thresh:
        correlation = 0.0
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
        correlation = r if np.isfinite(r) else 0.0

    return rmse, correlation

def is_dof_intended(
    movement_matrix: NDArray,
    dof_idx: int,
) -> bool:
    """Check if a DOF was prompted (intended) during a trial
    
    Args:
        movement_matrix: Single trial's movement matrix, shape(12, 4).
            Column 0 is amplitude; non-zero means the DOF was prompted.
        dof_idx: 1-based DOF idx

    Returns:
        True if the DOF had non-zero amplitude in this trial
    """
    return bool(movement_matrix[dof_idx - 1, 0] != 0)

def evaluate(
    predictions: NDArray,
    true_kinematics: NDArray,
    kinematic_indices: NDArray,
    trial_boundaries: list[tuple[int, int, int]],
    movement_matrices: NDArray,
    burn_in: int = 10,
) -> dict:
    """Compute per-trial, per-DOF metrics with intended/unintended classification.

    Burn-in is applied only to the first trial (the filter needs initial samples for state convergence).

    Args:
        predictions: shape(n_test_samples, n_dofs)
        true_kinematics: shape(n_test_samples, n_dofs)
        kinematic_indices: 1-based DOF indices, shape (n_dofs, )
        trial_boundaries: List of (start, end, original_trial_idx)
        movement_matrices: shape(n_total_trials, 12, 4)
        burn_in: int, samples to exclude from first trial

    Returns:
        Dict with per_trial, aggregate, and summary sections.
    """
    per_trial = []

    for boundary_idx, (t_start, t_end, orig_trial_idx) in enumerate(trial_boundaries):
        if boundary_idx == 0:
            t_start = min(t_start + burn_in, t_end) # TODO: @FRM, remove the need for burn-in by using steady state MKF
        if t_start >= t_end:
            logger.warning(
                "Trial %d has no samples after burn-in (start=%d, end=%d). Skipping.",
                orig_trial_idx, t_start, t_end,
            )
            continue

        trial_true = true_kinematics[t_start:t_end, :]
        trial_pred = predictions[t_start:t_end, :]

        mvmnt_mat = movement_matrices[orig_trial_idx]

        # TODO: @FRM, refactor this section to make metrics extensible
        dof_metrics = {}
        for i, dof_idx in enumerate(kinematic_indices):
            rmse, corr = compute_trial_dof_metrics(
                trial_true[:, i], trial_pred[:, i]
            )
            intended = is_dof_intended(mvmnt_mat, int(dof_idx))

            dof_metrics[str(int(dof_idx))] = {
                "rmse": rmse,
                "correlation": corr,
                "intended": intended,
            }
        
        # separate intended vs unintended means
        rmses_intended = [v["rmse"] for v in dof_metrics.values() if v["intended"]]
        rmses_unintended = [v["rmse"] for v in dof_metrics.values() if not v["intended"]]
        corrs_intended = [v["correlation"] for v in dof_metrics.values() if v["intended"]]
        corrs_unintended = [v["correlation"] for v in dof_metrics.values() if not v["intended"]]
        rmses_all = [v["rmse"] for v in dof_metrics.values()]
        corrs_all = [v["correlation"] for v in dof_metrics.values()]

        per_trial.append({
            "trial_index": boundary_idx,
            "original_trial_idx": orig_trial_idx,
            "n_samples": t_end - t_start,
            "dof_metrics": dof_metrics,
            "rmse_mean_all": float(np.mean(rmses_all)) if rmses_all else None,
            "correlation_mean_all": float(np.mean(corrs_all)) if corrs_all else None,
            "rmse_mean_intended": float(np.mean(rmses_intended)) if rmses_intended else None,
            "correlation_mean_intended": float(np.mean(corrs_intended)) if corrs_intended else None,
            "rmse_mean_unintended": float(np.mean(rmses_unintended)) if rmses_unintended else None,
            "correlation_mean_unintended": float(np.mean(corrs_unintended)) if corrs_unintended else None, 
        })

    # --- Aggregate across trials ---
    aggregate_per_dof = {}
    for dof_str in [str(int(d)) for d in kinematic_indices]:
        intended_vals = [
            t["dof_metrics"][dof_str]
            for t in per_trial if dof_str in t["dof_metrics"]
            and t["dof_metrics"][dof_str]["intended"]
        ]
        unintended_vals = [
            t["dof_metrics"][dof_str]
            for t in per_trial if dof_str in t["dof_metrics"]
            and not t["dof_metrics"][dof_str]["intended"]
        ]
        all_vals = [
            t["dof_metrics"][dof_str]
            for t in per_trial if dof_str in t["dof_metrics"]
        ]

        aggregate_per_dof[dof_str] = {
            "intended": _aggregate_metric_list(intended_vals),
            "unintended": _aggregate_metric_list(unintended_vals),
            "all": _aggregate_metric_list(all_vals),
        }
    # --- Summary ---
    all_intended_rmses = [
        t["rmse_mean_intended"] for t in per_trial
        if t["rmse_mean_intended"] is not None
    ]
    all_unintended_rmses = [
        t["rmse_mean_unintended"] for t in per_trial
        if t["rmse_mean_unintended"] is not None
    ]
    all_rmses = [
        t["rmse_mean_all"] for t in per_trial
        if t["rmse_mean_all"] is not None
    ]
    all_intended_corrs = [
        t["correlation_mean_intended"] for t in per_trial
        if t["correlation_mean_intended"] is not None
    ]
    all_unintended_corrs = [
        t["correlation_mean_unintended"] for t in per_trial
        if t["correlation_mean_unintended"] is not None
    ]
    all_corrs = [
        t["correlation_mean_all"] for t in per_trial
        if t["correlation_mean_all"] is not None
    ]

    summary = {
        "n_trials": len(per_trial),
        "n_burn_in_excluded": burn_in,
        "intended":{
            "rmse_mean": float(np.mean(all_intended_rmses)) if all_intended_rmses else None,
            "rmse_std": float(np.std(all_intended_rmses)) if all_intended_rmses else None,
            "correlation_mean": float(np.mean(all_intended_corrs)) if all_intended_corrs else None,
            "correlation_std": float(np.std(all_intended_corrs)) if all_intended_corrs else None,
        },
        "unintended":{
            "rmse_mean": float(np.mean(all_unintended_rmses)) if all_unintended_rmses else None,
            "rmse_std": float(np.std(all_unintended_rmses)) if all_unintended_rmses else None,
            "correlation_mean": float(np.mean(all_unintended_corrs)) if all_unintended_corrs else None,
            "correlation_std": float(np.std(all_unintended_corrs)) if all_unintended_corrs else None,
        },
        "all":{
            "rmse_mean": float(np.mean(all_rmses)) if all_rmses else None,
            "rmse_std": float(np.std(all_rmses)) if all_rmses else None,
            "correlation_mean": float(np.mean(all_corrs)) if all_corrs else None,
            "correlation_std": float(np.std(all_corrs)) if all_corrs else None,
        },
    }

    return {
        "per_trial":  per_trial,
        "aggregate_per_dof": aggregate_per_dof,
        "summary": summary,
    }

def _aggregate_metric_list(vals: list[dict]) -> dict:
    """Aggregate a list of {rmse, correlation, intended} dicts."""
    if not vals:
        return{"n": 0, "rmse_mean": None, "rmse_std": None,
               "correlation_mean": None, "correlation_std": None}
    rmses = [v["rmse"] for v in vals]
    corrs = [v["correlation"] for v in vals]
    return{
        "n": len(vals),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "correlation_mean": float(np.mean(corrs)),
        "correlation_std": float(np.std(corrs)),
    }

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate decoder predictions with intended/unintended classification."
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions HDF5 from decode_features.py")
    parser.add_argument("--events", type=Path, required=True, help="Path to events HDF5 from preprocess_shared_data.m")
    parser.add_argument("--burn-in", type=int, default=10, help="Burn-in samples to exclude from first trial until MKF settles")
    parser.add_argument("--output-metrics", type=Path, required=True)
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args= parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Load predictions ---
    logger.info("Loading predictions from %s", args.predictions)
    pred_data = load_predictions(args.predictions)

    # --- Handle null results ---
    if pred_data["null_result"] or pred_data["predictions"].size == 0:
        logger.warning("Null prediction result. Writing null metrics")
        null_metrics = {
            "per_trial": [],
            "aggregate_per_dof": {},
            "summary":{
                "n_trials": 0,
                "n_burn_in_excluded": args.burn_in,
                "null_result": True,
                "null_reason": "no_predictions",
                "intended": {"rmse_mean": None, "rmse_std": None, "correlation_mean": None, "correlation_std": None},
                "unintended": {"rmse_mean": None, "rmse_std": None, "correlation_mean": None, "correlation_std": None},
                "all": {"rmse_mean": None, "rmse_std": None, "correlation_mean": None, "correlation_std": None},
            }
        }
        args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_metrics, "w") as f:
            json.dump(null_metrics, f, indent=2)
        logger.info("Null metrics written to %s", args.output_metrics)
        return
    # --- Load events ---
    logger.info("Loading events from %s", args.events)
    trial_starts, trial_stops, movement_matrices = load_events(args.events)

    # --- Align trial timestamps ---
    alignment_lag = pred_data["alignment_lag"]
    test_nip_time = pred_data["test_nip_time"]

    trial_starts, trial_stops = shift_trial_timestamps(
        trial_starts, trial_stops, test_nip_time, alignment_lag
    )

    # --- Map trials to test array indices ---
    trial_boundaries = build_test_trial_boundaries(
        test_nip_time, trial_starts, trial_stops
    )
    logger.info("Found %d test trials", len(trial_boundaries))

    # --- Evaluate predictions ---
    performance_metrics = evaluate(
        predictions=pred_data["predictions"],
        true_kinematics=pred_data["true_kinematics"],
        kinematic_indices=pred_data["kinematic_indices"],
        trial_boundaries=trial_boundaries,
        movement_matrices=movement_matrices,
        burn_in=args.burn_in,
    )

    # --- Add identifiers ---
    feature_set = args.predictions.stem.replace("_predictions", "")
    dof_config = args.predictions.parent.parent.name
    performance_metrics["feature_set"] = feature_set
    performance_metrics["dof_config"] = dof_config
    performance_metrics["kinematic_indices"] = pred_data["kinematic_indices"].tolist()

    # --- Write ---
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metrics, "w") as f:
        json.dump(performance_metrics, f, indent=2)

    intended = performance_metrics["summary"]["intended"]
    unintended = performance_metrics["summary"]["unintended"]
    logger.info(
        "Results: intended RMSE=%.4f (+/-%.4f), r=%.4f (+/-%.4f) | "
        "unintended RMSE=%.4f (+/-%.4f) unintended r=%.4f (+/-%.4f) " 
        "across %d trials",
        intended.get("rmse_mean") or 0, intended.get("rmse_std") or 0,
        intended.get("correlation_mean") or 0, intended.get("correlation_std") or 0,
        unintended.get("rmse_mean") or 0, unintended.get("rmse_std") or 0,
        unintended.get("correlation_mean") or 0, unintended.get("correlation_std") or 0,   
        performance_metrics["summary"]["n_trials"],
    )
    logger.info("Metrics written to %s", args.output_metrics)

if __name__ == "__main__":
    main()

