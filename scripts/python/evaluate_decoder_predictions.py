"""Evaluate decoder predictions against ground truth kinematics.

Pipeline rule: evaluate_decoder_predictions.smk
Inputs: predictions HDF5 (from decode_features), events HDF5 (from preprocess)
Outputs: metrics JSON

Architecture
---
Each metric is a function registered via the `@register_metric` decorator.
The evaluation loop computes every metric in the registry.
Adding a new metric requires only writing the function and decorating it. No further changes to the code are needed.

Metric function signature
---
    def some_metric(y_true: NDArray, y_pred: NDArray, dt: float) -> float
        y_true: 1-D array (single DOF, single Trial)
        y_pred: 1-D array (single DOF, single Trial)
        dt: sample period in seconds
The function may ignore arguments it doesn't need e.g., smoothness metrics ignore y_true.
The function returns `float('nan')` for trials where the metric is undefined e.g., dimensionless jerk for zero-amplitude DOFs.

Computes per-trial, per-DOF metrics with intended vs unintended
classification based on movement matrices.
"""

from __future__ import annotations

import argparse
import json
import logging
import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EPS = np.finfo(np.float64).eps

# Deadband / target-zone radius shared by d_rmse, ptit, and maxh
# Matches the target radius used in the real-time task.
# TODO(@FRM): Figure out how to make this a parameter that's read from the data acquisition files, instead of hardcoded.
TARGET_RADIUS: float = 0.10
# ===
# Metric Registry
# ===
@dataclass(frozen=True, slots=True) # frozen: soft-immutability, slots:"memory pre-allocation"
class MetricSpec:
    """Descriptor for one registered metric"""
    name: str
    fn: Callable[[NDArray, NDArray, float], float]
    higher_is_better: bool = False

# Insertion-ordered dict so JSON output is deterministic.
_METRIC_REGISTRY: dict[str, MetricSpec] = {}

def register_metric(name: str, *, higher_is_better: bool = False):
    """Decorator that adds a metric function to the global registry."""
    def _decorator(fn: Callable[[NDArray, NDArray, float], float]):
        if name in _METRIC_REGISTRY:
            raise ValueError(f"Duplicate metric name: {name!r}") # !r: print representation instead of string
        _METRIC_REGISTRY[name] = MetricSpec(
            name=name, fn=fn, higher_is_better=higher_is_better
        )
        return fn
    return _decorator

def get_registry() -> dict[str, MetricSpec]:
    """Return a *copy* of the registry (for testing / introspection)"""
    return dict(_METRIC_REGISTRY)

# ===
# Metric Implementation
# ===

# --- accuracy
@register_metric("rmse")
def _rmse(y_true: NDArray, y_pred:NDArray, dt:float) -> float:
    """Root Mean Squared Error"""
    y_error = y_true - y_pred
    return float(np.sqrt(np.mean(y_error**2)))

@register_metric("d_rmse")
def _d_rmse(y_true: NDArray, y_pred: NDArray, dt:float) -> float:
    """Deadband/Thresholded RMSE - error only counts if outside TARGET_RADIUS"""
    y_error = np.maximum(0.0, np.abs(y_true - y_pred) - TARGET_RADIUS)
    return float(np.sqrt(np.mean(y_error)**2))

@register_metric("r_squared", higher_is_better=True)
def _r_squared(y_true: NDArray, y_pred: NDArray, dt:float) -> float:
    """Coefficient of Determination (R^2)

    Unlike Pearson R, R^2 penalises bias and gain errors. A decoder that perfectly tracks shape but with wrong scale
    gets r ~=1 but R^2 < 1.
    """
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    if ss_tot < _EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot

@register_metric("correlation", higher_is_better=True)
def _correlation(y_true: NDArray, y_pred:NDArray, dt:float) -> float:
    """Pearson Correlation Coefficient"""
    std_thresh = math.sqrt(_EPS)
    if np.std(y_true) < std_thresh or np.std(y_pred) < std_thresh:
        return 0.0
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return r if np.isfinite(r) else 0.0
# --- smoothness ---
@register_metric("dimensionless_jerk")
def _dimensionless_jerk(y_true: NDArray, y_pred: NDArray, dt:float) -> float:
    """Dimensionless squared jerk of the predicted trajectory

    From Hogan & Sternard (2009):
        dimensionless jerk = $$\frac{\left( \int_{t_1}^{t_2} \dddot{x}(t)^2 \, dt \right) D^5}{A^2}$$
        D: movement duration
        A: trajectory amplitude (max - min) of predicted signal
        Lower values = smoother signal

        Compute on `y_pred`
        Returns NaN when A ~=0 (stationary DOF) because the metric is undefined without appreciable movement.
    """
    if len(y_pred) < 4: # Need at least four position samples to derive Jerk
        return float('nan')
    amplitude = float(np.max(y_pred) - np.min(y_pred))
    if amplitude < math.sqrt(_EPS):
        return float('nan')

    duration = (len(y_pred) - 1) * dt

    # Third finite difference. TODO(@FRM): Look into Savitzky-Golay polynomial method
    jerk = np.diff(y_pred, n=3) / (dt**3)
    integrated_squared_jerk = float(np.sum(jerk**2)) * dt
    return integrated_squared_jerk * (duration**5) / (amplitude ** 2)
# --- target-based metrics ---
@register_metric("ptit", higher_is_better=True)
def _percent_time_in_target(y_true: NDArray, y_pred:NDArray, dt: float) -> float:
    """Fraction of samples within TARGET_RADIUS"""
    in_target = np.abs(y_true - y_pred) <= TARGET_RADIUS
    return float(np.mean(in_target))

@register_metric("max_hold", higher_is_better=True)
def _max_hold(y_true: NDArray, y_pred: NDArray, dt: float) -> float:
    """Max Hold - longest uninterrupted run inside TARGET RADIUS in seconds.
    returns 0.0 if nos sample is within the target zone
    """
    in_target = np.abs(y_true - y_pred) <= TARGET_RADIUS
    if not np.any(in_target):
        return 0.0

    # TODO(@FRM): validate this trick
    padded = np.concatenate(([False], in_target, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs ==1)[0]
    stops = np.where(diffs == -1)[0]
    max_run_samples = int(np.max(stops - starts))
    return max_run_samples * dt

# --- alignment-based metrics ---
def _dtw_distance(y_true: NDArray, y_pred:NDArray, dt: float) -> float:
    """Dynamic Time Warping distance, normalized by path length.

    Measures trajectory similarity allowing for temporal warping.
    Unlike RMSE, DTW does not penalize constant temporal lag as severely.
    Comparing RMSE vs DTW can reveal whether a decoder's errors are primarily due to latency vs. shape mismatch.

    Basic O(n*m) implmentation - adequate for trials of a few hundred samples @30 Hz
    """
    n = len(y_true)
    m = len(y_pred)

    # Cost matrix
    cost = np.full((n+1, m+1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    for i in range(1, n +1):
        for j in range (1, m +1):
            c = (y_true[i-1] - y_pred[j-1]) ** 2
            cost[i, j] = c + min(cost[i-1,  j], cost[i, j-1], cost[i-1, j-1])
    # Normalize by the warping path length
    return float(np.sqrt(cost[n,m] / (n+m)))

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

    # # Snap to nearest existing NIP timestamp
    # start_idxs = np.clip(np.searchsorted(test_nip_time, shifted_starts), 0, len(test_nip_time) - 1)
    # stop_idxs = np.clip(np.searchsorted(test_nip_time, shifted_stops), 0, len(test_nip_time) - 1)
    # trial_starts = test_nip_time[start_idxs]
    # trial_stops = test_nip_time[stop_idxs]

    return shifted_starts, shifted_stops

# --- Metrics Computation ---
def compute_trial_dof_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    dt: float,
) -> dict[str, float]:
    """Compute all registered metrics for a single trial x DOF.

    Args:
        y_true: True kinematics, shape (n_samples,)
        y_pred: Predicted kinematics, shape (n_samples,)

    Returns:
        Dict of metrics by name. Values that are NaN indicate the metric was undefined for this slice (e.g. zero-amplitude jerk)
    """
    results: dict[str, float] = {}
    for spec in _METRIC_REGISTRY.values():
        try:
            val = spec.fn(y_true, y_pred, dt)
        except Exception:
            logger.debug("Metric %r raised on trial slice - storing NaN", spec.name, exc_info=True)
            val = float("nan")
        results[spec.name] = val
    return results


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
    dt: float,
    burn_in: int = 10,
) -> dict:
    """Compute per-trial, per-DOF metrics with intended/unintended classification.

    Burn-in is applied only to the first trial (the filter needs initial samples for state convergence).

    Args:
        predictions: shape(n_test_samples, n_dofs)
        true_kinematics: shape(n_test_samples, n_dofs)
        kinematic_indices: 1-based DOF indices, shape (n_dofs, )
        trial_boundaries: List of (start, end, original_trial_idx) from `build_test_trial_boundaries`
        movement_matrices: shape(n_total_trials, 12, 4)
        dt: sample period in seconds
        burn_in: int, samples to exclude from first trial (Kalman Convergence)

    Returns:
        Dict with per_trial, aggregate, and summary sections.
    """
    metric_names = list(_METRIC_REGISTRY.keys())
    per_trial: list[dict] = []

    for boundary_idx, (t_start, t_end, orig_trial_idx) in enumerate(trial_boundaries):
        if boundary_idx == 0:
            t_start = min(t_start + burn_in, t_end) # TODO: @FRM, remove the need for burn-in by using steady state MKF
        if t_end - t_start < 2:
            logger.warning(
                "Trial %d has < 2 samples after burn-in (start=%d, end=%d). Skipping to match plotter",
                orig_trial_idx, t_start, t_end,
            )
            continue

        trial_true = true_kinematics[t_start:t_end, :]
        trial_pred = predictions[t_start:t_end, :]
        mvmnt_mat = movement_matrices[orig_trial_idx]

        # --- per-DOF metrics
        dof_metrics: dict[str,dict] = {}
        for i, dof_idx in enumerate(kinematic_indices):
            metrics = compute_trial_dof_metrics(trial_true[:, i], trial_pred[:, i], dt)
            metrics["intended"] = is_dof_intended(mvmnt_mat, int(dof_idx))
            dof_metrics[str(int(dof_idx))] = metrics

        # --- aggregate across DOFs within this trial ---
        trial_record: dict = {
            "trial_index": boundary_idx,
            "original_trial_idx": orig_trial_idx,
            "n_samples": t_end - t_start,
            "dof_metrics": dof_metrics,
        }

        for subset_label, predicate in [
            ("intended", lambda v: v["intended"]),
            ("unintended", lambda v: not v["intended"]),
            ("all", lambda _: True),
        ]:
            subset_dofs = [v for v in dof_metrics.values() if predicate(v)]
            for m_name in metric_names:
                vals = [d[m_name] for d in subset_dofs if np.isfinite(d[m_name])]
                key = f"{m_name}_mean_{subset_label}"
                trial_record[key] = float(np.mean(vals)) if vals else None
        per_trial.append(trial_record)


    # --- Aggregate across trials, per DOF ---
    aggregate_per_dof: dict  = {}
    for dof_str in [str(int(d)) for d in kinematic_indices]:
        for subset_label, predicate in [
            ("intended", lambda v: v.get("intended", False)),
            ("unintended", lambda v: not v.get("intended", True)),
            ("all", lambda _: True),
        ]:
            vals = [
                t["dof_metrics"][dof_str]
                for t in per_trial
                if dof_str in t["dof_metrics"] and predicate(t["dof_metrics"][dof_str])
            ]
            aggregate_per_dof.setdefault(dof_str, {})[subset_label] = _aggregate_metric_list(
                vals, metric_names
            )

    # --- Summary across all trials ---
    summary: dict = {
        "n_trials": len(per_trial),
        "n_burn_in_excluded": burn_in,
        "metric_names": metric_names,
        "target_radius": TARGET_RADIUS,
        "dt": dt,
    }
    for subset_label in ("intended", "unintended", "all"):
        subset_summary: dict = {}
        for m_name in metric_names:
            key = f"{m_name}_mean_{subset_label}"
            values = [t[key] for t in per_trial if t[key] is not None]
            subset_summary[f"{m_name}_mean"] = float(np.mean(values)) if values else None
            subset_summary[f"{m_name}_std"] = float(np.std(values)) if values else None
        summary[subset_label] = subset_summary

    return {
        "per_trial":  per_trial,
        "aggregate_per_dof": aggregate_per_dof,
        "summary": summary,
    }

def _aggregate_metric_list(
        vals: list[dict], metric_names: list[str]
) -> dict:
    """Aggregate a list of per-DOF metric dicts."""
    if not vals:
        result: dict = {"n": 0}
        for m in metric_names:
            result[f"{m}_mean"] = None
            result[f"{m}_std"] = None
        return result
    result = {"n": len(vals)}
    for m in metric_names:
        finite_vals = [v[m] for v in vals if np.isfinite(v[m])]
        result[f"{m}_mean"] = float(np.mean(finite_vals)) if finite_vals else None
        result[f"{m}_std"] = float(np.std(finite_vals)) if finite_vals else None
    return result

def _build_null_metrics(burn_in: int) -> dict:
    """Return a valid-schema result dict for session with no predictions."""
    metric_names = list(_METRIC_REGISTRY.keys())
    null_subset: dict = {}
    for m in metric_names:
        null_subset[f"{m}_mean"] = None
        null_subset[f"{m}_std"] = None

    return {
        "per_trial": [],
        "aggregate_per_dof": {},
        "summary":{
            "n_trials": 0,
            "n_burn_in_excluded": burn_in,
            "null_result": True,
            "null_reason": "no_predictions",
            "metric_names": metric_names,
            "intended": dict(null_subset),
            "unintended": dict(null_subset),
            "all": dict(null_subset),
        },
    }

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate decoder predictions with intended/unintended classification."
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions HDF5 from decode_features.py")
    parser.add_argument("--events", type=Path, required=True, help="Path to events HDF5 from preprocess_shared_data.m")
    parser.add_argument("--burn-in", type=int, default=10, help="Burn-in samples to exclude from first trial until MKF settles")
    parser.add_argument("--target-radius", type=float, default=TARGET_RADIUS, help="Deadband  / target-zone radius (used by d_rmse, ptit, max_hold)")
    parser.add_argument("--output-metrics", type=Path, required=True)
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args= parse_args(argv)

    # Allow CLI override of target radius. TODO(@FRM): This is a code smell. Figure out how to remove it.
    global TARGET_RADIUS
    TARGET_RADIUS = args.target_radius

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
        null_metrics = _build_null_metrics(args.burn_in)
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

    # --- Compute sample period ---
    # Recall that Feedback Decode aims, but does not enforce a 33ms loop rate. It is 33 ms+/- jitter. The Ripple should have exaclty a 30kHz rate though
    dt = np.float(np.median(np.diff(test_nip_time)))
    dt_seconds = dt / np.float(30e3)
    logger.info("Sample period: %.4f s (%.1f Hz)", dt_seconds, 1.0 / dt_seconds)

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
        dt=dt_seconds,
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

    # --- Log Summary --- TODO(@FRM): Refactor to make extensible
    intended = performance_metrics["summary"]["intended"]
    unintended = performance_metrics["summary"]["unintended"]
    logger.info(
        "Results: intended RMSE=%.4f (+/-%.4f), intended d_RMSE=%.4f (+/-%.4f), r=%.4f (+/-%.4f) | "
        "unintended RMSE=%.4f (+/-%.4f), unintended d_RMSE=%.4f (+/-%.4f) unintended r=%.4f (+/-%.4f) " 
        "across %d trials",
        intended.get("rmse_mean") or 0, intended.get("rmse_std") or 0,
        intended.get("d_rmse_mean") or 0, intended.get("d_rmse_std") or 0,
        intended.get("correlation_mean") or 0, intended.get("correlation_std") or 0,
        unintended.get("rmse_mean") or 0, unintended.get("rmse_std") or 0,
        unintended.get("d_rmse_mean") or 0, unintended.get("d_rmse_std") or 0,
        unintended.get("correlation_mean") or 0, unintended.get("correlation_std") or 0,   
        performance_metrics["summary"]["n_trials"],
    )
    logger.info("Metrics written to %s", args.output_metrics)

if __name__ == "__main__":
    main()

