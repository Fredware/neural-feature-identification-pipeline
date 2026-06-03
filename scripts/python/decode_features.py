"""Decode kinematic predictions from neural features using a Modified Kalman Filter.

Pipeline rule: decode_features
Inputs:  features HDF5, kinematics HDF5, events HDF5, selected channels JSON
Outputs: predictions HDF5
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
from modified_kalman_filter import ModifiedKalmanFilter
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__) # TODO: @FRM, remember to switch to structlog


# --- Data Loading ---

def load_selection_result(filepath: Path) -> dict:
    """Load the output of select_features.py.

    Args:
        filepath: Path to selected.json.

    Returns:
        Dict with selected channels, split info, and alignment metadata.
    """
    with open(filepath) as f:
        return json.load(f)


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
# --- Main Entry Point ---

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode kinematics from neural features via Modified Kalman Filter."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True, help="Path to selected.json from select_features.")

    parser.add_argument("--apply-gain-threshold", action="store_true", help="Apply MKF gain-threshold post-processing.") # if the user types --apply-latching, python sets this variable to True. If they leave it out, it defaults to False.
    parser.add_argument("--apply-latching", action="store_true", help="Apply latching (smoothing) filter.")
    parser.add_argument("--latch-constant", type=float, default=1.0, help="Latching filter constant.")
    
    parser.add_argument("--output-predictions", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the decode pipeline step."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,                                 # importance threshold for the messages (DEBUG < INFO < WARNING < ERROR)
        format="[%(asctime)s] [%(levelname)s] %(message)s", # [time of message] [importance level] actual message.
        datefmt="%Y-%m-%d %H:%M:%S",                        # clock style for %(asctime)s
    )

    # --- Load selection results ---
    logger.info("Loading selection results from %s", args.selected)
    selection = load_selection_result(args.selected)

    kinematic_indices = selection["kinematic_indices"]
    selected_joint = selection["selected_channels_joint"]
    train_mask = np.array(selection["split"]["train_mask"])
    test_mask = np.array(selection["split"]["test_mask"])
    alignment_lag = selection["alignment"]["lag_samples"]

    # --- Handle null selection ---
    if len(selected_joint) == 0:
        logger.warning(
            "No features were selected (null selection result). "
            "Writing empty predictions"
        )
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output_predictions, "w") as f:
            f.create_dataset("predictions", data=np.array([]))
            f.create_dataset("true_kinematics", data=np.array([]))
            f.create_dataset("kinematic_indices", data=np.array(kinematic_indices))
            f.create_dataset("test_nip_time", data=np.array([]))
            f.attrs["alignment_lag"] = alignment_lag
            f.attrs["null_result"] = True
            f.attrs["null_reason"] = "no_features_passed_preselection"
        logger.info("Null metrics written to %s", args.output_predictions)
        return

    # --- Load data ---
    logger.info("Loading features from %s", args.features)
    features = load_features(args.features)

    logger.info("Loading kinematics from %s", args.kinematics)
    kinematics, nip_time = load_kinematics(args.kinematics)

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
    test_nip = nip_time[test_mask]
    logger.info("Predicting on %d test samples", test_feat.shape[0])

    predictions = decoder.predict(test_feat)

    # --- Write predictions ---
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_predictions, "w") as f:
        f.create_dataset("predictions", data=predictions)
        f.create_dataset("true_kinematics", data=test_kin)
        f.create_dataset("kinematic_indices", data=np.array(kinematic_indices))
        f.create_dataset("test_nip_time", data=test_nip)
        f.attrs["alignment_lag"] = alignment_lag
        f.attrs["null_result"] = False
    logger.info("Predictions written to %s", args.output_predictions)


if __name__ == "__main__":
    main()

