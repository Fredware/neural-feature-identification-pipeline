"""
Select neural features for kinematic decoding via Forward Feature Selection with Gram-Schmidt Orthogonalization.

Pipeline rule: select_features
Inputs: features HDF5, kinematics HDF5, events HDF5
Outputs: JSON with selected feature idxs, alignment metadata, train/test split info
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging  # TODO: refactor to use structlog
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# --- Data Structures ---
@dataclass
class AlignmentInfo:
    """Metadata from the feature-to-kinematic alignment step."""
    lag_samples: int
    n_samples_after_align: int
    mean_abs_corr_at_lag: float

@dataclass
class TrialGroup:
    """A group of trials sharing the same movement signature."""
    signature: tuple[int, ...]
    trial_indices: list[int]
    is_rest: bool
    is_compound: bool

    @property
    def n_trials(self) -> int:
        return len(self.trial_indices)

@dataclass
class SplitInfo:
    """Train/Test split metadata"""
    train_ratio: float
    split_type: str
    include_combined: bool
    n_train_samples: int
    n_test_samples: int
    train_mask: list[bool] = field(default_factory=list)
    test_mask: list[bool] = field(default_factory=list)

@dataclass
class SelectionResult:
    """Output of the feature selection step"""
    feature_set: str
    dof_config: str
    kinematic_indices: list[int]

    alignment: AlignmentInfo
    split: SplitInfo

    selected_channels_per_dof: dict[str, list[int]] # TODO:@FRM, assess if you really need this line.
    selected_channels_joint: list[int]

    n_candidates_after_preselection: int
    correlation_threshold: float
    max_features: int

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# --- Data Loading ---
def load_features(filepath: Path) -> tuple[NDArray, NDArray]:
    """Load features and computation times from the 'extract_features' output.

    Args:
        filepath: Path
            Path to {feature_set}.h5 produced by extract_features.m
    Returns:
        features: ndarray, shape (n_samples, n_features)
            Feature matrix. Convention: rows=time, columns=features.
        computation_times: ndarray, shape(n_samples, )
            Per-frame computation time in seconds
    """
    with h5py.File(filepath, "r") as f:
        features = f["features"][:]
        computation_times = f["computation_times"][:]    

    computation_times = computation_times.ravel()

    return features, computation_times

def load_kinematics(filepath: Path) -> tuple[NDArray, NDArray]:
    """Load Kinematic labels and NIP timestamps from the preprocess_session output

    Args:
        filepath: Path
            Path to kinematics.h5 produced by preprocess_session.smk
    Returns:
        kinematics: ndarray, shape(n_samples, n_dofs)
            Kinematic labels matrix. Convention: rows=time, cols=DOFs
        nip_time: ndarray, shape(n_samples, )
            NIP timestamps at the KDF frame rate (~30 Hz)
    """
    with h5py.File(filepath, "r") as f:
        kinematics = f["kinematics"][:]
        nip_time = f["nip_time"][:]
    nip_time = nip_time.ravel()
    
    # MATLAB writes kinematics as (n_dofs, n_samples). Transpose to (n_samples, n_dofs)
    # The code below assumes 12 DOFs from MSMS model.
    if kinematics.ndim == 2 and kinematics.shape[0] < kinematics.shape[1]:
        kinematics = kinematics.T
    return kinematics, nip_time

def load_events(filepath: Path) -> tuple[NDArray, NDArray, NDArray]:
    """Load trial start/stop timestamps and movement parameters from the preprocess_session output

    Args:
        filepath: Path
            Path to events.h5 produced by preprocess_session.smk
    Returns:
        A tuple of (trial_starts, trial_stops, movement_matrices) where:
            trial_starts: ndarray of shape(n_trials, )
            trial_stops: ndarray of shape(n_trials, )
            movement_matrices: ndarray of shape (n_trials, 12, 4)
                Columns are [amplitude, delay, rise_time, hold_time].
                Amplitude sign encodes direction (+flex, -ext)
    """
    with h5py.File(filepath, "r") as f:
        trial_starts = f["trial_start_idxs"][:].ravel()
        trial_stops = f["trial_stop_idxs"][:].ravel()
        movement_matrices = f["movement_matrices"][:]

    # Ensure matrices are of shape (n_trials, 12, 4) since MATLAB may write transposed
    if movement_matrices.shape[-1] != 4:
        movement_matrices = np.transpose(movement_matrices, (2,1,0))
    return trial_starts, trial_stops, movement_matrices

def _get_signature_from_amplitudes(movement_matrix_row: NDArray) -> tuple[int, ...]:
    """Convert a single trial's movement parameters to a signed DOF signature

    Args:
        movement_matrix_row: ndarray of shape (12, 4)
            Full MvmntMat for one trial.
            Column 0 is amplitude where sign equals direction.
    Returns:
        Tuple of signed 1-based DOF idxs. Empty tuple for "rest" trials.
    """
    amplitudes = movement_matrix_row[:, 0]
    active = np.nonzero(amplitudes)[0]
    if len(active) == 0:
        return ()
    return tuple(int(np.sign(amplitudes[i])) * (i + 1) for i in active)

def _group_trials(movement_matrices: NDArray) -> list[TrialGroup]:
    """
    Group trials by movement signature.

    Trials with identical signed-DOF signatures are grouped together.
    Groups are ordered by first appearance.

    Args:
        movement_matrices: ndarray of shape (n_trials, 12, 4)
    Returns:
        List of TrialGroup, one per unique signature
    """
    groups: list[TrialGroup] = []
    sig_to_group: dict[tuple[int, ...], int] = {}

    for trial_idx in range(movement_matrices.shape[0]):
        sig  = _get_signature_from_amplitudes(movement_matrices[trial_idx, :])
        if sig in sig_to_group:
            groups[sig_to_group[sig]].trial_indices.append(trial_idx)
        else:
            sig_to_group[sig] = len(groups)
            groups.append(TrialGroup(
                signature=sig,
                trial_indices=[trial_idx],
                is_rest=(len(sig)==0),
                is_compound=(len(sig)>1),
            ))
    return groups

# --- Preprocessing (Align, Split) ---
def estimate_alignment_lag(
        kinematics: NDArray,
        features: NDArray,
        max_lag_samples: int=12,
) -> tuple[int, NDArray, NDArray]:
    """Estimate the lag between kinematics and features via cross-correlation

    Computes normalized cross-correlation between all feature-DOF pairs using FFT convolution.
    Then, it finds the lag that maximizes mean absolute correlation across all pairs.

    Args:
        kinematics: ndarray, shape (n_samples, n_dof)
        features: ndarray, shape (n_samples, n_features)
        max_lag_samples: int
            Maximum lag to search (in samples). Defaults to twelve which captures 2 std of Wang et al's global shift distribution (~400ms)
    Returns:
        best_lag: int
            Optimal lag in samples. Positives means features lead kinematics (anticipation).
        cropped_lags: ndarray
            Lag values used during search
        mean_abs_corr_per_lag: ndarray
            Mean of |r| at each lag
    """
    n_samples, n_features = features.shape
    _, n_dofs = kinematics.shape

    # --- Normalize (Z-score) along time axis ---
    features_eps = np.finfo(features.dtype).eps
    features_norm = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + features_eps)
    
    kinematics_eps = np.finfo(kinematics.dtype).eps
    kinematics_norm = (kinematics - np.mean(kinematics, axis=0)) / (np.std(kinematics, axis=0) + kinematics_eps)

    # --- Cross-correlate via FFT convolution ---
    # cross-correlation = ifft(fft(a) * conj(fft(b)))
    fft_len = 2*n_samples-1
    features_fft = np.fft.rfft(features_norm, n=fft_len, axis=0)
    kinematics_fft = np.fft.rfft(kinematics_norm, n=fft_len, axis=0)
    # Broadcast: (fft_bins, n_features, 1) * (fft_bins, 1, n_dofs)
    # conj(feat)*kin gives R[k] = sum_t feat[t]*kin[t+k]
    cross_fft = np.conj(features_fft[:, :, np.newaxis]) * kinematics_fft[:, np.newaxis, :]
    cross_corr = np.fft.irfft(cross_fft, n=fft_len, axis=0)
    # Reorder sot that lag=-(n-1) is first, lag=0 is centered, lag=+(n-1) is last
    cross_corr = np.roll(cross_corr, n_samples -1, axis=0)
    # Normalize by number of samples to get actual correlation coefficients
    cross_corr /= n_samples

    # --- Define search params (lag) ---
    lags = np.arange(-n_samples + 1, n_samples)
    # Crop to realistic search window (i.e., 0 to max_lag)
    # positive lags (features lag behind kinematics - reaction case)
    # negative lags (features lead kinematics - anticipation case)
    valid_idxs = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    cropped_corr = cross_corr[valid_idxs, :, :]
    cropped_lags = lags[valid_idxs]

    # --- Aggregate across all feature-dof pairs ---
    # Absolute value prevents negative and positive correlations form cancelling out
    mean_abs_corr_per_lag = np.mean(np.abs(cropped_corr), axis=(1,2))

    # --- Find the best lag ---
    best_lag_idx = np.argmax(mean_abs_corr_per_lag)
    best_global_lag = cropped_lags[best_lag_idx]

    logger.info(
        "Alignment: best lag = %d samples (mean |r| = %.3f)",
        best_global_lag,
        mean_abs_corr_per_lag[best_lag_idx]
    )
    
    return best_global_lag, cropped_lags, mean_abs_corr_per_lag


def align_kinematics_and_markers(
    kinematics: NDArray,
    trial_starts: NDArray,
    trial_stops: NDArray,
    nip_time: NDArray,
    lag: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Shift kinematics and event stamps to align with features.

    Features and NIP timestamps are kept intact since they are a fixed reference. 
    Kinematics are rolled (circshifted) and zero-padded. 
    Event timestamps are shifted by the equivalent NIP lag offset and clamped to the valid NIP range.

    Args:
        kinematics: ndarray of shape (n_samples, n_dofs)
        trial_starts: ndarray of shape (n_trials, )
            NIP timestamps.
        trial_stops: ndarray of shape (n_trials, )
            NIP timestamps.
        nip_time: ndarray of shape (n_samples, )
            NIP timestamps for each sample.
        lag: int
            Shift in samples. Positive means features lead kinematics (neural activity anticipates movement), 
            so kinematics are shifted backward (earlier in time) by 'lag' samples
    Returns:
        A tuple of (kinematics, trial_starts, trial_stops) with the shift applied.
        Kinematics are zero-padded at the vacated edge.
        Event timestamps are clamped to [nip_time[0], nip_time[-1]].
    """
    if lag == 0:
        return kinematics.copy(), trial_starts.copy(), trial_stops.copy()
    
    # --- Shift kinematics ---
    kinematics_shifted = np.roll(kinematics, -lag, axis=0)
    # Zero-pad vacated edge
    if lag > 0:
        # Shifted left: last 'lag' samples wrapped around
        kinematics_shifted[-lag:, :] = 0.0
    else:
        kinematics_shifted[:-lag,:] = 0.0
    logger.info(
        "Kinematics shifted by %d samples. Zero-padded %d edge samples.", 
        -lag, 
        abs(lag))    

    # --- Shift events ---
    # Shift by NIP offset equivalent to the lag.
    # Compute median inter-sample interval to convert lag in samples to lag in NIP time units. 
    # Median is robust to jitter.
    nip_dt = np.median(np.diff(nip_time))
    nip_shift = lag * nip_dt

    shifted_starts = trial_starts - nip_shift
    shifted_stops = trial_stops - nip_shift
    # Snap to nearest existing NIP timestamp
    start_idxs = np.clip(np.searchsorted(nip_time, shifted_starts), 0, len(nip_time)-1)
    stop_idxs = np.clip(np.searchsorted(nip_time, shifted_stops), 0, len(nip_time)-1)
    shifted_starts = nip_time[start_idxs]
    shifted_stops = nip_time[stop_idxs]
    logger.info(
        "Markers shifted by %d samples (%.3f NIP units). %d trials preserved",
        -lag, nip_shift, len(shifted_starts)
    )
    
    return kinematics_shifted, shifted_starts, shifted_stops

def build_train_test_split(
        nip_time: NDArray,
        trial_starts: NDArray,
        trial_stops: NDArray,
        movement_matrices: NDArray,
        train_ratio: float,
        split_type: str,
        include_combined: bool,
        include_rest: bool,
)-> SplitInfo:
    """Build boolean (train/test) masks from trial boundaries, while splitting per movement group.

    Trials are grouped by movement signature (signed-DOF vector).
    The split is applied independently within each group so that every movement type present in the test set is also represented in the training set.

    Args:
        nip_time: ndarray of shape (n_samples, )
            NIP timestamps
        trial_starts: ndarray of shape (n_trials, )
            Trial onset timestamps
        trial_stops: ndarray of shape (n_trials, )
            Trial completion timestamps
        movement_matrices: ndarray of shape (n_trials, 12, 4)
            MvmntMat for each trial.
        train_ratio: float
            Fraction of trials within each trial to assign to training
        split_type: str
            Split strategy. "train_first" assigns the first trials within a group to training.
            "train_last" assigns the last trials withing a group to training.
        include_combined: bool
            If true, include compound (i.e. multi-DOF) trials in both training and testing.
            If false, compound trials are excluded from training and testing. # TODO:@FRM, add an option to completely exclude them.
        include_rest: bool
            If true, rest trials conribute to both masks.

    Returns:
        SplitInfo dataclass with boolean masks over the sample axis.

    Raises:
        ValueError: If uknown split_type is provided.
    """
    n_samples = len(nip_time)
    groups = _group_trials(movement_matrices)

    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)

    group_summary = []

    for group in groups:
        # --- Decide whether to further process a group ---
        if group.is_rest:
            if not include_rest:
                continue
            use_for_train = True
            use_for_test = True
        elif group.is_compound:
            use_for_train = include_combined
            use_for_test = include_combined
            if not use_for_train and not use_for_test:
                continue
        else:
            use_for_train = True
            use_for_test = True
        # --- Split the group's trials ---
        n_trials = group.n_trials
        if split_type == "train_first":
            n_train = int(np.floor(n_trials*train_ratio))
        elif split_type == "train_last":
            n_train = int(np.floor(n_trials*train_ratio))
        else:
            raise ValueError(f"Unsupported split type: '{split_type}'")
        
        for j, trial_idx in enumerate(group.trial_indices):
            # Window: [Previous trial end, This trial end]
            if trial_idx > 0:
                window_start = trial_stops[trial_idx - 1]
            else:
                window_start = nip_time[0]
            window_end = trial_stops[trial_idx]

            in_window = (nip_time >= window_start) & (nip_time <= window_end)

            if split_type == "train_first":
                is_train = j < n_train
            elif split_type == "train_last":
                is_train = j >= (n_trials-n_train)
            
            if is_train and use_for_train:
                train_mask |= in_window
            elif use_for_test:
                test_mask |= in_window
        
        group_summary.append(
            f"sig={group.signature}, n={n_trials}, "
            f"train={min(n_train, n_trials) if use_for_train else 0}"
        )

    n_train = int(np.sum(train_mask))
    n_test = int(np.sum(test_mask))

    for line in group_summary:
        logger.info(" Group: %s", line)
    logger.info(
        "Split total: %d train, %d test samples (%.1f%%/%.1f%%)",
        n_train, n_test,
        100 * n_train / max(n_samples, 1),
        100 * n_test / max(n_samples, 1),
    )
    if n_train == 0:
        logger.warning("Training set is empty. Check train_ratio and trial structure")
    if n_test == 0:
        logger.warning("Testing set is empty. Check train_ratio and trial structure")
    return SplitInfo(
        train_ratio = train_ratio,
        split_type= split_type,
        include_combined=include_combined,
        n_train_samples=n_train,
        n_test_samples=n_test,
        train_mask=train_mask.tolist(),
        test_mask=test_mask.tolist(),
    )

def split_kinematics_directional(
        kinematics: NDArray,
        dof_idxs: list[int],
) -> NDArray:
    """Split bi-directional kinematic signals into positive and negative components.

    A DOF with both flexion (+) and extension (-) is split into two rows:
    one containing only the positive samples (negatives zeroed out) and
    one containing the absolute value of the negative samples (positives zeroed out)
    DOFs with only one direction produce a single row.

    Used for pre-selection by correlation: a channel that correlates with flexion but
    not extension will have low absolute Pearson correlation on the combined signal,
    causing it to be missed.

    Based on corrChanSelZeroMeanDarpa_jag.m lines 37-50

    Args:
        kinematics: ndarray of shape (n_samples, all_dofs)
        dof_idxs: list of int
            1-based DOF indices to extract and split. These correspond to the KDF
            kinematic indices (e.g., [1, 2, 3, 6, 10, 12] for 6DOF-DEKA)
    
    Returns:
        ndarray of shape (n_samples, n_split_components) where
        n_split_components >= len(dof_indices)
    
    Raises:
        ValueError: if non-zero kinematic data is not found for the requested DOFs
    """
    columns =[]
    for dof_idx in dof_idxs:
        # Convert 1-based idx into 0-based col idx
        signal = kinematics[:, dof_idx-1]
        pos_mask = signal > 0
        neg_mask = signal < 0

        if np.any(pos_mask):
            pos_col = np.zeros_like(signal)
            pos_col[pos_mask] = signal[pos_mask]
            columns.append(pos_col)
        if np.any(neg_mask):
            neg_col = np.zeros_like(signal)
            neg_col[neg_mask] = np.abs(signal[neg_mask])
            columns.append(neg_col)
    
    if not columns:
        raise ValueError(
            f"Could not find non-zero kinematic data for DOF indices {dof_idxs}"
        )
    return np.column_stack(columns)

def preselect_by_correlation(
    kinematics_split: NDArray,
    features: NDArray,
    min_corr_threshold: float,
    max_channels: int | None=None,
) -> NDArray:
    """
    Pre-select feature channels exceding a correlation threshold with any DOF
    
    Computes the absolute pearson correlation between each feature channel
    (zero-meaned) and each directionally-split kinematic signal.
    A channel is retained if it exceeds ``theshold`` for at least one DOF component.
    Channels are ordered via interleaved round-robin across DOFs (1 channel per DOF at a time)

    Based on corrChanSelZeroMeanDarpa_jag.m (window_type='all' path)

    Args:
        kinematics_split: ndarray of shape(n_samples, n_split_dofs)
            Directionally split, zero-meaned kinematics
        features: ndarray of shape (n_samples, n_features)
        threshold: Minimum absolute correlation to retrain a channel.
        max_channels: Maximum number of channels to return. None for no limit.
    
    Returns:
        ndarray of int with indices into the feature columns that pass the threshold, 
        ordered by interleaved DOF ranking.

    """

    n_samples, n_features = features.shape
    n_dofs = kinematics_split.shape[1]

    # Zero-mean kinematics and features along time
    kin = kinematics_split - kinematics_split.mean(axis=0, keepdims=True)
    feat = features - features.mean(axis=0, keepdims=True)

    # Norms along time axis
    feat_norm = np.sqrt(np.sum(feat**2, axis=0, keepdims=True))
    feat_norm = np.maximum(feat_norm, np.finfo(feat_norm.dtype).eps)
    kin_norm = np.sqrt(np.sum(kin**2, axis=0, keepdims=True))
    kin_norm = np.maximum(kin_norm, np.finfo(kin_norm.dtype).eps)

    # Correlation matrix: (n_features, n_dofs)
    rzx = (feat.T @ kin) / (feat_norm.T @ kin_norm)
    rzx = np.abs(np.nan_to_num(rzx, nan=0.0))
    rzx[rzx < min_corr_threshold] = 0.0

    # Sort each DOF column descending
    sorted_idxs = np.argsort(-rzx, axis=0)
    sorted_vals = np.take_along_axis(rzx, sorted_idxs, axis=0)

    # Interleave: round-robin across DOFs, 1 channel per DOF per pass
    selected = []
    selected_set: set[int] = set()

    for row in range(n_features):
        for dof in range(n_dofs):
            if row >= sorted_idxs.shape[0]:
                continue
            ch = int(sorted_idxs[row, dof])
            if sorted_vals[row, dof] > 0 and ch not in selected_set:
                selected.append(ch)
                selected_set.add(ch)

    candidate_idxs = np.array(selected, dtype=int)
    if max_channels is not None and len (candidate_idxs) > max_channels:
        candidate_idxs = candidate_idxs[:max_channels]
    
    logger.info(
        "Pre-selection: % d / %d channels pass threshold=%0.2f",
        len(candidate_idxs),
        n_features,
        min_corr_threshold
    )
    return candidate_idxs

def gram_schmidt_select(
        kinematics: NDArray,
        features: NDArray,
        candidate_idxs: NDArray, 
        dof_idxs: list[int],
        max_chans: int=48,
        per_dof: bool=False,
) -> tuple[dict[str, list[int]], list[int]]:
    """Forward Feature Selection via Gram Schmidt orthogonalization

    At each step, select the candidate feature most correlated with the residual kinematics (after projecting out previously selected features).

    Adapted from gramSchmDarpa_jag.m

    Args:
        kinematics: ndarray of shape (n_samples, n_active_dofs). 
            Kinematic labels for the active DOFs.
            Kinematic labels are already subsetted, NOT directionally split Gram-Schmidt operates on the original bi-directional signal.
        features: ndarray of shape (n_samples, n_features). 
            Full feature matrix.
        candidate_idxs: ndarray of int
            Feature column indices that passed correlation pre-selection
        dof_idxs: list[int]
            1-based DOF indices for labeling output keys
        max_chans: int
            Maximum features to select per DOF (or total for joint mode)
        per_dof: bool
            If True, run selection independently per DOF. If False, run joint selection across all DOFs simultaneously

    Returns:
        A tuple of (channels_per_dof, channels_joint) where:
            channels_per_dof: Dict 
                Maps DOF index (str) to list of selected feature column indices. Empty if per_dof=False
            channels_joint: List
                Selected feature column indices for joint mode. Empty if list per_dof = True
    """
    # Get subset of feature candidates and transpose to (n_candidate, n_samples) for the inner loop (it operates on row vectors)
    Z = features[:, candidate_idxs].T

    # Zero-mean features along time (Now axis=1)
    Z = Z - Z.mean(axis=1, keepdims=True)
    Z[np.isnan(Z)] = 0.0

    channels_per_dof: dict[str, list[int]] = {}
    channels_joint: list[int] = []

    if per_dof:
        for i, dof_idx in enumerate(dof_idxs):
            # (1, n_samples) =  single DOF as a row. Use slice indexing to preserve singleton dimension
            X = kinematics[:, i:i+1].T
            X = X - X.mean(axis=1, keepdims=True)

            selected = _gram_schmidt_inner(X, Z, max_chans)
            channels_per_dof[str(dof_idx)] = [
                int(candidate_idxs[s]) for s in selected # gram_schmidt_inner returns the position of the feature subset that passed pre-selection. Map them back to the positions of the full feature set.
            ]
    else:
        # (n_dofs, n_samples)
        X = kinematics.T.copy()
        X = X - X.mean(axis=1, keepdims=True)

        selected = _gram_schmidt_inner(X, Z, max_chans)
        channels_joint = [int(candidate_idxs[s]) for s in selected]
    
    return channels_per_dof, channels_joint


def _gram_schmidt_inner(
        X: NDArray,
        Z: NDArray,
        max_channels: int,
)-> list[int]:
    """Core Gram-Schmidt loop operating on zero-meaned row-vector signals.

    Internally uses (n_rows, n_samples) layout for efficient dot products.

    Args:
        X: ndarray of shape (n_dofs, n_samples)
            Zero-meaned kinematic labels
        Y: ndarray of shape (n_candidates, n_samples)
            Zero-meaned candidate feaatures
        max_channels: int
            Maximum number of channels to select
    Returns:
        List of 0-based indices into Z rows, in selection order
    """
    n_candidates, n_samples = Z.shape

    mu = np.zeros_like(X)
    newZ = Z.copy()

    # Normalize rows
    norms = np.linalg.norm(newZ, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    newZ = newZ / norms

    selected: list[int] = []

    for step in range(min(max_channels, n_candidates)):
        # Residual = target - running projection
        resid = X - mu

        # Normalize residual nows
        resid_norm = np.linalg.norm(resid, axis=1, keepdims=True)
        resid_norm = np.maximum(resid_norm, 1e-12)
        resid_norm = resid / resid_norm

        # Correlation (n_candidates, n_dofs)
        corrs = newZ @ resid_norm.T
        corrs = np.nan_to_num(corrs, nan=0.0)

        flat_abs = np.abs(corrs).ravel()
        sorted_flat = np.argsort(-flat_abs)

        best_idx = None
        for flat_idx in sorted_flat:
            cand_idx = flat_idx // corrs.shape[1]
            if cand_idx not in selected:
                best_idx = cand_idx
                break
        
        if best_idx is None:
            break

        selected.append(int(best_idx))

        z_selected = newZ[best_idx, :]
        denom = z_selected @ z_selected + 1e-12
        w = resid @ z_selected / denom
        mu = mu + np.outer(w, z_selected)

        proj_coeffs = newZ @ z_selected / denom
        newZ = newZ - np.outer(proj_coeffs, z_selected)

        norms = np.linalg.norm(newZ, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        newZ = newZ / norms
        newZ[best_idx, :] = 0.0

    return selected
        

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select neural features with Forward Feature Selection via Gram-Schmidt Orthogonalization"
    )
    parser.add_argument(
        "--features", type=Path, required=True,
        help="Path to features.hdf5",
    )
    parser.add_argument(
        "--kinematics", type=Path, required=True,
        help="Path to kinematics.hdf5",
    )
    parser.add_argument(
        "--events", type=Path, required=True,
        help="Path to events.hdf5",
    )
    parser.add_argument(
        "--kinematic-indices", type=int, nargs="+", required=True,
        help="1-based kinematic DOF indices to use (e.g., 1 2 3 4 5 6 10 12)",
    )
    parser.add_argument(
        "--max-features", type=int, default=48,
        help="Maximum features to select per DOF",
    )
    parser.add_argument(
        "--min-correlation", type=float, default=0.3,
        help="Minimum |r| for correlation pre-selection",
    )
    parser.add_argument(
        "--max-lag", type=int, default=12,
        help="Maximum alignment lag to search (samples)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Fraction of trials for training",
    )
    parser.add_argument(
        "--split-type", type=str, default="train_first",
        help="Train test split strategy",
    )
    parser.add_argument(
        "--include-combined", type=str, default="False",
        help="Include combined movement trials"
    )
    parser.add_argument(
        "--include-rest", type=str, default="True",
        help="Include rest period in trials"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path for JSON output",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Run the feature selection step"""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    include_combined = args.include_combined.lower() in ("true", "1", "yes")
    include_rest = args.include_rest.lower() in ("true", "1", "yes")

    # --- Load data ---
    logger.info("Loading features from %s", args.features)
    features, _ = load_features(args.features)

    logger.info("Loading kinematics from %s", args.kinematics)
    kinematics, nip_time = load_kinematics(args.kinematics)

    logger.info("Loading events from %s", args.events)
    trial_starts, trial_stops, movement_amplitudes = load_events(args.events)

    n_kin_samples = kinematics.shape[0]
    MAX_SAMPLE_MISMATCH = 3 # NFR drops 1-2 leading frames; allow small discrepancy

    if features.shape[0] == n_kin_samples:
        pass # shape is already (n_samples, n_features)
    elif features.shape[1] == n_kin_samples:
        logger.info(
            "Transposing features from %s to (%d, %d)",
            features.shape, n_kin_samples, features.shape[0],
        )
        features= features.T
    # TODO: @FRM, Eventually remove the next two guards by fixing NFR computation.
    elif abs(features.shape[0] - n_kin_samples) <= MAX_SAMPLE_MISMATCH:
        n_diff = n_kin_samples - features.shape[0]
        logger.warning(
            "Features have %d fewer samples than kinematics. "
            "Trimming %d leading kinematic/nip_time samples.",
            n_diff, n_diff,
        )
        kinematics = kinematics[n_diff:, :]
        nip_time = nip_time[n_diff:]
    elif abs(features.shape[1] - n_kin_samples) <= MAX_SAMPLE_MISMATCH:
        features = features.T
        n_diff = n_kin_samples - features.shape[0]
        if n_diff > 0:
            logger.warning(
                "Features transposed. Trimming %d leading kinematic samples.",
                n_diff,
            )
            kinematics = kinematics[n_diff:, :]
            nip_time = nip_time[n_diff:]
    else:
        raise ValueError(
            f"Features shape {features.shape} does not match "
            f"kinematics n_samples={n_kin_samples} on either axis."
        )

    logger.info(
        "Loaded: features %s, kinematics %s, %d trials",
        features.shape, kinematics.shape, len(trial_starts),
    )

    n_samples_original = features.shape[0]

    # --- Align kinematics to features --- TODO:@FRM, consider limiting alignment to only active DOFs
    lag, cropped_lags, mean_abs_corr = estimate_alignment_lag(
        kinematics, features, max_lag_samples=args.max_lag
    )

    kinematics_original = kinematics.copy()
    trial_starts_original = trial_starts.copy()
    trial_stops_original = trial_stops.copy()

    kinematics, trial_starts, trial_stops = align_kinematics_and_markers(kinematics, trial_starts, trial_stops, nip_time, lag)
    
    alignment_info = AlignmentInfo(
        lag_samples = lag,
        n_samples_after_align =  n_samples_original,
        mean_abs_corr_at_lag = float(mean_abs_corr[np.argmax(mean_abs_corr)])
    )

    # # --- (Debug only) Plot data before and after alignment ---
    # fig, axes = plt.subplots(3, 1, figsize=(14, 8), tight_layout=True)

    # # Pick representative signals: highest-variance DOF and most-correlated feature
    # dof_idx = np.argmax(np.var(kinematics_original, axis=0))
    # feat_corrs = features.T @ (kinematics_original[:, dof_idx] - kinematics_original[:, dof_idx].mean())
    # feat_norms = np.sqrt(np.sum((features - features.mean(axis=0))**2, axis=0))
    # feat_norms = np.maximum(feat_norms, 1e-12)
    # corrs_with_dof = np.abs(feat_corrs / (feat_norms * np.sqrt(np.sum((kinematics_original[:, dof_idx] - kinematics_original[:, dof_idx].mean())**2))))
    # feat_idx = np.argmax(corrs_with_dof)

    # axes[0].plot(cropped_lags, mean_abs_corr)
    # axes[0].axvline(lag, color="r", linestyle="--", label=f"best lag = {lag}")
    # axes[0].set_xlabel("Lag (samples)")
    # axes[0].set_ylabel("Mean |r|")
    # axes[0].set_title("Cross-correlation lag search")
    # axes[0].legend()

    # axes[1].plot(kinematics_original[:, dof_idx], label=f"kin DOF {dof_idx}", alpha=0.8)
    # axes[1].plot(features[:, feat_idx] / (np.std(features[:, feat_idx]) + 1e-12) * np.std(kinematics_original[:, dof_idx]),
    #              label=f"feat {feat_idx} (scaled)", alpha=0.6)
    # axes[1].set_title("Before alignment")
    # axes[1].legend()

    # axes[2].plot(kinematics[:, dof_idx], label=f"kin DOF {dof_idx} (shifted)", alpha=0.8)
    # axes[2].plot(features[:, feat_idx] / (np.std(features[:, feat_idx]) + 1e-12) * np.std(kinematics[:, dof_idx]),
    #              label=f"feat {feat_idx} (scaled)", alpha=0.6)
    # axes[2].set_title(f"After alignment (lag={lag})")
    # axes[2].legend()

    # for ts in trial_starts_original:
    #     idx = np.searchsorted(nip_time, ts)
    #     if idx < n_samples_original:
    #         axes[1].axvline(idx, color="g", alpha=0.6, linewidth=1.5)
    # for ts in trial_stops_original:
    #     idx = np.searchsorted(nip_time, ts)
    #     if idx < n_samples_original:
    #         axes[1].axvline(idx, color="r", alpha=0.6, linewidth=1.5)

    # for ts in trial_starts:
    #     idx = np.searchsorted(nip_time, ts)
    #     if idx < n_samples_original:
    #         axes[2].axvline(idx, color="g", alpha=0.6, linewidth=1.5)
    # for ts in trial_stops:
    #     idx = np.searchsorted(nip_time, ts)
    #     if idx < n_samples_original:
    #         axes[2].axvline(idx, color="r", alpha=0.6, linewidth=1.5)


    # fig.savefig("./reports/figures/temp/alignment_debug.png", dpi=300)
    # logger.info("Debug plot saved to ./reports/figures/temp/alignment_debug.png")

    # --- Build train/test split ---
    split_info = build_train_test_split(
        nip_time, trial_starts, trial_stops, movement_amplitudes,
        train_ratio = args.train_ratio,
        split_type = args.split_type,
        include_combined = include_combined,
        include_rest=include_rest
    )

    train_mask = np.array(split_info.train_mask)

    # --- Extract subset of active DOFs/kinematics ---
    active_kinematics = kinematics[:, [idx-1 for idx in args.kinematic_indices]]

    # --- Correlation pre-selection (on training data) ---
    kin_split = split_kinematics_directional(kinematics, args.kinematic_indices)
    kin_split_train = kin_split[train_mask, :]
    features_train = features[train_mask, :]

    candidate_idxs = preselect_by_correlation(
        kin_split_train,
        features_train,
        min_corr_threshold = args.min_correlation,
    )

    if len(candidate_idxs) == 0:
        logger.error(
            "No features passed correlation pre-selection threshold at threshold=%.2f."
            "Writing null result (expected for datasets with degraded signal quality)",
            args.min_correlation,
        )
        result = SelectionResult(
            feature_set=args.features.stem,
            dof_config=args.output.parent.parent.name,
            kinematic_indices=args.kinematic_indices,
            alignment=alignment_info,
            split=split_info,
            selected_channels_per_dof={},
            selected_channels_joint= [],
            n_candidates_after_preselection=0,
            correlation_threshold=args.min_correlation,
            max_features=args.max_features
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2, cls=NumpyEncoder)
        logger.info("Null selection written to %s", args.output)
        return

    # --- Gram Schmidt Selection (on training data only) ---
    active_kinematics_train = active_kinematics[train_mask, :]

    channels_per_dof, _ = gram_schmidt_select(
        active_kinematics_train,
        features_train,
        candidate_idxs,
        dof_idxs = args.kinematic_indices,
        max_chans = args.max_features,
        per_dof = True,
    )

    _, channels_joint = gram_schmidt_select(
        active_kinematics_train,
        features_train,
        candidate_idxs,
        dof_idxs=args.kinematic_indices,
        max_chans=args.max_features,
        per_dof=False,
    )

    feature_set = args.features.stem
    dof_config = args.output.parent.parent.name

    # Package results
    result = SelectionResult(
        feature_set=feature_set,
        dof_config=dof_config,
        kinematic_indices=args.kinematic_indices,
        alignment=alignment_info,
        split=split_info, 
        selected_channels_per_dof=channels_per_dof,
        selected_channels_joint=channels_joint,
        n_candidates_after_preselection=len(candidate_idxs),
        correlation_threshold=args.min_correlation,
        max_features=args.max_features,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_dict = asdict(result)

    with open(args.output, "w") as f:
        json.dump(result_dict, f, indent=2, cls=NumpyEncoder)

    logger.info("Selection results written to %s", args.output)
    logger.info(
        "Per-DOF channels: %s",
        {k: len(v) for k, v in channels_per_dof.items()},
    )
    logger.info("Joint channels: %d selected", len(channels_joint))

if __name__ == "__main__":
    main()

