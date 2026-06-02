"""Modified Kalman Filter for Neural Decoding

Implements a linear Kalman filter with optional gain-threshold post-processing following the implementation described in:
George, et al. (2020) "Intuitive neuromyoelectric control of a dexterous bionic arm using a modified Kalman filter," J. Neurosci. Methods

Designed as an sklearn-compatible RegressionMixin

State Model: x[t] = A @ x[t-1] + w, w ~N(0, W)
Observation: z[t] = H @ x[t]   + v, v ~N(0, Q)

Convention: axis 0 is always time. All public APIs follow (n_samples, n_dims)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Machine epsilon for float64, used as the base unit for numerical guards and regularization
_EPS = np.finfo(np.float64).eps

@dataclass
class KalmanFilterState:
    """Snapshot of filter state at a single timestamp (for diagnostics)."""
    x_posterior: NDArray
    P_posterior: NDArray
    K: NDArray

class ModifiedKalmanFilter:
    """Linear Kalman filter with optional gain-threshold post-processing

    Follows the sklearn estimator pattern: fit(X, y) then predict(X).
    X = features (observations), y = kinematics (states)

    The "modified" part refers to the per-DOF gain and threshold transformation
    applied after the Kalman update, which maps raw filter output to the [-1, 1] 
    prosthetic control range. This post-processing is optional and disabled by default 
    for offline evaluation.

    Args:
        apply_gain_threshold: Whether to apply gain-threshold post-processing after
            each Kalman update. Defaults to False for offline evaluation.
        gains: Per-DOF gains as (n_dofs, 2) array where col 0 are gains for the flexion
            direction and col 1 for the extension direction. If None, defaults to ones.
        thresholds: Per-DOF thresholds as (n_dofs, 2) array for the dead-band. If None,
            defaults to 0.2
        apply_latching: Whether to apply the latching (adaptive smoothing) filter.
            Defaults to False. 
            Described in Nieveen, et al. (2020), [Nonlinear Latching Filter], IEEE TNSRE
        latch_constant: Latching filter constant controlling smoothing intensity. Only used
            if apply_latching is True.
        store_diagnostics: Whether to store per-timestamp filter state (x, P, K) during
            predict. Useful for debugging, but memory intensive.

    Attributes:
        A_: State transition matrix, shape (n_dofs, n_dofs)
        W_: Process noise covariance, shape (n_dofs, n_dofs)
        H_: Observation model, shape (n_features, n_dofs)
        Q_: Observation noise covariance, shape(n_features, n_dofs)
        n_dofs_: Number of DOFs (state dimensions)
        n_features_: Number of features (observation dimensions)
        is_fitted_: Wheher fit() has been called
        diagnostics_: List of KalmanFilterState if store_diagnostic is True
    """
    def __init__(
        self,
        apply_gain_threshold: bool = False,
        gains: NDArray | None = None,
        thresholds: NDArray | None = None,
        apply_latching: bool = False,
        latch_constant: float = 1.0,
        store_diagnostics: bool = False,
    ):
        self.apply_gain_threshold = apply_gain_threshold
        self.gains = gains
        self.thresholds = thresholds
        self.apply_latching = apply_latching
        self.latch_constant = latch_constant
        self.store_diagnostics_ = store_diagnostics
        self.is_fitted_ = False

    def fit(self, X: NDArray, y: NDArray) -> "ModifiedKalmanFilter":
        """Fit the Kalman Filter parameters from training data.

        Estimates state transition (A) and observation model (H) via Ordinary Least Squares, 
        and process noise covariance (W) and observation noise covariance (Q) via 
        Minimum Mean Square Error estimation of the residual noise.

        Args:
            X: Features array of shape (n_samples, n_features)
            y: Kinematics array of shape (n_samples, n_dofs)

        Return:
            self
        
        Raises:
            ValueError: If X and y have different sample counts.
        """
        n_samples, n_features = X.shape
        _, n_dofs = y.shape

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}"
            )
        if n_samples <= 1:
            raise ValueError(
                f"Need at least 2 samples, got {n_samples}"
            )
        
        # Work in (n_dims, n_samples) layout for the matrix equations. Also make notation compliant with Wu et al.
        M = n_samples
        x = y.T # (n_dofs, M)
        z = X.T # (n_features, M)

        # --- State transition matrix ---
        # Governing equation:  A = x[1,:] @ x[:-1].T @ inv(x[:1]@x[:-1].T)
        x_prev = x[:, :M-1]
        x_curr = x[:, 1:M]

        A1 = x_curr @ x_prev.T
        A2 = x_prev @ x_prev.T
        A, _, A_rank, A_sv = np.linalg.lstsq(A2.T, A1.T, rcond=None)
        self.A_ = A.T

        logger.info(
            "A: shape=(%d, %d), rank=%d / %d",
            n_dofs, n_dofs, A_rank, n_dofs,
        )

        # --- Process noise matrix ---
        # Governing equation:  W = (1/(M-1)) * (x_curr @ x_curr.T - A @ x_prev @ x_curr.T)
        W1 = x_curr @ x_curr.T
        W2 = x_prev @ x_curr.T
        self.W_ = (1.0 / (M - 1)) * (W1 - self.A_ @ W2)
        self.W_ = 0.5 * (self.W_ + self.W_.T) # make symmetric
        # Regularize near-singular W (inactive DOFs)
        # Threshold: element is "near-zero" if below 100 * eps * max(|diag(W)|)
        # Regularization: add 100 * eps * max(|diag(W)|) * I to preserve the scale
        W_diag = np.diag(self.W_)
        W_scale = max(np.max(np.abs(W_diag)), _EPS)
        near_zero_thresh = 100 * _EPS * W_scale
        n_zero = np.sum(np.abs(W_diag) < near_zero_thresh)
        if n_zero > 0:
            logger.warning(
                "W has %d near-zero diagonal entries (thresh=%.2e)."
                "Adding regularization=%.2e.",
                n_zero, near_zero_thresh, near_zero_thresh,
            )
            self.W_ += np.eye(n_dofs) * near_zero_thresh

        # --- Observation matrix ---
        # Governing equation:  H = Pzx @ inv(Rxx)
        Pzx = z @ x.T
        Rxx = x @ x.T
        H, _, H_rank, _ = np.linalg.lstsq(Rxx.T, Pzx.T, rcond=None)
        self.H_ = H.T

        logger.info(
            "H: shape=(%d, %d), rank=%d/%d",
            n_features, n_dofs, H_rank, min(n_features, n_dofs),
        )

        # --- Observation Noise ---
        # Governing equation:  Q = (1/M) * (Rzz - H @ Pzx.T)
        Rzz = z @ z.T
        self.Q_ = (1.0 / M) * (Rzz - self.H_ @ Pzx.T)
        self.Q_ = 0.5 * (self.Q_ + self.Q_.T) # make symmetric
        # Regularize if negative diagonal entries.
        # Shift by |min(diag(Q))| + eps-scaled guard
        Q_diag = np.diag(self.Q_)
        n_neg = np.sum(Q_diag < 0)
        if n_neg > 0:
            Q_scale = max(np.max(np.abs(Q_diag)), _EPS)
            reg = np.abs(np.min(Q_diag)) + 100 * _EPS * Q_scale
            logger.warning(
                "Q has %d negative diagonal entries. Adding regularization=%.2e",
                n_neg, reg
            )
            self.Q_ += np.eye(n_features) * reg

        # Store dimensions
        self.n_dofs_ = n_dofs
        self.n_features_ = n_features

        # Set up default gains and thresholds
        if self.gains is None:
            self.gains_ = np.ones((n_dofs, 2))
        else:
            self.gains_ = np.array(self.gains)

        if self.thresholds is None:
            self.thresholds_ = 0.2*np.ones((n_dofs, 2))
        else:
            self.thresholds_ = np.array(self.thresholds)

        # Initialize filter state (called once at fit time)
        self._x_hat = np.zeros(n_dofs)
        self._P = np.zeros((n_dofs, n_dofs))
        self.is_fitted_ = True
        return self
    
    def predict(self, X:NDArray) -> NDArray:
        """Run the Kalman filter causally over the test features

        Resets filter state to zeros before prediction. Optionally applies gain-threshold post-processing and latching filter.

        Arga:
            X: Features array of shape (n_samples, n_features)
        
        Returns:
            Predicted kinematics of shape (n_samples, n_dofs)

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If n_features doesn't match training
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_dofs_))

        # Reset state for each predict call
        x_hat = np.zeros(self.n_dofs_)
        P = np.zeros((self.n_dofs_, self.n_dofs_))

        A = self.A_
        W = self.W_
        H = self.H_
        Q = self.Q_
        I = np.eye(self.n_dofs_)

        # Gain limits for clipping: [-1/gain_ext, 1/gain_flex]
        klim_lo = -1.0 / self.gains_[:, 1]
        klim_hi = 1.0 / self.gains_[:, 0]

        if self.store_diagnostics_:
            self.store_diagnostics_ = []
        
        x_hat_prev = np.zeros(self.n_dofs_) # for latching filter

        for t in range(n_samples):
            z_t = X[t, :]

            # --- Time update (predict) ---
            x_prior = A @ x_hat
            P_prior = A @ P @ A.T + W

            # --- Measurement update (correct) ---
            S = H @ P_prior @ H.T + Q
            K_T, _, _, _ = np.linalg.lstsq(S.T, (P_prior @ H.T).T, rcond=None)
            K = K_T.T

            innovation = z_t - H @ x_prior
            x_hat = x_prior + K @ innovation
            P = (I - K @ H) @ P_prior

            # --- Clip to gain limits ---
            x_hat = np.clip(x_hat, klim_lo, klim_hi)

            # --- Store raw filter output ---
            predictions[t, :] = x_hat

            # --- Optional gain-threshold post-processing ---
            if self.apply_gain_threshold:
                predictions[t, :] = self._apply_gain_threshold(
                    x_hat, self.gains_, self.thresholds_
                )
            
            # --- Optional latching filter ---
            if self.apply_latching:
                predictions[t, :] = self._apply_latching(
                    predictions[t, :], x_hat_prev, self.latch_constant
                )
            
            x_hat_prev = predictions[t, :].copy()

            # --- Store diagnostics ---
            if self.store_diagnostics_:
                self.store_diagnostics_.append(KalmanFilterState(
                    x_posterior=x_hat.copy(),
                    P_posterior=P.copy(),
                    K=K.copy(),
                ))

        logger.info("Predicted %d samples", n_samples)
        return predictions
    
    def get_params(self) -> dict:
        """Return fitted model parameters (sklearn convention)"""
        if not self.is_fitted_:
            return {}
        return {
            "A": self.A_,
            "W": self.W_,
            "H": self.H_,
            "Q": self.Q_,
            "n_dofs": self.n_dofs_,
            "n_features": self.n_features_,
        }
    
    @staticmethod
    def _apply_gain_threshold(
        x_hat: NDArray,
        gains: NDArray,
        thresholds: NDArray,
    )   ->  NDArray:
        """Apply per-DOF directional gain and threshold transformation

        Maps the raw Kalman output to the [-1, 1] prosthetic control range
        with per-DOF, per-direction gain and dead-zone threshold.

        The transformation for each DOF is
            flexion: y = (x * gain_flex - thresh_flex) / (1 - thresh_flex)
            extension: y = (x * gain_ext + thresh_ext) / (1 - thresh_ext)
        
        Zero-crossings induced by the transformation are corrected (set to 0),
        and the output clipped to [-1, 1]

        Args:
            x_hat: Raw fiter output of shape (n_dofs,)
            gains: Per-DOF gains of shape (n_dofs, 2). Col 0 = flex, col 1 = ext
            threholds: Per-DOF thresholds of shape (n_dofs, 2)

        Returns:
            Transformed output of shape (n_dofs,)
        """
        is_positive = x_hat >=0
        val_flex = (x_hat * gains[:, 0] - thresholds[:, 0]) / (1 - thresholds[:, 0])
        val_ext = (x_hat * gains[:, 1] + thresholds[:, 1]) / (1 - thresholds[:, 1])

        result = np.where(is_positive, val_flex, val_ext)

        # Correct zero-crossing induced by transformation
        neg_cross = (result < 0) & is_positive
        pos_cross = (result > 0) & (~is_positive)
        result = np.where(neg_cross | pos_cross, 0.0, result)

        return np.clip(result, -1.0, 1.0)
    
    @staticmethod
    def _apply_latching(
        x_hat: NDArray,
        x_hat_prev: NDArray,
        filter_constant: float,
    ) -> NDArray:
        """Apply adaptive smoothing (latching filter) to reduce jitter
        References Nieveen, et al. 2020. "A Nonlinear Latching Filter to Remove Jitter From Movement Estimates for Prostheses", IEEE TNSRE.

        The gain is proportional to the square of the state change, providing
        stronger smoothing for small movement and allowing large movements to
        pass through.

        Args:
            x_hat: Current prediction of shape (n_dofs,)
            x_hat_prev: Previous prediction, shape (n_dofs,)
            filter_constant: Controls smoothing intensity

        Returns:
            Smoothed prediction, shape (n_dofs,)
        """        
        delta = x_hat - x_hat_prev
        adaptive_gain = np.minimum(filter_constant * delta ** 2, 1.0)
        result = x_hat_prev + adaptive_gain * delta
        return np.clip(result, -1.0, 1.0)

# --- Per trial evaluation metrics ---
@dataclass
class TrialMetrics:
    """Metrics for a single trial"""
    trial_index: int
    n_samples: int
    rmse_per_dof: dict[str, float]
    correlation_per_dof: dict[str, float]
    rmse_mean: float
    correlation_mean: float

def _compute_single_trial_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    dof_indices: list[int],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute RMSE and correlation for one trial segment.

    Args:
        y_true: True kinematics for this trial, shpae (n_trial_samples, n_dofs)
        y_pred: Predicted kinematics for this trial, shape (n_trial_samples, n_dofs)
        dof_indices: 1-based DOF indices for labeling

    Returns:
        Tuple of (rmse_per_dof, corr_per_dof) dicts
    """
    rmse = {}
    corr = {}
    std_thresh = np.sqrt(_EPS)

    for i, dof_idx in enumerate(dof_indices):
        residuals = y_true[:, i] - y_pred[:, i]
        rmse[str(dof_idx)] = float(np.sqrt(np.mean(residuals**2)))

        if np.std(y_true[:, i]) < std_thresh or np.std(y_pred[:, i]) < std_thresh:
            corr[str(dof_idx)] = 0.0
        else:
            r = float(np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1])
            corr[str(dof_idx)] = r if np.isfinite(r) else 0.0
    return rmse, corr

def compute_decode_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    dof_indices: list[int],
    trial_boundaries: list[tuple[int, int]],
    burn_in: int =15, 
) ->  dict:
    """Compute per-trial, per-DOF RMSE and Pearson correlation

    Metrics are computed independently within each trial, then aggregated.
    The burn-in is applied only to the first trial (the filter needs the initial
    samples for state convergence but subsequent trials benefit from the converged state)

    Args:
        y_true: True kinematics, shape (n_samples, n_dofs)
        y_pred: Predicted kinematics, shape (n_samples, n_dofs)
        dof_indices: 1-based DOF indices for labeling results
        trial_boundaries: List of (start_idx, end_idx) tuples defining each trial's
            sample range within the test data. Indices are 0-based and end-exclusive
        burn_in: Number of leading samples to exclude from the first trial.
    
    Returns:
        Dict with per-trial metrics, per-DOF aggregates, and summary
    """
    trial_metrics: list[dict] = []
    std_thresh = np.sqrt(_EPS)

    for t_idx, (t_start, t_end) in enumerate(trial_boundaries):
        # Apply burn-in only to the first trial
        if t_idx == 0:
            t_start = min(t_start + burn_in, t_end)
        
        if t_start >= t_end:
            logger.warning(
                "Trial %d has no samples after burn-in (start=%d, end=%d). Skipping",
                t_idx, t_start, t_end,
            )
            continue

        trial_true = y_true[t_start:t_end, :]
        trial_pred = y_pred[t_start:t_end, :]

        rmse, corr = _compute_single_trial_metrics(
            trial_true, trial_pred, dof_indices
        )

        trial_metrics.append({
            "trial_index": t_idx,
            "n_samples": t_end - t_start,
            "rmse_per_dof": rmse,
            "correlation_per_dof": corr,
            "rmse_mean": float(np.mean(list(rmse.values()))),
            "correlation_mean": float(np.mean(list(corr.values()))),
        })
    
    # Aggregate across trials
    aggregate_rmse_per_dof = {}
    aggregate_corr_per_dof = {}

    for dof_str in [str(d) for d in dof_indices]:
        dof_rmses = [t["rmse_per_dof"][dof_str] for t in trial_metrics]
        dof_corrs = [t["correlation_per_dof"][dof_str] for t in trial_metrics]

        aggregate_rmse_per_dof[dof_str] = {
            "mean": float(np.mean(dof_rmses)),
            "std": float(np.std(dof_rmses)),
            "min": float(np.min(dof_rmses)),
            "max": float(np.max(dof_rmses)),
            "values": dof_rmses
        }
        aggregate_corr_per_dof[dof_str] = {
            "mean": float(np.mean(dof_corrs)),
            "std": float(np.std(dof_corrs)),
            "min": float(np.min(dof_corrs)),
            "max": float(np.max(dof_corrs)),
            "values": dof_corrs,
        }
    
    all_trial_rmses = [t["rmse_mean"] for t in trial_metrics]
    all_trial_corrs = [t["correlation_mean"] for t in trial_metrics]

    return{
        "per_trial": trial_metrics,
        "aggregate_rmse_per_dof": aggregate_rmse_per_dof,
        "aggregate_correlation_per_dof": aggregate_corr_per_dof,
        "summary":{
            "rmse_mean": float(np.mean(all_trial_rmses)),
            "rmse_std": float(np.std(all_trial_rmses)),
            "correlation_mean": float(np.mean(all_trial_corrs)),
            "correlation_std": float(np.std(all_trial_corrs)),
            "n_trials": len(trial_metrics),
            "n_burn_in_excluded": burn_in,
        },
    }

