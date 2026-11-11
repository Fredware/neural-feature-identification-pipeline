"""Due-diligence checks for every feature set (MAV is the motivating case).

For each backed-up session, configuration and feature set, using the pipeline's
own loading/alignment code (decode_features.py) and its Kalman filter:

  n_sel        features selected by Gram-Schmidt (max 48)
  lag          alignment lag chosen by select_features (samples of ~33 ms)
  r_feat_max   best |r| of ANY feature column with ANY active DOF (training samples)
  r_feat_sel   median best-|r| of the SELECTED features
  r_train      decoder correlation with target on TRAINING data (mean over DOFs)
  r_test       same on TEST data
  pred_sd      SD of test predictions (near 0 = silent decoder)

How to read it for a feature that decodes poorly:
  - r_feat_max low       -> the feature itself carries little movement information
  - r_feat_max high but n_sel/r_feat_sel low -> selection problem
  - r_train high, r_test low                 -> overfitting / non-stationarity
  - r_train low despite good selected features -> decoder/scaling problem

Usage (repo root):
    python scripts/python/feature_due_diligence.py ~/sfn_session36 ~/sfn_session144
Results: structured log events on stdout + reports/stats_sfn2026/final/feature_due_diligence.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import structlog

structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.dev.ConsoleRenderer(),
])
log = structlog.get_logger(__name__)
_EPS = np.finfo(np.float64).eps
OUT_CSV = Path("reports/stats_sfn2026/final/feature_due_diligence.csv")

sys.path.insert(0, "scripts/python")
from decode_features import (load_features, load_kinematics,  # noqa: E402
                             load_selection_result, resolve_feature_orientation)
from modified_kalman_filter import ModifiedKalmanFilter  # noqa: E402

FEATS = ["NFR", "SBP-RAW", "DWT-DB4", "MAV"]


def mean_r(a, b):
    rs = [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])
          if np.std(a[:, j]) > 0 and np.std(b[:, j]) > 0]
    return float(np.mean(rs)) if rs else float("nan")


def best_abs_r(X, Y):
    """Per feature column: max |r| over DOFs. X: (n, p), Y: (n, d)."""
    Xz = (X - X.mean(0)) / (X.std(0) + _EPS)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + _EPS)
    R = (Xz.T @ Yz) / len(X)
    return np.nanmax(np.abs(R), axis=1)


rows = []
for root in map(lambda p: Path(p).expanduser(), sys.argv[1:]):
    kin_all, nip_all = load_kinematics(root / "kinematics.h5")
    for cfg in ["3DOF", "6DOF", "8DOF"]:
        for fs in FEATS:
            sel_path = root / cfg / "selected" / f"{fs}.json"
            if not sel_path.exists():
                continue
            sel = load_selection_result(sel_path)
            idx, joint = sel["kinematic_indices"], sel["selected_channels_joint"]
            tr = np.array(sel["split"]["train_mask"]); te = np.array(sel["split"]["test_mask"])
            lag = sel["alignment"]["lag_samples"]
            F, n_trim = resolve_feature_orientation(load_features(root / "features" / f"{fs}.h5"),
                                                    kin_all.shape[0])
            K = kin_all[n_trim:].copy()
            if lag:                                           # same alignment as decode_features.py
                K = np.roll(K, -lag, axis=0)
                if lag > 0: K[-lag:] = 0.0
                else: K[:-lag] = 0.0
            Y = K[:, [i - 1 for i in idx]]
            r_all = best_abs_r(F[tr], Y[tr])
            if not joint:
                row = dict(session=root.name, config=cfg, feature=fs, n_sel=0, lag=lag,
                           r_feat_max=round(float(np.nanmax(r_all)), 3), null_selection=True)
                log.warning("null_selection", **row); rows.append(row)
                continue
            X = F[:, joint]
            dec = ModifiedKalmanFilter().fit(X[tr], Y[tr])
            p_tr, p_te = dec.predict(X[tr]), dec.predict(X[te])
            row = dict(session=root.name, config=cfg, feature=fs, n_sel=len(joint), lag=lag,
                       r_feat_max=round(float(np.nanmax(r_all)), 3),
                       r_feat_sel=round(float(np.median(r_all[joint])), 3),
                       r_train=round(mean_r(p_tr, Y[tr]), 3), r_test=round(mean_r(p_te, Y[te]), 3),
                       pred_sd=round(float(np.std(p_te)), 4), null_selection=False)
            log.info("feature_checked", **row); rows.append(row)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = ["session", "config", "feature", "n_sel", "lag", "r_feat_max", "r_feat_sel",
          "r_train", "r_test", "pred_sd", "null_selection"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
log.info("results_written", path=str(OUT_CSV), n_rows=len(rows))

