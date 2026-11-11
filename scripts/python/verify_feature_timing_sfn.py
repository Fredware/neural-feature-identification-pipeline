"""Due-diligence round 2: frame timing (SBP) and window length (MAV).

Findings from reading the extraction code (repo main, commit edf36c3):

  SBP-RAW (+frm_sbp/compute_sbp_features.m) ignores the KDF NIP timestamps. It places
  frames on a fixed grid every 990 samples (66 samples at 2 kHz) from the start of the
  NS5 segment, then pads or trims to the number of KDF frames. NFR, MAV and DWT anchor
  every frame to its NIP timestamp. If the real frame interval is not exactly 990 samples,
  or the KDF has gaps, SBP slides out of register with the kinematics as the session goes on.

  MAV (+zmh_mav/makeRollingPowerFeatures_zmh.m) sizes its buffer as 9 columns, then assigns
  ~990 samples into it, so MATLAB grows the buffer to one frame. The feature is mean |HP|
  over ~1 frame (~33 ms), not the configured 300 ms (config: mav.window_length_sec = 0.3).

This script measures, from the pipeline's own files:

  1. frames   per session: KDF frame-interval stats, gaps, and the SBP offset predicted
              by the 990-sample grid at the end of the session.
  2. drift    per session: in 6 chunks, the lag that best aligns SBP with MAV on the same
              channel (MAV is timestamp-anchored). Control: DWT level 4 (0.94-1.9 kHz) vs MAV.
              If SBP's lag changes across chunks as predicted and the control's doesn't,
              the grid bug is confirmed for that session.
  3. windows  per session x config: best |r| between feature columns and active DOFs on
              training rows, scanned over lags -12..12, for
                MAV as extracted / MAV with the 300 ms window restored (causal 9-frame mean) /
                SBP as extracted / SBP re-timed onto the NIP timestamps / NFR / DWT.
              This estimates how much each bug costs before any re-extraction.

Usage (repo root, conda env active):
    python scripts/python/verify_feature_timing_sfn.py ~/sfn_session36 ~/sfn_session144
    # frame timing for every session still on scratch:
    python scripts/python/verify_feature_timing_sfn.py /scratch/general/vast/u1424875/neural-feature-identification-pipeline/*/
Outputs: structured log events on stdout + reports/stats_sfn2026/final/feature_timing_{frames,drift,windows}.csv
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

sys.path.insert(0, "scripts/python")
from decode_features import (load_features, load_kinematics,  # noqa: E402
                             load_selection_result, resolve_feature_orientation)

_EPS = np.finfo(np.float64).eps
OUT_DIR = Path("reports/stats_sfn2026/final")
SBP_GRID = 990            # samples per SBP frame (66 at 2 kHz x 15), hard-coded in compute_sbp_features.m
MAX_LOOKBACK = 37 * 30    # samples; frame intervals above this are gaps (same cap as NFR/MAV)
MAV_WINDOW_FRAMES = 9     # floor(0.3 / 0.033), the buffer length the MAV code intended
DWT_LEVELS = 10           # channel-major: ch1 lv1..lv10, ch2 lv1..lv10, ...
DWT_REF_LEVEL = 4         # 30 kHz / 2^5..2^4 = 0.94-1.9 kHz, overlaps the SBP and MAV bands
N_CHUNKS = 6
LAG_SCAN = range(-12, 13) # select_features' lag bound
CONFIGS = ["3DOF", "6DOF", "8DOF"]


# ---------------------------------------------------------------------------- helpers
def zscore(X):
    X = np.asarray(X, dtype=np.float64)
    sd = np.nanstd(X, axis=0)
    keep = sd > _EPS
    Z = np.zeros_like(X)
    Z[:, keep] = (X[:, keep] - np.nanmean(X[:, keep], axis=0)) / sd[keep]
    return np.nan_to_num(Z), keep


def xcorr_best_lag(s, m, a, b, L):
    """Lag k in [-L, L] maximising the channel-averaged corr(s[r + k], m[r]) for r in [a, b)."""
    n, c = b - a, s.shape[1]
    seg = np.zeros((n + 2 * L, c))
    lo, hi = a - L, b + L
    src_lo, src_hi = max(lo, 0), min(hi, len(s))
    seg[src_lo - lo:src_hi - lo] = s[src_lo:src_hi]
    nfft = 1 << int(np.ceil(np.log2(2 * n + 2 * L)))
    cc = np.fft.irfft(np.fft.rfft(seg, nfft, axis=0) * np.conj(np.fft.rfft(m[a:b], nfft, axis=0)),
                      nfft, axis=0)[:2 * L + 1]
    curve = cc.sum(axis=1) / (n * c)
    j = int(np.argmax(curve))
    return j - L, float(curve[j])


def best_abs_r(X, Y):
    """Per feature column: max |r| over DOFs. Rows with any non-finite value are dropped."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    Xz = (X - X.mean(0)) / (X.std(0) + _EPS)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + _EPS)
    return np.nanmax(np.abs((Xz.T @ Yz) / len(X)), axis=1)


def roll_like_pipeline(K, lag):
    """Same kinematics shift as decode_features.py."""
    K = np.roll(K, -lag, axis=0)
    if lag > 0:
        K[-lag:] = 0.0
    elif lag < 0:
        K[:-lag] = 0.0
    return K


def causal_mean(F, w):
    """Mean of the current and previous w-1 rows (fewer at the start)."""
    c = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), F]), axis=0)
    idx = np.arange(1, len(F) + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def retime_sbp(F_sbp, nip):
    """Pick, for each KDF frame, the SBP row whose window ends nearest the frame's NIP time."""
    rows = np.rint((nip - nip[0]) / SBP_GRID).astype(int)
    out = np.full((len(nip), F_sbp.shape[1]), np.nan)
    ok = rows < len(F_sbp)   # rows past the end were trimmed away by compute_sbp_features
    out[ok] = F_sbp[rows[ok]]
    return out, int((~ok).sum())


def load_oriented(root, fs, n_kin):
    path = root / "features" / f"{fs}.h5"
    if not path.exists():
        return None, None
    return resolve_feature_orientation(load_features(path), n_kin)


# ---------------------------------------------------------------------------- checks
def check_frames(root, nip):
    d = np.diff(nip.astype(np.int64))
    span = int(nip[-1] - nip[0])
    row = dict(session=root.name, n_frames=len(nip), span_s=round(span / 30e3, 1),
               interval_mean=round(float(d.mean()), 2), interval_median=float(np.median(d)),
               interval_sd=round(float(d.std()), 1), n_gaps=int((d > MAX_LOOKBACK).sum()),
               gap_s=round(float(d[d > MAX_LOOKBACK].sum()) / 30e3, 1),
               sbp_offset_end_frames=round(span / SBP_GRID - (len(nip) - 1), 1))
    log.info("frame_timing", **row)
    return row


def check_drift(root, nip, n_kin):
    sbp, t_sbp = load_oriented(root, "SBP-RAW", n_kin)
    mav, t_mav = load_oriented(root, "MAV", n_kin)
    dwt, t_dwt = load_oriented(root, "DWT-DB4", n_kin)
    if sbp is None or mav is None:
        log.warning("drift_skipped", session=root.name, reason="SBP-RAW or MAV features missing")
        return []
    # put every feature on kinematic rows (n_trim leading rows have no feature)
    def on_kin_rows(F, t):
        out = np.full((n_kin, F.shape[1]), np.nan)
        out[t:t + len(F)] = F[:n_kin - t]
        return out
    sbp, mav = on_kin_rows(sbp, t_sbp), on_kin_rows(mav, t_mav)
    n_ch = min(sbp.shape[1], mav.shape[1])
    Zs, ks = zscore(sbp[:, :n_ch]); Zm, km = zscore(mav[:, :n_ch])
    keep = ks & km
    Zs, Zm = Zs[:, keep], Zm[:, keep]
    Zd = None
    if dwt is not None:
        dwt = on_kin_rows(dwt, t_dwt)
        lvl = dwt[:, DWT_REF_LEVEL - 1::DWT_LEVELS][:, :n_ch]
        Zd, kd = zscore(lvl)
        Zd = Zd[:, keep]

    span_pred = abs(nip[-1] - nip[0]) / SBP_GRID - (len(nip) - 1)
    L = int(min(max(30, 1.3 * abs(span_pred) + 30), 3000))
    edges = np.linspace(0, n_kin, N_CHUNKS + 1).astype(int)
    rows = []
    for c in range(N_CHUNKS):
        a, b = edges[c], edges[c + 1]
        mid = (a + b) // 2
        pred = (nip[mid] - nip[0]) / SBP_GRID - mid   # SBP row that holds frame `mid`, minus mid
        lag_sbp, r_sbp = xcorr_best_lag(Zs, Zm, a, b, L)
        lag_dwt, r_dwt = xcorr_best_lag(Zd, Zm, a, b, L) if Zd is not None else (None, None)
        row = dict(session=root.name, chunk=c, t_mid_s=round(float(nip[mid] - nip[0]) / 30e3, 1),
                   sbp_pred_offset=round(float(pred), 1), sbp_lag=int(lag_sbp), sbp_r=round(r_sbp, 3),
                   dwt_lag=lag_dwt, dwt_r=None if r_dwt is None else round(r_dwt, 3),
                   lag_window=L, n_channels=int(keep.sum()))
        log.info("sbp_drift", **row)
        rows.append(row)
    return rows


def check_windows(root, kin, nip):
    n_kin = kin.shape[0]
    variants = {}
    for fs in ["NFR", "SBP-RAW", "DWT-DB4", "MAV"]:
        F, t = load_oriented(root, fs, n_kin)
        if F is not None:
            variants[fs] = (F, t)
    if "MAV" in variants:
        F, t = variants["MAV"]
        variants["MAV-300ms"] = (causal_mean(F, MAV_WINDOW_FRAMES), t)
    if "SBP-RAW" in variants:
        F, _ = variants["SBP-RAW"]   # SBP row r = window ending 990*r samples after nip[0]
        F_rt, n_lost = retime_sbp(F, nip)
        variants["SBP-retimed"] = (F_rt, 0)
        log.info("sbp_retimed", session=root.name, frames_without_sbp=n_lost)

    rows = []
    for cfg in CONFIGS:
        sel_path = next((root / cfg / "selected" / f"{fs}.json" for fs in ["MAV", "NFR", "DWT-DB4"]
                         if (root / cfg / "selected" / f"{fs}.json").exists()), None)
        if sel_path is None:
            continue
        sel = load_selection_result(sel_path)
        idx = sel["kinematic_indices"]
        # the saved mask covers that feature's trimmed rows; put it on kinematic rows
        tr_sel = np.array(sel["split"]["train_mask"], dtype=bool)
        tr_kin = np.concatenate([np.zeros(n_kin - len(tr_sel), dtype=bool), tr_sel])
        for name, (F, t) in variants.items():
            K = kin[t:]
            tr = tr_kin[t:]
            best = (None, -1.0, None)
            for lag in LAG_SCAN:
                Y = roll_like_pipeline(K.copy(), lag)[:, [i - 1 for i in idx]]
                r = best_abs_r(F[tr], Y[tr])
                r_max = float(np.nanmax(r))
                if r_max > best[1]:
                    top10 = float(np.mean(np.sort(r[np.isfinite(r)])[-10:]))
                    best = (lag, r_max, top10)
            row = dict(session=root.name, config=cfg, variant=name, best_lag=best[0],
                       at_scan_edge=abs(best[0]) == max(LAG_SCAN),
                       r_max=round(best[1], 3), r_top10=round(best[2], 3))
            log.info("feature_window", **row)
            rows.append(row)
    return rows


def write_csv(name, rows):
    if not rows:
        return
    path = OUT_DIR / f"feature_timing_{name}.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("results_written", path=str(path), n_rows=len(rows))


def main(roots):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, drift, windows = [], [], []
    for root in roots:
        kin_path = root / "kinematics.h5"
        if not kin_path.exists():
            log.warning("session_skipped", session=root.name, reason="no kinematics.h5")
            continue
        kin, nip = load_kinematics(kin_path)
        nip = nip.astype(np.float64)
        frames.append(check_frames(root, nip))
        drift += check_drift(root, nip, kin.shape[0])
        windows += check_windows(root, kin, nip)
    write_csv("frames", frames)
    write_csv("drift", drift)
    write_csv("windows", windows)


if __name__ == "__main__":
    main([Path(p).expanduser() for p in sys.argv[1:]])

