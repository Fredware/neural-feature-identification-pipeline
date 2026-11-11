#!/usr/bin/env bash
# One session: Kalman output raw (what the dataset used) vs the MKF post-processing
# (gain-threshold dead-band, then + latching filter), fixed default parameters:
# gains = 1, thresholds = 0.2, latch constant = 1.0 (chosen before seeing results).
# Scored with the pipeline's own evaluate script. Run from the repo root:
#   bash scripts/bash/run_mkf_decoder.sh ~/sfn_session144
set -euo pipefail
R=${1:-$HOME/sfn_session36}; export R
PY=scripts/python
declare -A FLAGS=( [raw]="" [gt]="--apply-gain-threshold" [gt_latch]="--apply-gain-threshold --apply-latching" )
for mode in raw gt gt_latch; do
  for cfg in 3DOF 6DOF 8DOF; do
    mkdir -p $R/mkf_$mode/$cfg/decode $R/current_eval_mkf_$mode
    for fs in NFR SBP-RAW DWT-DB4 MAV; do
      out=$R/mkf_$mode/$cfg/decode/${fs}_predictions.h5
      python $PY/decode_features.py --features $R/features/$fs.h5 \
        --kinematics $R/kinematics.h5 --selected $R/$cfg/selected/$fs.json \
        ${FLAGS[$mode]} --output-predictions $out > $R/mkf_$mode/$cfg/${fs}.log 2>&1 || true
      [ -f $out ] || { echo "decode failed: $mode $cfg $fs (see log)"; continue; }
      python $PY/evaluate_decoder_predictions.py --predictions $out \
        --events $R/events.h5 --burn-in 15 \
        --output-metrics $R/current_eval_mkf_$mode/${cfg}_${fs}_metrics.json > /dev/null 2>&1 \
        || echo "evaluate failed: $mode $cfg $fs"
    done
  done
done

python - <<'PY'
import json, os
R = os.environ["R"]
def get(mode, cfg, fs, which):
    p = f"{R}/current_eval_mkf_{mode}/{cfg}_{fs}_metrics.json" if mode else f"{R}/null/{cfg}_NULL_metrics.json"
    try: return json.load(open(p))["summary"][which]["rmse_mean"]
    except Exception: return float("nan")
print("Intended RMSE (poster metric)              | Unintended RMSE (rest of the hand)")
print(f"{'cfg':5} {'feature':8} {'raw':>6} {'gt':>6} {'gt+lat':>7} {'null':>6} | {'raw':>6} {'gt':>6} {'gt+lat':>7}")
for cfg in ["3DOF", "6DOF", "8DOF"]:
    for fs in ["NFR", "SBP-RAW", "DWT-DB4", "MAV"]:
        i = [get(m, cfg, fs, "intended") for m in ("raw", "gt", "gt_latch")]
        u = [get(m, cfg, fs, "unintended") for m in ("raw", "gt", "gt_latch")]
        n = get(None, cfg, fs, "intended")
        print(f"{cfg:5} {fs:8} {i[0]:6.3f} {i[1]:6.3f} {i[2]:7.3f} {n:6.3f} | {u[0]:6.3f} {u[1]:6.3f} {u[2]:7.3f}")
    print()
PY

