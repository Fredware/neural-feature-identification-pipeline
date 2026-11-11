#!/usr/bin/env bash
# Session 36: decode every config x feature with centering = none (must reproduce
# the backed-up predictions exactly) and rest; score with the pipeline's own
# evaluate script; print a comparison table. Run from the repo root.
set -euo pipefail
R=~/sfn_session36
PY=scripts/python
for mode in none rest; do
  for cfg in 3DOF 6DOF 8DOF; do
    for fs in NFR SBP-RAW DWT-DB4 MAV; do
      out=$R/center_$mode/$cfg/${fs}_predictions.h5
      mkdir -p $R/center_$mode/$cfg
      python $PY/decode_features.py --features $R/features/$fs.h5 \
        --kinematics $R/kinematics.h5 --selected $R/$cfg/selected/$fs.json \
        --center $mode --output-predictions $out > $R/center_$mode/$cfg/${fs}.log 2>&1 \
        || true
      [ -f $out ] || { echo "decode failed: $mode $cfg $fs (see log)"; continue; }
      python $PY/evaluate_decoder_predictions.py --predictions $out \
        --events $R/events.h5 --burn-in 15 \
        --output-metrics $R/center_$mode/${cfg}_${fs}_metrics.json > /dev/null 2>&1
    done
  done
done

python - <<'PY'
import json, os
import h5py, numpy as np
R = os.path.expanduser("~/sfn_session36")
def irmse(p):
    try: return json.load(open(p))["summary"]["intended"]["rmse_mean"]
    except Exception: return float("nan")
print(f"{'cfg':5} {'feature':8} {'original':>9} {'none':>7} {'rest':>7} {'do-nothing':>11}  same-as-original?")
for cfg in ["3DOF", "6DOF", "8DOF"]:
    null = irmse(f"{R}/null/{cfg}_NULL_metrics.json")
    for fs in ["NFR", "SBP-RAW", "DWT-DB4", "MAV"]:
        o = irmse(f"{R}/current_eval/{cfg}_{fs}_metrics.json")
        n = irmse(f"{R}/center_none/{cfg}_{fs}_metrics.json")
        r = irmse(f"{R}/center_rest/{cfg}_{fs}_metrics.json")
        try:
            a = h5py.File(f"{R}/{cfg}/decode/{fs}_predictions.h5")["predictions"][()]
            b = h5py.File(f"{R}/center_none/{cfg}/{fs}_predictions.h5")["predictions"][()]
            same = f"max|diff|={np.max(np.abs(a-b)):.1e}" if a.shape == b.shape else f"shape {a.shape} vs {b.shape}"
        except Exception as e:
            same = f"n/a ({type(e).__name__})"
        print(f"{cfg:5} {fs:8} {o:9.3f} {n:7.3f} {r:7.3f} {null:11.3f}  {same}")
    print()
PY

