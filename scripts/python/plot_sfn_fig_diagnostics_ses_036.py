"""Sanity check of session 36 decoder outputs (reads ~10 MB).

For every config x feature x DOF prints:
  zero%     share of prediction samples that are exactly 0
  pred_sd   how much the prediction moves at all
  r         correlation with the target over all test samples
  rmse      RMSE over all test samples
  rmse_0    RMSE of a "do-nothing" decoder that always outputs 0
  int_rmse  the pipeline's intended-trial RMSE for that DOF (current_eval json)

Usage:  python diag_session36.py ~/sfn_session36
"""
import json
from pathlib import Path
import sys

import h5py
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/sfn_session36").expanduser()
NAMES = {1: "Thumb F/E", 2: "Index F/E", 3: "Middle F/E", 4: "Ring F/E",
         5: "Little F/E", 6: "Thumb Int", 10: "Wrist Pitch", 12: "Wrist Roll"}

print(f"{'cfg':5} {'feature':8} {'DOF':12} {'zero%':>6} {'pred_sd':>8} {'r':>6} "
      f"{'rmse':>6} {'rmse_0':>6} {'int_rmse':>8}")
for cfg in ["3DOF", "6DOF", "8DOF"]:
    for feat in ["NFR", "SBP-RAW", "DWT-DB4", "MAV"]:
        with h5py.File(root / cfg / "decode" / f"{feat}_predictions.h5", "r") as f:
            ks = f["kinematic_indices"][()]
            p = f["predictions"][()]
            y = f["true_kinematics"][()]
        m = json.load(open(root / "current_eval" / f"{cfg}_{feat}_metrics.json"))
        per_dof = m.get("aggregate_per_dof", {})
        for j, k in enumerate(ks):
            pj, yj = p[:, j], y[:, j]
            zero = 100 * np.mean(pj == 0)
            r = np.corrcoef(pj, yj)[0, 1] if np.std(pj) > 0 else float("nan")
            rmse = np.sqrt(np.mean((pj - yj) ** 2))
            rmse0 = np.sqrt(np.mean(yj ** 2))
            d = per_dof.get(str(k), per_dof.get(int(k), {}))
            irmse = d.get("intended", {}).get("rmse_mean", float("nan"))
            print(f"{cfg:5} {feat:8} {NAMES.get(int(k), k):12} {zero:6.1f} "
                  f"{np.std(pj):8.4f} {r:6.2f} {rmse:6.3f} {rmse0:6.3f} {irmse:8.3f}")
    print()

