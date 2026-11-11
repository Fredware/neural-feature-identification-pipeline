"""Write a "do-nothing decoder" for session 36: same files as a real decoder,
but every prediction is 0 (hand at rest). Scoring it with the pipeline's own
evaluate script gives the intended-RMSE floor that any real decoder must beat.

Usage:  python make_null_predictions.py ~/sfn_session36
Writes: ~/sfn_session36/null/{3DOF,6DOF,8DOF}/NULL_predictions.h5
"""
from pathlib import Path
import sys

import h5py
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/sfn_session36").expanduser()
for cfg in ["3DOF", "6DOF", "8DOF"]:
    src = root / cfg / "decode" / "DWT-DB4_predictions.h5"   # template: same test set
    dst = root / "null" / cfg / "NULL_predictions.h5"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as s, h5py.File(dst, "w") as d:
        for k, v in s.attrs.items():
            d.attrs[k] = v
        for k in s.keys():
            data = s[k][()]
            d[k] = np.zeros_like(data) if k == "predictions" else data
    print("wrote", dst)

