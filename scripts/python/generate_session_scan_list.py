"""List decoded sessions (job_id = row in the curated manifest) for the raw-signal scan.

Usage (repo root):  python make_session_scan_list.py
Writes:             reports/stats_sfn2026/session_scan_list.csv
"""
import csv
from pathlib import Path
import sqlite3

import pandas as pd

DB = "data/raw/curated_file_list_jag.db"
OUT = Path("reports/stats_sfn2026/session_scan_list.csv")

rows = sqlite3.connect(DB).execute(
    "select rowid, participant_id, session_dir, training_filename from files order by rowid"
).fetchall()
m = pd.read_csv("reports/aggregated_decoder_metrics.csv",
                usecols=["job_id", "feature_set", "intended", "rmse", "null_result"])
m = m[(~m.null_result) & m.intended]
perf = m.groupby(["job_id", "feature_set"]).rmse.mean().unstack()   # mean intended RMSE

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["job_id", "participant", "session_dir", "training_file",
                "rmse_NFR", "rmse_SBP", "rmse_DWT", "rmse_MAV"])
    for rid, pid, sdir, tfile in rows:
        if rid not in perf.index or pid == "P202201":
            continue
        p = perf.loc[rid]
        w.writerow([rid, pid, sdir.strip("\\").replace("\\", "/"), tfile,
                    *(round(p.get(k, float("nan")), 4) for k in ["NFR", "SBP-RAW", "DWT-DB4", "MAV"])])
print("wrote", OUT, "with", sum(1 for _ in open(OUT)) - 1, "sessions")

