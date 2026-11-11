"""Do sliver trials reach the statistics?

A "sliver" is a trial window that contains only a few test samples because the test mask
(built per trial, then shifted by the alignment lag) spills into the neighbouring trial
(see #5). decode.build_test_trial_boundaries counts a trial if it has ANY test sample,
so each sliver becomes one extra scored trial with very few samples.

Reads reports/aggregated_decoder_metrics.csv (the model's input) and reports, per feature,
how many scored trials have fewer than --min-samples samples.

Usage (repo root):  python scripts/python/check_sliver_trials_sfn.py [--min-samples 30]
"""
import argparse
from pathlib import Path

import pandas as pd
import structlog

structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.dev.ConsoleRenderer(),
])
log = structlog.get_logger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=Path, default=Path("reports/aggregated_decoder_metrics.csv"))
ap.add_argument("--min-samples", type=int, default=30, help="~1 s at 30 Hz")
ap.add_argument("--out", type=Path, default=Path("reports/stats_sfn2026/final/sliver_trials.csv"))
a = ap.parse_args()

d = pd.read_csv(a.csv)
d = d[~d["null_result"].astype(bool)]
trials = (d.drop_duplicates(["job_id", "feature_set", "dof_config", "trial_index"])
            [["participant_id", "job_id", "feature_set", "dof_config", "trial_index",
              "original_trial_index", "n_trial_samples"]])
log.info("trial_length_quantiles", **{f"q{int(q * 100)}": float(trials["n_trial_samples"].quantile(q))
                                      for q in (0, .01, .05, .5)})
for fs, g in trials.groupby("feature_set"):
    small = g[g["n_trial_samples"] < a.min_samples]
    log.info("slivers", feature=fs, scored_trials=len(g), slivers=len(small),
             pct=round(100 * len(small) / len(g), 2), sessions_affected=small["job_id"].nunique(),
             median_sliver_samples=None if small.empty else float(small["n_trial_samples"].median()))
s144 = trials[(trials["job_id"].astype(str) == "144") & (trials["dof_config"] == "6DOF")]
log.info("session_144_6DOF_trials_per_feature", **s144.groupby("feature_set").size().to_dict())
a.out.parent.mkdir(parents=True, exist_ok=True)
trials[trials["n_trial_samples"] < a.min_samples].to_csv(a.out, index=False)
log.info("written", path=str(a.out))

