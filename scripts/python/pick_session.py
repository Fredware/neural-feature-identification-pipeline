"""Rank sessions for Figures 2-3 from the raw-signal scan.

Good channel: fires >= 2/s during movement and >= 1.5x its rest rate, spike
amplitude >= 5x noise, envelope not shared with the other channels (r < 0.5),
max |raw| < 2 mV, noise >= 1.5 uV (not dead). "Clean" also requires max |raw| < 300 uV.
Usage (repo root): python scripts/python/pick_session.py
"""
import glob
import pandas as pd

fs = sorted(glob.glob("reports/stats_sfn2026/session_scan_channels/job_*.csv"))
d = pd.concat([pd.read_csv(f).assign(job=int(f[-7:-4])) for f in fs])
d["ratio"] = (d.rate_move + 0.1) / (d.rate_rest + 0.1)
d["good"] = ((d.rate_move >= 2) & (d.ratio >= 1.5) & (d.amp >= 5) & (d.env_common < 0.5)
             & (d.max_uv < 2000) & (d.noise_uv >= 1.5))
d["clean"] = d.good & (d.max_uv < 300)

m = pd.read_csv("reports/aggregated_decoder_metrics.csv",
                usecols=["job_id", "participant_id", "session_dir", "days_since_implant",
                         "dof_config", "feature_set", "intended", "rmse", "null_result"])
m = m[~m.null_result]
info = m.groupby("job_id").agg(participant=("participant_id", "first"),
                               session_dir=("session_dir", "first"),
                               day=("days_since_implant", "first"),
                               configs=("dof_config", lambda x: "/".join(sorted(set(x)))))
rm = m[m.intended].groupby(["job_id", "feature_set"]).rmse.mean().unstack().round(3)

r = (d.groupby("job").agg(n_good=("good", "sum"), n_clean=("clean", "sum"))
     .join(info).join(rm))
r.to_csv("reports/stats_sfn2026/session_scan_ranking.csv")
print(r.sort_values(["n_clean", "n_good"], ascending=False).head(12).to_string())

# good channels vs time since implant, per participant (for a possible panel)
print("\nGood channels per session by participant (median, max):")
print(r.groupby("participant").n_good.agg(["median", "max", "count"]).to_string())

