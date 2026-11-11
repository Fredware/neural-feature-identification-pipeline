"""Trials per session and DOF configuration, for the SfN dataset table.

Uses the pipeline's OWN functions from scripts/python/select_features.py
(load_events, _group_trials), so the grouping and split are exactly what ran:
  - trials are grouped by movement signature (signed DOFs from the movement
    matrices in events.h5); rest = empty signature, compound = > 1 DOF
  - compound trials are excluded (include_combined from config.yaml)
  - rest trials are split too when include_rest is true
  - per group: n_train = floor(n_trials * train_ratio), n_test = the rest
The split is the same for every DOF configuration of a session (grouping uses
all 12 DOFs); a configuration's "trials" are the movement trials whose DOF is
one of that configuration's DOFs. Alignment shifts trial times, not counts.

Run from the repo root (needs events.h5 per session on scratch):
    python scripts/python/count_trials_sfn.py
Writes reports/stats_sfn2026/final/table_trials_per_config.csv and _per_session.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "scripts/python")
from select_features import _group_trials, load_events  # noqa: E402  (pipeline code)

cfg = yaml.safe_load(open("config.yaml"))
S = Path(cfg["paths"]["scratch_root"])
CONFIGS = {k: v["kinematic_indices"] for k, v in cfg["dof_configs"].items()}
split_cfg = cfg["analysis"]["train_test_split"]
RATIO = split_cfg["train_ratio"]
INCLUDE_REST = split_cfg.get("include_rest", True)
INCLUDE_COMBINED = split_cfg["include_combined"]
print(f"Split parameters from config.yaml: train_ratio={RATIO}, split_type={split_cfg['split_type']}, "
      f"include_rest={INCLUDE_REST}, include_combined={INCLUDE_COMBINED}")

m = pd.read_csv("reports/aggregated_decoder_metrics.csv",
                usecols=["job_id", "participant_id", "dof_config", "original_trial_index",
                         "intended", "rmse", "correlation", "null_result"])
m = m[(~m.null_result) & m.intended & m.rmse.notna() & m.correlation.notna()]
m = m[m.groupby("participant_id").job_id.transform("nunique") > 1]
scored = m.groupby(["job_id", "dof_config"]).original_trial_index.nunique()
jobs = m.groupby("job_id").participant_id.first()
used_cfgs = m.groupby("job_id").dof_config.unique()


def sig_dofs(sig):
    """DOF numbers in a signature. Printed once below so the format can be checked."""
    return sorted({abs(int(x)) for x in sig})


rows, missing, shown = [], [], False
for job, pid in jobs.items():
    ev = S / str(job) / "events.h5"
    if not ev.exists():
        missing.append(job)
        continue
    _, _, mats = load_events(ev)
    groups = _group_trials(mats)
    if not shown:
        print(f"Example signatures (job {job}):", [g.signature for g in groups[:6]])
        shown = True
    rest = [g for g in groups if g.is_rest]
    single = [g for g in groups if not g.is_rest and not g.is_compound]
    n_rest = sum(g.n_trials for g in rest)
    n_comp = sum(g.n_trials for g in groups if g.is_compound)
    for cname, idxs in CONFIGS.items():
        if cname not in used_cfgs[job]:
            continue
        gs = [g for g in single if set(sig_dofs(g.signature)) <= set(idxs)]
        total = sum(g.n_trials for g in gs)
        n_train = sum(int(np.floor(g.n_trials * RATIO)) for g in gs)
        rows.append(dict(
            job_id=job, participant_id=pid, dof_config=cname, n_dofs=len(idxs),
            dofs_present=len({d for g in gs for d in sig_dofs(g.signature)}),
            movement_trials=total, train=n_train, test=total - n_train,
            rest_trials=n_rest if INCLUDE_REST else 0, compound_excluded=n_comp,
            scored_test_trials=scored.get((job, cname), np.nan)))

ps = pd.DataFrame(rows)
out = Path("reports/stats_sfn2026/final"); out.mkdir(parents=True, exist_ok=True)
ps.to_csv(out / "table_trials_per_session.csv", index=False)


def med_iqr(x):
    x = pd.Series(x).dropna()
    return f"{x.median():.0f} [{x.quantile(.25):.0f}-{x.quantile(.75):.0f}]"


tab = (ps.groupby(["participant_id", "dof_config"])
       .apply(lambda d: pd.Series({
           "sessions": len(d),
           "movement trials/session": med_iqr(d.movement_trials),
           "per DOF": f"{(d.movement_trials / d.n_dofs).median():.1f}",
           "train / test": f"{d.train.median():.0f} / {d.test.median():.0f}",
           "rest trials/session": med_iqr(d.rest_trials),
           "DOFs present": med_iqr(d.dofs_present),
           "scored test (metrics)": med_iqr(d.scored_test_trials),
       }), include_groups=False).reset_index())
tab.to_csv(out / "table_trials_per_config.csv", index=False)
print(tab.to_string(index=False))
if missing:
    print(f"\n{len(missing)} sessions have no events.h5 on scratch yet: {missing}")

