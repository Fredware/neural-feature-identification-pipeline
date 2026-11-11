#!/usr/bin/env bash
# Re-run the pipeline for ONE session, back it up, score it, and draw Figs 2-3.
# Run from the repo root, inside tmux (so it survives a dropped connection):
#   tmux new -s sfn
#   source init_session_env.sh
#   bash scripts/bash/run_session_for_figs.sh 135 pipeline   # submits Slurm jobs, waits
#   bash scripts/bash/run_session_for_figs.sh 135 backup     # copy outputs + evaluate
#   bash scripts/bash/run_session_for_figs.sh 135 figs       # candidates + Fig 2 + Fig 3
# Detach from tmux with Ctrl-b then d; reattach later with: tmux attach -t sfn
set -euo pipefail
JOB=${1:?job id}; STAGE=${2:?pipeline|backup|figs}
S=/scratch/general/vast/u1424875/neural-feature-identification-pipeline
B=$HOME/sfn_session$JOB
OUT=reports/stats_sfn2026/final/session_$JOB
CFGS="3DOF 6DOF 8DOF"; FEATS="NFR SBP-RAW DWT-DB4 MAV"
# session info from the manifest (session_dir, NS5 file name)
read -r SDIR NS5 < <(python -c "
import pandas as pd; m=pd.read_csv('reports/manifest.tsv', sep='\t').set_index('job_id')
r=m.loc[$JOB]; print(r.session_dir, r.full_stream_filename)")
DAY=$(python -c "
import pandas as pd; m=pd.read_csv('reports/aggregated_decoder_metrics.csv', usecols=['job_id','days_since_implant','participant_id'])
r=m[m.job_id==$JOB].iloc[0]; print(f'{r.participant_id}, day {int(r.days_since_implant)}')")
echo "Job $JOB: $SDIR ($NS5), $DAY"

case $STAGE in
pipeline)
  TARGETS=""
  for c in $CFGS; do for f in $FEATS; do TARGETS="$TARGETS $S/$JOB/$c/decode/${f}_predictions.h5"; done; done
  snakemake -s workflow/Snakefile --profile workflow/profiles/slurm --latency-wait 30 \
    --rerun-incomplete $TARGETS
  ;;
backup)
  mkdir -p "$B/features" "$B/current_eval"
  cp "$S/$JOB/kinematics.h5" "$S/$JOB/events.h5" "$B/"
  cp "$S/$JOB"/features/*.h5 "$B/features/"
  for c in $CFGS; do
    mkdir -p "$B/$c"; cp -r "$S/$JOB/$c/decode" "$S/$JOB/$c/selected" "$B/$c/"
    for f in $FEATS; do
      python scripts/python/evaluate_decoder_predictions.py \
        --predictions "$B/$c/decode/${f}_predictions.h5" --events "$B/events.h5" \
        --burn-in 15 --output-metrics "$B/current_eval/${c}_${f}_metrics.json" > /dev/null 2>&1 \
        && echo "evaluated $c $f" || echo "EVALUATE FAILED: $c $f"
    done
  done
  du -sh "$B"
  ;;
figs)
  mkdir -p "$OUT"
  module load matlab/R2024b
  ARGS="'sess_root','$B','session_dir','$SDIR','ns5_file','$NS5','out','$OUT'"
  matlab -batch "addpath('scripts/matlab'); get_sfn_fig_002_channel_candidates($ARGS)"
  matlab -batch "addpath('scripts/matlab'); plot_sfn_fig_002_signal_flow($ARGS,'label','Session $JOB ($DAY)')"
  for c in 3DOF 8DOF; do
    python scripts/python/plot_sfn_fig_003_preds.py "$B" "$OUT" --config $c
  done
  echo "Figures in $OUT"
  ;;
*) echo "stage must be pipeline, backup or figs"; exit 1 ;;
esac

