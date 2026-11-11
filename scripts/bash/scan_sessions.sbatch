#!/bin/bash
#SBATCH --job-name=sfn_scan
#SBATCH --clusters=notchpeak
#SBATCH --partition=notchpeak-guest
#SBATCH --qos=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --constraint=rogersa
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=reports/stats_sfn2026/session_scan_%j.log
# Guest nodes can be preempted: the scan appends one row per session and skips
# finished ones, so just resubmit (sbatch scripts/bash/scan_sessions.sbatch).
module load matlab/R2024b
matlab -batch "addpath('scripts/matlab'); scan_sessions_raw_qc()"

