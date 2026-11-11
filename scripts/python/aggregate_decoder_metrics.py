#!/usr/bin/env python3
"""Aggregate decode metrics from all pipeline outputs into a tidy CSV.

Walks the scratch directory tree, collects all *_metrics.json files,
joins with the manifest for session metadata (participant, implant date),
and produces a single long-format CSV suitable for statistical analysis
and visualization.

Output columns:
    job_id, participant_id, session_date, days_since_implant,
    feature_set, dof_config, trial_index, original_trial_index,
    n_trial_samples, dof, rmse, correlation, intended,
    trial_rmse_mean_all, trial_rmse_mean_intended, trial_rmse_mean_unintended,
    trial_correlation_mean_all, trial_correlation_mean_intended,
    null_result, predictions_path

Usage:
    python aggregate_metrics.py \
        --scratch-root /scratch/.../neural-feature-identification-pipeline \
        --manifest reports/categorized_manifest.tsv \
        --participant-info reports/participant_info.yaml \
        --output reports/aggregate_metrics.csv
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import re

import polars as pl
import yaml

logger = logging.getLogger(__name__)


def parse_session_date(session_dir: str) -> str | None:
    """Extract recording date from session_dir like 'P201601/20160914-103030'."""
    match = re.search(r"(\d{8})-\d{6}", session_dir)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    return None


def load_participant_info(filepath: Path) -> dict:
    """Load participant implant dates from YAML."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def collect_metrics(scratch_root: Path) -> list[dict]:
    """Walk scratch tree and collect all decode metrics JSONs."""
    records = []
    pattern = "*/*/decode/*_metrics.json"

    for metrics_path in sorted(scratch_root.glob(pattern)):
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", metrics_path, e)
            continue

        parts = metrics_path.parts
        decode_idx = parts.index("decode")
        job_id = int(parts[decode_idx - 2])
        dof_config = parts[decode_idx - 1]
        feature_set = metrics_path.stem.replace("_metrics", "")

        records.append({
            "job_id": job_id,
            "dof_config": dof_config,
            "feature_set": feature_set,
            "metrics": metrics,
            "predictions_path": str(
                metrics_path.parent / f"{feature_set}_predictions.h5"
            ),
        })

    logger.info("Collected %d metrics files", len(records))
    return records


def flatten_to_rows(
    records: list[dict],
    manifest: pl.DataFrame,
    participant_info: dict | None = None,
) -> list[dict]:
    """Flatten per-trial metrics into row dicts for Polars ingestion."""
    manifest_dict = {row["job_id"]: row for row in manifest.to_dicts()}

    implant_dates: dict[str, datetime] = {}
    if participant_info:
        for pid, info in participant_info.items():
            date_str = info.get("implant_date")
            if date_str:
                implant_dates[pid] = datetime.strptime(str(date_str), "%Y-%m-%d")

    rows = []

    for rec in records:
        job_id = rec["job_id"]
        dof_config = rec["dof_config"]
        feature_set = rec["feature_set"]
        metrics = rec["metrics"]
        predictions_path = rec["predictions_path"]

        if job_id not in manifest_dict:
            logger.warning("job_id %d not in manifest, skipping", job_id)
            continue

        job_info = manifest_dict[job_id]
        participant_id = job_info["participant_id"]
        session_dir = job_info["session_dir"]
        session_date = parse_session_date(session_dir)

        days_since_implant = None
        if session_date and participant_id in implant_dates:
            sd = datetime.strptime(session_date, "%Y-%m-%d")
            days_since_implant = (sd - implant_dates[participant_id]).days

        base = {
            "job_id": job_id,
            "participant_id": participant_id,
            "session_dir": session_dir,
            "session_date": session_date,
            "days_since_implant": days_since_implant,
            "feature_set": feature_set,
            "dof_config": dof_config,
            "predictions_path": predictions_path,
        }

        # Check for null result
        summary = metrics.get("summary", {})
        is_null = summary.get("null_result", False)

        if is_null or not metrics.get("per_trial"):
            kin_indices = metrics.get("kinematic_indices", [])
            for dof in kin_indices:
                rows.append({
                    **base,
                    "trial_index": -1,
                    "original_trial_index": -1,
                    "n_trial_samples": 0,
                    "dof": int(dof),
                    "rmse": None,
                    "d_rmse": None,
                    "correlation": None,
                    "intended": None,
                    "trial_rmse_mean_all": None,
                    "trial_rmse_mean_intended": None,
                    "trial_rmse_mean_unintended": None,
                    "trial_correlation_mean_all": None,
                    "trial_correlation_mean_intended": None,
                    "null_result": True,
                })
            continue

        # Per-trial metrics with intended/unintended tagging
        for trial in metrics["per_trial"]:
            for dof_str, dof_data in trial["dof_metrics"].items():
                # breakpoint()
                rows.append({
                    **base,
                    "trial_index": trial["trial_index"],
                    "original_trial_index": trial["original_trial_idx"],
                    "n_trial_samples": trial["n_samples"],
                    "dof": int(dof_str),
                    "rmse": dof_data["rmse"],
                    "d_rmse": dof_data["d_rmse"],
                    "correlation": dof_data["correlation"],
                    "intended": dof_data["intended"],
                    "trial_rmse_mean_all": trial.get("rmse_mean_all"),
                    "trial_rmse_mean_intended": trial.get("rmse_mean_intended"),
                    "trial_rmse_mean_unintended": trial.get("rmse_mean_unintended"),
                    "trial_d_rmse_mean_all": trial.get("d_rmse_mean_all"),
                    "trial_d_rmse_mean_intended": trial.get("d_rmse_mean_intended"),
                    "trial_d_rmse_mean_unintended": trial.get("d_rmse_mean_unintended"),
                    "trial_correlation_mean_all": trial.get("correlation_mean_all"),
                    "trial_correlation_mean_intended": trial.get("correlation_mean_intended"),
                    "trial_correlation_mean_unintended": trial.get("correlation_mean_unintended"),
                    "null_result": False,
                })

    return rows


def build_tidy_frame(rows: list[dict]) -> pl.DataFrame:
    """Build a typed Polars DataFrame from flattened row dicts."""
    schema = {
        "job_id": pl.Int64,
        "participant_id": pl.Utf8,
        "session_dir": pl.Utf8,
        "session_date": pl.Utf8,
        "days_since_implant": pl.Int64,
        "feature_set": pl.Utf8,
        "dof_config": pl.Utf8,
        "trial_index": pl.Int64,
        "original_trial_index": pl.Int64,
        "n_trial_samples": pl.Int64,
        "dof": pl.Int64,
        "rmse": pl.Float64,
        "d_rmse": pl.Float64,
        "correlation": pl.Float64,
        "intended": pl.Boolean,
        "trial_rmse_mean_all": pl.Float64,
        "trial_rmse_mean_intended": pl.Float64,
        "trial_rmse_mean_unintended": pl.Float64,
        "trial_d_rmse_mean_all": pl.Float64,
        "trial_d_rmse_mean_intended": pl.Float64,
        "trial_d_rmse_mean_unintended": pl.Float64,
        "trial_correlation_mean_all": pl.Float64,
        "trial_correlation_mean_intended": pl.Float64,
        "trial_correlation_mean_unintended": pl.Float64,
        "null_result": pl.Boolean,
        "predictions_path": pl.Utf8,
    }

    df = pl.DataFrame(rows, schema=schema)

    return df.with_columns(
        pl.col("session_date").str.to_date("%Y-%m-%d").alias("session_date")
    )


def print_summary(df: pl.DataFrame) -> None:
    """Print summary statistics."""
    n_sessions = df.select("job_id").n_unique()
    n_null = df.filter(pl.col("null_result")).select("job_id").n_unique()

    logger.info("Total rows: %d", len(df))
    logger.info("Sessions: %d", n_sessions)
    logger.info("Null results: %d / %d sessions", n_null, n_sessions)

    valid = df.filter(~pl.col("null_result"))

    # Intended metrics
    intended = valid.filter(pl.col("intended"))
    if len(intended) > 0:
        summary = (
            intended.group_by("feature_set").agg(
                pl.col("correlation").mean().alias("r_mean"),
                pl.col("correlation").std().alias("r_std"),
                pl.col("rmse").mean().alias("rmse_mean"),
                pl.col("rmse").std().alias("rmse_std"),
                pl.col("d_rmse").mean().alias("d_rmse_mean"),
                pl.col("d_rmse").std().alias("d_rmse_std"),
                pl.len().alias("n"),
            ).sort("feature_set")
        )
        logger.info("\n  Intended DOF metrics:")
        for row in summary.to_dicts():
            logger.info(
                "    %s: r=%.3f (±%.3f), RMSE=%.4f (±%.4f), d_RMSE=%.4f (±%.4f), n=%d",
                row["feature_set"],
                row["r_mean"] or 0, row["r_std"] or 0,
                row["rmse_mean"] or 0, row["rmse_std"] or 0,
                row["d_rmse_mean"] or 0, row["d_rmse_std"] or 0,
                row["n"],
            )

    # Unintended metrics
    unintended = valid.filter(~pl.col("intended"))
    if len(unintended) > 0:
        summary = (
            unintended.group_by("feature_set").agg(
                pl.col("rmse").mean().alias("rmse_mean"),
                pl.col("rmse").std().alias("rmse_std"),
                pl.col("d_rmse").mean().alias("d_rmse_mean"),
                pl.col("d_rmse").std().alias("d_rmse_std"),
                pl.len().alias("n"),
            ).sort("feature_set")
        )
        logger.info("\n  Unintended DOF metrics (crosstalk):")
        for row in summary.to_dicts():
            logger.info(
                "    %s: RMSE=%.4f (±%.4f), d_RMSE=%.4f (±%.4f), n=%d",
                row["feature_set"],
                row["rmse_mean"] or 0, row["rmse_std"] or 0,
                row["d_rmse_mean"] or 0, row["d_rmse_std"] or 0,
                row["n"],
            )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate decode metrics into tidy CSV."
    )
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--participant-info", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    manifest = pl.read_csv(args.manifest, separator="\t")

    participant_info = None
    if args.participant_info and args.participant_info.exists():
        participant_info = load_participant_info(args.participant_info)
        logger.info("Loaded participant info for %d participants", len(participant_info))

    records = collect_metrics(args.scratch_root)

    if not records:
        logger.warning("No metrics found. Check scratch_root path.")
        return

    rows = flatten_to_rows(records, manifest, participant_info)
    df = build_tidy_frame(rows)

    logger.info(
        "Flattened to %d rows: %d sessions × %d feature sets × %d dof configs",
        len(df),
        df.select("job_id").n_unique(),
        df.select("feature_set").n_unique(),
        df.select("dof_config").n_unique(),
    )

    print_summary(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.output)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()

