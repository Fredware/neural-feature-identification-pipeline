#!/usr/bin/env python3
"""Predict SLURM resource requirements from input file sizes and historical job data.

Reads extract_features logs to get actual memory/duration, correlates with
input file sizes from the manifest, and fits a simple linear model to predict
optimal mem_mb and runtime for each job × feature_set combination.

Usage:
    # Step 1: Build the training data
    python predict_job_resources.py build \
        --manifest reports/manifest.tsv \
        --log-dir reports/logs/extract_features/ \
        --data-root /uufs/chpc.utah.edu/common/home/george-group1/data-usea \
        --output reports/resource_model.json

    # Step 2: Annotate the manifest with predictions
    python predict_job_resources.py annotate \
        --manifest reports/manifest.tsv \
        --model reports/resource_model.json \
        --output reports/manifest_annotated.tsv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 1. Parse logs for actual resource usage
# =============================================================================

MEMORY_PATTERN = re.compile(r'megabytes=(?P<mb>[\d.]+)')
DURATION_PATTERN = re.compile(
    r'Feature extraction completed.*?duration_sec=(?P<sec>[\d.]+)'
)
TOTAL_DURATION_PATTERN = re.compile(
    r'Feature extraction process finished.*?total_duration_sec=(?P<sec>[\d.]+)'
)
FILENAME_PATTERN = re.compile(r'^(?P<job_id>\d+)_(?P<feature_set>.+)\.log$')


def parse_log(filepath: Path) -> dict | None:
    """Extract resource usage from a single extract_features log."""
    text = filepath.read_text()

    memory_values = [float(m.group("mb")) for m in MEMORY_PATTERN.finditer(text)]
    duration_match = DURATION_PATTERN.search(text)
    total_match = TOTAL_DURATION_PATTERN.search(text)

    if not memory_values:
        return None

    # First 4 memory values are raw/scaled training + raw/scaled baseline
    peak_memory_mb = sum(memory_values[:4]) if len(memory_values) >= 4 else sum(memory_values)

    return {
        "peak_memory_mb": peak_memory_mb,
        "extraction_duration_sec": float(duration_match.group("sec")) if duration_match else None,
        "total_duration_sec": float(total_match.group("sec")) if total_match else None,
    }


def collect_log_data(log_dir: Path) -> pd.DataFrame:
    """Parse all logs into a DataFrame."""
    records = []
    for log_file in sorted(log_dir.glob("*.log")):
        fname_match = FILENAME_PATTERN.match(log_file.name)
        if not fname_match:
            continue

        job_id = int(fname_match.group("job_id"))
        feature_set = fname_match.group("feature_set")
        parsed = parse_log(log_file)

        if parsed is None:
            continue

        records.append({
            "job_id": job_id,
            "feature_set": feature_set,
            **parsed,
        })

    return pd.DataFrame(records)


# =============================================================================
# 2. Collect input file sizes
# =============================================================================

def collect_file_sizes(manifest: pd.DataFrame, data_root: str) -> pd.DataFrame:
    """Get NS5 and KDF file sizes for each job."""
    records = []
    data_root = Path(data_root)

    for _, row in manifest.iterrows():
        session_path = data_root / row["session_dir"]

        ns5_path = session_path / row["full_stream_filename"]
        kdf_path = session_path / row["training_filename"]
        baseline_path = session_path / row["baseline_filename"]

        records.append({
            "job_id": row["job_id"],
            "kdf_bytes": ns5_path.stat().st_size if ns5_path.exists() else None,
            "kdf_bytes": kdf_path.stat().st_size if kdf_path.exists() else None,
            "baseline_bytes": baseline_path.stat().st_size if baseline_path.exists() else None,
        })

    return pd.DataFrame(records)


# =============================================================================
# 3. Fit resource prediction model
# =============================================================================

def fit_model(df: pd.DataFrame) -> dict:
    """Fit per-feature-set linear models: resource = a * kdf_bytes + b.

    Returns a dict of model parameters and recommended safety margins.
    """
    models = {}

    for feature_set in df["feature_set"].unique():
        fs_df = df[df["feature_set"] == feature_set].dropna(
            subset=["kdf_bytes", "peak_memory_mb"]
        )

        if len(fs_df) < 3:
            continue

        x = fs_df["kdf_bytes"].values / 1e9  # GB
        y_mem = fs_df["peak_memory_mb"].values / 1024  # GB

        # Linear fit: y = a*x + b
        mem_coeffs = np.polyfit(x, y_mem, 1)
        mem_r2 = 1 - np.sum((y_mem - np.polyval(mem_coeffs, x))**2) / np.sum((y_mem - y_mem.mean())**2)

        # Duration model (use total_duration if available, else extraction)
        dur_col = "total_duration_sec" if fs_df["total_duration_sec"].notna().sum() > 3 else "extraction_duration_sec"
        dur_df = fs_df.dropna(subset=[dur_col])

        dur_coeffs = None
        dur_r2 = None
        if len(dur_df) >= 3:
            y_dur = dur_df[dur_col].values / 60  # minutes
            x_dur = dur_df["kdf_bytes"].values / 1e9
            dur_coeffs = np.polyfit(x_dur, y_dur, 1).tolist()
            dur_r2 = float(1 - np.sum((y_dur - np.polyval(dur_coeffs, x_dur))**2) / np.sum((y_dur - y_dur.mean())**2))

        # Percentiles for safety margins
        mem_p95 = float(np.percentile(y_mem, 95))
        residuals = y_mem - np.polyval(mem_coeffs, x)

        models[feature_set] = {
            "mem_coeffs_gb": mem_coeffs.tolist(),  # [slope, intercept] in GB
            "mem_r2": float(mem_r2),
            "mem_p95_gb": mem_p95,
            "mem_residual_std_gb": float(np.std(residuals)),
            "dur_coeffs_min": dur_coeffs,
            "dur_r2": dur_r2,
            "n_samples": len(fs_df),
            "safety_margin": 2,  # account for unlogged intermediate allocations
        }

    return models


def predict_resources(kdf_bytes: float, feature_set: str, model: dict) -> dict:
    """Predict mem_mb and runtime_min for a single job.

    Args:
        kdf_bytes: NS5 file size in bytes.
        feature_set: Feature set identifier.
        model: Fitted model dict from fit_model.

    Returns:
        Dict with predicted_mem_mb and predicted_runtime_min.
    """
    if feature_set not in model:
        return {"predicted_mem_mb": 80000, "predicted_runtime_min": 120}

    params = model[feature_set]
    x = kdf_bytes / 1e9  # GB
    margin = params["safety_margin"]

    # Memory prediction
    mem_gb = np.polyval(params["mem_coeffs_gb"], x)
    # Add residual std for safety, then apply margin
    mem_gb = (mem_gb + params["mem_residual_std_gb"]) * margin
    # Floor at 52 GB (MATLAB overhead), ceil at 128 GB (node limit)
    mem_mb = int(np.clip(mem_gb * 1024, 52000, 128000))
    # Round up to nearest 4 GB for clean SLURM allocation
    mem_mb = int(np.ceil(mem_mb / 4000) * 4000)

    # Duration prediction
    if params["dur_coeffs_min"] is not None:
        dur_min = np.polyval(params["dur_coeffs_min"], x) * margin
        dur_min = int(np.clip(dur_min, 5, 240))
    else:
        dur_min = 120

    return {
        "predicted_mem_mb": mem_mb,
        "predicted_runtime_min": dur_min,
    }


# =============================================================================
# 4. CLI
# =============================================================================

def cmd_build(args):
    """Build the resource prediction model from logs and file sizes."""
    manifest = pd.read_csv(args.manifest, sep="\t")
    log_data = collect_log_data(Path(args.log_dir))
    file_sizes = collect_file_sizes(manifest, args.data_root)

    # Merge
    df = log_data.merge(file_sizes, on="job_id", how="left")

    print(f"Collected {len(log_data)} log entries, {len(file_sizes)} file size records")
    print(f"Merged: {len(df)} rows with {df['kdf_bytes'].notna().sum()} valid file sizes")

    # Fit
    model = fit_model(df)

    for fs, params in model.items():
        print(f"\n{fs}:")
        print(f"  Memory model: {params['mem_coeffs_gb'][0]:.3f} * NS5_GB + {params['mem_coeffs_gb'][1]:.3f} (R²={params['mem_r2']:.3f})")
        if params["dur_coeffs_min"]:
            print(f"  Duration model: {params['dur_coeffs_min'][0]:.3f} * NS5_GB + {params['dur_coeffs_min'][1]:.3f} (R²={params['dur_r2']:.3f})")
        print(f"  Samples: {params['n_samples']}, P95 memory: {params['mem_p95_gb']:.1f} GB")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nModel saved to {output_path}")


def cmd_annotate(args):
    """Annotate manifest with predicted resource requirements."""
    manifest = pd.read_csv(args.manifest, sep="\t")

    with open(args.model) as f:
        model = json.load(f)

    file_sizes = collect_file_sizes(manifest, args.data_root)
    manifest = manifest.merge(file_sizes, on="job_id", how="left")

    feature_sets = list(model.keys())
    for fs in feature_sets:
        mem_col = f"{fs}_mem_mb"
        dur_col = f"{fs}_runtime_min"
        manifest[mem_col] = None
        manifest[dur_col] = None

        for idx, row in manifest.iterrows():
            if pd.notna(row.get("kdf_bytes")):
                pred = predict_resources(row["kdf_bytes"], fs, model)
                manifest.at[idx, mem_col] = pred["predicted_mem_mb"]
                manifest.at[idx, dur_col] = pred["predicted_runtime_min"]

    output_path = Path(args.output)
    manifest.to_csv(output_path, sep="\t", index=False)
    print(f"Annotated manifest saved to {output_path}")
    print(f"Columns added: {[c for c in manifest.columns if '_mem_mb' in c or '_runtime_min' in c]}")


def main():
    parser = argparse.ArgumentParser(description="Predict SLURM resource requirements")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Fit resource model from logs")
    build_p.add_argument("--manifest", required=True)
    build_p.add_argument("--log-dir", required=True)
    build_p.add_argument("--data-root", required=True)
    build_p.add_argument("--output", required=True)

    annot_p = sub.add_parser("annotate", help="Annotate manifest with predictions")
    annot_p.add_argument("--manifest", required=True)
    annot_p.add_argument("--model", required=True)
    annot_p.add_argument("--data-root", required=True)
    annot_p.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "annotate":
        cmd_annotate(args)


if __name__ == "__main__":
    main()

