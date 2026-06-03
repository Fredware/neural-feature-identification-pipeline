#!/usr/bin/env python3
"""Generate an interactive HTML dashboard from aggregate decode metrics.

Produces a faceted scatter plot:
  - x-axis: days since implant
  - y-axis: RMSE or correlation (toggle)
  - color: feature set (NFR, SBP-RAW, DWT-DB4, MAV)
  - row facets: participant
  - column facets: DOF config (3DOF, 6DOF, 8DOF)
  - trend lines: LOESS or linear fit per feature set
  - hover: session details (job_id, session_dir, trial metrics)
  - click: prints the trace viewer command to console

Usage:
    python plot_dashboard.py \
        --metrics reports/aggregate_metrics.csv \
        --output reports/figures/dashboard.html
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Install plotly: pip install plotly")
    raise


FEATURE_COLORS = {
    "NFR": "#E53935",
    "SBP-RAW": "#1E88E5",
    "DWT-DB4": "#43A047",
    "MAV": "#FB8C00",
}

DOF_ORDER = ["3DOF", "6DOF", "8DOF"]
METRIC_LABELS = {
    "correlation": "Pearson Correlation (r)",
    "rmse": "RMSE",
}


def aggregate_per_session(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trial metrics to per-session means.

    Groups by (job_id, feature_set, dof_config) and computes
    mean ± std across trials for each DOF, plus the session-level
    mean across all DOFs.

    Args:
        df: Tidy per-trial DataFrame from aggregate_metrics.py.

    Returns:
        DataFrame with one row per (session × feature_set × dof_config).
    """
    # Filter out null results for aggregation
    valid = df[~df["null_result"]].copy()

    session_agg = valid.groupby(
        ["job_id", "participant_id", "session_dir", "session_date",
         "days_since_implant", "feature_set", "dof_config"],
        dropna=False,
    ).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        correlation_mean=("correlation", "mean"),
        correlation_std=("correlation", "std"),
        n_trials=("trial_index", "nunique"),
        n_dofs=("dof", "nunique"),
    ).reset_index()

    # Add null results back as NaN rows
    nulls = df[df["null_result"]].drop_duplicates(
        subset=["job_id", "feature_set", "dof_config"]
    )[["job_id", "participant_id", "session_dir", "session_date",
       "days_since_implant", "feature_set", "dof_config"]].copy()
    nulls["rmse_mean"] = float("nan")
    nulls["rmse_std"] = float("nan")
    nulls["correlation_mean"] = float("nan")
    nulls["correlation_std"] = float("nan")
    nulls["n_trials"] = 0
    nulls["n_dofs"] = 0

    return pd.concat([session_agg, nulls], ignore_index=True)


def build_dashboard(
    df_sessions: pd.DataFrame,
    metric: str = "correlation",
    output_path: Path | None = None,
):
    """Build the faceted scatter plot with trend lines.

    Args:
        df_sessions: Per-session aggregated DataFrame.
        metric: "correlation" or "rmse".
        output_path: If provided, write standalone HTML.
    """
    y_col = f"{metric}_mean"
    y_err_col = f"{metric}_std"
    y_label = METRIC_LABELS.get(metric, metric)

    # Filter to valid (non-NaN) data
    plot_df = df_sessions.dropna(subset=[y_col, "days_since_implant"]).copy()

    # Order DOF configs
    plot_df["dof_config"] = pd.Categorical(
        plot_df["dof_config"], categories=DOF_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values(["dof_config", "participant_id", "days_since_implant"])

    # Build hover text
    plot_df["hover_text"] = plot_df.apply(
        lambda r: (
            f"Job: {r['job_id']}<br>"
            f"Session: {r['session_dir']}<br>"
            f"Day {r['days_since_implant']:.0f}<br>"
            f"{y_label}: {r[y_col]:.3f} ± {r[y_err_col]:.3f}<br>"
            f"Trials: {r['n_trials']}, DOFs: {r['n_dofs']}<br>"
            f"<br><i>Trace command:</i><br>"
            f"python scripts/python/plot_traces.py "
            f"--job-id {r['job_id']} "
            f"--feature-set {r['feature_set']} "
            f"--dof-config {r['dof_config']}"
        ),
        axis=1,
    )

    fig = px.scatter(
        plot_df,
        x="days_since_implant",
        y=y_col,
        error_y=y_err_col,
        color="feature_set",
        facet_row="participant_id",
        facet_col="dof_config",
        hover_data={"hover_text": True, "days_since_implant": False, y_col: False},
        color_discrete_map=FEATURE_COLORS,
        labels={
            "days_since_implant": "Days Since Implant",
            y_col: y_label,
            "feature_set": "Feature Set",
        },
        title=f"Neural Feature Comparison — {y_label} Over Time",
        category_orders={"dof_config": DOF_ORDER},
        opacity=0.6,
    )

    # Add LOESS-style trend lines per feature set per facet
    for feature_set in plot_df["feature_set"].unique():
        for pid in plot_df["participant_id"].unique():
            for dof_cfg in plot_df["dof_config"].unique():
                subset = plot_df[
                    (plot_df["feature_set"] == feature_set)
                    & (plot_df["participant_id"] == pid)
                    & (plot_df["dof_config"] == dof_cfg)
                ].sort_values("days_since_implant")

                if len(subset) < 3:
                    continue

                x = subset["days_since_implant"].values
                y = subset[y_col].values

                # Simple moving average as lightweight trend
                window = max(3, len(x) // 5)
                if len(x) >= window:
                    y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
                    x_smooth = x[(window - 1) // 2: (window - 1) // 2 + len(y_smooth)]

                    # Find facet row/col indices
                    row_idx = sorted(plot_df["participant_id"].unique()).index(pid)
                    col_idx = DOF_ORDER.index(dof_cfg) if dof_cfg in DOF_ORDER else 0
                    n_rows = plot_df["participant_id"].nunique()

                    fig.add_trace(
                        go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode="lines",
                            line=dict(
                                color=FEATURE_COLORS.get(feature_set, "gray"),
                                width=2,
                                dash="dot",
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=n_rows - row_idx,
                        col=col_idx + 1,
                    )

    fig.update_layout(
        height=350 * plot_df["participant_id"].nunique(),
        width=1200,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_traces(
        marker=dict(size=6),
        error_y=dict(thickness=0.8, width=2),
        hovertemplate="%{customdata[0]}",
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs=True)
        print(f"Dashboard saved to {output_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive decode metrics dashboard."
    )
    parser.add_argument("--metrics", type=Path, required=True,
                        help="Path to aggregate_metrics.csv.")
    parser.add_argument("--metric", type=str, default="correlation",
                        choices=["correlation", "rmse"],
                        help="Which metric to plot on the y-axis.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output HTML path. Shows interactively if not set.")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics, parse_dates=["session_date"])
    df_sessions = aggregate_per_session(df)

    print(f"Loaded {len(df)} per-trial rows, aggregated to {len(df_sessions)} sessions")

    build_dashboard(df_sessions, metric=args.metric, output_path=args.output)


if __name__ == "__main__":
    main()

