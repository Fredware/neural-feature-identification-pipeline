#!/usr/bin/env python3
"""Generate an interactive HTML dashboard from aggregate decode metrics.

Produces a faceted scatter plot:
  - x-axis: days since implant
  - y-axis: per-trial RMSE or correlation (mean across DOFs)
  - color: feature set (NFR, SBP-RAW, DWT-DB4, MAV)
  - row facets: participant
  - column facets: DOF config (3DOF, 6DOF, 8DOF)
  - trend lines: LOWESS or polynomial regression spanning full x range
  - dropdown: toggle intended-only, unintended-only, or all DOFs
  - hover: session details + trace viewer command

Each dot is one trial, not one session.

Usage:
    python plot_dashboard.py \
        --metrics reports/aggregate_metrics.csv \
        --metric rmse --filter intended \
        --output reports/figures/dashboard_intended.html
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("Install plotly: pip install plotly")
    raise

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    HAS_LOWESS = True
except ImportError:
    HAS_LOWESS = False


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
    "d_rmse": "Deadband RMSE (tau=0.10)",
}
FILTER_LABELS = {
    "intended": "Intended DOFs Only",
    "unintended": "Unintended DOFs Only (Crosstalk)",
    "all": "All DOFs",
}


def aggregate_per_trial(df: pd.DataFrame, intent_filter: str = "all") -> pd.DataFrame:
    """Aggregate per-DOF metrics to per-trial means.

    Each row in the output represents one trial, with RMSE and correlation
    averaged across the relevant DOFs (intended, unintended, or all).
    """
    valid = df[~df["null_result"]].copy()

    if intent_filter == "intended":
        valid = valid[valid["intended"] == True]
    elif intent_filter == "unintended":
        valid = valid[valid["intended"] == False]

    if len(valid) == 0:
        return pd.DataFrame()

    trial_agg = valid.groupby(
        ["job_id", "participant_id", "session_dir", "session_date",
         "days_since_implant", "feature_set", "dof_config", "trial_index"],
        dropna=False,
    ).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        d_rmse_mean=("d_rmse", "mean"),
        d_rmse_std=("d_rmse", "std"),
        correlation_mean=("correlation", "mean"),
        correlation_std=("correlation", "std"),
        n_dofs=("dof", "nunique"),
        n_intended=("intended", "sum"),
    ).reset_index()

    return trial_agg


def fit_trend_line(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray | None:
    """Fit a trend line over the full x range.

    Uses LOWESS if statsmodels is available, otherwise degree-2 polynomial.

    Args:
        x: Observed x values.
        y: Observed y values.
        x_grid: Evenly spaced x values spanning the full range.

    Returns:
        Smoothed y values on x_grid, or None if fitting fails.
    """
    if len(x) < 5:
        return None

    try:
        if HAS_LOWESS:
            # LOWESS: fraction = adaptive based on sample size
            frac = max(0.2, min(0.6, 15.0 / len(x)))
            smoothed = sm_lowess(y, x, frac=frac, return_sorted=True)
            # Interpolate onto the full grid
            y_grid = np.interp(x_grid, smoothed[:, 0], smoothed[:, 1])
        else:
            # Polynomial fallback
            degree = min(3, len(x) // 5)
            coeffs = np.polyfit(x, y, degree)
            y_grid = np.polyval(coeffs, x_grid)

        return y_grid

    except Exception:
        return None


def build_dashboard(
    df_trials: pd.DataFrame,
    metric: str = "correlation",
    intent_filter: str = "all",
    output_path: Path | None = None,
):
    """Build the faceted scatter plot with trend lines spanning the full x range."""
    y_col = f"{metric}_mean"
    y_label = METRIC_LABELS.get(metric, metric)
    filter_label = FILTER_LABELS.get(intent_filter, intent_filter)

    plot_df = df_trials.dropna(subset=[y_col, "days_since_implant"]).copy()

    if len(plot_df) == 0:
        print(f"No data to plot for filter={intent_filter}")
        return

    plot_df["dof_config"] = pd.Categorical(
        plot_df["dof_config"], categories=DOF_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values(
        ["dof_config", "participant_id", "days_since_implant"]
    )

    plot_df["hover_text"] = plot_df.apply(
        lambda r: (
            f"Job: {r['job_id']} | Trial: {r['trial_index']}<br>"
            f"Session: {r['session_dir']}<br>"
            f"Day {r['days_since_implant']:.0f}<br>"
            f"{y_label}: {r[y_col]:.4f}<br>"
            f"DOFs in average: {r['n_dofs']}<br>"
            f"<br><i>Trace command:</i><br>"
            f"python scripts/python/plot_kalman_predictions.py "
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
        title=f"Neural Feature Comparison — {y_label} Over Time ({filter_label})",
        category_orders={"dof_config": DOF_ORDER},
        opacity=0.3,
    )

    # Add LOWESS/polynomial trend lines spanning the full x range per facet
    participants = sorted(plot_df["participant_id"].unique())
    n_rows = len(participants)

    for feature_set in plot_df["feature_set"].unique():
        for pid in participants:
            for dof_cfg in plot_df["dof_config"].dropna().unique():
                subset = plot_df[
                    (plot_df["feature_set"] == feature_set)
                    & (plot_df["participant_id"] == pid)
                    & (plot_df["dof_config"] == dof_cfg)
                ].sort_values("days_since_implant")

                if len(subset) < 5:
                    continue

                x = subset["days_since_implant"].values.astype(float)
                y = subset[y_col].values

                # Full x range for this facet
                facet_df = plot_df[
                    (plot_df["participant_id"] == pid)
                    & (plot_df["dof_config"] == dof_cfg)
                ]
                x_min = facet_df["days_since_implant"].min()
                x_max = facet_df["days_since_implant"].max()
                x_grid = np.linspace(x_min, x_max, 200)

                y_trend = fit_trend_line(x, y, x_grid)
                if y_trend is None:
                    continue

                row_idx = participants.index(pid)
                col_idx = DOF_ORDER.index(dof_cfg) if dof_cfg in DOF_ORDER else 0

                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=y_trend,
                        mode="lines",
                        line=dict(
                            color=FEATURE_COLORS.get(feature_set, "gray"),
                            width=2.5,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=n_rows - row_idx,
                    col=col_idx + 1,
                )

    fig.update_layout(
        height=400 * n_rows,
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
        marker=dict(size=4),
        hovertemplate="%{customdata[0]}",
        selector=dict(mode="markers"),
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
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="correlation",
                        choices=["correlation", "rmse", "d_rmse"])
    parser.add_argument("--filter", type=str, default="all",
                        choices=["intended", "unintended", "all"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics, parse_dates=["session_date"])

    n_intended = df["intended"].sum() if "intended" in df.columns else 0
    n_unintended = (~df["intended"]).sum() if "intended" in df.columns else 0
    n_null = df["null_result"].sum()
    print(f"Loaded {len(df)} rows")
    print(f"Intended: {n_intended}, Unintended: {n_unintended}, Null: {n_null}")

    df_trials = aggregate_per_trial(df, intent_filter=args.filter)
    print(f"Aggregated to {len(df_trials)} trial points")

    # Diagnostic: flag suspiciously low RMSE values
    if args.metric == "rmse" and len(df_trials) > 0:
        near_zero = df_trials[df_trials["rmse_mean"] < 0.01]
        if len(near_zero) > 0:
            print(f"\nWARNING: {len(near_zero)} trials have RMSE < 0.01.")
            print("This typically means the intended DOF had near-constant")
            print("kinematics during that trial (hold phase or rest).")
            print("Inspect with:")
            for _, row in near_zero.head(5).iterrows():
                print(f"  job_id={row['job_id']}, trial={row['trial_index']}, "
                      f"feature_set={row['feature_set']}, dof_config={row['dof_config']}, "
                      f"rmse={row['rmse_mean']:.6f}")

    build_dashboard(
        df_trials,
        metric=args.metric,
        intent_filter=args.filter,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

