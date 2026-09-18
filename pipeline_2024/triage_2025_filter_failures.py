from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ROOT = PROJECT_ROOT / "data" / "pipeline_2025_comparison_results" / "riversp_comparison"
SELECTED_CSV = PROJECT_ROOT / "data" / "tau_range_apply_2024_model_sampled_2025" / "selected_2025_demo_reaches.csv"
OUT_CSV = COMPARISON_ROOT / "2025_filter_failure_candidates.csv"


def main() -> None:
    wide = pd.read_csv(COMPARISON_ROOT / "reach_year_noise_reduction_raw_to_updated.csv")
    agreement = pd.read_csv(COMPARISON_ROOT / "reach_year_agreement_with_riversp.csv")
    selected = pd.read_csv(SELECTED_CSV)[
        ["reach_id", "selection_category", "agreement_rmse_reduction_pct", "noise_rmse_reduction_pct"]
    ]

    agreement_wide = agreement.pivot(
        index=["reach_id", "year"],
        columns="comparison",
        values=["n", "rmse", "mad", "iqr", "std"],
    )
    agreement_wide.columns = [f"{comparison}_{metric}" for metric, comparison in agreement_wide.columns]
    agreement_wide = agreement_wide.reset_index()

    df = wide.merge(agreement_wide, on=["reach_id", "year"], how="left").merge(selected, on="reach_id", how="left")

    metrics = ["std", "rmse", "mad", "iqr"]
    for metric in metrics:
        df[f"updated_worse_than_riversp_{metric}"] = df[f"{metric}_updated_filter"] > df[f"{metric}_riversp"]
        df[f"updated_vs_riversp_{metric}_ratio"] = df[f"{metric}_updated_filter"] / df[f"{metric}_riversp"]

    worse_cols = [f"updated_worse_than_riversp_{metric}" for metric in metrics]
    df["worse_than_riversp_metric_count"] = df[worse_cols].sum(axis=1)
    df["tail_worse_than_riversp"] = df["updated_worse_than_riversp_std"] & df["updated_worse_than_riversp_rmse"]
    df["robust_worse_than_riversp"] = df["updated_worse_than_riversp_mad"] & df["updated_worse_than_riversp_iqr"]

    df["candidate_failure_tier"] = np.select(
        [
            df["worse_than_riversp_metric_count"] >= 3,
            df["tail_worse_than_riversp"],
            df["updated_minus_riversp_rmse"] > 1.0,
        ],
        [
            "consistent_worse_than_riversp",
            "tail_worse_than_riversp",
            "poor_agreement_with_riversp",
        ],
        default="not_flagged",
    )

    cols = [
        "reach_id",
        "year",
        "selection_category",
        "n_updated_filter",
        "n_riversp",
        "candidate_failure_tier",
        "worse_than_riversp_metric_count",
        "std_updated_filter",
        "std_riversp",
        "rmse_updated_filter",
        "rmse_riversp",
        "mad_updated_filter",
        "mad_riversp",
        "iqr_updated_filter",
        "iqr_riversp",
        "updated_minus_riversp_rmse",
        "updated_minus_riversp_mad",
        "updated_minus_riversp_iqr",
        "agreement_rmse_reduction_pct",
        "noise_rmse_reduction_pct",
    ]
    tier_order = {
        "consistent_worse_than_riversp": 0,
        "tail_worse_than_riversp": 1,
        "poor_agreement_with_riversp": 2,
        "not_flagged": 3,
    }
    out = df[cols].copy()
    out["tier_order"] = out["candidate_failure_tier"].map(tier_order)
    out = out.sort_values(
        ["tier_order", "worse_than_riversp_metric_count", "updated_minus_riversp_rmse"],
        ascending=[True, False, False],
    ).drop(columns=["tier_order"])

    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()

