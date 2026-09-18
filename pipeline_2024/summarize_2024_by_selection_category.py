from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ROOT = PROJECT_ROOT / "data" / "riversp_comparison_updated_filter"
EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "tau_range_update_experiment"
OUT_DIR = COMPARISON_ROOT / "category_summaries"


def robust_mad(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def iqr(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))


def spread(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    if arr.size == 0:
        return pd.Series({"n": 0, "mean": np.nan, "std": np.nan, "rmse": np.nan, "mad": np.nan, "iqr": np.nan})
    return pd.Series(
        {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
            "rmse": float(np.sqrt(np.mean(arr**2))),
            "mad": robust_mad(pd.Series(arr)),
            "iqr": iqr(pd.Series(arr)),
        }
    )


def has_2024_band(reach_id: str) -> bool:
    return (EXPERIMENT_ROOT / f"reach_{reach_id}" / "2024" / "universal_band.csv").exists()


def build_category_table() -> pd.DataFrame:
    agreement = pd.read_csv(COMPARISON_ROOT / "reach_year_agreement_with_riversp.csv")
    noise = pd.read_csv(COMPARISON_ROOT / "reach_year_noise_reduction_raw_to_updated.csv")
    lengths = pd.read_csv(COMPARISON_ROOT / "reach_lengths.csv")

    agreement_wide = agreement.pivot(
        index=["reach_id", "year"],
        columns="comparison",
        values=["n", "std", "rmse", "mad", "iqr"],
    )
    agreement_wide.columns = [f"{comparison}_{metric}" for metric, comparison in agreement_wide.columns]
    agreement_wide = agreement_wide.reset_index()

    df = agreement_wide.merge(noise, on=["reach_id", "year"], how="outer").merge(lengths, on="reach_id", how="left")
    df["reach_id"] = df["reach_id"].astype("int64").astype(str)
    df["has_2024_band"] = df["reach_id"].map(has_2024_band)

    raw = pd.to_numeric(df["raw_minus_riversp_rmse"], errors="coerce")
    updated = pd.to_numeric(df["updated_minus_riversp_rmse"], errors="coerce")
    df["agreement_rmse_reduction_frac"] = (raw - updated) / raw
    df["agreement_rmse_reduction_pct"] = 100.0 * df["agreement_rmse_reduction_frac"]
    df["noise_rmse_reduction_frac"] = pd.to_numeric(df["rmse_reduction_frac_raw_to_updated"], errors="coerce")
    df["noise_rmse_reduction_pct"] = 100.0 * df["noise_rmse_reduction_frac"]

    df["eligible_for_2025_selection_logic"] = (
        df["has_2024_band"]
        & (pd.to_numeric(df["updated_minus_riversp_n"], errors="coerce") >= 12)
        & raw.notna()
        & updated.notna()
        & (raw > 0)
        & df["noise_rmse_reduction_frac"].notna()
    )
    df["classifiable_by_improvement_logic"] = (
        df["has_2024_band"]
        & raw.notna()
        & updated.notna()
        & (raw > 0)
        & df["noise_rmse_reduction_frac"].notna()
    )
    df["fewer_than_12_matches"] = pd.to_numeric(df["updated_minus_riversp_n"], errors="coerce") < 12

    conditions = [
        df["classifiable_by_improvement_logic"]
        & (df["agreement_rmse_reduction_frac"] >= 0.75)
        & (df["noise_rmse_reduction_frac"] > 0),
        df["classifiable_by_improvement_logic"]
        & (df["agreement_rmse_reduction_frac"] >= 0.25)
        & (df["noise_rmse_reduction_frac"] > 0),
        df["classifiable_by_improvement_logic"]
        & (df["agreement_rmse_reduction_frac"] > 0)
        & (df["noise_rmse_reduction_frac"] > 0),
        df["classifiable_by_improvement_logic"],
    ]
    choices = ["strong_both", "moderate_both", "small_both", "no_both_improvement"]
    df["selection_category"] = np.select(conditions, choices, default="unclassifiable")

    order = {
        "strong_both": 0,
        "moderate_both": 1,
        "small_both": 2,
        "no_both_improvement": 3,
        "unclassifiable": 4,
    }
    df["selection_category_order"] = df["selection_category"].map(order)
    return df.sort_values(["selection_category_order", "reach_id", "year"]).reset_index(drop=True)


def build_pooled_residual_spread(category_table: pd.DataFrame) -> pd.DataFrame:
    residuals = pd.read_csv(COMPARISON_ROOT / "matched_pixc_riversp_slopes_with_residuals.csv")
    residuals["reach_id"] = residuals["reach_id"].astype("int64").astype(str)
    category_cols = [
        "reach_id",
        "year",
        "selection_category",
        "eligible_for_2025_selection_logic",
        "fewer_than_12_matches",
    ]
    residuals = residuals.merge(category_table[category_cols], on=["reach_id", "year"], how="left")

    method_columns = {
        "raw_pixc": "raw_pixc_residual_from_reach_year_mean_m_per_km",
        "updated_filter": "updated_filter_residual_from_reach_year_mean_m_per_km",
        "riversp": "riversp_residual_from_reach_year_mean_m_per_km",
        "raw_minus_riversp": "raw_minus_riversp_m_per_km",
        "updated_minus_riversp": "updated_minus_riversp_m_per_km",
    }

    rows = []
    for (category, year), part in residuals.groupby(["selection_category", "year"], dropna=False):
        for method, column in method_columns.items():
            stats = spread(part[column])
            stats["selection_category"] = category
            stats["year"] = int(year)
            stats["method"] = method
            stats["n_reaches"] = int(part.loc[part[column].notna(), "reach_id"].nunique())
            rows.append(stats)

    out = pd.DataFrame(rows)
    return out[
        ["selection_category", "year", "method", "n_reaches", "n", "mean", "std", "rmse", "mad", "iqr"]
    ].sort_values(["selection_category", "method"])


def build_reach_year_method_spread(category_table: pd.DataFrame) -> pd.DataFrame:
    spread_by_reach = pd.read_csv(COMPARISON_ROOT / "reach_year_residual_spread_by_method.csv")
    spread_by_reach["reach_id"] = spread_by_reach["reach_id"].astype("int64").astype(str)
    out = spread_by_reach.merge(
        category_table[
            [
                "reach_id",
                "year",
                "selection_category",
                "eligible_for_2025_selection_logic",
                "fewer_than_12_matches",
            ]
        ],
        on=["reach_id", "year"],
        how="left",
    )
    return out.sort_values(["selection_category", "reach_id", "method"]).reset_index(drop=True)


def summarize_reach_year_spread(reach_year_method: pd.DataFrame) -> pd.DataFrame:
    metrics = ["std", "rmse", "mad", "iqr"]
    rows = []
    for (category, year, method), part in reach_year_method.groupby(["selection_category", "year", "method"], dropna=False):
        row = {
            "selection_category": category,
            "year": int(year),
            "method": method,
            "n_reaches": int(part["reach_id"].nunique()),
            "n_observations": int(part["n"].sum()),
        }
        for metric in metrics:
            vals = pd.to_numeric(part[metric], errors="coerce")
            row[f"mean_reach_{metric}"] = float(vals.mean())
            row[f"median_reach_{metric}"] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["selection_category", "method"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    category_table = build_category_table()
    category_table.to_csv(OUT_DIR / "reach_year_selection_categories_2024.csv", index=False)

    category_counts = (
        category_table.groupby(["selection_category", "year"], dropna=False)
        .agg(
            n_reaches=("reach_id", "nunique"),
            total_riversp_matches=("n_riversp", "sum"),
            median_agreement_rmse_reduction_pct=("agreement_rmse_reduction_pct", "median"),
            median_noise_rmse_reduction_pct=("noise_rmse_reduction_pct", "median"),
        )
        .reset_index()
    )
    category_counts.to_csv(OUT_DIR / "category_counts_2024.csv", index=False)

    pooled = build_pooled_residual_spread(category_table)
    pooled.to_csv(OUT_DIR / "pooled_residual_spread_by_category_method_2024.csv", index=False)

    reach_year_method = build_reach_year_method_spread(category_table)
    reach_year_method.to_csv(OUT_DIR / "reach_year_residual_spread_by_category_method_2024.csv", index=False)

    reach_year_summary = summarize_reach_year_spread(reach_year_method)
    reach_year_summary.to_csv(OUT_DIR / "reach_year_spread_summary_by_category_method_2024.csv", index=False)

    print(f"Wrote {OUT_DIR / 'reach_year_selection_categories_2024.csv'}")
    print(f"Wrote {OUT_DIR / 'category_counts_2024.csv'}")
    print(f"Wrote {OUT_DIR / 'pooled_residual_spread_by_category_method_2024.csv'}")
    print(f"Wrote {OUT_DIR / 'reach_year_residual_spread_by_category_method_2024.csv'}")
    print(f"Wrote {OUT_DIR / 'reach_year_spread_summary_by_category_method_2024.csv'}")
    print()
    print(category_counts.to_string(index=False))


if __name__ == "__main__":
    main()
