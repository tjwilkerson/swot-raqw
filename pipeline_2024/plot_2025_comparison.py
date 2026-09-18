from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COMPARISON_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "pipeline_2025_comparison_results"
    / "riversp_comparison"
)

METHOD_COLUMNS = {
    "raw_pixc": "raw_pixc_slope_m_per_km",
    "updated_filter": "updated_postfilter_slope_m_per_km",
    "riversp": "riversp_slope_m_per_km",
}


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residuals = pd.read_csv(root / "matched_pixc_riversp_slopes_with_residuals.csv")
    method_summary = pd.read_csv(root / "overall_residual_spread_by_method.csv")
    wide = pd.read_csv(root / "reach_year_noise_reduction_raw_to_updated.csv")
    agreement_summary = pd.read_csv(root / "overall_agreement_with_riversp.csv")
    return residuals, method_summary, wide, agreement_summary


def residuals_long_frame(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_COLUMNS:
        resid_col = f"{method}_residual_from_reach_year_mean_m_per_km"
        part = residuals[["reach_id", "year", "file", resid_col]].copy()
        part = part.rename(columns={resid_col: "residual_m_per_km"})
        part["method"] = method
        rows.append(part)
    return pd.concat(rows, ignore_index=True).dropna(subset=["residual_m_per_km"])


def agreement_long_frame(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in {
        "raw_minus_riversp": "raw_minus_riversp_m_per_km",
        "updated_minus_riversp": "updated_minus_riversp_m_per_km",
    }.items():
        part = residuals[["reach_id", "year", "file", col]].copy()
        part = part.rename(columns={col: "difference_m_per_km"})
        part["comparison"] = label
        rows.append(part)
    return pd.concat(rows, ignore_index=True).dropna(subset=["difference_m_per_km"])


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def save_2025_plots(root: Path) -> list[Path]:
    residuals, method_summary, wide, _agreement_summary = load_inputs(root)
    residuals_long = residuals_long_frame(residuals)
    agreement_long = agreement_long_frame(residuals)

    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = [
        residuals_long.loc[residuals_long["method"] == method, "residual_m_per_km"].dropna()
        for method in ["raw_pixc", "updated_filter", "riversp"]
    ]
    ax.boxplot(plot_data, tick_labels=["Raw PIXC", "Updated filter", "RiverSP"], showfliers=False)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_ylabel("Slope residual from reach/year mean (m/km)")
    ax.set_title("2025 Residual Spread by Method")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "residual_spread_by_method_boxplot.png"
    fig.savefig(path, dpi=200)
    written.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    agreement_plot_data = [
        agreement_long.loc[agreement_long["comparison"] == label, "difference_m_per_km"].dropna()
        for label in ["raw_minus_riversp", "updated_minus_riversp"]
    ]
    ax.boxplot(agreement_plot_data, tick_labels=["Raw - RiverSP", "Updated - RiverSP"], showfliers=False)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_ylabel("Slope difference from RiverSP (m/km)")
    ax.set_title("2025 Agreement With RiverSP")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "agreement_with_riversp_boxplot.png"
    fig.savefig(path, dpi=200)
    written.append(path)
    plt.close(fig)

    if "mad_reduction_pct_raw_to_updated" in wide.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = clean_numeric(wide["mad_reduction_pct_raw_to_updated"])
        ax.hist(vals, bins=30, color="#2A9D8F", edgecolor="white")
        ax.axvline(0, color="0.3", linewidth=1)
        ax.set_xlabel("MAD reduction from raw to updated (%)")
        ax.set_ylabel("Reach/year count")
        ax.set_title("2025 MAD Reduction")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "mad_reduction_histogram.png"
        fig.savefig(path, dpi=200)
        written.append(path)
        plt.close(fig)

    spread_for_plot = method_summary.set_index("method").loc[
        ["raw_pixc", "updated_filter", "riversp"], ["std", "mad", "iqr"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(spread_for_plot.index))
    bar_w = 0.25
    for offset, metric in zip([-bar_w, 0, bar_w], ["std", "mad", "iqr"]):
        ax.bar(x + offset, spread_for_plot[metric], width=bar_w, label=metric.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(["Raw PIXC", "Updated filter", "RiverSP"])
    ax.set_ylabel("Residual spread (m/km)")
    ax.set_title("2025 Pooled Reach/Year Residual Spread")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "pooled_residual_spread_metrics.png"
    fig.savefig(path, dpi=200)
    written.append(path)
    plt.close(fig)

    reduction_specs = [
        ("std_reduction_pct_raw_to_updated", "STD"),
        ("rmse_reduction_pct_raw_to_updated", "RMSE"),
        ("mad_reduction_pct_raw_to_updated", "MAD"),
        ("iqr_reduction_pct_raw_to_updated", "IQR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (column, label) in zip(axes.ravel(), reduction_specs):
        vals = clean_numeric(wide[column])
        ax.hist(vals, bins=35, color="#2A9D8F", edgecolor="white", alpha=0.9)
        ax.axvline(0, color="0.25", linewidth=1)
        ax.axvline(vals.median(), color="#E76F51", linewidth=2, label=f"median {vals.median():.1f}%")
        ax.set_title(f"{label} Reduction")
        ax.set_ylabel("Reach count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Reduction from raw to updated (%)")
    fig.suptitle("2025 Reach-Level Noise Reduction", y=1.02)
    fig.tight_layout()
    path = plots_dir / "reach_level_reduction_histograms.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    written.append(path)
    plt.close(fig)

    count_rows = []
    for column, label in reduction_specs:
        vals = clean_numeric(wide[column])
        count_rows.append(
            {
                "metric": label,
                "improved": int((vals > 0).sum()),
                "worsened": int((vals < 0).sum()),
                "unchanged": int((vals == 0).sum()),
            }
        )
    counts_df = pd.DataFrame(count_rows).set_index("metric")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts_df.index, counts_df["improved"], label="Improved", color="#2A9D8F")
    ax.bar(counts_df.index, -counts_df["worsened"], label="Worsened", color="#E76F51")
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_ylabel("Reach count")
    ax.set_title("2025 Reach-Level Outcomes")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "reach_level_improved_worsened_counts.png"
    fig.savefig(path, dpi=200)
    written.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, title in zip(axes, ["mad", "iqr"], ["MAD", "IQR"]):
        raw_col = f"{metric}_raw_pixc"
        upd_col = f"{metric}_updated_filter"
        plot_df = wide[[raw_col, upd_col]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(plot_df[raw_col], plot_df[upd_col], s=24, alpha=0.7, color="#457B9D", edgecolors="none")
        lim = np.nanmax([plot_df[raw_col].max(), plot_df[upd_col].max()])
        ax.plot([0, lim], [0, lim], color="0.25", linewidth=1, linestyle="--")
        ax.set_xlabel(f"Raw PIXC {title} (m/km)")
        ax.set_ylabel(f"Updated filter {title} (m/km)")
        ax.set_title(f"2025 Reach-Level {title}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "raw_vs_updated_robust_spread_scatter.png"
    fig.savefig(path, dpi=200)
    written.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label, color in [
        ("raw_pixc", "Raw PIXC", "#6C757D"),
        ("updated_filter", "Updated filter", "#2A9D8F"),
        ("riversp", "RiverSP", "#E76F51"),
    ]:
        vals = (
            residuals_long.loc[residuals_long["method"] == method, "residual_m_per_km"]
            .abs()
            .dropna()
            .sort_values()
            .to_numpy()
        )
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        ax.plot(vals, y, label=label, color=color, linewidth=2)
    ax.set_xlim(0, residuals_long["residual_m_per_km"].abs().quantile(0.98))
    ax.set_xlabel("Absolute residual from reach/year mean (m/km)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("2025 Absolute Residual Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "absolute_residual_cdf_by_method.png"
    fig.savefig(path, dpi=200)
    written.append(path)
    plt.close(fig)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate 2025 comparison plots from cached CSV outputs.")
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    args = parser.parse_args()

    written = save_2025_plots(args.comparison_root)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

