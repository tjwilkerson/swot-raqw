from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ROOT = PROJECT_ROOT / "data" / "riversp_comparison_updated_filter"
CATEGORY_ROOT = COMPARISON_ROOT / "category_summaries"
PLOTS_DIR = COMPARISON_ROOT / "plots"

CATEGORIES = [
    ("strong_both", "Strong both"),
    ("moderate_both", "Moderate both"),
    ("small_both", "Small both"),
    ("no_both_improvement", "No both improvement"),
]
METHODS = [
    ("raw_pixc", "Raw PIXC", "raw_pixc_residual_from_reach_year_mean_m_per_km", "#6C757D"),
    ("updated_filter", "Updated filter", "updated_filter_residual_from_reach_year_mean_m_per_km", "#2A9D8F"),
    ("riversp", "RiverSP", "riversp_residual_from_reach_year_mean_m_per_km", "#E76F51"),
]


def load_plot_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals = pd.read_csv(COMPARISON_ROOT / "matched_pixc_riversp_slopes_with_residuals.csv")
    categories = pd.read_csv(CATEGORY_ROOT / "reach_year_selection_categories_2024.csv")

    residuals["reach_id"] = residuals["reach_id"].astype("int64").astype(str)
    categories["reach_id"] = categories["reach_id"].astype("int64").astype(str)

    category_cols = ["reach_id", "year", "selection_category"]
    residuals = residuals.merge(categories[category_cols], on=["reach_id", "year"], how="left")
    residuals = residuals[residuals["selection_category"].isin([cat for cat, _ in CATEGORIES])].copy()
    return residuals, categories


def make_boxplot() -> Path:
    residuals, categories = load_plot_data()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    method_offsets = np.linspace(-0.22, 0.22, len(METHODS))
    box_width = 0.17
    category_positions = np.arange(1, len(CATEGORIES) + 1)

    for method_idx, (_method, _method_label, column, color) in enumerate(METHODS):
        data = []
        positions = []
        for cat_idx, (category, _category_label) in enumerate(CATEGORIES):
            vals = pd.to_numeric(
                residuals.loc[residuals["selection_category"] == category, column],
                errors="coerce",
            ).dropna()
            data.append(vals)
            positions.append(category_positions[cat_idx] + method_offsets[method_idx])

        box = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor("0.2")
            patch.set_linewidth(1.0)
        for median in box["medians"]:
            median.set_color("0.05")
            median.set_linewidth(1.4)
        for whisker in box["whiskers"]:
            whisker.set_color("0.25")
        for cap in box["caps"]:
            cap.set_color("0.25")

    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xticks(category_positions)
    category_counts = categories.groupby("selection_category")["reach_id"].nunique().to_dict()
    ax.set_xticklabels(
        [f"{label}\n(n={category_counts.get(category, 0):,})" for category, label in CATEGORIES]
    )
    ax.set_ylabel("Slope residual from reach/year mean (m/km)")
    ax.set_title("2024 Residual Spread by Method and Reach Category")
    ax.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        Patch(facecolor=color, edgecolor="0.2", alpha=0.72, label=label)
        for _method, label, _column, color in METHODS
    ]
    ax.legend(handles=legend_handles, loc="best", framealpha=0.94)

    fig.tight_layout()
    out_png = PLOTS_DIR / "residual_spread_by_method_and_category_boxplot.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return out_png


def main() -> None:
    out_png = make_boxplot()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
