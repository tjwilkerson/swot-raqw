from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPLIED_ROOT = PROJECT_ROOT / "data" / "pipeline_2025_comparison_results"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "chile_reaches_with_valid_discharge_run" / "processed"
DEFAULT_YEAR = 2025


def plot_trim_comparison_2025(
    pts_df: pd.DataFrame,
    row: pd.Series,
    out_path: Path,
) -> None:
    df = pts_df[["s_m", "height"]].dropna().sort_values("s_m")
    if df.empty:
        return

    x = df["s_m"].to_numpy(dtype="float64")
    y = df["height"].to_numpy(dtype="float64")
    x_km = x / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_km, y, s=10, alpha=0.18, color="#888888", label="All points")

    def _plot_window(
        tau_low: float,
        tau_high: float,
        detrend_slope_m_per_km: float,
        color: str,
        label: str,
    ) -> None:
        if not np.isfinite(tau_low) or not np.isfinite(tau_high) or not np.isfinite(detrend_slope_m_per_km):
            return
        y_detr = y - (detrend_slope_m_per_km / 1000.0) * x
        lo = float(np.quantile(y_detr, tau_low))
        hi = float(np.quantile(y_detr, tau_high))
        keep = (y_detr >= lo) & (y_detr <= hi)
        ax.scatter(x_km[keep], y[keep], s=12, alpha=0.55, color=color, label=label)

    _plot_window(
        float(row["updated_tau_low"]),
        float(row["updated_tau_high"]),
        float(row["updated_detrend_slope_m_per_km"]),
        "#1F77B4",
        "Filtered points",
    )

    reach_id = str(row["reach_id"])
    file_name = Path(str(row["file"])).name
    ax.set_title(f"Reach {reach_id} | {file_name}", fontsize=10)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("WSE (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.clear()
    plt.close(fig)
    gc.collect()


def result_csvs(applied_root: Path, year: int) -> list[Path]:
    return sorted(applied_root.glob(f"reach_*/{year}/updated_tau_window_results.csv"))


def generate_plots(
    applied_root: Path,
    processed_root: Path,
    year: int,
    overwrite: bool = False,
    max_plots: int | None = None,
) -> tuple[int, int, int]:
    written = 0
    skipped = 0
    missing_points = 0

    for result_csv in result_csvs(applied_root, year):
        reach_id = result_csv.parts[-3].replace("reach_", "", 1)
        reach_processed_dir = processed_root / f"reach_{reach_id}" / str(year)
        plots_dir = result_csv.parent / "plots"
        results = pd.read_csv(result_csv)
        if results.empty:
            continue

        for _, row in results.iterrows():
            stem = Path(str(row["file"])).stem
            out_path = plots_dir / f"{stem}_trim_compare.png"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            pts_path = reach_processed_dir / "pixc_points" / f"{stem}_pts.csv"
            if not pts_path.exists():
                missing_points += 1
                continue

            pts_df = pd.read_csv(pts_path)
            plot_trim_comparison_2025(pts_df, row, out_path)
            written += 1
            if max_plots is not None and written >= max_plots:
                return written, skipped, missing_points

    return written, skipped, missing_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2025 per-observation PIXC point/trim plots like the 2024 trim_compare plots."
    )
    parser.add_argument("--applied-root", type=Path, default=DEFAULT_APPLIED_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-plots", type=int, default=None)
    args = parser.parse_args()

    written, skipped, missing_points = generate_plots(
        applied_root=args.applied_root,
        processed_root=args.processed_root,
        year=args.year,
        overwrite=args.overwrite,
        max_plots=args.max_plots,
    )
    print(f"Wrote plots: {written}")
    print(f"Skipped existing plots: {skipped}")
    print(f"Missing point CSVs: {missing_points}")


if __name__ == "__main__":
    main()
