from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATE_PATTERN = re.compile(r"_(\d{8})T\d{6}_")


def observation_date(file_name: str) -> str | None:
    match = DATE_PATTERN.search(Path(file_name).name)
    if match is None:
        return None
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def apply_raqw_window(points: pd.DataFrame, result: pd.Series) -> pd.DataFrame:
    output = points[["s_m", "height"]].dropna().sort_values("s_m").copy()
    x = output["s_m"].to_numpy(dtype="float64")
    y = output["height"].to_numpy(dtype="float64")

    detrended = y - (float(result["raqw_detrend_slope_m_per_km"]) / 1000.0) * x
    low = float(np.quantile(detrended, float(result["raqw_tau_low"])))
    high = float(np.quantile(detrended, float(result["raqw_tau_high"])))
    retained = (detrended >= low) & (detrended <= high)

    output["raqw_detrended_height_m"] = detrended
    output["raqw_lower_bound_m"] = low
    output["raqw_upper_bound_m"] = high
    output["raqw_retained"] = retained
    return output


def export_observations(
    run_root: Path,
    reach_id: str,
    date: str,
    output_dir: Path,
) -> list[Path]:
    year = int(date[:4])
    results_path = (
        run_root / "results" / f"reach_{reach_id}" / str(year) / "raqw_results.csv"
    )
    if not results_path.exists():
        raise FileNotFoundError(f"RAQW results not found: {results_path}")

    results = pd.read_csv(results_path, dtype={"reach_id": str})
    matches = results.loc[results["file"].map(observation_date) == date].copy()
    if matches.empty:
        available = sorted(
            value for value in results["file"].map(observation_date).dropna().unique()
        )
        raise ValueError(
            f"No RAQW observation found for reach {reach_id} on {date}. "
            f"Available dates: {', '.join(available)}"
        )

    points_dir = (
        run_root / "processed" / f"reach_{reach_id}" / str(year) / "pixc_points"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for _, result in matches.sort_values("file").iterrows():
        file_stem = Path(result["file"]).stem
        points_path = points_dir / f"{file_stem}_pts.csv"
        if not points_path.exists():
            raise FileNotFoundError(f"PIXC point file not found: {points_path}")

        filtered = apply_raqw_window(pd.read_csv(points_path), result)
        expected = int(result["raqw_n_kept"])
        actual = int(filtered["raqw_retained"].sum())
        if actual != expected:
            raise RuntimeError(
                f"Retained-point check failed for {file_stem}: "
                f"expected {expected}, calculated {actual}."
            )

        filtered.insert(0, "source_file", result["file"])
        filtered.insert(0, "observation_date", date)
        filtered.insert(0, "reach_id", str(reach_id))
        output_path = output_dir / f"{file_stem}_raqw_points.csv"
        filtered.to_csv(output_path, index=False)
        written.append(output_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a saved RAQW window to the original PIXC points."
    )
    parser.add_argument("--reach-id", required=True)
    parser.add_argument("--date", required=True, help="Observation date as YYYY-MM-DD.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("data") / "raqw_reach_labels",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        pd.Timestamp(args.date)
    except ValueError as exc:
        raise ValueError("--date must be a valid date formatted as YYYY-MM-DD.") from exc

    output_dir = args.output_dir or (
        args.run_root / "exports" / f"reach_{args.reach_id}" / args.date
    )
    paths = export_observations(
        run_root=args.run_root.resolve(),
        reach_id=str(args.reach_id),
        date=args.date,
        output_dir=output_dir.resolve(),
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
