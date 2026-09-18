from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from pipeline_2024.train_tau_filter_2024 import (
    TauRangeConfig,
    prepare_file_band,
    select_tau_window_updated,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "chile_reaches_with_valid_discharge_run" / "processed"
DEFAULT_TRAINED_ROOT = PROJECT_ROOT / "data" / "tau_range_update_experiment"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "tau_range_apply_2024_model_sampled_2025"
DEFAULT_SELECTED_REACHES = DEFAULT_OUTPUT_ROOT / "selected_2025_demo_reaches.csv"


def load_config(config_json: Path | None, args: argparse.Namespace) -> TauRangeConfig:
    cfg = TauRangeConfig(
        processed_root=args.processed_root,
        output_root=args.output_root,
        year=args.apply_year,
    )
    if config_json and config_json.exists():
        raw = json.loads(config_json.read_text(encoding="utf-8"))
        allowed = {field.name for field in TauRangeConfig.__dataclass_fields__.values()}
        values = {key: value for key, value in raw.items() if key in allowed}
        for key in ("project_root", "processed_root", "output_root"):
            if key in values:
                values[key] = Path(values[key])
        if "reaches" in values:
            values["reaches"] = tuple(str(r) for r in values["reaches"])
        values["year"] = int(raw.get("apply_year", args.apply_year))
        cfg = TauRangeConfig(**values)

    return replace(
        cfg,
        processed_root=args.processed_root,
        output_root=args.output_root,
        year=args.apply_year,
        reaches=tuple(args.reaches) if args.reaches else cfg.reaches,
    )


def selected_reach_ids(selected_csv: Path) -> tuple[str, ...]:
    df = pd.read_csv(selected_csv, usecols=["reach_id"])
    return tuple(df["reach_id"].dropna().astype("int64").astype(str).drop_duplicates())


def list_ready_reaches(cfg: TauRangeConfig, trained_root: Path, reach_ids: tuple[str, ...]) -> tuple[str, ...]:
    ready = []
    for reach_id in reach_ids:
        apply_dir = cfg.processed_root / f"reach_{reach_id}" / str(cfg.year)
        band_csv = trained_root / f"reach_{reach_id}" / "2024" / "universal_band.csv"
        if (
            (apply_dir / "quantile_slopes_long.csv").exists()
            and (apply_dir / "pixc_points").is_dir()
            and band_csv.exists()
        ):
            ready.append(str(reach_id))
    return tuple(ready)


def run_apply_reach(cfg: TauRangeConfig, trained_root: Path, reach_id: str) -> pd.DataFrame:
    apply_dir = cfg.processed_root / f"reach_{reach_id}" / str(cfg.year)
    train_dir = trained_root / f"reach_{reach_id}" / "2024"
    out_dir = cfg.output_root / f"reach_{reach_id}" / str(cfg.year)
    out_dir.mkdir(parents=True, exist_ok=True)

    slopes_df = pd.read_csv(apply_dir / "quantile_slopes_long.csv")
    original_results = pd.read_csv(apply_dir / "tau_trim_results.csv")
    q_band = pd.read_csv(train_dir / "universal_band.csv")
    prepared = prepare_file_band(slopes_df, q_band, reach_id=reach_id)

    rows: list[dict[str, float | str | int]] = []
    for file_name, file_band in prepared.groupby("file", sort=True):
        file_band = file_band.sort_values("quantile").reset_index(drop=True)
        pts_path = apply_dir / "pixc_points" / f"{Path(file_name).stem}_pts.csv"
        if not pts_path.exists():
            continue

        pts_df = pd.read_csv(pts_path)
        updated = select_tau_window_updated(file_band, pts_df, cfg)
        original = original_results[original_results["file"] == file_name]
        original_row = original.iloc[0] if not original.empty else None

        row: dict[str, float | str | int] = {
            "reach_id": str(reach_id),
            "year": int(cfg.year),
            "file": file_name,
            "K_universal": float(file_band["k_universal"].dropna().iloc[0]),
            "updated_tau_low": updated["tau_low"],
            "updated_tau_high": updated["tau_high"],
            "updated_width": updated["width"],
            "updated_score": updated["score"],
            "updated_keep_frac": updated["keep_frac"],
            "updated_trim_resid_iqr_m": updated["trim_resid_iqr_m"],
            "updated_postfilter_slope_m_per_km": updated["ols_slope_postfilter_m_per_km"],
            "updated_delta_slope_m_per_km": updated["delta_slope_m_per_km"],
            "updated_detrend_slope_m_per_km": updated["detrend_slope_m_per_km"],
        }
        if original_row is not None:
            row.update(
                {
                    "original_tau_low": float(original_row["tau_low"]),
                    "original_tau_high": float(original_row["tau_high"]),
                    "original_keep_frac": float(original_row["n_keep"] / original_row["n_raw"]),
                    "original_postfilter_slope_m_per_km": float(original_row["ols_slope_postfilter_m_per_km"]),
                    "original_delta_slope_m_per_km": float(original_row["delta_slope_m_per_km"]),
                    "original_detrend_slope_m_per_km": float(original_row["detrend_slope_m_per_km"]),
                    "tau_low_shift": float(updated["tau_low"] - original_row["tau_low"]),
                    "tau_high_shift": float(updated["tau_high"] - original_row["tau_high"]),
                }
            )
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    result.to_csv(out_dir / "updated_tau_window_results.csv", index=False)
    return result


def existing_reach_result(cfg: TauRangeConfig, reach_id: str) -> pd.DataFrame | None:
    result_csv = cfg.output_root / f"reach_{reach_id}" / str(cfg.year) / "updated_tau_window_results.csv"
    if not result_csv.exists() or result_csv.stat().st_size == 0:
        return None
    try:
        result = pd.read_csv(result_csv)
    except Exception:
        return None
    return result if not result.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply 2024 trained tau bands to processed 2025 PIXC.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--trained-root", type=Path, default=DEFAULT_TRAINED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_REACHES)
    parser.add_argument("--config-json", type=Path, default=PROJECT_ROOT / "data" / "tau_range_apply_2024_model" / "apply_2024_to_2025_config.json")
    parser.add_argument("--apply-year", type=int, default=2025)
    parser.add_argument("--reach", dest="reaches", action="append")
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config_json, args)
    reach_ids = tuple(args.reaches) if args.reaches else selected_reach_ids(args.selected_csv)
    ready = list_ready_reaches(cfg, args.trained_root, reach_ids)
    missing = sorted(set(reach_ids) - set(ready))
    if missing:
        print(f"Skipping {len(missing)} reaches missing 2025 inputs or 2024 band: {missing}")

    per_reach = []
    errors = []
    for idx, reach_id in enumerate(ready, start=1):
        if not args.rerun_existing:
            existing = existing_reach_result(cfg, reach_id)
            if existing is not None:
                print(f"[{idx}/{len(ready)}] reusing existing {cfg.year} reach {reach_id} ({len(existing):,} rows)")
                per_reach.append(existing)
                continue

        print(f"[{idx}/{len(ready)}] applying 2024 band to {cfg.year} reach {reach_id}")
        try:
            result = run_apply_reach(cfg, args.trained_root, reach_id)
            if not result.empty:
                per_reach.append(result)
        except Exception as exc:
            errors.append({"reach_id": str(reach_id), "error": repr(exc)})
            print(f"  ERROR: {exc}")

    combined = pd.concat(per_reach, ignore_index=True) if per_reach else pd.DataFrame()
    args.output_root.mkdir(parents=True, exist_ok=True)
    combined_csv = args.output_root / "selected_reaches_applied_2024_to_2025_results.csv"
    combined.to_csv(combined_csv, index=False)
    pd.DataFrame(errors).to_csv(args.output_root / "selected_reaches_apply_2024_to_2025_errors.csv", index=False)
    with (args.output_root / "selected_reaches_apply_2024_to_2025_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    print(f"Wrote {combined_csv}")
    print(f"Rows: {len(combined):,}")


if __name__ == "__main__":
    main()
