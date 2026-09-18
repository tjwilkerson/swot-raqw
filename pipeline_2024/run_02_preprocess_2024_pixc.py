from __future__ import annotations

import argparse
from pathlib import Path

import earthaccess

from pipeline_2024.legacy_notebook_code.pixc_quantile_chile_discharge_core import (
    ChileDischargePixcQuantileConfig,
    batch_process_reach_csv_years_parallel,
)
from pipeline_2024.paths import CHILE_REACHES_GPKG, RUN_ROOT, VALID_DISCHARGE_REACHES_CSV


def build_config(run_root: Path = RUN_ROOT) -> ChileDischargePixcQuantileConfig:
    return ChileDischargePixcQuantileConfig(
        project_root=Path(__file__).resolve().parents[1],
        reach_gpkg=CHILE_REACHES_GPKG,
        run_root=run_root,
        utm_epsg=32718,
        out_prefix="tau_trim_dynK_p95",
        use_dynamic_k=True,
        target_coverage=0.95,
        min_files_per_quantile=5,
        min_points_per_granule=50,
        min_points_after_trim=10,
        make_plots=False,
        save_outputs=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess 2024 PIXC for valid-discharge Chile reaches.")
    parser.add_argument("--reach-csv", type=Path, default=VALID_DISCHARGE_REACHES_CSV)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--backend", choices=("thread", "process"), default="thread")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--earthaccess-login", action="store_true")
    args = parser.parse_args()

    if args.earthaccess_login:
        earthaccess.login(strategy="netrc")

    cfg = build_config(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    out = batch_process_reach_csv_years_parallel(
        cfg,
        reach_csv=args.reach_csv,
        years=[args.year],
        backend=args.backend,
        max_workers=args.max_workers,
        delete_nc_after_success=True,
        delete_nc_after_failure=True,
    )

    summary_csv = args.run_root / f"preprocess_{args.year}_summary.csv"
    failures_csv = args.run_root / f"preprocess_{args.year}_failures.csv"
    out["summary_df"].to_csv(summary_csv, index=False)
    out["fail_df"].to_csv(failures_csv, index=False)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {failures_csv}")


if __name__ == "__main__":
    main()

