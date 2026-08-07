"""Shared stage orchestration used by both the CLI and graphical interface."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from .config import RAQWConfig
from .filter import FilterProgressCallback, run_filter
from .geometry import geometry_provenance_path
from .preprocess import preprocess_reach_year
from .provenance import write_manifest


StageCallback = Callable[[str, str], None]


def _notify(callback: StageCallback | None, stage: str, message: str) -> None:
    if callback is not None:
        callback(stage, message)


def preprocess_stage(
    cfg: RAQWConfig,
    *,
    login: bool = True,
    nc_paths: list[Path] | None = None,
    callback: StageCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _notify(callback, "preprocess", "Resolving reach geometry and PIXC inputs")
    summary, slopes = preprocess_reach_year(cfg, login=login, nc_paths=nc_paths)
    raw_inputs = nc_paths if nc_paths is not None else sorted(cfg.raw_dir.glob("*.nc"))
    geometry_input = geometry_provenance_path(cfg)
    manifest_inputs = ([geometry_input] if geometry_input is not None else []) + raw_inputs
    write_manifest(
        manifest_inputs,
        cfg.processed_dir / "preprocessing_input_manifest.json",
        root=Path.cwd(),
        metadata={"reach_id": cfg.reach_id, "year": cfg.year, "stage": "preprocess"},
    )
    _notify(
        callback,
        "preprocess",
        f"Accepted {slopes['file'].nunique():,} overpasses from {len(summary):,} candidates",
    )
    return summary, slopes


def fit_stage(
    cfg: RAQWConfig,
    *,
    make_plots: bool = False,
    reference_path: Path | None = None,
    callback: StageCallback | None = None,
    progress_callback: FilterProgressCallback | None = None,
) -> pd.DataFrame:
    _notify(callback, "filter", "Fitting the reference and selecting quantile windows")
    results = run_filter(
        cfg,
        make_plots=make_plots,
        reference_path=reference_path,
        progress_callback=progress_callback,
    )
    slopes_path = cfg.processed_dir / "quantile_slopes_long.csv"
    if not slopes_path.exists():
        slopes_path = cfg.processed_dir / "quantile_slopes.csv"
    inputs = [slopes_path, *sorted(cfg.points_dir.glob("*_pts.csv"))]
    if reference_path is not None:
        inputs.append(Path(reference_path))
    write_manifest(
        inputs,
        cfg.results_dir / "filter_input_manifest.json",
        root=Path.cwd(),
        metadata={
            "reach_id": cfg.reach_id,
            "year": cfg.year,
            "stage": "apply-reference" if reference_path else "fit",
        },
    )
    _notify(callback, "filter", f"Wrote {len(results):,} filtered observations")
    return results


def run_workflow(
    cfg: RAQWConfig,
    *,
    login: bool = True,
    nc_paths: list[Path] | None = None,
    make_plots: bool = False,
    callback: StageCallback | None = None,
    progress_callback: FilterProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, slopes = preprocess_stage(
        cfg,
        login=login,
        nc_paths=nc_paths,
        callback=callback,
    )
    results = fit_stage(
        cfg,
        make_plots=make_plots,
        callback=callback,
        progress_callback=progress_callback,
    )
    _notify(callback, "complete", f"Analysis complete: {cfg.results_dir}")
    return summary, slopes, results
