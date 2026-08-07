from pathlib import Path

import numpy as np
import pandas as pd

from raqw.config import RAQWConfig
from raqw.filter import (
    build_envelope,
    load_envelope,
    run_filter,
    select_window,
    trim_and_refit,
)


def make_config(tmp_path: Path, **overrides) -> RAQWConfig:
    values = {
        "reach_id": "12345678901",
        "year": 2024,
        "reach_file": tmp_path / "reaches.geojson",
        "run_root": tmp_path,
    }
    values.update(overrides)
    return RAQWConfig(**values)


def test_trim_and_refit_recovers_linear_slope() -> None:
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 10_000.0, 500)
    y = 100.0 + 0.002 * x + rng.normal(0.0, 0.08, x.size)
    y[::25] += 8.0
    points = pd.DataFrame({"s_m": x, "height": y})

    result = trim_and_refit(points, 0.05, 0.90, 2.0, minimum_points=10)

    assert result is not None
    assert result["n_kept"] == 425
    assert abs(result["filtered_slope_m_per_km"] - 2.0) < 0.02


def test_build_and_reload_reference_envelope(tmp_path: Path) -> None:
    rows = []
    for file_index in range(6):
        for quantile in (0.1, 0.5, 0.9):
            rows.append(
                {
                    "file": f"file_{file_index}.nc",
                    "reach_id": "12345678901",
                    "quantile": quantile,
                    "slope_m_per_km": 2.0 + 0.02 * file_index + 0.1 * quantile,
                }
            )
    cfg = make_config(tmp_path)
    envelope, k_value = build_envelope(pd.DataFrame(rows), cfg)
    path = tmp_path / "reference.csv"
    envelope.to_csv(path, index=False)

    loaded, loaded_k = load_envelope(path)

    assert len(loaded) == 3
    assert np.isclose(loaded_k, k_value)
    assert set(loaded.columns) >= {
        "quantile",
        "median_slope_m_per_km",
        "lower_slope_m_per_km",
        "upper_slope_m_per_km",
    }


def test_select_window_uses_contiguous_core_and_expansion(tmp_path: Path) -> None:
    quantiles = np.round(np.arange(0.1, 1.0, 0.1), 1)
    slopes = np.array([3.0, 2.05, 2.02, 2.00, 2.00, 2.01, 2.03, 2.08, 3.2])
    curve = pd.DataFrame(
        {
            "file": "synthetic.nc",
            "quantile": quantiles,
            "slope_m_per_km": slopes,
            "median_slope_m_per_km": 2.0,
            "mad_slope_m_per_km": 0.05,
            "lower_slope_m_per_km": 1.8,
            "upper_slope_m_per_km": 2.2,
            "k_universal": 4.0,
            "z_score": np.abs(slopes - 2.0) / 0.05,
        }
    )
    x = np.linspace(0.0, 5_000.0, 200)
    points = pd.DataFrame(
        {
            "s_m": x,
            "height": 50.0 + 0.002 * x + 0.05 * np.sin(x / 200.0),
        }
    )
    cfg = make_config(
        tmp_path,
        core_tau_low=0.2,
        core_tau_high=0.8,
        min_core_width=0.2,
    )

    progress_events = []
    result = select_window(curve, points, cfg, progress_callback=progress_events.append)

    assert 0.2 <= result["tau_low"] <= result["tau_high"] <= 0.8
    assert result["window_width"] >= 0.2
    assert np.isfinite(result["filtered_slope_m_per_km"])
    assert progress_events[-1]["candidates_complete"] == progress_events[-1]["candidates_total"]


def write_synthetic_processed_year(cfg: RAQWConfig, file_count: int) -> None:
    cfg.points_dir.mkdir(parents=True, exist_ok=True)
    quantiles = np.round(np.arange(0.1, 1.0, 0.1), 1)
    slope_rows = []
    for file_index in range(file_count):
        file_name = f"synthetic_{cfg.year}_{file_index}.nc"
        slopes = 2.0 + 0.01 * file_index + 0.03 * (quantiles - 0.5) ** 2
        for quantile, slope in zip(quantiles, slopes):
            slope_rows.append(
                {
                    "file": file_name,
                    "reach_id": cfg.reach_id,
                    "quantile": quantile,
                    "slope_m_per_km": slope,
                }
            )
        x = np.linspace(0.0, 5_000.0, 150)
        points = pd.DataFrame(
            {
                "s_m": x,
                "height": 100.0 + 0.002 * x + 0.04 * np.sin(x / 150.0),
            }
        )
        points.to_csv(cfg.points_dir / f"{Path(file_name).stem}_pts.csv", index=False)
    pd.DataFrame(slope_rows).to_csv(
        cfg.processed_dir / "quantile_slopes_long.csv",
        index=False,
    )


def test_applies_frozen_reference_to_another_year(tmp_path: Path) -> None:
    training = make_config(
        tmp_path,
        min_files_per_quantile=5,
        core_tau_low=0.2,
        core_tau_high=0.8,
        min_core_width=0.2,
    )
    write_synthetic_processed_year(training, file_count=5)
    progress_events = []
    training_results = run_filter(training, progress_callback=progress_events.append)
    reference_path = training.results_dir / "raqw_envelope.csv"

    application = RAQWConfig(
        **{
            **training.as_serializable_dict(),
            "year": 2025,
            "reach_file": training.reach_file,
            "run_root": training.run_root,
        }
    )
    write_synthetic_processed_year(application, file_count=1)
    application_results = run_filter(application, reference_path=reference_path)

    assert len(training_results) == 5
    completed = [event for event in progress_events if event["event"] == "overpass_complete"]
    assert len(completed) == 5
    assert completed[-1]["file_number"] == completed[-1]["file_total"] == 5
    assert completed[-1]["eta_seconds"] == 0.0
    assert len(application_results) == 1
    assert application_results["reference_year"].iloc[0] == 2024
    assert (application.results_dir / "run_config.json").exists()
