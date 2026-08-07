from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from .config import RAQWConfig


FilterProgressCallback = Callable[[dict[str, object]], None]


def robust_mad(values: np.ndarray, epsilon: float = 1e-12) -> float:
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return epsilon
    median = np.median(values)
    return float(max(np.median(np.abs(values - median)), epsilon))


def valid_regression_axis(
    values: np.ndarray,
    minimum_unique: int = 2,
) -> bool:
    values = np.asarray(values, dtype="float64")
    return (
        values.size >= minimum_unique
        and np.unique(values).size >= minimum_unique
        and np.ptp(values) > 1e-9
    )


def build_envelope(slopes: pd.DataFrame, cfg: RAQWConfig) -> tuple[pd.DataFrame, float]:
    clean = slopes.dropna(subset=["file", "quantile", "slope_m_per_km"]).copy()
    rows = []
    for quantile, group in clean.groupby("quantile"):
        values = group["slope_m_per_km"].to_numpy(dtype="float64")
        if len(values) < cfg.min_files_per_quantile:
            continue
        median = float(np.median(values))
        mad = robust_mad(values, epsilon=cfg.mad_epsilon)
        k_quantile = float(np.quantile(np.abs(values - median) / mad, cfg.target_coverage))
        rows.append(
            {
                "quantile": float(quantile),
                "median_slope_m_per_km": median,
                "mad_slope_m_per_km": mad,
                "k_quantile": k_quantile,
            }
        )
    envelope = pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)
    if envelope.empty:
        raise ValueError(
            f"No quantile has at least {cfg.min_files_per_quantile} valid overpasses."
        )

    k_values = envelope["k_quantile"].to_numpy(dtype="float64")
    lower = np.quantile(k_values, cfg.band_trim_fraction)
    upper = np.quantile(k_values, 1.0 - cfg.band_trim_fraction)
    trimmed = k_values[(k_values >= lower) & (k_values <= upper)]
    k_universal = min(
        float(np.mean(trimmed if len(trimmed) else k_values)),
        cfg.universal_k_cap,
    )
    envelope["lower_slope_m_per_km"] = (
        envelope["median_slope_m_per_km"] - k_universal * envelope["mad_slope_m_per_km"]
    )
    envelope["upper_slope_m_per_km"] = (
        envelope["median_slope_m_per_km"] + k_universal * envelope["mad_slope_m_per_km"]
    )
    envelope["k_universal"] = k_universal
    return envelope, k_universal


def prepare_file_curves(
    slopes: pd.DataFrame,
    envelope: pd.DataFrame,
    reach_id: str,
) -> pd.DataFrame:
    curves = slopes.loc[slopes["reach_id"].astype(str) == str(reach_id)].copy()
    curves = curves.merge(envelope, on="quantile", how="inner")
    curves["z_score"] = (
        curves["slope_m_per_km"] - curves["median_slope_m_per_km"]
    ).abs() / curves["mad_slope_m_per_km"].clip(lower=1e-12)
    return curves.sort_values(["file", "quantile"])


def load_envelope(path: Path) -> tuple[pd.DataFrame, float]:
    """Load and validate a previously fitted reach reference envelope."""

    envelope = pd.read_csv(path)
    required = {
        "quantile",
        "median_slope_m_per_km",
        "mad_slope_m_per_km",
        "lower_slope_m_per_km",
        "upper_slope_m_per_km",
        "k_universal",
    }
    missing = sorted(required - set(envelope.columns))
    if missing:
        raise ValueError(
            f"Reference envelope {path} is missing columns: {', '.join(missing)}"
        )
    envelope = envelope.dropna(subset=sorted(required)).sort_values("quantile")
    if envelope.empty:
        raise ValueError(f"Reference envelope {path} contains no valid rows.")
    k_values = envelope["k_universal"].dropna().unique()
    if len(k_values) != 1:
        raise ValueError("Reference envelope must contain one k_universal value.")
    return envelope.reset_index(drop=True), float(k_values[0])


def trim_and_refit(
    points: pd.DataFrame,
    tau_low: float,
    tau_high: float,
    detrend_slope_m_per_km: float,
    minimum_points: int,
    minimum_unique: int = 2,
) -> dict[str, float] | None:
    clean = points[["s_m", "height"]].dropna().sort_values("s_m")
    x = clean["s_m"].to_numpy(dtype="float64")
    y = clean["height"].to_numpy(dtype="float64")
    if not valid_regression_axis(x, minimum_unique=minimum_unique):
        return None

    raw_slope = float(linregress(x, y).slope * 1000.0)
    detrended = y - (detrend_slope_m_per_km / 1000.0) * x
    low = float(np.quantile(detrended, tau_low))
    high = float(np.quantile(detrended, tau_high))
    keep = (detrended >= low) & (detrended <= high)
    if int(keep.sum()) < minimum_points or not valid_regression_axis(
        x[keep],
        minimum_unique=minimum_unique,
    ):
        return None

    filtered_slope = float(linregress(x[keep], y[keep]).slope * 1000.0)
    kept_residuals = detrended[keep]
    return {
        "n_raw": int(len(clean)),
        "n_kept": int(keep.sum()),
        "keep_fraction": float(keep.mean()),
        "raw_slope_m_per_km": raw_slope,
        "filtered_slope_m_per_km": filtered_slope,
        "slope_change_m_per_km": filtered_slope - raw_slope,
        "trimmed_residual_iqr_m": float(
            np.quantile(kept_residuals, 0.75) - np.quantile(kept_residuals, 0.25)
        ),
    }


def window_metrics(
    curve: pd.DataFrame,
    start: int,
    end: int,
    points: pd.DataFrame,
    cfg: RAQWConfig,
) -> dict[str, float]:
    window = curve.iloc[start : end + 1]
    slopes = window["slope_m_per_km"].to_numpy(dtype="float64")
    z_scores = window["z_score"].to_numpy(dtype="float64")
    k_universal = float(window["k_universal"].iloc[0])
    scale = robust_mad(
        curve["slope_m_per_km"].to_numpy(dtype="float64"),
        epsilon=cfg.mad_epsilon,
    )
    differences = np.diff(slopes)
    roughness = float(np.median(np.abs(differences)) / scale) if differences.size else 0.0
    outside = np.maximum(z_scores - k_universal, 0.0)

    endpoint_jump = 0.0
    if start > 0:
        endpoint_jump += abs(
            curve.iloc[start]["slope_m_per_km"] - curve.iloc[start - 1]["slope_m_per_km"]
        ) / scale
    if end < len(curve) - 1:
        endpoint_jump += abs(
            curve.iloc[end + 1]["slope_m_per_km"] - curve.iloc[end]["slope_m_per_km"]
        ) / scale

    tau_low = float(window["quantile"].iloc[0])
    tau_high = float(window["quantile"].iloc[-1])
    centrality = float(
        1.0 - np.mean(np.abs(window["quantile"].to_numpy(dtype="float64") - 0.5) / 0.5)
    )
    tail_penalty = max(0.0, cfg.tail_guard - tau_low) + max(
        0.0, tau_high - (1.0 - cfg.tail_guard)
    )
    detrend_slope = float(np.median(slopes))
    trim = trim_and_refit(
        points,
        tau_low,
        tau_high,
        detrend_slope,
        cfg.min_points_after_trim,
        cfg.min_unique_coordinates,
    )
    if trim is None:
        trim_iqr = np.nan
        trim_reward = 0.0
        stability = 0.0
    else:
        trim_iqr = trim["trimmed_residual_iqr_m"]
        trim_reward = 1.0 / (1.0 + trim_iqr)
        stability = 1.0 / (1.0 + abs(trim["slope_change_m_per_km"]))

    mean_z_normalized = float(np.mean(z_scores / max(k_universal, 1e-12)))
    width = tau_high - tau_low
    score = (
        cfg.width_weight * width
        + cfg.closeness_weight / (1.0 + mean_z_normalized)
        + cfg.smoothness_weight / (1.0 + roughness)
        + cfg.centrality_weight * centrality
        + cfg.trim_spread_weight * trim_reward
        + cfg.trim_stability_weight * stability
        - cfg.outside_band_weight * float(np.mean(outside / max(k_universal, 1e-12)))
        - cfg.tail_penalty_weight * tail_penalty
        - cfg.endpoint_jump_weight * endpoint_jump
    )
    return {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "window_width": width,
        "centrality": centrality,
        "score": float(score),
        "outside_fraction": float(np.mean(z_scores > k_universal)),
        "roughness": roughness,
        "endpoint_jump": float(endpoint_jump),
        "detrend_slope_m_per_km": detrend_slope,
        "keep_fraction": trim["keep_fraction"] if trim else np.nan,
        "trimmed_residual_iqr_m": trim_iqr,
        "raw_slope_m_per_km": trim["raw_slope_m_per_km"] if trim else np.nan,
        "filtered_slope_m_per_km": trim["filtered_slope_m_per_km"] if trim else np.nan,
        "slope_change_m_per_km": trim["slope_change_m_per_km"] if trim else np.nan,
        "n_raw": trim["n_raw"] if trim else 0,
        "n_kept": trim["n_kept"] if trim else 0,
    }


def choose_initial_window(
    curve: pd.DataFrame,
    points: pd.DataFrame,
    cfg: RAQWConfig,
    progress_callback: FilterProgressCallback | None = None,
) -> tuple[int, int, dict[str, float]]:
    candidate_indices = curve.index[
        curve["quantile"].between(cfg.core_tau_low, cfg.core_tau_high)
    ].tolist()
    quantiles = curve["quantile"].to_numpy(dtype="float64")
    valid_candidate_count = sum(
        1
        for start in candidate_indices
        for end in candidate_indices
        if end >= start and quantiles[end] - quantiles[start] >= cfg.min_core_width
    )
    evaluated = 0
    candidates = []
    for start in candidate_indices:
        for end in candidate_indices:
            if end < start:
                continue
            if quantiles[end] - quantiles[start] < cfg.min_core_width:
                continue
            metrics = window_metrics(curve, start, end, points, cfg)
            evaluated += 1
            if progress_callback is not None and (
                evaluated == 1
                or evaluated == valid_candidate_count
                or evaluated % 250 == 0
            ):
                progress_callback(
                    {
                        "event": "candidate_progress",
                        "candidates_complete": evaluated,
                        "candidates_total": valid_candidate_count,
                    }
                )
            priority = 0 if metrics["outside_fraction"] <= cfg.max_core_outside_fraction else 1
            candidates.append((priority, -metrics["score"], start, end, metrics))
    if not candidates:
        raise ValueError("No valid RAQW candidate windows were found.")
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, start, end, metrics = candidates[0]
    return start, end, metrics


def evaluate_expansion(
    curve: pd.DataFrame,
    points: pd.DataFrame,
    cfg: RAQWConfig,
    start: int,
    end: int,
    current: dict[str, float],
    direction: int,
) -> tuple[int, int, dict[str, float]] | None:
    candidate_start = start - 1 if direction < 0 else start
    candidate_end = end + 1 if direction > 0 else end
    if candidate_start < 0 or candidate_end >= len(curve):
        return None
    metrics = window_metrics(curve, candidate_start, candidate_end, points, cfg)
    new_index = candidate_start if direction < 0 else candidate_end
    new_z = float(curve.iloc[new_index]["z_score"])
    passes = (
        metrics["outside_fraction"] <= cfg.max_expand_outside_fraction
        and new_z <= cfg.max_expand_z_multiplier * float(curve["k_universal"].iloc[0])
        and metrics["endpoint_jump"] <= cfg.max_expand_endpoint_jump
        and metrics["score"] - current["score"] >= cfg.min_expand_score_gain
    )
    return (candidate_start, candidate_end, metrics) if passes else None


def select_window(
    curve: pd.DataFrame,
    points: pd.DataFrame,
    cfg: RAQWConfig,
    progress_callback: FilterProgressCallback | None = None,
) -> dict[str, float]:
    start, end, metrics = choose_initial_window(
        curve,
        points,
        cfg,
        progress_callback=progress_callback,
    )
    while True:
        options = [
            option
            for option in (
                evaluate_expansion(curve, points, cfg, start, end, metrics, -1),
                evaluate_expansion(curve, points, cfg, start, end, metrics, 1),
            )
            if option is not None
        ]
        if not options:
            return metrics
        start, end, metrics = max(options, key=lambda item: item[2]["score"])


def plot_result(
    curve: pd.DataFrame,
    points: pd.DataFrame,
    result: dict[str, float],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(curve["quantile"], curve["slope_m_per_km"], color="#457B9D")
    axes[0].plot(
        curve["quantile"],
        curve["median_slope_m_per_km"],
        color="#333333",
        linewidth=1.2,
    )
    axes[0].fill_between(
        curve["quantile"],
        curve["lower_slope_m_per_km"],
        curve["upper_slope_m_per_km"],
        color="#B8B8B8",
        alpha=0.3,
    )
    axes[0].axvspan(result["tau_low"], result["tau_high"], color="#2A9D8F", alpha=0.25)
    axes[0].set(xlabel="Quantile", ylabel="Slope (m/km)", title="RAQW quantile window")
    axes[0].grid(alpha=0.25)

    clean = points[["s_m", "height"]].dropna().sort_values("s_m")
    x = clean["s_m"].to_numpy(dtype="float64")
    y = clean["height"].to_numpy(dtype="float64")
    detrended = y - (result["detrend_slope_m_per_km"] / 1000.0) * x
    low = np.quantile(detrended, result["tau_low"])
    high = np.quantile(detrended, result["tau_high"])
    keep = (detrended >= low) & (detrended <= high)
    axes[1].scatter(x / 1000.0, y, s=8, alpha=0.15, color="#777777")
    axes[1].scatter(x[keep] / 1000.0, y[keep], s=10, alpha=0.55, color="#2A9D8F")
    axes[1].set(xlabel="Distance (km)", ylabel="WSE (m)", title="RAQW retained points")
    axes[1].grid(alpha=0.25)
    figure.suptitle(Path(curve["file"].iloc[0]).name)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_filter(
    cfg: RAQWConfig,
    make_plots: bool = False,
    reference_path: Path | None = None,
    progress_callback: FilterProgressCallback | None = None,
) -> pd.DataFrame:
    slopes_path = cfg.processed_dir / "quantile_slopes.csv"
    if not slopes_path.exists():
        legacy_path = cfg.processed_dir / "quantile_slopes_long.csv"
        slopes_path = legacy_path if legacy_path.exists() else slopes_path
    if not slopes_path.exists():
        raise FileNotFoundError(f"No quantile slope table found in {cfg.processed_dir}.")

    slopes = pd.read_csv(slopes_path, dtype={"reach_id": str})
    if reference_path is None:
        envelope, k_universal = build_envelope(slopes, cfg)
        reference_source = "fitted from current reach-year"
        reference_year = int(cfg.year)
    else:
        reference_path = Path(reference_path).resolve()
        envelope, k_universal = load_envelope(reference_path)
        if "reach_id" in envelope.columns:
            reference_reaches = set(envelope["reach_id"].dropna().astype(str))
            if reference_reaches and reference_reaches != {str(cfg.reach_id)}:
                raise ValueError(
                    f"Reference envelope is for reach IDs {sorted(reference_reaches)}, "
                    f"not {cfg.reach_id}."
                )
        reference_source = str(reference_path)
        if "reference_year" in envelope.columns:
            years = envelope["reference_year"].dropna().astype(int).unique()
            reference_year = int(years[0]) if len(years) == 1 else None
        else:
            reference_year = (
                int(reference_path.parent.name)
                if reference_path.parent.name.isdigit()
                else None
            )
    curves = prepare_file_curves(slopes, envelope, cfg.reach_id)
    if curves.empty:
        raise ValueError(
            "No observation quantiles overlap the supplied reference envelope."
        )
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.results_dir / "plots"
    if make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    grouped_curves = list(curves.groupby("file", sort=True))
    total_files = len(grouped_curves)
    filter_started = perf_counter()
    rows = []
    for file_number, (file_name, curve) in enumerate(grouped_curves, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "overpass_start",
                    "file": file_name,
                    "file_number": file_number,
                    "file_total": total_files,
                    "elapsed_seconds": perf_counter() - filter_started,
                }
            )
        curve = curve.sort_values("quantile").reset_index(drop=True)
        points_path = cfg.points_dir / f"{Path(file_name).stem}_pts.csv"
        if not points_path.exists():
            print(f"Skipping {file_name}: missing {points_path.name}")
            continue
        points = pd.read_csv(points_path)
        def candidate_progress(event: dict[str, object]) -> None:
            if progress_callback is not None:
                progress_callback(
                    {
                        **event,
                        "file": file_name,
                        "file_number": file_number,
                        "file_total": total_files,
                        "elapsed_seconds": perf_counter() - filter_started,
                    }
                )

        result = select_window(
            curve,
            points,
            cfg,
            progress_callback=candidate_progress,
        )
        rows.append(
            {
                "reach_id": str(cfg.reach_id),
                "year": int(cfg.year),
                "file": file_name,
                "reference_year": reference_year,
                "k_universal": k_universal,
                **{f"raqw_{key}": value for key, value in result.items()},
            }
        )
        if make_plots:
            plot_result(curve, points, result, plots_dir / f"{Path(file_name).stem}_raqw.png")
        if progress_callback is not None:
            elapsed = perf_counter() - filter_started
            progress_callback(
                {
                    "event": "overpass_complete",
                    "file": file_name,
                    "file_number": file_number,
                    "file_total": total_files,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": (elapsed / file_number) * (total_files - file_number),
                }
            )

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("RAQW did not produce any overpass results.")
    results = results.sort_values("file").reset_index(drop=True)
    results.to_csv(cfg.results_dir / "raqw_results.csv", index=False)
    envelope_output = envelope.copy()
    if "reach_id" not in envelope_output.columns:
        envelope_output.insert(0, "reach_id", str(cfg.reach_id))
    if "reference_year" not in envelope_output.columns:
        envelope_output.insert(1, "reference_year", reference_year)
    envelope_output.to_csv(cfg.results_dir / "raqw_envelope.csv", index=False)
    run_configuration = cfg.as_serializable_dict()
    run_configuration["reference_source"] = reference_source
    with (cfg.results_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(run_configuration, stream, indent=2)
    return results
