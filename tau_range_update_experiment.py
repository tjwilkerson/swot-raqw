from __future__ import annotations

import json
import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

plt.ioff()


@dataclass(frozen=True)
class TauRangeConfig:
    project_root: Path = Path(r"C:\SWOT_universal_PIXC_quantile_filter")
    processed_root: Path = Path(
        r"C:\SWOT_universal_PIXC_quantile_filter\data\chile_reaches_with_valid_discharge_run\processed"
    )
    output_root: Path = Path(
        r"C:\SWOT_universal_PIXC_quantile_filter\data\tau_range_update_experiment"
    )
    reaches: tuple[str, ...] = ()
    year: int = 2024

    target_coverage: float = 0.95
    min_files_per_quantile: int = 5
    trim_frac: float = 0.05
    min_points_after_trim: int = 10

    core_tau_low: float = 0.01
    core_tau_high: float = 0.99
    min_core_width: float = 0.05
    tail_guard: float = 0.00
    max_core_outside_frac: float = 0.35
    max_expand_outside_frac: float = 0.50
    max_expand_z: float = 4.00
    max_expand_endpoint_jump: float = 50.0
    min_expand_gain: float = -0.50

    width_weight: float = 10.0
    closeness_weight: float = 0.5
    smoothness_weight: float = 1.0
    centrality_weight: float = 0.0
    outside_band_weight: float = 0.25
    tail_penalty_weight: float = 0.0
    endpoint_jump_weight: float = 0.0
    trim_spread_weight: float = 1.5
    trim_stability_weight: float = 1.0


def robust_mad(values: np.ndarray, eps: float = 1e-12) -> float:
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return eps
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(max(mad, eps))


def has_valid_regression_x(x: np.ndarray, min_unique_x: int = 2, min_x_span: float = 1e-9) -> bool:
    x = np.asarray(x, dtype="float64")
    if x.size < 2:
        return False
    if np.unique(x).size < min_unique_x:
        return False
    return bool((np.max(x) - np.min(x)) > min_x_span)


def build_universal_k_band(
    slopes_df: pd.DataFrame,
    target_coverage: float = 0.95,
    min_files_per_quantile: int = 5,
    trim_frac: float = 0.05,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, float]:
    df = slopes_df.dropna(subset=["file", "quantile", "slope_m_per_km"]).copy()

    rows: list[dict[str, float]] = []
    for tau, g in df.groupby("quantile"):
        s = g["slope_m_per_km"].to_numpy(dtype="float64")
        if len(s) < min_files_per_quantile:
            continue

        med = float(np.median(s))
        mad = robust_mad(s, eps=eps)
        z = np.abs(s - med) / mad
        k_tau = float(np.quantile(z, target_coverage))

        rows.append(
            {
                "quantile": float(tau),
                "median": med,
                "mad": mad,
                "k_tau": k_tau,
            }
        )

    q = pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)
    if q.empty:
        raise ValueError("No valid quantiles to build band.")

    ks = q["k_tau"].to_numpy(dtype="float64")
    lo = float(np.quantile(ks, trim_frac))
    hi = float(np.quantile(ks, 1 - trim_frac))
    ks2 = ks[(ks >= lo) & (ks <= hi)]
    k_universal = float(np.mean(ks2) if len(ks2) else np.mean(ks))
    k_universal = min(k_universal, 8.0)

    q["lower"] = q["median"] - k_universal * q["mad"]
    q["upper"] = q["median"] + k_universal * q["mad"]
    q["k_universal"] = k_universal
    return q, k_universal


def detrend_trim_refit(
    pts_df: pd.DataFrame,
    tau_low: float,
    tau_high: float,
    detrend_slope_m_per_km: float,
    min_points_after_trim: int = 10,
) -> dict[str, float] | None:
    df = pts_df[["s_m", "height"]].dropna().sort_values("s_m")
    if df.empty:
        return None

    x = df["s_m"].to_numpy(dtype="float64")
    y = df["height"].to_numpy(dtype="float64")

    if not has_valid_regression_x(x):
        return None

    lr0 = linregress(x, y)
    raw_slope = lr0.slope * 1000.0

    m_det = detrend_slope_m_per_km / 1000.0
    y_detr = y - m_det * x

    lo = float(np.quantile(y_detr, tau_low))
    hi = float(np.quantile(y_detr, tau_high))
    keep = (y_detr >= lo) & (y_detr <= hi)
    n_keep = int(keep.sum())
    if n_keep < min_points_after_trim:
        return None
    if not has_valid_regression_x(x[keep]):
        return None

    lr1 = linregress(x[keep], y[keep])
    postfilter_slope = lr1.slope * 1000.0
    kept_resid = y_detr[keep]
    spread_iqr = float(np.quantile(kept_resid, 0.75) - np.quantile(kept_resid, 0.25))
    spread_std = float(np.std(kept_resid))

    return {
        "n_raw": int(len(df)),
        "n_keep": n_keep,
        "keep_frac": float(n_keep / len(df)),
        "ols_slope_raw_m_per_km": float(raw_slope),
        "ols_slope_postfilter_m_per_km": float(postfilter_slope),
        "delta_slope_m_per_km": float(postfilter_slope - raw_slope),
        "trim_resid_iqr_m": spread_iqr,
        "trim_resid_std_m": spread_std,
    }


def load_reach_inputs(cfg: TauRangeConfig, reach_id: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    reach_dir = cfg.processed_root / f"reach_{reach_id}" / str(cfg.year)
    slopes_df = pd.read_csv(reach_dir / "quantile_slopes_long.csv")
    orig_results = pd.read_csv(reach_dir / "tau_trim_results.csv")
    return slopes_df, orig_results, reach_dir


def list_valid_solution_reach_ids(processed_root: Path, year: int = 2024) -> tuple[str, ...]:
    return tuple(
        sorted(
            p.name.replace("reach_", "", 1)
            for p in Path(processed_root).glob("reach_*")
            if p.is_dir()
            and (p / str(year) / "tau_trim_results.csv").exists()
            and (p / str(year) / "tau_trim_results.csv").stat().st_size > 0
        )
    )


def has_updated_tau_window_results(output_root: Path, reach_id: str, year: int) -> bool:
    results_csv = Path(output_root) / f"reach_{reach_id}" / str(year) / "updated_tau_window_results.csv"
    return results_csv.exists() and results_csv.stat().st_size > 0


def list_pending_experiment_reach_ids(cfg: TauRangeConfig) -> tuple[str, ...]:
    return tuple(
        rid for rid in cfg.reaches if not has_updated_tau_window_results(cfg.output_root, rid, cfg.year)
    )


def prepare_file_band(slopes_df: pd.DataFrame, q_band: pd.DataFrame, reach_id: str) -> pd.DataFrame:
    file_df = (
        slopes_df[slopes_df["reach_id"].astype(str) == str(reach_id)]
        .copy()
        .sort_values(["file", "quantile"])
    )
    merged = file_df.merge(
        q_band[["quantile", "median", "mad", "lower", "upper", "k_universal"]],
        on="quantile",
        how="left",
    )
    merged["mad_safe"] = merged["mad"].clip(lower=1e-12)
    merged["z_score"] = (merged["slope_m_per_km"] - merged["median"]).abs() / merged["mad_safe"]
    return merged


def compute_window_metrics(
    file_band: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    points_df: pd.DataFrame,
    cfg: TauRangeConfig,
) -> dict[str, float]:
    win = file_band.iloc[start_idx : end_idx + 1].copy()
    tau_low = float(win["quantile"].iloc[0])
    tau_high = float(win["quantile"].iloc[-1])
    slopes = win["slope_m_per_km"].to_numpy(dtype="float64")
    z = win["z_score"].to_numpy(dtype="float64")
    k_universal = float(win["k_universal"].iloc[0])

    slope_scale = robust_mad(file_band["slope_m_per_km"].to_numpy(dtype="float64"))
    diffs = np.diff(slopes)
    roughness = float(np.median(np.abs(diffs)) / slope_scale) if diffs.size else 0.0
    outside = np.maximum(z - k_universal, 0.0)
    outside_frac = float(np.mean(z > k_universal))
    mean_z_norm = float(np.mean(z / max(k_universal, 1e-12)))

    endpoint_jump = 0.0
    if start_idx > 0:
        endpoint_jump += abs(
            file_band.iloc[start_idx]["slope_m_per_km"]
            - file_band.iloc[start_idx - 1]["slope_m_per_km"]
        ) / slope_scale
    if end_idx < len(file_band) - 1:
        endpoint_jump += abs(
            file_band.iloc[end_idx + 1]["slope_m_per_km"]
            - file_band.iloc[end_idx]["slope_m_per_km"]
        ) / slope_scale

    tail_penalty = max(0.0, cfg.tail_guard - tau_low) + max(0.0, tau_high - (1.0 - cfg.tail_guard))
    width = float(tau_high - tau_low)
    centrality = float(1.0 - np.mean(np.abs(win["quantile"].to_numpy(dtype="float64") - 0.5) / 0.5))

    detrend_slope = float(np.median(slopes))
    trim_stats = detrend_trim_refit(
        pts_df=points_df,
        tau_low=tau_low,
        tau_high=tau_high,
        detrend_slope_m_per_km=detrend_slope,
        min_points_after_trim=cfg.min_points_after_trim,
    )
    if trim_stats is None:
        trim_penalty = 10.0
        trim_reward = 0.0
        keep_frac = 0.0
        trim_resid_iqr = np.nan
        postfilter_slope = np.nan
        delta_slope = np.nan
    else:
        trim_penalty = trim_stats["trim_resid_iqr_m"]
        trim_reward = 1.0 / (1.0 + trim_penalty)
        keep_frac = trim_stats["keep_frac"]
        trim_resid_iqr = trim_stats["trim_resid_iqr_m"]
        postfilter_slope = trim_stats["ols_slope_postfilter_m_per_km"]
        delta_slope = trim_stats["delta_slope_m_per_km"]

    if np.isfinite(delta_slope):
        trim_stability = 1.0 / (1.0 + abs(delta_slope))
    else:
        trim_stability = 0.0

    score = (
        cfg.width_weight * width
        + cfg.closeness_weight * (1.0 / (1.0 + mean_z_norm))
        + cfg.smoothness_weight * (1.0 / (1.0 + roughness))
        + cfg.centrality_weight * centrality
        + cfg.trim_spread_weight * trim_reward
        + cfg.trim_stability_weight * trim_stability
        - cfg.outside_band_weight * float(np.mean(outside / max(k_universal, 1e-12)))
        - cfg.tail_penalty_weight * tail_penalty
        - cfg.endpoint_jump_weight * endpoint_jump
    )

    return {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "width": width,
        "centrality": centrality,
        "mean_z_norm": mean_z_norm,
        "outside_frac": outside_frac,
        "roughness": roughness,
        "endpoint_jump": float(endpoint_jump),
        "score": float(score),
        "detrend_slope_m_per_km": detrend_slope,
        "keep_frac": float(keep_frac),
        "trim_resid_iqr_m": float(trim_resid_iqr) if np.isfinite(trim_resid_iqr) else np.nan,
        "ols_slope_postfilter_m_per_km": float(postfilter_slope) if np.isfinite(postfilter_slope) else np.nan,
        "delta_slope_m_per_km": float(delta_slope) if np.isfinite(delta_slope) else np.nan,
    }


def choose_core_window(
    file_band: pd.DataFrame,
    points_df: pd.DataFrame,
    cfg: TauRangeConfig,
) -> tuple[int, int, dict[str, float]]:
    core = file_band[
        (file_band["quantile"] >= cfg.core_tau_low)
        & (file_band["quantile"] <= cfg.core_tau_high)
    ].reset_index()
    if core.empty:
        raise ValueError("No quantiles in configured core window.")

    candidates: list[tuple[float, int, int, dict[str, float]]] = []
    for i in range(len(core)):
        for j in range(i, len(core)):
            start_idx = int(core.iloc[i]["index"])
            end_idx = int(core.iloc[j]["index"])
            metrics = compute_window_metrics(file_band, start_idx, end_idx, points_df, cfg)
            if metrics["width"] < cfg.min_core_width:
                continue
            priority = 0.0 if metrics["outside_frac"] <= cfg.max_core_outside_frac else 1.0
            candidates.append((priority, -metrics["score"], start_idx, end_idx, metrics))

    if not candidates:
        raise ValueError("No valid candidate core windows found.")

    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, start_idx, end_idx, metrics = candidates[0]
    return start_idx, end_idx, metrics


def evaluate_expansion(
    file_band: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    points_df: pd.DataFrame,
    cfg: TauRangeConfig,
    direction: str,
    current_metrics: dict[str, float],
) -> tuple[int, int, dict[str, float]] | None:
    if direction == "left":
        if start_idx == 0:
            return None
        cand_start, cand_end = start_idx - 1, end_idx
    elif direction == "right":
        if end_idx >= len(file_band) - 1:
            return None
        cand_start, cand_end = start_idx, end_idx + 1
    else:
        raise ValueError(f"Unknown direction: {direction}")

    metrics = compute_window_metrics(file_band, cand_start, cand_end, points_df, cfg)
    gain = metrics["score"] - current_metrics["score"]
    z_new = float(file_band.iloc[cand_start if direction == "left" else cand_end]["z_score"])

    passes = (
        metrics["outside_frac"] <= cfg.max_expand_outside_frac
        and z_new <= cfg.max_expand_z * float(file_band["k_universal"].iloc[0])
        and metrics["endpoint_jump"] <= cfg.max_expand_endpoint_jump
        and gain >= cfg.min_expand_gain
    )
    if not passes:
        return None
    return cand_start, cand_end, metrics


def select_tau_window_updated(
    file_band: pd.DataFrame,
    points_df: pd.DataFrame,
    cfg: TauRangeConfig,
) -> dict[str, float]:
    start_idx, end_idx, metrics = choose_core_window(file_band, points_df, cfg)

    while True:
        left = evaluate_expansion(file_band, start_idx, end_idx, points_df, cfg, "left", metrics)
        right = evaluate_expansion(file_band, start_idx, end_idx, points_df, cfg, "right", metrics)
        options = [opt for opt in (left, right) if opt is not None]
        if not options:
            break

        options.sort(key=lambda item: item[2]["score"], reverse=True)
        start_idx, end_idx, metrics = options[0]

    return metrics


def plot_tau_comparison(
    file_band: pd.DataFrame,
    original: pd.Series | None,
    updated: dict[str, float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        file_band["quantile"],
        file_band["slope_m_per_km"],
        color="#4C9BD6",
        linewidth=2,
        label="File slope curve",
    )
    ax.plot(
        file_band["quantile"],
        file_band["median"],
        color="#444444",
        linewidth=1.5,
        alpha=0.8,
        label="Ensemble median",
    )
    ax.fill_between(
        file_band["quantile"],
        file_band["lower"],
        file_band["upper"],
        color="#C5C5C5",
        alpha=0.25,
        label="Universal band",
    )

    if original is not None and np.isfinite(original["tau_low"]):
        ax.axvspan(
            float(original["tau_low"]),
            float(original["tau_high"]),
            color="#F4A261",
            alpha=0.25,
            label="Original tau window",
        )

    ax.axvspan(
        updated["tau_low"],
        updated["tau_high"],
        color="#2A9D8F",
        alpha=0.22,
        label="Updated tau window",
    )

    ax.set_xlabel("Quantile (tau)")
    ax.set_ylabel("Slope (m/km)")
    ax.set_title(Path(file_band["file"].iloc[0]).name)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.clear()
    plt.close(fig)
    gc.collect()


def plot_trim_comparison(
    pts_df: pd.DataFrame,
    original: pd.Series | None,
    updated: dict[str, float],
    out_path: Path,
) -> None:
    df = pts_df[["s_m", "height"]].dropna().sort_values("s_m")
    x = df["s_m"].to_numpy(dtype="float64")
    y = df["height"].to_numpy(dtype="float64")
    x_km = x / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_km, y, s=10, alpha=0.18, color="#888888", label="All points")

    def _plot_window(window: pd.Series | dict[str, float], color: str, label: str) -> None:
        slope_det = float(window["detrend_slope_m_per_km"])
        y_detr = y - (slope_det / 1000.0) * x
        lo = float(np.quantile(y_detr, float(window["tau_low"])))
        hi = float(np.quantile(y_detr, float(window["tau_high"])))
        keep = (y_detr >= lo) & (y_detr <= hi)
        ax.scatter(x_km[keep], y[keep], s=12, alpha=0.55, color=color, label=label)

    _plot_window(updated, "#1F77B4", "Filtered points")

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("WSE (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.clear()
    plt.close(fig)
    gc.collect()


def run_reach_experiment(cfg: TauRangeConfig, reach_id: str) -> pd.DataFrame:
    slopes_df, original_results, reach_dir = load_reach_inputs(cfg, reach_id)
    q_band, k_universal = build_universal_k_band(
        slopes_df=slopes_df,
        target_coverage=cfg.target_coverage,
        min_files_per_quantile=cfg.min_files_per_quantile,
        trim_frac=cfg.trim_frac,
    )
    prepared = prepare_file_band(slopes_df, q_band, reach_id=reach_id)

    out_dir = cfg.output_root / f"reach_{reach_id}" / str(cfg.year)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for file_name, file_band in prepared.groupby("file", sort=True):
        file_band = file_band.sort_values("quantile").reset_index(drop=True)
        pts_path = reach_dir / "pixc_points" / f"{Path(file_name).stem}_pts.csv"
        if not pts_path.exists():
            continue

        pts_df = pd.read_csv(pts_path)
        updated = select_tau_window_updated(file_band, pts_df, cfg)
        original = original_results[original_results["file"] == file_name]
        original_row = original.iloc[0] if not original.empty else None

        row: dict[str, float | str] = {
            "reach_id": str(reach_id),
            "year": int(cfg.year),
            "file": file_name,
            "K_universal": float(k_universal),
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

        plot_tau_comparison(file_band, original_row, updated, plots_dir / f"{Path(file_name).stem}_tau_compare.png")
        plot_trim_comparison(pts_df, original_row, updated, plots_dir / f"{Path(file_name).stem}_trim_compare.png")
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    results_df.to_csv(out_dir / "updated_tau_window_results.csv", index=False)
    q_band.to_csv(out_dir / "universal_band.csv", index=False)
    return results_df


def main() -> None:
    cfg = TauRangeConfig()
    if not cfg.reaches:
        cfg = TauRangeConfig(reaches=list_valid_solution_reach_ids(cfg.processed_root, year=cfg.year))
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    per_reach_results: list[pd.DataFrame] = []
    pending_reaches = list_pending_experiment_reach_ids(cfg)
    for reach_id in pending_reaches:
        reach_results = run_reach_experiment(cfg, reach_id)
        per_reach_results.append(reach_results)

    if not per_reach_results:
        return

    combined = pd.concat(per_reach_results, ignore_index=True)
    combined.to_csv(cfg.output_root / "combined_results.csv", index=False)

    with (cfg.output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)


if __name__ == "__main__":
    main()
