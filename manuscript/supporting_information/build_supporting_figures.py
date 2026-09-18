"""Build the figures and count macros used by the GRL Supporting Information.

The script reads the preserved 2024 processing outputs in this repository.  It
does not download SWOT data and does not modify any analysis output.  Run from
the repository root with the project Python environment, for example:

    python manuscript/supporting_information/build_supporting_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SI_ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = SI_ROOT / "figures"

PROCESSED_ROOT = PROJECT_ROOT / "data" / "chile_reaches_with_valid_discharge_run" / "processed"
TAU_ROOT = PROJECT_ROOT / "data" / "tau_range_update_experiment"
COMPARISON_ROOT = PROJECT_ROOT / "data" / "riversp_comparison_updated_filter"
REACH_LIST = PROJECT_ROOT / "data" / "Petrohue_SWORD_reaches" / "chile_reaches_with_valid_discharge.csv"
ALL_RESULTS = COMPARISON_ROOT / "all_reach_updated_tau_window_results.csv"
MATCHED_RESULTS = COMPARISON_ROOT / "matched_pixc_riversp_slopes.csv"
RUN_CONFIG = TAU_ROOT / "run_config.json"

BLUE = "#176B8A"
LIGHT_BLUE = "#A9D6E5"
ORANGE = "#E76F51"
GREEN = "#2A9D8F"
DARK = "#263238"
GRAY = "#737B80"
LIGHT_GRAY = "#E9EEF0"


def tex_escape(value: object) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ]:
        text = text.replace(old, new)
    return text


def classify_error(error: object) -> str:
    if pd.isna(error) or not str(error).strip():
        return "passed"
    value = str(error).lower()
    if "hydrocron" in value and "qa gate" not in value:
        return "RiverSP match unavailable/invalid"
    if "qa gate" in value:
        return "RiverSP quality gate"
    if "classification filter" in value:
        return "PIXC classification filter"
    if "inside reach buffer" in value:
        return "Reach buffer filter"
    if "coverage failed" in value:
        return "Reach coverage filter"
    return "Other processing failure"


def collect_metrics() -> tuple[dict[str, int | float], pd.DataFrame]:
    summary_paths = sorted(PROCESSED_ROOT.glob("reach_*/2024/summary.csv"))
    summaries: list[pd.DataFrame] = []
    for path in summary_paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["summary_path"] = str(path.relative_to(PROJECT_ROOT))
        summaries.append(frame)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary["screen_result"] = summary["error"].map(classify_error)

    all_results = pd.read_csv(ALL_RESULTS, low_memory=False)
    matched = pd.read_csv(MATCHED_RESULTS, low_memory=False)
    reach_candidates = pd.read_csv(REACH_LIST)

    successful = summary[summary["screen_result"] == "passed"].copy()
    successful["n_points"] = pd.to_numeric(successful.get("n_points"), errors="coerce")
    below_50 = int((successful["n_points"] < 50).sum())

    counts = summary["screen_result"].value_counts().to_dict()
    metrics: dict[str, int | float] = {
        "candidate_reaches": int(len(reach_candidates)),
        "reaches_with_summary": int(len(summary_paths)),
        "downloaded_reach_granule_records": int(len(summary)),
        "prescreen_passed_records": int(len(successful)),
        "prescreen_passed_reaches": int(successful["reach_id"].astype(str).nunique()),
        "final_candidate_observations": int(len(all_results)),
        "final_candidate_reaches": int(all_results["reach_id"].astype(str).nunique()),
        "successful_filtered_slopes": int(all_results["updated_postfilter_slope_m_per_km"].notna().sum()),
        "failed_filtered_slopes": int(all_results["updated_postfilter_slope_m_per_km"].isna().sum()),
        "matched_observations": int(len(matched)),
        "matched_reaches": int(matched["reach_id"].astype(str).nunique()),
        "successful_records_below_50_points": below_50,
        "hydrocron_invalid": int(counts.get("RiverSP match unavailable/invalid", 0)),
        "quality_gate": int(counts.get("RiverSP quality gate", 0)),
        "classification_filter": int(counts.get("PIXC classification filter", 0)),
        "buffer_filter": int(counts.get("Reach buffer filter", 0)),
        "coverage_filter": int(counts.get("Reach coverage filter", 0)),
        "other_failure": int(counts.get("Other processing failure", 0)),
    }
    return metrics, summary


def write_metrics(metrics: dict[str, int | float]) -> None:
    (SI_ROOT / "supporting_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    macro_names = {
        "candidate_reaches": "CandidateReachCount",
        "reaches_with_summary": "SummaryReachCount",
        "downloaded_reach_granule_records": "DownloadedReachGranuleCount",
        "prescreen_passed_records": "PreScreenPassCount",
        "prescreen_passed_reaches": "PreScreenReachCount",
        "final_candidate_observations": "FinalCandidateCount",
        "final_candidate_reaches": "FinalReachCount",
        "successful_filtered_slopes": "SuccessfulFilteredCount",
        "failed_filtered_slopes": "FailedFilteredCount",
        "matched_observations": "MatchedObservationCount",
        "matched_reaches": "MatchedReachCount",
        "successful_records_below_50_points": "BelowFiftyCount",
        "hydrocron_invalid": "HydrocronInvalidCount",
        "quality_gate": "QualityGateCount",
        "classification_filter": "ClassificationFilterCount",
        "buffer_filter": "BufferFilterCount",
        "coverage_filter": "CoverageFilterCount",
        "other_failure": "OtherFailureCount",
    }
    lines = ["% Generated by build_supporting_figures.py; do not edit by hand."]
    for key, macro in macro_names.items():
        value = metrics[key]
        rendered = f"{value:,}" if isinstance(value, int) else str(value)
        lines.append(rf"\newcommand{{\{macro}}}{{{rendered}}}")
    (SI_ROOT / "supporting_metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.6,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + h - 0.035, title, va="top", ha="left", fontsize=10.5, weight="bold", color=color)
    ax.text(x + 0.025, y + h - 0.082, body, va="top", ha="left", fontsize=8.25, color=DARK, linespacing=1.20)


def add_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=GRAY))


def figure_s1_workflow(metrics: dict[str, int | float], summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x, w, h = 0.05, 0.63, 0.105
    ys = [0.85, 0.69, 0.53, 0.37, 0.21, 0.05]
    boxes = [
        (
            "1. Reach and observation discovery",
            f"{metrics['candidate_reaches']:,} SWORD reaches with valid DAWG/SOS discharge\n"
            f"{metrics['downloaded_reach_granule_records']:,} reach-granule records assessed (2024)",
            BLUE,
        ),
        (
            "2. RiverSP match and quality gate",
            "Nearest Hydrocron record within +/-30 min\n"
            "Require reach_q in {0, 1}; buffer width = RiverSP width/2",
            BLUE,
        ),
        (
            "3. PIXC correction and spatial pre-screen",
            "Correct three tides; retain classes >2, excluding class 5\n"
            "Use EPSG:32718; clip to buffer; require coverage >=0.95",
            BLUE,
        ),
        (
            "4. Reach-year quantile characterization",
            "Fit WSE ~ distance for tau=0.01,...,0.99\n"
            "Build reach-year median/MAD and 95% reference band",
            GREEN,
        ),
        (
            "5. Tau-range selection and point filtering",
            "Score width, agreement, smoothness, spread, and stability\n"
            "Expand the window while all constraints pass",
            GREEN,
        ),
        (
            "6. Final slope and comparison set",
            f"Detrend, trim by selected residual quantiles, fit retained-point OLS\n"
            f"{metrics['successful_filtered_slopes']:,} slopes; {metrics['final_candidate_reaches']:,} reaches; "
            f"{metrics['failed_filtered_slopes']:,} final-fit failures",
            ORANGE,
        ),
    ]
    for idx, (y, content) in enumerate(zip(ys, boxes)):
        title, body, color = content
        add_box(ax, x, y, w, h, title, body, color)
        if idx < len(ys) - 1:
            add_arrow(ax, 0.365, y - 0.005, 0.365, ys[idx + 1] + h + 0.005)

    failure_counts = (
        summary.loc[summary["screen_result"] != "passed", "screen_result"]
        .value_counts()
        .reindex(
            [
                "RiverSP match unavailable/invalid",
                "RiverSP quality gate",
                "PIXC classification filter",
                "Reach buffer filter",
                "Reach coverage filter",
                "Other processing failure",
            ],
            fill_value=0,
        )
    )
    inset = fig.add_axes([0.71, 0.59, 0.27, 0.27])
    inset.barh(np.arange(len(failure_counts)), failure_counts.to_numpy(), color=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.8)
    inset.set_yticks(np.arange(len(failure_counts)))
    inset.set_yticklabels([])
    inset.invert_yaxis()
    inset.tick_params(axis="x", labelsize=7.5)
    inset.tick_params(axis="y", length=0)
    inset.grid(axis="x", alpha=0.25)
    inset.set_title("Pre-screen failures", fontsize=8.5, weight="bold")
    short_labels = ["No RiverSP", "reach_q", "Class", "Buffer", "Coverage", "Other"]
    xmax = max(float(failure_counts.max()) * 1.20, 1.0)
    inset.set_xlim(0, xmax)
    for i, (label, value) in enumerate(zip(short_labels, failure_counts.to_numpy())):
        if value < 800:
            inset.text(xmax * 0.018, i, f"{label}  ({value:,})", va="center", ha="left", fontsize=7.2, color=DARK, weight="bold")
        else:
            inset.text(xmax * 0.018, i, label, va="center", ha="left", fontsize=7.2, color=DARK, weight="bold")
            inset.text(value, i, f" {value:,}", va="center", fontsize=7.2)
    for spine in ["top", "right"]:
        inset.spines[spine].set_visible(False)

    ax.text(0.72, 0.51, "Accounting after pre-screen", fontsize=9.5, weight="bold", color=DARK)
    ax.text(
        0.72,
        0.485,
        f"{metrics['prescreen_passed_records']:,} records passed\n"
        f"{metrics['prescreen_passed_reaches']:,} reaches represented\n"
        f"{metrics['final_candidate_observations']:,} entered final comparison\n"
        f"{metrics['successful_records_below_50_points']:,} passing records had <50 points",
        fontsize=8.1,
        va="top",
        color=DARK,
        linespacing=1.35,
    )

    fig.suptitle("Supporting workflow and observation accounting", fontsize=14, weight="bold", color=DARK, y=0.985)
    fig.savefig(FIGURE_ROOT / "figure_S1_workflow.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def robust_mad(values: np.ndarray, eps: float = 1e-12) -> float:
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    return float(max(np.median(np.abs(values - median)), eps))


def build_reference_band(slopes: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, float]] = []
    for tau, group in slopes.dropna(subset=["file", "quantile", "slope_m_per_km"]).groupby("quantile"):
        values = group["slope_m_per_km"].to_numpy(dtype=float)
        if len(values) < int(config["min_files_per_quantile"]):
            continue
        median = float(np.median(values))
        mad = robust_mad(values)
        k_tau = float(np.quantile(np.abs(values - median) / mad, float(config["target_coverage"])))
        rows.append({"quantile": float(tau), "median": median, "mad": mad, "k_tau": k_tau})
    band = pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)
    k_values = band["k_tau"].to_numpy(dtype=float)
    trim = float(config["trim_frac"])
    lo, hi = np.quantile(k_values, [trim, 1 - trim])
    kept = k_values[(k_values >= lo) & (k_values <= hi)]
    universal_k = min(float(np.mean(kept if len(kept) else k_values)), 8.0)
    band["lower"] = band["median"] - universal_k * band["mad"]
    band["upper"] = band["median"] + universal_k * band["mad"]
    return band, universal_k


def observation_paths(reach_id: str, file_name: str) -> tuple[Path, Path]:
    reach_dir = PROCESSED_ROOT / f"reach_{reach_id}" / "2024"
    return (
        reach_dir / "pixc_points" / f"{Path(file_name).stem}_pts.csv",
        reach_dir / "quantile_slopes_long.csv",
    )


def get_result_row(results: pd.DataFrame, reach_id: str, file_name: str) -> pd.Series:
    match = results[
        (results["reach_id"].astype(str) == str(reach_id))
        & (results["file"].astype(str) == str(file_name))
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one result for reach={reach_id}, file={file_name}; found {len(match)}")
    return match.iloc[0]


def retained_mask(points: pd.DataFrame, row: pd.Series) -> np.ndarray:
    x = points["s_m"].to_numpy(dtype=float)
    y = points["height"].to_numpy(dtype=float)
    y_detrended = y - (float(row["updated_detrend_slope_m_per_km"]) / 1000.0) * x
    low = float(np.quantile(y_detrended, float(row["updated_tau_low"])))
    high = float(np.quantile(y_detrended, float(row["updated_tau_high"])))
    return (y_detrended >= low) & (y_detrended <= high)


def plot_profile(
    ax: plt.Axes,
    reach_id: str,
    file_name: str,
    row: pd.Series,
    compact_title: str,
    annotation_corner: str = "lower right",
) -> None:
    points_path, _ = observation_paths(reach_id, file_name)
    points = pd.read_csv(points_path)[["s_m", "height"]].dropna().sort_values("s_m")
    keep = retained_mask(points, row)
    x_km = points["s_m"].to_numpy(dtype=float) / 1000.0
    y = points["height"].to_numpy(dtype=float)
    ax.scatter(x_km, y, s=7, color=GRAY, alpha=0.20, linewidths=0, label="All spatially screened PIXC")
    ax.scatter(x_km[keep], y[keep], s=8, color=BLUE, alpha=0.72, linewidths=0, label="Retained PIXC")
    if keep.sum() >= 2:
        slope, intercept = np.polyfit(x_km[keep], y[keep], 1)
        line_x = np.array([x_km.min(), x_km.max()])
        ax.plot(line_x, slope * line_x + intercept, color=ORANGE, linewidth=1.7, label="Final retained-point OLS")
    ax.set_title(compact_title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("Along-channel distance (km)", fontsize=11.5)
    ax.set_ylabel("Corrected WSE (m)", fontsize=11.5)
    ax.tick_params(labelsize=10.5)
    ax.grid(alpha=0.22)
    annotation = (
        rf"$\tau$={float(row['updated_tau_low']):.2f}--{float(row['updated_tau_high']):.2f}" + "\n"
        + f"retained={int(keep.sum()):,}/{len(points):,}\n"
        + f"OLS={float(row['updated_postfilter_slope_m_per_km']):.3f} m km$^{{-1}}$"
    )
    annotation_position = {
        "lower right": (0.985, 0.03, "right", "bottom"),
        "upper left": (0.015, 0.97, "left", "top"),
    }
    x_text, y_text, horizontal_alignment, vertical_alignment = annotation_position[annotation_corner]
    ax.text(
        x_text, y_text, annotation,
        transform=ax.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": LIGHT_GRAY, "alpha": 0.92},
    )


def plot_tau(ax: plt.Axes, reach_id: str, file_name: str, row: pd.Series, config: dict[str, object], compact_title: str) -> None:
    _, slopes_path = observation_paths(reach_id, file_name)
    slopes = pd.read_csv(slopes_path)
    band, universal_k = build_reference_band(slopes, config)
    file_curve = slopes[slopes["file"] == file_name].sort_values("quantile")
    merged = file_curve.merge(band, on="quantile", how="left")
    ax.fill_between(merged["quantile"], merged["lower"], merged["upper"], color=LIGHT_GRAY, label="Reach-year universal band")
    ax.plot(merged["quantile"], merged["median"], color=GRAY, linewidth=1.2, label="Reach-year median")
    ax.plot(merged["quantile"], merged["slope_m_per_km"], color=BLUE, linewidth=1.7, label="Overpass quantile slope")
    ax.axvspan(float(row["updated_tau_low"]), float(row["updated_tau_high"]), color=GREEN, alpha=0.22, label="Selected tau range")
    ax.set_title(compact_title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel(r"Quantile $\tau$", fontsize=11.5)
    ax.set_ylabel("Quantile-regression slope (m km$^{-1}$)", fontsize=11.5)
    ax.tick_params(labelsize=10.5)
    ax.grid(alpha=0.22)


def figure_s2_examples(results: pd.DataFrame, config: dict[str, object]) -> None:
    examples = [
        (
            "66220000061",
            "SWOT_L2_HR_PIXC_014_173_089R_20240423T150328_20240423T150339_PGD0_01.nc",
            "Example 1",
        ),
        (
            "66405400041",
            "SWOT_L2_HR_PIXC_011_382_185L_20240228T121637_20240228T121648_PGD0_01.nc",
            "Example 2",
        ),
    ]
    # Designed for full text-width placement in the SI. Keeping the canvas
    # near page width prevents labels from becoming too small when LaTeX
    # scales the raster to \textwidth.
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), constrained_layout=True)
    for i, (reach_id, file_name, title) in enumerate(examples):
        row = get_result_row(results, reach_id, file_name)
        # Keep the statistics box away from the fitted profile in both examples.
        annotation_corner = "upper left"
        plot_profile(
            axes[i, 0],
            reach_id,
            file_name,
            row,
            f"({chr(97 + 2*i)}) {title}: filtered profile",
            annotation_corner=annotation_corner,
        )
        plot_tau(axes[i, 1], reach_id, file_name, row, config, f"({chr(98 + 2*i)}) {title}: quantile-slope selection")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="outside lower center", ncol=2, frameon=False, fontsize=10)
    fig.savefig(FIGURE_ROOT / "figure_S2_tau_examples.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_s3_diagnostic_cases(results: pd.DataFrame) -> None:
    cases = [
        (
            "66140000321",
            "SWOT_L2_HR_PIXC_016_117_073L_20240602T082955_20240602T083006_PGD0_01.nc",
            "Success: waterfall signal preserved",
        ),
        (
            "66405400051",
            "SWOT_L2_HR_PIXC_011_173_124L_20240221T005405_20240221T005416_PGD0_01.nc",
            "Failure: large retained vertical spread",
        ),
        (
            "66190200291",
            "SWOT_L2_HR_PIXC_017_145_080R_20240624T051640_20240624T051651_PGD0_01.nc",
            "Failure: large internal sampling gap",
        ),
    ]
    # Equal-width stacked panels give each diagnostic case the same visual
    # weight. The compact canvas leaves room for the caption on the SI page.
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.4), constrained_layout=True)
    for i, (reach_id, file_name, title) in enumerate(cases):
        row = get_result_row(results, reach_id, file_name)
        annotation_corner = "upper left"
        plot_profile(
            axes[i],
            reach_id,
            file_name,
            row,
            f"({chr(97+i)}) {title}",
            annotation_corner=annotation_corner,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    fig.savefig(FIGURE_ROOT / "figure_S3_failure_modes.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics, summary = collect_metrics()
    write_metrics(metrics)
    results = pd.read_csv(ALL_RESULTS, low_memory=False)
    config = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))

    figure_s1_workflow(metrics, summary)
    figure_s2_examples(results, config)
    figure_s3_diagnostic_cases(results)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
