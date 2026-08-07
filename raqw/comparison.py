"""Match RAQW slopes to RiverSP and generate reproducible comparison outputs."""

from __future__ import annotations

from datetime import timedelta
from io import StringIO
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
RIVERSP_FIELDS = (
    "reach_id,time_str,wse,slope,slope_u,slope_r_u,"
    "slope2,slope2_u,slope2_r_u,width,reach_q"
)


PIXC_TIME_PATTERN = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


def parse_pixc_time(file_name: str) -> pd.Timestamp:
    match = PIXC_TIME_PATTERN.search(str(file_name))
    if match is None:
        return pd.NaT
    return pd.to_datetime(
        match.group(1),
        format="%Y%m%dT%H%M%S",
        utc=True,
        errors="coerce",
    )


def hydrocron_response_to_table(text: str) -> pd.DataFrame:
    text = (text or "").strip()
    if not text:
        return pd.DataFrame()
    if text.startswith("{"):
        payload = json.loads(text)
        text = (payload.get("results", {}).get("csv", "") or "").strip()
    return pd.read_csv(StringIO(text)) if text else pd.DataFrame()


def fetch_riversp_table(
    filter_results: pd.DataFrame,
    collection_name: str = "SWOT_L2_HR_RiverSP_D",
    hydrocron_url: str = HYDROCRON_URL,
    fields: str = RIVERSP_FIELDS,
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    """Retrieve RiverSP reach rows covering each PIXC reach-year window."""

    observations = normalize_filter_results(filter_results)
    observations = observations.dropna(subset=["pixc_time_utc"]).copy()
    observations["year"] = observations["pixc_time_utc"].dt.year
    windows = (
        observations.groupby(["reach_id", "year"], as_index=False)
        .agg(start=("pixc_time_utc", "min"), end=("pixc_time_utc", "max"))
        .sort_values(["reach_id", "year"])
    )
    parts: list[pd.DataFrame] = []
    for _, window in windows.iterrows():
        start = pd.Timestamp(window["start"]).floor("D")
        end = pd.Timestamp(window["end"]).floor("D") + pd.Timedelta(days=1)
        params = {
            "feature": "Reach",
            "feature_id": str(window["reach_id"]),
            "start_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "output": "csv",
            "collection_name": collection_name,
            "fields": fields,
        }
        response = requests.get(hydrocron_url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        table = hydrocron_response_to_table(response.text)
        if table.empty:
            continue
        table["query_reach_id"] = str(window["reach_id"])
        table["query_year"] = int(window["year"])
        parts.append(table)
    if not parts:
        return pd.DataFrame()
    output = pd.concat(parts, ignore_index=True)
    if "reach_id" not in output.columns:
        output["reach_id"] = output["query_reach_id"]
    output["reach_id"] = output["reach_id"].fillna(output["query_reach_id"]).astype(str)
    if "slope" in output.columns:
        output["riversp_slope_m_per_km"] = (
            pd.to_numeric(output["slope"], errors="coerce") * 1000.0
        )
    output["riversp_time_utc"] = pd.to_datetime(
        output.get("time_str"),
        utc=True,
        errors="coerce",
    )
    return output


def normalize_filter_results(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize standalone RAQW and preserved publication result schemas."""

    output = results.copy()
    aliases = {
        "updated_postfilter_slope_m_per_km": "filtered_pixc_slope_m_per_km",
        "raqw_filtered_slope_m_per_km": "filtered_pixc_slope_m_per_km",
        "raw_pixc_slope_m_per_km": "raw_pixc_slope_m_per_km",
        "raqw_raw_slope_m_per_km": "raw_pixc_slope_m_per_km",
    }
    for source, target in aliases.items():
        if target not in output.columns and source in output.columns:
            output[target] = output[source]
    required = {"reach_id", "file", "filtered_pixc_slope_m_per_km"}
    missing = sorted(required - set(output.columns))
    if missing:
        raise ValueError(f"Filter result table is missing: {', '.join(missing)}")
    output["reach_id"] = output["reach_id"].astype(str)
    if "pixc_time_utc" in output.columns:
        parsed = pd.to_datetime(output["pixc_time_utc"], utc=True, errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    output["pixc_time_utc"] = parsed.fillna(output["file"].map(parse_pixc_time))
    output["obs_date_utc"] = output["pixc_time_utc"].dt.strftime("%Y-%m-%d")
    return output


def normalize_riversp_table(
    riversp: pd.DataFrame,
    slope_column: str = "riversp_slope_m_per_km",
    slope_scale: float = 1.0,
) -> pd.DataFrame:
    output = riversp.copy()
    if "reach_id" not in output.columns and "query_reach_id" in output.columns:
        output["reach_id"] = output["query_reach_id"]
    time_column = next(
        (
            name
            for name in ("riversp_time_utc", "time_utc", "time_str")
            if name in output.columns
        ),
        None,
    )
    if time_column is None:
        raise ValueError("RiverSP table must include riversp_time_utc, time_utc, or time_str.")
    if "reach_id" not in output.columns or slope_column not in output.columns:
        raise ValueError(
            f"RiverSP table must include reach_id and {slope_column!r}."
        )
    output["reach_id"] = output["reach_id"].astype(str)
    output["riversp_time_utc"] = pd.to_datetime(
        output[time_column],
        utc=True,
        errors="coerce",
    )
    output["riversp_slope_m_per_km"] = (
        pd.to_numeric(output[slope_column], errors="coerce") * float(slope_scale)
    )
    if "reach_q" not in output.columns:
        if "riversp_reach_q" in output.columns:
            output["reach_q"] = output["riversp_reach_q"]
        else:
            output["reach_q"] = np.nan
    output["reach_q"] = pd.to_numeric(output["reach_q"], errors="coerce")
    return output.dropna(subset=["riversp_time_utc", "riversp_slope_m_per_km"])


def attach_riversp_matches(
    filter_results: pd.DataFrame,
    riversp: pd.DataFrame,
    allowed_reach_q: tuple[int, ...] = (0, 1),
    maximum_time_delta: timedelta = timedelta(minutes=30),
) -> pd.DataFrame:
    results = normalize_filter_results(filter_results)
    reference = riversp.copy()
    grouped = {
        reach_id: group.sort_values("riversp_time_utc")
        for reach_id, group in reference.groupby("reach_id", sort=False)
    }

    rows: list[dict[str, object]] = []
    maximum_seconds = maximum_time_delta.total_seconds()
    for _, result in results.iterrows():
        row = result.to_dict()
        candidates = grouped.get(str(result["reach_id"]), pd.DataFrame()).copy()
        if candidates.empty or pd.isna(result["pixc_time_utc"]):
            row.update(
                {
                    "riversp_time_utc": pd.NaT,
                    "riversp_reach_q": np.nan,
                    "riversp_slope_m_per_km": np.nan,
                    "riversp_time_delta_s": np.nan,
                }
            )
            rows.append(row)
            continue

        candidates["time_delta_s"] = (
            candidates["riversp_time_utc"] - result["pixc_time_utc"]
        ).abs().dt.total_seconds()
        candidates["qa_rank"] = np.where(
            candidates["reach_q"].isin(allowed_reach_q),
            0,
            1,
        )
        match = candidates.sort_values(
            ["qa_rank", "time_delta_s", "riversp_time_utc"]
        ).iloc[0]
        if match["qa_rank"] != 0 or match["time_delta_s"] > maximum_seconds:
            row.update(
                {
                    "riversp_time_utc": pd.NaT,
                    "riversp_reach_q": np.nan,
                    "riversp_slope_m_per_km": np.nan,
                    "riversp_time_delta_s": np.nan,
                }
            )
        else:
            row.update(
                {
                    "riversp_time_utc": match["riversp_time_utc"],
                    "riversp_reach_q": match["reach_q"],
                    "riversp_slope_m_per_km": match["riversp_slope_m_per_km"],
                    "riversp_time_delta_s": match["time_delta_s"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_summary(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = {
        "filtered_pixc": "filtered_pixc_slope_m_per_km",
        "raw_pixc": "raw_pixc_slope_m_per_km",
    }
    for method, column in methods.items():
        if column not in matched.columns:
            continue
        pair = matched[[column, "riversp_slope_m_per_km"]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        difference = pair[column] - pair["riversp_slope_m_per_km"]
        rows.append(
            {
                "method": method,
                "n": int(len(pair)),
                "bias_m_per_km": float(difference.mean()) if len(pair) else np.nan,
                "mae_m_per_km": float(difference.abs().mean()) if len(pair) else np.nan,
                "rmse_m_per_km": (
                    float(np.sqrt(np.mean(np.square(difference))))
                    if len(pair)
                    else np.nan
                ),
                "pearson_r": (
                    float(pair[column].corr(pair["riversp_slope_m_per_km"]))
                    if len(pair) > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_agreement(matched: pd.DataFrame, output_path: Path) -> Path:
    methods = [
        ("filtered_pixc_slope_m_per_km", "Filtered PIXC", "#2A9D8F"),
        ("raw_pixc_slope_m_per_km", "Raw PIXC", "#737B80"),
    ]
    methods = [item for item in methods if item[0] in matched.columns]
    figure, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.7))
    axes_array = np.atleast_1d(axes)
    for axis, (column, label, color) in zip(axes_array, methods):
        pair = matched[[column, "riversp_slope_m_per_km"]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        axis.scatter(
            pair["riversp_slope_m_per_km"],
            pair[column],
            s=12,
            alpha=0.35,
            color=color,
            linewidths=0,
        )
        if not pair.empty:
            bounds = np.asarray(
                [pair.min().min(), pair.max().max()],
                dtype="float64",
            )
            axis.plot(bounds, bounds, color="#263238", linewidth=1, linestyle="--")
        axis.set_title(label)
        axis.set_xlabel("RiverSP slope (m km$^{-1}$)")
        axis.set_ylabel(f"{label} slope (m km$^{{-1}}$)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)
    return output_path


def run_comparison(
    filter_results_path: Path,
    riversp_path: Path,
    output_dir: Path,
    slope_column: str = "riversp_slope_m_per_km",
    slope_scale: float = 1.0,
    maximum_time_delta_minutes: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(filter_results_path, dtype={"reach_id": str})
    riversp = normalize_riversp_table(
        pd.read_csv(riversp_path, dtype={"reach_id": str, "query_reach_id": str}),
        slope_column=slope_column,
        slope_scale=slope_scale,
    )
    matched = attach_riversp_matches(
        results,
        riversp,
        maximum_time_delta=timedelta(minutes=maximum_time_delta_minutes),
    )
    summary = agreement_summary(matched)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_dir / "matched_pixc_riversp_slopes.csv", index=False)
    summary.to_csv(output_dir / "overall_agreement_with_riversp.csv", index=False)
    plot_agreement(matched, output_dir / "pixc_riversp_agreement.png")
    return matched, summary
