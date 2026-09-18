# Auto-extracted from riversp_updated_filter_comparison.ipynb.
# Keep old notebooks untouched; refactor clean code around this copy.

# %% Notebook cell 1
from __future__ import annotations

import io
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pandas.errors import EmptyDataError

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

RIVERSP_SLOPE_FIELDS = ("slope", "slope_u", "slope_r_u", "slope2", "slope2_u", "slope2_r_u")
RIVERSP_RENAMED_SLOPE_FIELDS = tuple(f"riversp_{field}" for field in RIVERSP_SLOPE_FIELDS)
RIVERSP_CACHE_COLUMNS = (
    "reach_id",
    "time_utc",
    "wse",
    *RIVERSP_RENAMED_SLOPE_FIELDS,
    "width",
    "reach_q",
)
RIVERSP_MATCH_COLUMNS = (
    "riversp_time_utc",
    "riversp_reach_q",
    "riversp_slope_native",
    "riversp_slope_m_per_km",
    "riversp_slope_u_native",
    "riversp_slope_u_m_per_km",
    "riversp_slope_r_u_native",
    "riversp_slope_r_u_m_per_km",
    "riversp_slope2_native",
    "riversp_slope2_m_per_km",
    "riversp_slope2_u_native",
    "riversp_slope2_u_m_per_km",
    "riversp_slope2_r_u_native",
    "riversp_slope2_r_u_m_per_km",
    "riversp_time_delta_s",
)


@dataclass(frozen=True)
class ComparisonConfig:
    project_root: Path = Path(r"C:\SWOT_universal_PIXC_quantile_filter")
    processed_root: Path = Path(r"C:\SWOT_universal_PIXC_quantile_filter\data\chile_reaches_with_valid_discharge_run\processed")
    experiment_root: Path = Path(r"C:\SWOT_universal_PIXC_quantile_filter\data\pipeline_2025_comparison_results")
    experiment_results_csv: Path = Path(r"C:\SWOT_universal_PIXC_quantile_filter\data\pipeline_2025_comparison_results\selected_reaches_applied_2024_to_2025_results.csv")
    output_root: Path = Path(r"C:\SWOT_universal_PIXC_quantile_filter\data\pipeline_2025_comparison_results\riversp_comparison")

    hydrocron_url: str = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
    collection_name: str = "SWOT_L2_HR_RiverSP_D"
    fields: str = "reach_id,time_str,wse,slope,slope_u,slope_r_u,slope2,slope2_u,slope2_r_u,width,reach_q"
    timeout_s: int = 60
    max_retries: int = 5
    sleep_min_s: float = 0.15
    sleep_max_s: float = 0.45

    allowed_reach_q: tuple[int, ...] = (0, 1)
    riversp_slope_to_m_per_km: float = 1000.0


cfg = ComparisonConfig()
cfg.output_root.mkdir(parents=True, exist_ok=True)
cfg

# %% Notebook cell 3
def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def request_with_retries(
    url: str,
    params: dict[str, Any] | None = None,
    timeout_s: int = 60,
    max_retries: int = 5,
) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params or {}, timeout=timeout_s)
            if response.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {response.status_code}")
            return response
        except Exception as exc:
            last_err = exc
            backoff = min(30, (2 ** (attempt - 1)) * 0.7) + random.uniform(0, 0.5)
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {max_retries} retries. Last error: {last_err}")


def hydrocron_response_to_df(text: str) -> pd.DataFrame:
    text = (text or "").strip()
    if not text:
        return pd.DataFrame()

    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return pd.DataFrame()
        csv_text = (obj.get("results", {}).get("csv", "") or "").strip()
        if not csv_text:
            return pd.DataFrame()
        try:
            return pd.read_csv(io.StringIO(csv_text))
        except EmptyDataError:
            return pd.DataFrame()

    try:
        return pd.read_csv(io.StringIO(text))
    except EmptyDataError:
        return pd.DataFrame()


def normalize_hydrocron_slope_df(df: pd.DataFrame, reach_id: str) -> pd.DataFrame:
    if df is None or df.empty or "time_str" not in df.columns:
        return pd.DataFrame(columns=RIVERSP_CACHE_COLUMNS)

    out = df.copy()
    out = out[out["time_str"].astype(str) != "no_data"].copy()
    t = pd.to_datetime(out["time_str"], errors="coerce", utc=True)
    out = out.loc[t.notna()].copy()
    out["time_utc"] = t.loc[t.notna()].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()

    if "reach_id" not in out.columns:
        out["reach_id"] = str(reach_id)
    else:
        out["reach_id"] = out["reach_id"].fillna(str(reach_id)).astype(str)

    out = out.rename(columns={field: f"riversp_{field}" for field in RIVERSP_SLOPE_FIELDS if field in out.columns})

    for column in ["wse", *RIVERSP_RENAMED_SLOPE_FIELDS, "width", "reach_q"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out[out["riversp_slope"].notna()].copy()
    return out[list(RIVERSP_CACHE_COLUMNS)].reset_index(drop=True)


def fetch_riversp_day(reach_id: str, date: pd.Timestamp, cfg: ComparisonConfig = cfg) -> pd.DataFrame:
    day = pd.Timestamp(date).tz_convert("UTC") if pd.Timestamp(date).tzinfo else pd.Timestamp(date).tz_localize("UTC")
    start_dt = day.normalize().to_pydatetime()
    end_dt = (day.normalize() + pd.Timedelta(days=1)).to_pydatetime()
    params = {
        "feature": "Reach",
        "feature_id": str(reach_id),
        "start_time": iso_z(start_dt),
        "end_time": iso_z(end_dt),
        "output": "csv",
        "collection_name": cfg.collection_name,
        "fields": cfg.fields,
    }
    time.sleep(random.uniform(cfg.sleep_min_s, cfg.sleep_max_s))
    response = request_with_retries(cfg.hydrocron_url, params=params, timeout_s=cfg.timeout_s, max_retries=cfg.max_retries)
    response.raise_for_status()
    raw = hydrocron_response_to_df(response.text)
    return normalize_hydrocron_slope_df(raw, str(reach_id))


def fetch_riversp_window(reach_id: str, start_dt: datetime, end_dt: datetime, cfg: ComparisonConfig = cfg) -> pd.DataFrame:
    params = {
        "feature": "Reach",
        "feature_id": str(reach_id),
        "start_time": iso_z(start_dt),
        "end_time": iso_z(end_dt),
        "output": "csv",
        "collection_name": cfg.collection_name,
        "fields": cfg.fields,
    }
    time.sleep(random.uniform(cfg.sleep_min_s, cfg.sleep_max_s))
    response = request_with_retries(cfg.hydrocron_url, params=params, timeout_s=cfg.timeout_s, max_retries=cfg.max_retries)
    response.raise_for_status()
    raw = hydrocron_response_to_df(response.text)
    return normalize_hydrocron_slope_df(raw, str(reach_id))

# %% Notebook cell 5
PIXC_TIME_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


def parse_pixc_start_time(file_name: str) -> pd.Timestamp:
    match = PIXC_TIME_RE.search(str(file_name))
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%S", utc=True, errors="coerce")


def load_updated_results(cfg: ComparisonConfig = cfg) -> pd.DataFrame:
    files = sorted(cfg.experiment_root.glob("reach_*/[0-9][0-9][0-9][0-9]/updated_tau_window_results.csv"))
    rows = []
    for path in files:
        try:
            part = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if part.empty:
            continue
        rows.append(part)

    if rows:
        df = pd.concat(rows, ignore_index=True)
        df = df.drop_duplicates(["reach_id", "year", "file"], keep="first")
        rebuilt_csv = cfg.output_root / "all_reach_updated_tau_window_results.csv"
        df.to_csv(rebuilt_csv, index=False)
        print(f"Loaded {len(df):,} rows from {len(files):,} per-reach result files.")
        print(f"Wrote rebuilt combined table: {rebuilt_csv}")
    else:
        df = pd.read_csv(cfg.experiment_results_csv)
        print(f"Loaded fallback combined_results.csv with {len(df):,} rows.")
    df["reach_id"] = df["reach_id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["pixc_time_utc"] = df["file"].map(parse_pixc_start_time)
    return df


def load_raw_slope_rows(cfg: ComparisonConfig = cfg) -> pd.DataFrame:
    rows = []
    for path in cfg.processed_root.glob("reach_*/[0-9][0-9][0-9][0-9]/tau_trim_results.csv"):
        try:
            part = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if part.empty or "file" not in part.columns or "ols_slope_raw_m_per_km" not in part.columns:
            continue
        part = part[["reach_id", "year", "file", "ols_slope_raw_m_per_km"]].copy()
        part["reach_id"] = part["reach_id"].astype(str)
        part["year"] = pd.to_numeric(part["year"], errors="coerce").astype("Int64")
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["reach_id", "year", "file", "raw_pixc_slope_m_per_km"])
    raw = pd.concat(rows, ignore_index=True)
    raw = raw.rename(columns={"ols_slope_raw_m_per_km": "raw_pixc_slope_m_per_km"})
    return raw.drop_duplicates(["reach_id", "year", "file"], keep="first")


def load_summary_times(cfg: ComparisonConfig = cfg) -> pd.DataFrame:
    rows = []
    for path in cfg.processed_root.glob("reach_*/[0-9][0-9][0-9][0-9]/summary.csv"):
        try:
            part = pd.read_csv(path, usecols=lambda c: c in {"reach_id", "file", "t0_utc", "reach_q"})
        except Exception:
            continue
        if part.empty or "file" not in part.columns:
            continue
        year = int(path.parent.name)
        part["year"] = year
        part["reach_id"] = part["reach_id"].astype(str)
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["reach_id", "year", "file", "summary_t0_utc", "summary_reach_q"])
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"t0_utc": "summary_t0_utc", "reach_q": "summary_reach_q"})
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out.drop_duplicates(["reach_id", "year", "file"], keep="first")


updated = load_updated_results(cfg)
raw_slopes = load_raw_slope_rows(cfg)
summary_times = load_summary_times(cfg)

comparison = updated.merge(raw_slopes, on=["reach_id", "year", "file"], how="left")
comparison = comparison.merge(summary_times, on=["reach_id", "year", "file"], how="left")
summary_dt = pd.to_datetime(comparison["summary_t0_utc"], utc=True, errors="coerce")
comparison["pixc_time_utc"] = summary_dt.fillna(comparison["pixc_time_utc"])
comparison["obs_date_utc"] = comparison["pixc_time_utc"].dt.strftime("%Y-%m-%d")

print(f"updated experiment rows: {len(updated):,}")
print(f"comparison rows: {len(comparison):,}")
print(f"rows with raw PIXC slope: {comparison['raw_pixc_slope_m_per_km'].notna().sum():,}")
print(f"rows with PIXC observation date: {comparison['obs_date_utc'].notna().sum():,}")
comparison.head()

# %% Notebook cell 7
REFRESH_RIVERSP_CACHE = True
RIVERSP_CACHE_CSV = cfg.output_root / "riversp_daily_cache.csv"


def build_riversp_daily_cache(comparison: pd.DataFrame, refresh: bool = False, cfg: ComparisonConfig = cfg) -> pd.DataFrame:
    if RIVERSP_CACHE_CSV.exists() and not refresh:
        cache = pd.read_csv(RIVERSP_CACHE_CSV)
        cache["query_reach_id"] = cache["query_reach_id"].astype(str)
        return cache

    obs_days = (
        comparison[["reach_id", "year", "obs_date_utc"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["reach_id", "year", "obs_date_utc"])
        .reset_index(drop=True)
    )
    windows = (
        obs_days
        .assign(obs_dt=lambda d: pd.to_datetime(d["obs_date_utc"], utc=True, errors="coerce"))
        .dropna(subset=["obs_dt"])
        .groupby(["reach_id", "year"], as_index=False)
        .agg(start_date=("obs_dt", "min"), end_date=("obs_dt", "max"), n_obs_days=("obs_dt", "size"))
    )
    print(f"Unique observation reach-days to cover: {len(obs_days):,}")
    print(f"Hydrocron reach/year window queries to run: {len(windows):,}")

    rows = []
    errors = []
    for idx, row in windows.iterrows():
        rid = str(row["reach_id"])
        start_dt = pd.Timestamp(row["start_date"]).normalize().to_pydatetime()
        end_dt = (pd.Timestamp(row["end_date"]).normalize() + pd.Timedelta(days=1)).to_pydatetime()
        try:
            window_df = fetch_riversp_window(rid, start_dt, end_dt, cfg=cfg)
            if not window_df.empty:
                window_df = window_df.copy()
                window_df["query_reach_id"] = rid
                window_df["query_date_utc"] = pd.to_datetime(window_df["time_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
                window_df["query_window_start_utc"] = iso_z(start_dt)
                window_df["query_window_end_utc"] = iso_z(end_dt)
                rows.append(window_df)
        except Exception as exc:
            errors.append({"query_reach_id": rid, "year": row["year"], "start_date": row["start_date"], "end_date": row["end_date"], "error": str(exc)})
        if (idx + 1) % 50 == 0:
            print(f"finished {idx + 1:,}/{len(windows):,} reach/year queries")

    cache = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[*RIVERSP_CACHE_COLUMNS, "query_reach_id", "query_date_utc", "query_window_start_utc", "query_window_end_utc"]
    )
    if not cache.empty:
        wanted = obs_days[["reach_id", "obs_date_utc"]].rename(columns={"reach_id": "query_reach_id", "obs_date_utc": "query_date_utc"})
        wanted["query_reach_id"] = wanted["query_reach_id"].astype(str)
        cache["query_reach_id"] = cache["query_reach_id"].astype(str)
        cache = cache.merge(wanted.drop_duplicates(), on=["query_reach_id", "query_date_utc"], how="inner")
    cache.to_csv(RIVERSP_CACHE_CSV, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(cfg.output_root / "riversp_daily_cache_errors.csv", index=False)
        print(f"Hydrocron errors: {len(errors):,}; see riversp_daily_cache_errors.csv")

    return cache


riversp_cache = build_riversp_daily_cache(comparison, refresh=REFRESH_RIVERSP_CACHE, cfg=cfg)
print(f"RiverSP cached rows: {len(riversp_cache):,}")
riversp_cache.head()

# %% Notebook cell 9
def choose_riversp_match(candidates: pd.DataFrame, pixc_time: pd.Timestamp, cfg: ComparisonConfig = cfg) -> pd.Series | None:
    if candidates.empty:
        return None
    cand = candidates.copy()
    cand["time_dt"] = pd.to_datetime(cand["time_utc"], utc=True, errors="coerce")
    cand["qa_rank"] = np.where(cand["reach_q"].isin(cfg.allowed_reach_q), 0, 1)
    if pd.notna(pixc_time):
        cand["time_delta_s"] = (cand["time_dt"] - pixc_time).abs().dt.total_seconds()
    else:
        cand["time_delta_s"] = np.inf
    cand = cand.sort_values(["qa_rank", "time_delta_s", "time_utc"], na_position="last")
    return cand.iloc[0]


def attach_riversp_matches(comparison: pd.DataFrame, riversp_cache: pd.DataFrame, cfg: ComparisonConfig = cfg) -> pd.DataFrame:
    cache = riversp_cache.copy()
    if cache.empty:
        out = comparison.copy()
        for column in RIVERSP_MATCH_COLUMNS:
            out[column] = np.nan
        return out

    cache["query_reach_id"] = cache["query_reach_id"].astype(str)
    grouped = {key: part for key, part in cache.groupby(["query_reach_id", "query_date_utc"], sort=False)}

    rows = []
    for _, row in comparison.iterrows():
        out_row = row.to_dict()
        key = (str(row["reach_id"]), row["obs_date_utc"])
        match = choose_riversp_match(grouped.get(key, pd.DataFrame()), row["pixc_time_utc"], cfg=cfg)
        if match is not None:
            out_row["riversp_time_utc"] = match["time_utc"]
            out_row["riversp_reach_q"] = match["reach_q"]
            for field in RIVERSP_RENAMED_SLOPE_FIELDS:
                native_col = f"{field}_native"
                per_km_col = f"{field}_m_per_km"
                native_value = pd.to_numeric(match.get(field, np.nan), errors="coerce")
                out_row[native_col] = native_value
                out_row[per_km_col] = float(native_value) * cfg.riversp_slope_to_m_per_km if pd.notna(native_value) else np.nan
            out_row["riversp_time_delta_s"] = match["time_delta_s"]
        else:
            for column in RIVERSP_MATCH_COLUMNS:
                out_row[column] = pd.NA if column == "riversp_time_utc" else np.nan
        rows.append(out_row)
    return pd.DataFrame(rows)


matched = attach_riversp_matches(comparison, riversp_cache, cfg=cfg)
matched_csv = cfg.output_root / "matched_pixc_riversp_slopes.csv"
matched.to_csv(matched_csv, index=False)

print(f"matched rows: {len(matched):,}")
print(f"rows with RiverSP slope: {matched['riversp_slope_m_per_km'].notna().sum():,}")
print(f"rows with RiverSP enhanced slope2: {matched['riversp_slope2_m_per_km'].notna().sum():,}")
print(f"wrote: {matched_csv}")
matched[["reach_id", "year", "file", "obs_date_utc", "raw_pixc_slope_m_per_km", "updated_postfilter_slope_m_per_km", "riversp_slope_m_per_km", "riversp_slope2_m_per_km", "riversp_reach_q", "riversp_time_delta_s"]].head()

# %% Notebook cell 11
METHOD_COLUMNS = {
    "raw_pixc": "raw_pixc_slope_m_per_km",
    "updated_filter": "updated_postfilter_slope_m_per_km",
    "riversp": "riversp_slope_m_per_km",
    "riversp_slope2": "riversp_slope2_m_per_km",
}
PLOT_METHODS = ("updated_filter", "riversp", "riversp_slope2")
PLOT_METHOD_LABELS = {
    "updated_filter": "Updated filter",
    "riversp": "RiverSP slope",
    "riversp_slope2": "RiverSP slope2",
}
PLOT_METHOD_COLORS = {
    "updated_filter": "#2A9D8F",
    "riversp": "#E76F51",
    "riversp_slope2": "#457B9D",
}


def add_within_method_residuals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for method, col in METHOD_COLUMNS.items():
        mean_col = f"{method}_reach_year_mean_m_per_km"
        resid_col = f"{method}_residual_from_reach_year_mean_m_per_km"
        out[mean_col] = out.groupby(["reach_id", "year"])[col].transform("mean")
        out[resid_col] = out[col] - out[mean_col]
    out["raw_minus_riversp_m_per_km"] = out["raw_pixc_slope_m_per_km"] - out["riversp_slope_m_per_km"]
    out["updated_minus_riversp_m_per_km"] = out["updated_postfilter_slope_m_per_km"] - out["riversp_slope_m_per_km"]
    out["raw_minus_riversp_slope2_m_per_km"] = out["raw_pixc_slope_m_per_km"] - out["riversp_slope2_m_per_km"]
    out["updated_minus_riversp_slope2_m_per_km"] = out["updated_postfilter_slope_m_per_km"] - out["riversp_slope2_m_per_km"]
    return out


def spread_stats(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    if x.size == 0:
        return pd.Series({"n": 0, "mean": np.nan, "std": np.nan, "rmse": np.nan, "mad": np.nan, "iqr": np.nan})
    med = np.median(x)
    return pd.Series(
        {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
            "rmse": float(np.sqrt(np.mean(x**2))),
            "mad": float(np.median(np.abs(x - med))),
            "iqr": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        }
    )


residuals = add_within_method_residuals(matched)
residuals.to_csv(cfg.output_root / "matched_pixc_riversp_slopes_with_residuals.csv", index=False)

long_rows = []
for method in METHOD_COLUMNS:
    resid_col = f"{method}_residual_from_reach_year_mean_m_per_km"
    part = residuals[["reach_id", "year", "file", resid_col]].copy()
    part = part.rename(columns={resid_col: "residual_m_per_km"})
    part["method"] = method
    long_rows.append(part)
residuals_long = pd.concat(long_rows, ignore_index=True).dropna(subset=["residual_m_per_km"])

reach_year_summary = (
    residuals_long
    .groupby(["reach_id", "year", "method"])["residual_m_per_km"]
    .apply(spread_stats)
    .unstack()
    .reset_index()
)
reach_year_summary.to_csv(cfg.output_root / "reach_year_residual_spread_by_method.csv", index=False)

method_summary = (
    residuals_long
    .groupby("method")["residual_m_per_km"]
    .apply(spread_stats)
    .unstack()
    .reset_index()
)
method_summary.to_csv(cfg.output_root / "overall_residual_spread_by_method.csv", index=False)

method_summary

# %% Notebook cell 13
agreement_rows = []
for label, col in {
    "raw_minus_riversp": "raw_minus_riversp_m_per_km",
    "updated_minus_riversp": "updated_minus_riversp_m_per_km",
    "updated_minus_riversp_slope2": "updated_minus_riversp_slope2_m_per_km",
}.items():
    part = residuals[["reach_id", "year", "file", col]].copy()
    part = part.rename(columns={col: "difference_m_per_km"})
    part["comparison"] = label
    agreement_rows.append(part)

agreement_long = pd.concat(agreement_rows, ignore_index=True).dropna(subset=["difference_m_per_km"])

agreement_summary = (
    agreement_long
    .groupby("comparison")["difference_m_per_km"]
    .apply(spread_stats)
    .unstack()
    .reset_index()
)
agreement_summary.to_csv(cfg.output_root / "overall_agreement_with_riversp.csv", index=False)

agreement_by_reach_year = (
    agreement_long
    .groupby(["reach_id", "year", "comparison"])["difference_m_per_km"]
    .apply(spread_stats)
    .unstack()
    .reset_index()
)
agreement_by_reach_year.to_csv(cfg.output_root / "reach_year_agreement_with_riversp.csv", index=False)

agreement_summary

# %% Notebook cell 15
wide = reach_year_summary.pivot(index=["reach_id", "year"], columns="method", values=["n", "std", "rmse", "mad", "iqr"]).reset_index()
wide.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in wide.columns.to_flat_index()]

for metric in ["std", "rmse", "mad", "iqr"]:
    raw_col = f"{metric}_raw_pixc"
    upd_col = f"{metric}_updated_filter"
    if raw_col in wide.columns and upd_col in wide.columns:
        wide[f"{metric}_reduction_frac_raw_to_updated"] = (wide[raw_col] - wide[upd_col]) / wide[raw_col]
        wide[f"{metric}_reduction_pct_raw_to_updated"] = 100.0 * wide[f"{metric}_reduction_frac_raw_to_updated"]

noise_reduction_csv = cfg.output_root / "reach_year_noise_reduction_raw_to_updated.csv"
wide.to_csv(noise_reduction_csv, index=False)
print(f"wrote: {noise_reduction_csv}")

reduction_cols = [c for c in wide.columns if c.endswith("_reduction_pct_raw_to_updated")]
wide[["reach_id", "year"] + reduction_cols].describe()

# %% Notebook cell 17
plots_dir = cfg.output_root / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
plot_data = [
    residuals_long.loc[residuals_long["method"] == method, "residual_m_per_km"].dropna()
    for method in PLOT_METHODS
]
ax.boxplot(plot_data, labels=[PLOT_METHOD_LABELS[method] for method in PLOT_METHODS], showfliers=False)
ax.axhline(0, color="0.3", linewidth=1)
ax.set_ylabel("Slope residual from reach/year mean (m/km)")
ax.set_title("Residual Spread by Slope Product")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(plots_dir / "residual_spread_by_method_boxplot.png", dpi=200)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
agreement_plot_data = [
    agreement_long.loc[agreement_long["comparison"] == label, "difference_m_per_km"].dropna()
    for label in ["updated_minus_riversp", "updated_minus_riversp_slope2"]
]
ax.boxplot(
    agreement_plot_data,
    labels=["Updated - RiverSP slope", "Updated - RiverSP slope2"],
    showfliers=False,
)
ax.axhline(0, color="0.3", linewidth=1)
ax.set_ylabel("Updated-filter slope difference (m/km)")
ax.set_title("Updated Filter Agreement with RiverSP Slopes")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(plots_dir / "agreement_with_riversp_boxplot.png", dpi=200)
plt.show()

if "mad_reduction_pct_raw_to_updated" in wide.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = wide["mad_reduction_pct_raw_to_updated"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(vals, bins=30, color="#2A9D8F", edgecolor="white")
    ax.axvline(0, color="0.3", linewidth=1)
    ax.set_xlabel("MAD reduction from raw to updated (%)")
    ax.set_ylabel("Reach/year count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "mad_reduction_histogram.png", dpi=200)
    plt.show()

# %% Notebook cell 19
plots_dir = cfg.output_root / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

spread_for_plot = method_summary.set_index("method").loc[list(PLOT_METHODS), ["std", "mad", "iqr"]]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(spread_for_plot.index))
bar_w = 0.25
for offset, metric in zip([-bar_w, 0, bar_w], ["std", "mad", "iqr"]):
    ax.bar(x + offset, spread_for_plot[metric], width=bar_w, label=metric.upper())
ax.set_xticks(x)
ax.set_xticklabels([PLOT_METHOD_LABELS[method] for method in PLOT_METHODS])
ax.set_ylabel("Residual spread (m/km)")
ax.set_title("Pooled Reach/Year Residual Spread")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(plots_dir / "pooled_residual_spread_metrics.png", dpi=200)
plt.show()

reduction_specs = [
    ("std_reduction_pct_raw_to_updated", "STD"),
    ("rmse_reduction_pct_raw_to_updated", "RMSE"),
    ("mad_reduction_pct_raw_to_updated", "MAD"),
    ("iqr_reduction_pct_raw_to_updated", "IQR"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, (column, label) in zip(axes.ravel(), reduction_specs):
    vals = pd.to_numeric(wide[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(vals, bins=35, color="#2A9D8F", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="0.25", linewidth=1)
    ax.axvline(vals.median(), color="#E76F51", linewidth=2, label=f"median {vals.median():.1f}%")
    ax.set_title(f"{label} Reduction")
    ax.set_ylabel("Reach count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
for ax in axes[-1, :]:
    ax.set_xlabel("Reduction from raw to updated (%)")
fig.tight_layout()
fig.savefig(plots_dir / "reach_level_reduction_histograms.png", dpi=200)
plt.show()

count_rows = []
for column, label in reduction_specs:
    vals = pd.to_numeric(wide[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    count_rows.append({"metric": label, "improved": int((vals > 0).sum()), "worsened": int((vals < 0).sum()), "unchanged": int((vals == 0).sum())})
counts_df = pd.DataFrame(count_rows).set_index("metric")
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(counts_df.index, counts_df["improved"], label="Improved", color="#2A9D8F")
ax.bar(counts_df.index, -counts_df["worsened"], label="Worsened", color="#E76F51")
ax.axhline(0, color="0.25", linewidth=1)
ax.set_ylabel("Reach count")
ax.set_title("Reach-Level Outcomes")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(plots_dir / "reach_level_improved_worsened_counts.png", dpi=200)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, metric, title in zip(axes, ["mad", "iqr"], ["MAD", "IQR"]):
    raw_col = f"{metric}_raw_pixc"
    upd_col = f"{metric}_updated_filter"
    plot_df = wide[[raw_col, upd_col]].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(plot_df[raw_col], plot_df[upd_col], s=20, alpha=0.65, color="#457B9D", edgecolors="none")
    lim = np.nanmax([plot_df[raw_col].max(), plot_df[upd_col].max()])
    ax.plot([0, lim], [0, lim], color="0.25", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Raw PIXC {title} (m/km)")
    ax.set_ylabel(f"Updated filter {title} (m/km)")
    ax.set_title(f"Reach-Level {title}: Raw vs Updated")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(plots_dir / "raw_vs_updated_robust_spread_scatter.png", dpi=200)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
for method in PLOT_METHODS:
    label = PLOT_METHOD_LABELS[method]
    color = PLOT_METHOD_COLORS[method]
    vals = residuals_long.loc[residuals_long["method"] == method, "residual_m_per_km"].abs().dropna().sort_values().to_numpy()
    if vals.size == 0:
        continue
    y = np.arange(1, vals.size + 1) / vals.size
    ax.plot(vals, y, label=label, color=color, linewidth=2)
ax.set_xlim(0, residuals_long["residual_m_per_km"].abs().quantile(0.98))
ax.set_xlabel("Absolute residual from reach/year mean (m/km)")
ax.set_ylabel("Cumulative fraction")
ax.set_title("Absolute Residual Distribution")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(plots_dir / "absolute_residual_cdf_by_method.png", dpi=200)
plt.show()

# %% Notebook cell 21
try:
    from scipy.stats import binomtest, wilcoxon

    checks = []
    for metric in ["std", "rmse", "mad", "iqr"]:
        raw_col = f"{metric}_raw_pixc"
        upd_col = f"{metric}_updated_filter"
        if raw_col not in wide.columns or upd_col not in wide.columns:
            continue
        paired = wide[[raw_col, upd_col]].replace([np.inf, -np.inf], np.nan).dropna()
        paired = paired[paired[raw_col] > 0]
        diff = paired[raw_col] - paired[upd_col]
        improved = int((diff > 0).sum())
        worsened = int((diff < 0).sum())
        unchanged = int((diff == 0).sum())
        sign_n = improved + worsened
        sign_p = binomtest(improved, sign_n, 0.5, alternative="greater").pvalue if sign_n else np.nan
        wilcoxon_p = wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue if sign_n else np.nan
        checks.append(
            {
                "metric": metric,
                "paired_reach_years": int(len(paired)),
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
                "median_raw_minus_updated": float(np.median(diff)) if len(diff) else np.nan,
                "sign_test_p_greater": sign_p,
                "wilcoxon_p_greater": wilcoxon_p,
            }
        )
    paired_check = pd.DataFrame(checks)
    paired_check.to_csv(cfg.output_root / "paired_noise_reduction_sign_check.csv", index=False)
    paired_check
except Exception as exc:
    print(f"Optional scipy check skipped: {exc}")

