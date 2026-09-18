# Auto-extracted core definitions from quantile_filter_by_reach_and_year_chile_discharge.ipynb.
# Contains definition cells only; no batch run cell.

# %% Notebook cell 0
from __future__ import annotations

import os
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd
import geopandas as gpd
import earthaccess
import netCDF4 as nc
import requests

from shapely.geometry import Polygon, box, MultiLineString
from shapely.ops import unary_union, linemerge
from scipy.stats import linregress
import statsmodels.formula.api as smf


# =========================
# Config + path conventions
# =========================

@dataclass(frozen=True)
class PixcQuantileConfig:
    project_root: Path

    # reach geometry source
    reach_gpkg: Path
    reach_layer: str | None = None
    reach_id_field: str = "reach_id"

    # earthdata collection short names
    pixc_short_name: str = "SWOT_L2_HR_PIXC_D"
    riversp_collection_name: str = "SWOT_L2_HR_RiverSP_D"   # Hydrocron collection_name

    # Hydrocron QA gating
    use_hydrocron_qa_gate: bool = True
    hydrocron_quality_field: str = "reach_q"
    hydrocron_allowed_quality: tuple[int, ...] = (0,1)   # strict: only 0 passes

    # processing params
    utm_epsg: int = 32718
    hydrocron_window_minutes: int = 30
    width_factor: float = 1.0
    height_correction: bool = True

    # Ãâ€ž-trim workflow params (your existing defaults)
    use_dynamic_k: bool = True          # NEW: if True, compute k(Ãâ€ž) from target coverage
    target_coverage: float = 0.95       # NEW: fraction of files to include at each Ãâ€ž (e.g., 0.95)
    k: float = 6                        # kept for fallback when use_dynamic_k=False

    min_files_per_quantile: int = 5
    min_points_per_granule: int = 50
    min_points_after_trim: int = 10
    make_plots: bool = True
    save_outputs: bool = True
    out_prefix: str = "tau_trim"

    def raw_pixc_dir(self, reach_id: str | int, year: int) -> Path:
        return self.project_root / "data" / "raw" / "pixc_d" / f"reach_{reach_id}" / str(year)

    def processed_dir(self, reach_id: str | int, year: int) -> Path:
        return self.project_root / "data" / "processed" / f"reach_{reach_id}" / str(year)

    def points_dir(self, reach_id: str | int, year: int) -> Path:
        return self.processed_dir(reach_id, year) / "pixc_points"


# =========================
# Helper functions
# =========================

def year_to_temporal(year: int):
    return (f"{year}-01-01T00:00:00", f"{year}-12-31T23:59:59")


def get_reach_geometry(cfg: PixcQuantileConfig, reach_id: str | int):
    gdf = gpd.read_file(cfg.reach_gpkg, layer=cfg.reach_layer)

    if cfg.reach_id_field not in gdf.columns:
        raise ValueError(
            f"Column '{cfg.reach_id_field}' not found in {cfg.reach_gpkg}. "
            f"Available columns: {list(gdf.columns)}"
        )

    # normalize types
    gdf[cfg.reach_id_field] = gdf[cfg.reach_id_field].astype(str)
    rid = str(reach_id)

    reach = gdf[gdf[cfg.reach_id_field] == rid]
    if reach.empty:
        raise ValueError(f"Reach {rid} not found in {cfg.reach_gpkg}.")

    reach = reach.to_crs(4326)
    return reach.iloc[0].geometry


def granule_footprint_from_umm(g):
    umm = g.render_dict.get("umm", {})
    geom = (
        umm.get("SpatialExtent", {})
           .get("HorizontalSpatialDomain", {})
           .get("Geometry", {})
    )

    gpolys = geom.get("GPolygons")
    if gpolys:
        polys = []
        for gp in gpolys:
            pts = gp.get("Boundary", {}).get("Points", [])
            if len(pts) < 3:
                continue
            coords = [(p["Longitude"], p["Latitude"]) for p in pts]
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            p = Polygon(coords)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
        return unary_union(polys) if polys else None

    rects = geom.get("BoundingRectangles")
    if rects:
        rect_geoms = []
        for r in rects:
            w = r.get("WestBoundingCoordinate")
            s = r.get("SouthBoundingCoordinate")
            e = r.get("EastBoundingCoordinate")
            n = r.get("NorthBoundingCoordinate")
            if None in (w, s, e, n):
                continue
            rect_geoms.append(box(w, s, e, n))
        return unary_union(rect_geoms) if rect_geoms else None

    return None


def apply_filter(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # your exact classification filter
    return df[(df["classification"] > 2) & (df["classification"] != 5)].copy()


HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"


def hydrocron_response_to_df(response_text: str) -> pd.DataFrame:
    text = response_text.strip()
    if text.startswith("{"):
        obj = json.loads(text)
        csv_text = obj.get("results", {}).get("csv", "")
        return pd.read_csv(StringIO(csv_text)) if csv_text else pd.DataFrame()
    return pd.read_csv(StringIO(text))


def hydrocron_width_and_quality_for_reach_time(
    reach_id: str | int,
    target_time_utc: datetime,
    window_minutes: int,
    collection_name: str,
    quality_field: str = "reach_q",
) -> dict:
    """
    Returns the closest Hydrocron Reach row to target_time_utc within +/- window_minutes.

    Output dict includes:
      - width (float)
      - quality (int or NaN if missing)
      - time_str (datetime UTC)
      - dt_sec (float)
    """
    start = (target_time_utc - timedelta(minutes=window_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (target_time_utc + timedelta(minutes=window_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    fields = f"reach_id,time_str,width,{quality_field}"

    params = {
        "feature": "Reach",
        "feature_id": str(reach_id),
        "start_time": start,
        "end_time": end,
        "output": "csv",
        "collection_name": collection_name,
        "fields": fields,
    }

    r = requests.get(HYDROCRON_URL, params=params, timeout=60)
    r.raise_for_status()

    dfw = hydrocron_response_to_df(r.text)
    if dfw.empty:
        raise ValueError("Hydrocron returned 0 rows for tight window.")

    # normalize time + choose closest row
    dfw["time_str"] = pd.to_datetime(dfw["time_str"], utc=True, errors="coerce")
    dfw = dfw.dropna(subset=["time_str"])
    if dfw.empty:
        raise ValueError("Hydrocron rows had invalid time_str.")

    dfw["dt_sec"] = (dfw["time_str"] - target_time_utc).abs().dt.total_seconds()
    row = dfw.sort_values("dt_sec").iloc[0]

    # width required
    if "width" not in dfw.columns:
        raise ValueError("Hydrocron response missing required field: width")

    width = float(row["width"])

    # quality optional (set NaN if missing)
    if quality_field in dfw.columns:
        try:
            quality = int(row[quality_field])
        except Exception:
            quality = np.nan
    else:
        quality = np.nan

    return {
        "width": width,
        "quality": quality,
        "time_str": row["time_str"],
        "dt_sec": float(row["dt_sec"]),
        "quality_field": quality_field,
    }

    r = requests.get(HYDROCRON_URL, params=params, timeout=60)
    r.raise_for_status()

    dfw = hydrocron_response_to_df(r.text)
    if dfw.empty:
        raise ValueError("Hydrocron returned 0 rows for tight window.")

    dfw["time_str"] = pd.to_datetime(dfw["time_str"], utc=True)
    dfw["dt_sec"] = (dfw["time_str"] - target_time_utc).abs().dt.total_seconds()
    row = dfw.sort_values("dt_sec").iloc[0]
    return float(row["width"])


_GRANULE_TIME_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


def parse_pixc_start_time_from_name(nc_path: Path) -> datetime:
    m = _GRANULE_TIME_RE.search(nc_path.name)
    if not m:
        raise ValueError(f"Could not parse times from filename: {nc_path.name}")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def build_reach_line_utm(cfg: PixcQuantileConfig, reach_id: str | int):
    rgdf = gpd.read_file(cfg.reach_gpkg, layer=cfg.reach_layer)
    rgdf[cfg.reach_id_field] = rgdf[cfg.reach_id_field].astype(str)
    reach = rgdf.loc[rgdf[cfg.reach_id_field] == str(reach_id)].copy()
    if reach.empty:
        raise ValueError(f"reach_id={reach_id} not found in {cfg.reach_gpkg}")

    if reach.crs is None or reach.crs.to_epsg() != 4326:
        reach = reach.to_crs(epsg=4326)

    reach_utm = reach.to_crs(epsg=cfg.utm_epsg)
    geom = reach_utm.geometry.iloc[0]
    if isinstance(geom, MultiLineString):
        geom = linemerge(geom)
    return geom


# =========================
# 1) Download step
# =========================

def download_pixc_by_reach_year_covers(
    cfg: PixcQuantileConfig,
    reach_id: str | int,
    year: int,
    drop_if_no_footprint: bool = True,
) -> list[Path]:
    out_dir = cfg.raw_pixc_dir(reach_id, year)
    out_dir.mkdir(parents=True, exist_ok=True)

    reach_geom = get_reach_geometry(cfg, reach_id)
    reach_bbox = reach_geom.bounds  # (minx, miny, maxx, maxy)

    earthaccess.login()

    results = earthaccess.search_data(
        short_name=cfg.pixc_short_name,
        bounding_box=reach_bbox,
        temporal=year_to_temporal(year),
    )

    kept, dropped, nofoot = [], [], []
    for g in results:
        fp = granule_footprint_from_umm(g)
        if fp is None:
            nofoot.append(g)
            if drop_if_no_footprint:
                dropped.append(g)
            else:
                kept.append(g)
            continue

        if fp.covers(reach_geom):
            kept.append(g)
        else:
            dropped.append(g)

    print(f"Reach: {reach_id} | Year: {year}")
    print(f"Search returned: {len(results)}")
    print(f"Kept (footprint covers reach): {len(kept)}")
    print(f"Dropped: {len(dropped)}")
    print(f"No-footprint: {len(nofoot)}")
    print(f"Downloading to: {out_dir}")

    paths = earthaccess.download(kept, str(out_dir), threads=2)  # or 2
    # earthaccess may return strings
    paths = [Path(p) for p in paths]

    print(f"Downloaded {len(paths)} files.")
    return paths


# =========================
# 2) Batch quantile-slope builder
# =========================

def process_one_pixc_file(
    cfg: PixcQuantileConfig,
    nc_path: Path,
    reach_line_utm,
    reach_id: str | int,
    out_points_dir: Path,
):
    # 1) Hydrocron width + QA for this granule timestamp
    t0 = parse_pixc_start_time_from_name(nc_path)

    hc = hydrocron_width_and_quality_for_reach_time(
        reach_id=reach_id,
        target_time_utc=t0,
        window_minutes=cfg.hydrocron_window_minutes,
        collection_name=cfg.riversp_collection_name,
        quality_field=cfg.hydrocron_quality_field,
    )

    width_m = hc["width"]
    reach_q = hc["quality"]  # may be NaN if field missing

    # Optional: gate on QA
    if cfg.use_hydrocron_qa_gate:
        # If quality is missing, treat as fail (strict). If you prefer "missing => allow", change logic.
        if not np.isfinite(reach_q) or (int(reach_q) not in set(cfg.hydrocron_allowed_quality)):
            raise ValueError(
                f"Hydrocron QA gate failed: {cfg.hydrocron_quality_field}={reach_q} "
                f"(allowed={cfg.hydrocron_allowed_quality})"
            )

    buffer_dist_m = (width_m / 2.0) * cfg.width_factor
    print(
        f"  Hydrocron width={width_m:.2f} m | "
        f"{cfg.hydrocron_quality_field}={reach_q} | "
        f"buffer_dist={buffer_dist_m:.2f} m"
    )

    # buffer around reach centerline
    reach_buffer_geom = reach_line_utm.buffer(buffer_dist_m, cap_style=2)
    reach_buffer_gdf = gpd.GeoDataFrame(
        {"reach_id": [str(reach_id)]},
        geometry=[reach_buffer_geom],
        crs=f"EPSG:{cfg.utm_epsg}",
    )

    # 2) Read PIXC arrays
    with nc.Dataset(str(nc_path), "r") as ds:
        pc = ds.groups["pixel_cloud"]

        lat = pc.variables["latitude"][:]
        lon = pc.variables["longitude"][:]
        height = pc.variables["height"][:]
        classification = pc.variables["classification"][:]

        if cfg.height_correction:
            solid_earth_tide = pc.variables["solid_earth_tide"][:]
            load_tide_fes = pc.variables["load_tide_fes"][:]
            pole_tide = pc.variables["pole_tide"][:]
            height = height - solid_earth_tide - load_tide_fes - pole_tide

    lat = np.ma.filled(lat, np.nan).astype("float64")
    lon = np.ma.filled(lon, np.nan).astype("float64")
    height = np.ma.filled(height, np.nan).astype("float64")
    classification = np.ma.filled(classification, np.nan).astype("float64")

    df = pd.DataFrame({"lat": lat, "lon": lon, "height": height, "classification": classification}).dropna()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=cfg.utm_epsg)

    # classification filter
    gdf = apply_filter(gdf)
    if gdf.empty:
        raise ValueError("No points after classification filter.")

    # spatial clip (safe fallback if no sindex)
    try:
        clipped = gpd.clip(gdf, reach_buffer_gdf)
    except Exception:
        clipped = gdf[gdf.within(reach_buffer_geom)].copy()

    if clipped.empty:
        raise ValueError("No points inside reach buffer.")

    # along-reach distance s_m: project to line
    # (works for LineString/MultiLineString merged into line above)
    line = reach_line_utm
    clipped["s_m"] = clipped.geometry.apply(lambda p: float(line.project(p)))
    clipped = clipped.sort_values("s_m")

    pts_df = clipped[["s_m", "height"]].dropna().copy()
    if pts_df.empty:
        raise ValueError("No valid (s_m, height) points.")

    # coverage QC: keep date only if processed points span >=95% of fixed reach length
    reach_length_m = float(line.length)
    max_s_m = float(pts_df["s_m"].max())
    coverage_frac = (max_s_m / reach_length_m) if reach_length_m > 0 else np.nan
    if (not np.isfinite(coverage_frac)) or (coverage_frac < 0.95):
        raise ValueError(
            f"Reach-length coverage failed: max_s_m={max_s_m:.2f} m, "
            f"reach_length_m={reach_length_m:.2f} m, coverage_frac={coverage_frac:.3f} (<0.95)"
        )

    # save points
    out_points_dir.mkdir(parents=True, exist_ok=True)
    pts_path = out_points_dir / f"{nc_path.stem}_pts.csv"
    pts_df.to_csv(pts_path, index=False)

    # quantile regression sweep
    q = np.round(np.arange(0.01, 1.00, 0.01), 2)
    reg_df = pts_df.rename(columns={"s_m": "x", "height": "y"}).copy()

    slopes = []
    for tau in q:
        # statsmodels quantreg
        res = smf.quantreg("y ~ x", reg_df).fit(q=tau, max_iter=10000)
        slope_m_per_m = float(res.params["x"])
        slopes.append(slope_m_per_m * 1000.0)  # m/km

    summary = {
        "file": nc_path.name,
        "t0_utc": t0.isoformat(),
        "reach_id": str(reach_id),
        "n_points": int(len(pts_df)),
        "reach_length_m": reach_length_m,
        "max_s_m": max_s_m,
        "reach_coverage_frac": coverage_frac,
        "reach_coverage_threshold": 0.95,
        "width_m": float(width_m),
        "buffer_dist_m": float(buffer_dist_m),
        "hydrocron_time_str": hc["time_str"].isoformat(),
        "hydrocron_dt_sec": float(hc["dt_sec"]),
        cfg.hydrocron_quality_field: (int(reach_q) if np.isfinite(reach_q) else np.nan),
        "pts_csv": str(pts_path),
    }

    return summary, q, np.array(slopes, dtype="float64"), pts_path


def run_pixc_batch_for_reach(
    cfg: PixcQuantileConfig,
    reach_id: str | int,
    year: int,
    nc_dir: Path | None = None,
):
    """
    Reads .nc files from nc_dir (defaults to cfg.raw_pixc_dir).
    Writes:
      processed/reach_{id}/{year}/summary.csv
      processed/reach_{id}/{year}/quantile_slopes_long.csv
      processed/reach_{id}/{year}/pixc_points/*.csv
    """
    if nc_dir is None:
        nc_dir = cfg.raw_pixc_dir(reach_id, year)
    nc_dir = Path(nc_dir)

    nc_paths = sorted(nc_dir.glob("*.nc"))
    if not nc_paths:
        raise FileNotFoundError(f"No .nc files found in: {nc_dir}")

    out_dir = cfg.processed_dir(reach_id, year)
    out_dir.mkdir(parents=True, exist_ok=True)
    points_dir = cfg.points_dir(reach_id, year)
    points_dir.mkdir(parents=True, exist_ok=True)

    reach_line_utm = build_reach_line_utm(cfg, reach_id)

    summaries = []
    slopes_long = []

    for i, nc_path in enumerate(nc_paths, start=1):
        print(f"\n[{i}/{len(nc_paths)}] {nc_path.name}")

        try:
            summary, quantiles, slopes, pts_path = process_one_pixc_file(
                cfg=cfg,
                nc_path=nc_path,
                reach_line_utm=reach_line_utm,
                reach_id=reach_id,
                out_points_dir=points_dir,
            )

            summaries.append(summary)

            tmp = pd.DataFrame({
                "file": nc_path.name,
                "t0_utc": summary["t0_utc"],
                "reach_id": str(reach_id),
                "quantile": quantiles,
                "slope_m_per_km": slopes,
            })
            slopes_long.append(tmp)

        except Exception as e:
            err = str(e)
            print("  ERROR:", err)
            summaries.append({
                "file": nc_path.name,
                "t0_utc": None,
                "reach_id": str(reach_id),
                "error": err,
            })
            continue

    summary_df = pd.DataFrame(summaries)
    slopes_df = pd.concat(slopes_long, ignore_index=True) if slopes_long else pd.DataFrame()

    summary_csv = out_dir / "summary.csv"
    slopes_csv = out_dir / "quantile_slopes_long.csv"
    summary_df.to_csv(summary_csv, index=False)
    slopes_df.to_csv(slopes_csv, index=False)

    print("\nSaved:")
    print(" -", summary_csv)
    print(" -", slopes_csv)
    print(" - points:", points_dir)

    return summary_df, slopes_df, out_dir, points_dir


import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress


import matplotlib.pyplot as plt

def detrend_trim_refit(
    pts_df,
    tau_low,
    tau_high,
    detrend_slope_m_per_km,
    min_points_after_trim=10,
):
    df = pts_df[["s_m", "height"]].dropna().sort_values("s_m")
    if df.empty:
        return None

    x = df["s_m"].to_numpy()
    y = df["height"].to_numpy()

    lr0 = linregress(x, y)
    raw_slope = lr0.slope * 1000.0

    m_det = detrend_slope_m_per_km / 1000.0
    y_detr = y - m_det * x

    lo = np.quantile(y_detr, tau_low)
    hi = np.quantile(y_detr, tau_high)
    keep = (y_detr >= lo) & (y_detr <= hi)

    if keep.sum() < min_points_after_trim:
        return None

    lr1 = linregress(x[keep], y[keep])
    postfilter_ols_slope_m_per_km = lr1.slope * 1000.0

    return {
        "n_raw": int(len(df)),
        "n_keep": int(keep.sum()),

        # OLS before filtering (all points)
        "ols_slope_raw_m_per_km": float(raw_slope),

        # OLS after filtering (kept points)  <-- THIS IS WHAT YOU WANT
        "ols_slope_postfilter_m_per_km": float(postfilter_ols_slope_m_per_km),

        "delta_slope_m_per_km": float(postfilter_ols_slope_m_per_km - raw_slope),
    }


# -------------------------------------------------------
# 1) Build universal-K Ãâ€ž band across files
# -------------------------------------------------------
def build_universal_k_band(
    slopes_df: pd.DataFrame,
    target_coverage: float = 0.95,
    min_files_per_quantile: int = 5,
    trim_frac: float = 0.05,
    eps: float = 1e-12,
):
    df = slopes_df.dropna(subset=["file", "quantile", "slope_m_per_km"]).copy()

    rows = []
    for tau, g in df.groupby("quantile"):
        s = g["slope_m_per_km"].to_numpy(dtype="float64")
        if len(s) < min_files_per_quantile:
            continue

        med = np.median(s)
        mad = np.median(np.abs(s - med))
        mad_safe = max(mad, eps)

        z = np.abs(s - med) / mad_safe
        k_tau = np.quantile(z, target_coverage)

        rows.append({
            "quantile": tau,
            "median": med,
            "mad": mad_safe,
            "k_tau": k_tau,
        })

    q = pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)
    if q.empty:
        raise ValueError("No valid quantiles to build band.")

    ks = q["k_tau"].to_numpy()
    lo = np.quantile(ks, trim_frac)
    hi = np.quantile(ks, 1 - trim_frac)
    ks2 = ks[(ks >= lo) & (ks <= hi)]
    K = np.mean(ks2) if len(ks2) else np.mean(ks)
    K = min(float(K), 8.0)

    q["lower"] = q["median"] - K * q["mad"]
    q["upper"] = q["median"] + K * q["mad"]

    return q[["quantile", "median", "lower", "upper"]], float(K)


# -------------------------------------------------------
# 2) Longest contiguous Ãâ€ž-run inside band (per file)
# -------------------------------------------------------
def find_file_tau_window(slopes_df, q_band):
    results = []

    for f, g in slopes_df.groupby("file"):
        g = g.sort_values("quantile").merge(q_band, on="quantile", how="left")

        q = g["quantile"].to_numpy()
        s = g["slope_m_per_km"].to_numpy()
        lo = g["lower"].to_numpy()
        hi = g["upper"].to_numpy()

        ok = np.isfinite(s) & (s >= lo) & (s <= hi)

        if not np.any(ok):
            results.append((f, np.nan, np.nan))
            continue

        idx = np.where(ok)[0]
        splits = np.where(np.diff(idx) != 1)[0] + 1
        runs = np.split(idx, splits)
        best = max(runs, key=len)

        tau_low = q[best[0]]
        tau_high = q[best[-1]]
        results.append((f, tau_low, tau_high))

    return pd.DataFrame(results, columns=["file", "tau_low", "tau_high"])


# -------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------
def plot_tau_curve(file_slopes, tau_low, tau_high, out_path):
    file_slopes = file_slopes.sort_values("quantile")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(file_slopes["quantile"], file_slopes["slope_m_per_km"], alpha=0.4)

    band = file_slopes[
        (file_slopes["quantile"] >= tau_low) &
        (file_slopes["quantile"] <= tau_high)
    ]

    if not band.empty:
        ax.plot(band["quantile"], band["slope_m_per_km"], linewidth=3)

    ax.axvline(tau_low, linestyle="--")
    ax.axvline(tau_high, linestyle="--")

    ax.set_xlabel("Quantile (Ãâ€ž)")
    ax.set_ylabel("Slope (m/km)")
    ax.set_title(Path(file_slopes["file"].iloc[0]).name)
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pixc_trim(pts_df, slope_det, tau_low, tau_high, out_path):
    df = pts_df[["s_m", "height"]].dropna().sort_values("s_m")

    x = df["s_m"].to_numpy()
    y = df["height"].to_numpy()
    x_km = x / 1000.0

    m_det = slope_det / 1000.0
    y_detr = y - m_det * x

    lo = np.quantile(y_detr, tau_low)
    hi = np.quantile(y_detr, tau_high)
    keep = (y_detr >= lo) & (y_detr <= hi)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_km, y, s=12, alpha=0.25)
    ax.scatter(x_km[keep], y[keep], s=14, alpha=0.85)

    if keep.sum() >= 2:
        lr = linregress(x[keep], y[keep])
        slope = lr.slope * 1000.0
        intercept = lr.intercept

        xx_km = np.linspace(x_km.min(), x_km.max(), 200)
        xx_m = xx_km * 1000.0
        yy = (slope / 1000.0) * xx_m + intercept
        ax.plot(xx_km, yy, linestyle="--", linewidth=2)

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("WSE (m)")
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------
# 3) Main Universal-K Workflow WITH PLOTS
# -------------------------------------------------------
def run_universal_k_tau_trim(
    slopes_df,
    points_dir,
    output_dir,
    target_coverage=0.95,
    min_files_per_quantile=5,
    min_points_after_trim=10,
    reach_id=None,
    year=None,
):
    q_band, K = build_universal_k_band(
        slopes_df,
        target_coverage=target_coverage,
        min_files_per_quantile=min_files_per_quantile,
    )

    file_windows = find_file_tau_window(slopes_df, q_band)

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for _, row in file_windows.iterrows():
        f = row["file"]
        tau_low = row["tau_low"]
        tau_high = row["tau_high"]

        if not np.isfinite(tau_low):
            continue

        slope_det = np.median(
            slopes_df[
                (slopes_df["file"] == f) &
                (slopes_df["quantile"] >= tau_low) &
                (slopes_df["quantile"] <= tau_high)
            ]["slope_m_per_km"]
        )

        pts_path = Path(points_dir) / f"{Path(f).stem}_pts.csv"
        if not pts_path.exists():
            continue

        pts = pd.read_csv(pts_path)

        # Make plots
        plot_tau_curve(
            slopes_df[slopes_df["file"] == f],
            tau_low,
            tau_high,
            plots_dir / f"{Path(f).stem}_tau_curve.png",
        )

        plot_pixc_trim(
            pts,
            slope_det,
            tau_low,
            tau_high,
            plots_dir / f"{Path(f).stem}_pixc_trim.png",
        )

        # Refit
        out = detrend_trim_refit(
            pts,
            tau_low,
            tau_high,
            slope_det,
            min_points_after_trim,
        )

        if out is not None:
            # attach identifiers + key params so it's "tracked" in results_df
            out["file"] = f
            out["tau_low"] = float(tau_low)
            out["tau_high"] = float(tau_high)
            out["detrend_slope_m_per_km"] = float(slope_det)
            out["K_universal"] = float(K)
            out["reach_id"] = str(reach_id) if reach_id is not None else np.nan
            out["year"] = int(year) if year is not None else np.nan

            results.append(out)

    results_df = pd.DataFrame(results)
    if "K_universal" not in results_df.columns:
        results_df["K_universal"] = float(K)
    if reach_id is not None and "reach_id" not in results_df.columns:
        results_df["reach_id"] = str(reach_id)
    if year is not None and "year" not in results_df.columns:
        results_df["year"] = int(year)

    # Save "tracked" slopes table
    results_csv = Path(output_dir) / "tau_trim_results.csv"
    results_df.to_csv(results_csv, index=False)

    print(f"Universal K = {K:.4f}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Results saved to: {results_csv}")

    return q_band, results_df

# =========================
# 4) One-call orchestrator
# =========================

def process_reach_year(
    cfg: PixcQuantileConfig,
    reach_id: str | int,
    year: int,
    force_redownload: bool = False,
    force_rebuild_slopes: bool = False,   # NEW
    min_slopes_rows: int = 10,            # NEW: sanity threshold
):
    """
    End-to-end:
      - download PIXC (covers reach) -> raw dir
      - build quantile slopes + save per-granule point CSVs (unless cached)
      - run Ãâ€ž-trim workflow and write outputs
    """
    raw_dir = cfg.raw_pixc_dir(reach_id, year)
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        already = list(raw_dir.glob("*.nc"))
        if force_redownload or len(already) == 0:
            download_pixc_by_reach_year_covers(cfg, reach_id, year)
        else:
            print(f"Using cached downloads: {len(already)} files in {raw_dir}")

        # Where processed outputs should live (matches your pipeline)
        out_dir = cfg.processed_dir(reach_id, year)
        out_dir.mkdir(parents=True, exist_ok=True)
        points_dir = cfg.points_dir(reach_id, year)
        slopes_csv = out_dir / "quantile_slopes_long.csv"

        # ---------------------------------------------------------
        # Check for cached slopes BEFORE running the batch builder
        # ---------------------------------------------------------
        use_cached_slopes = (
            (not force_rebuild_slopes)
            and slopes_csv.exists()
            and slopes_csv.stat().st_size > 0
        )

        summary_df = None
        slopes_df = None

        if use_cached_slopes:
            try:
                tmp = pd.read_csv(slopes_csv)
                required = {"file", "quantile", "slope_m_per_km"}
                if required.issubset(tmp.columns) and len(tmp) >= int(min_slopes_rows) and points_dir.exists():
                    print(f"[slopes] Using cached slopes: {slopes_csv} (rows={len(tmp)})")
                    slopes_df = tmp
                    # summary_df isn't needed for Ãâ€ž-trim; keep as None unless you want to also cache it
                else:
                    print("[slopes] Cache invalid/incomplete (missing cols, too few rows, or missing points_dir). Rebuilding.")
                    use_cached_slopes = False
            except Exception as e:
                print(f"[slopes] Failed reading cached slopes ({e}). Rebuilding.")
                use_cached_slopes = False

        if not use_cached_slopes:
            # quantile slope builder (also writes slopes_csv + per-granule points)
            summary_df, slopes_df, out_dir_built, points_dir_built = run_pixc_batch_for_reach(
                cfg=cfg,
                reach_id=reach_id,
                year=year,
                nc_dir=raw_dir,
            )
            # ensure we use the builder's returned dirs (source of truth)
            out_dir = out_dir_built
            points_dir = Path(points_dir_built)
            slopes_csv = Path(out_dir) / "quantile_slopes_long.csv"

            if not slopes_csv.exists():
                # just a hard failure if builder didn't write it
                raise FileNotFoundError(f"Expected slopes CSV not found after build: {slopes_csv}")

        # ---------------------------------------------------------
        # Ãâ€ž-trim filter step
        # ---------------------------------------------------------
        slopes_df = pd.read_csv(slopes_csv)

        q_band, results_df = run_universal_k_tau_trim(
            slopes_df=slopes_df,
            points_dir=str(points_dir),
            output_dir=str(out_dir),
            target_coverage=cfg.target_coverage,
            min_files_per_quantile=cfg.min_files_per_quantile,
            min_points_after_trim=cfg.min_points_after_trim,
            reach_id=reach_id,
            year=year,
        )
        return {
            "raw_dir": raw_dir,
            "processed_dir": out_dir,
            "points_dir": points_dir,
            "summary_df": summary_df,
            "slopes_df": slopes_df,
            "q_band": q_band,
            "results_df": results_df,
            "slopes_csv": slopes_csv,
        }
    finally:
        if raw_dir.exists():
            try:
                shutil.rmtree(raw_dir)
                print(f"  Cleanup: removed raw_dir {raw_dir}")
            except Exception as e:
                print(f"  WARN: could not remove raw_dir {raw_dir}: {e}")

# %% Notebook cell 1
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pandas as pd


@dataclass(frozen=True, kw_only=True)
class ChileDischargePixcQuantileConfig(PixcQuantileConfig):
    run_root: Path

    def raw_pixc_dir(self, reach_id: str | int, year: int) -> Path:
        return self.run_root / "raw" / "pixc_d" / f"reach_{reach_id}" / str(year)

    def processed_dir(self, reach_id: str | int, year: int) -> Path:
        return self.run_root / "processed" / f"reach_{reach_id}" / str(year)

    def points_dir(self, reach_id: str | int, year: int) -> Path:
        return self.processed_dir(reach_id, year) / "pixc_points"


def list_reach_ids_from_csv(csv_path: Path, reach_id_field="reach_id"):
    df = pd.read_csv(csv_path)
    if reach_id_field not in df.columns:
        raise ValueError(
            f"Column '{reach_id_field}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    return (
        df[reach_id_field]
        .dropna()
        .astype("int64")
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def delete_raw_dir(raw_dir: Path, verbose: bool = True) -> int:
    """
    Robust cleanup: delete the entire reach/year raw_dir tree.
    Returns number of files removed (best effort).
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return 0

    # count files for reporting (best-effort)
    try:
        n_files = sum(1 for _ in raw_dir.rglob("*") if _.is_file())
    except Exception:
        n_files = 0

    try:
        shutil.rmtree(raw_dir)
        if verbose:
            print(f"  Cleanup: removed raw_dir {raw_dir} ({n_files} files)")
        return n_files
    except Exception as e:
        if verbose:
            print(f"  WARN: could not remove raw_dir {raw_dir}: {e}")
        return 0


def _summarize_out(out, rid: str, y: int) -> dict:
    """
    Compact per-(reach,year) summary from process_reach_year output.
    Adjust to match your out schema as needed.
    """
    r = {"reach_id": rid, "year": y}

    if isinstance(out, dict) and "results_df" in out and out["results_df"] is not None:
        df = out["results_df"]
        r["n_files"] = int(df["file"].nunique()) if "file" in df.columns else int(len(df))

        for col in ["tau_low", "tau_high", "window_len", "true_frac", "pass_flag"]:
            if col in df.columns:
                r[f"mean_{col}"] = float(pd.to_numeric(df[col], errors="coerce").mean())

        if "K_universal" in df.columns and not df["K_universal"].dropna().empty:
            r["K_universal"] = float(pd.to_numeric(df["K_universal"], errors="coerce").dropna().iloc[0])

    return r


def _worker_process_reach_year(
    cfg,
    rid: str,
    y: int,
    *,
    delete_nc_after_success: bool,
    delete_nc_after_failure: bool,
):
    """
    Runs one (reach, year). Returns a dict:
      {"ok": True, "row": {...}}  OR  {"ok": False, "fail": {...}}
    """
    y = int(y)
    raw_dir = cfg.raw_pixc_dir(rid, y)

    try:
        out = process_reach_year(cfg, reach_id=rid, year=y)
        row = _summarize_out(out, rid, y)
        return {"ok": True, "row": row}
    except Exception as e:
        return {"ok": False, "fail": {"reach_id": rid, "year": y, "error": repr(e)}}
    finally:
        # Always remove raw downloads after this reach/year attempt.
        delete_raw_dir(raw_dir, verbose=True)


def batch_process_reach_csv_years_parallel(
    cfg,
    reach_csv,
    years,
    reach_id_field="reach_id",
    stop_on_error: bool = False,
    delete_nc_after_success: bool = True,
    delete_nc_after_failure: bool = False,
    *,
    backend: str = "thread",   # "thread" (easy) or "process" (CPU-bound)
    max_workers: int | None = None,
):
    """
    Parallel batch runner over (reach_id, year).

    backend="thread" is safest in notebooks + avoids pickling issues.
    backend="process" can speed CPU-heavy regression, but on Windows/notebooks
    you may hit pickling/spawn constraints depending on how cfg/classes are defined.
    """
    reach_ids = list_reach_ids_from_csv(reach_csv, reach_id_field=reach_id_field)
    years = [int(y) for y in years]

    print(f"Found {len(reach_ids)} reaches in {reach_csv}")
    tasks = [(rid, y) for rid in reach_ids for y in years]
    print(f"Submitting {len(tasks)} (reach,year) tasks...")

    if max_workers is None:
        # conservative default: don't DOS Earthdata/Hydrocron
        max_workers = min(6, (os.cpu_count() or 4))

    Executor = ThreadPoolExecutor if backend.lower().startswith("thread") else ProcessPoolExecutor

    rows: list[dict] = []
    fails: list[dict] = []

    with Executor(max_workers=max_workers) as ex:
        fut_to_task = {
            ex.submit(
                _worker_process_reach_year,
                cfg,
                rid,
                y,
                delete_nc_after_success=delete_nc_after_success,
                delete_nc_after_failure=delete_nc_after_failure,
            ): (rid, y)
            for (rid, y) in tasks
        }

        for fut in as_completed(fut_to_task):
            rid, y = fut_to_task[fut]
            try:
                res = fut.result()
            except Exception as e:
                # Should be rare because worker catches most exceptions,
                # but keep this as a safety net.
                res = {"ok": False, "fail": {"reach_id": rid, "year": int(y), "error": repr(e)}}

            if res.get("ok"):
                rows.append(res["row"])
            else:
                fails.append(res["fail"])
                print(f"FAILED reach {rid} year {y}: {res['fail']['error']}")
                if stop_on_error:
                    raise RuntimeError(f"Stopping on first error: reach {rid} year {y}")

    summary_df = pd.DataFrame(rows)
    fail_df = pd.DataFrame(fails)
    return {"summary_df": summary_df, "fail_df": fail_df}



