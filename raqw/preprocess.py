from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import earthaccess
import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from shapely.geometry import MultiLineString, Polygon, box
from shapely.ops import linemerge, unary_union

from .config import RAQWConfig
from .geometry import get_reach_geometry


GRANULE_TIME_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


def projected_epsg(cfg: RAQWConfig) -> int:
    if cfg.projected_crs:
        crs = cfg.projected_crs.upper()
        return int(crs.split(":", 1)[1] if crs.startswith("EPSG:") else crs)

    geometry = get_reach_geometry(cfg)
    reach = gpd.GeoSeries([geometry], crs="EPSG:4326")
    estimated = reach.estimate_utm_crs()
    epsg = estimated.to_epsg() if estimated is not None else None
    if epsg is None:
        raise ValueError("Could not determine a projected EPSG code for the reach.")
    return int(epsg)


def granule_footprint_from_umm(granule):
    geometry = (
        granule.render_dict.get("umm", {})
        .get("SpatialExtent", {})
        .get("HorizontalSpatialDomain", {})
        .get("Geometry", {})
    )
    polygons = geometry.get("GPolygons")
    if polygons:
        parts = []
        for polygon_data in polygons:
            points = polygon_data.get("Boundary", {}).get("Points", [])
            if len(points) < 3:
                continue
            coordinates = [(point["Longitude"], point["Latitude"]) for point in points]
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            polygon = Polygon(coordinates)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                parts.append(polygon)
        return unary_union(parts) if parts else None

    rectangles = geometry.get("BoundingRectangles")
    if rectangles:
        parts = []
        for rectangle in rectangles:
            values = (
                rectangle.get("WestBoundingCoordinate"),
                rectangle.get("SouthBoundingCoordinate"),
                rectangle.get("EastBoundingCoordinate"),
                rectangle.get("NorthBoundingCoordinate"),
            )
            if any(value is None for value in values):
                continue
            parts.append(box(*values))
        return unary_union(parts) if parts else None
    return None


def download_pixc(cfg: RAQWConfig, login: bool = True) -> list[Path]:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cfg.raw_dir.glob("*.nc"))
    if existing:
        print(f"Reusing {len(existing)} PIXC files from {cfg.raw_dir}")
        return existing

    reach_geometry = get_reach_geometry(cfg)
    if login:
        earthaccess.login(strategy="all")
    results = earthaccess.search_data(
        short_name=cfg.pixc_short_name,
        bounding_box=reach_geometry.bounds,
        temporal=(f"{cfg.year}-01-01T00:00:00", f"{cfg.year}-12-31T23:59:59"),
    )

    selected = []
    for granule in results:
        footprint = granule_footprint_from_umm(granule)
        if footprint is not None and footprint.covers(reach_geometry):
            selected.append(granule)
    if not selected:
        raise ValueError(
            f"No {cfg.pixc_short_name} granules fully cover reach "
            f"{cfg.reach_id} in {cfg.year}."
        )

    print(f"Search returned: {len(results)}")
    print(f"Kept (footprint covers reach): {len(selected)}")
    print(f"Downloading to: {cfg.raw_dir}")
    paths = earthaccess.download(selected, str(cfg.raw_dir), threads=2)
    return [Path(path) for path in paths]


def hydrocron_response_to_df(response_text: str) -> pd.DataFrame:
    text = response_text.strip()
    if text.startswith("{"):
        payload = json.loads(text)
        csv_text = payload.get("results", {}).get("csv", "")
        return pd.read_csv(StringIO(csv_text)) if csv_text else pd.DataFrame()
    return pd.read_csv(StringIO(text))


def hydrocron_width_and_quality(
    cfg: RAQWConfig,
    target_time_utc: datetime,
) -> dict[str, object]:
    start = (target_time_utc - timedelta(minutes=cfg.hydrocron_window_minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    end = (target_time_utc + timedelta(minutes=cfg.hydrocron_window_minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    params = {
        "feature": "Reach",
        "feature_id": str(cfg.reach_id),
        "start_time": start,
        "end_time": end,
        "output": "csv",
        "collection_name": cfg.riversp_collection_name,
        "fields": "reach_id,time_str,width,reach_q",
    }
    response = requests.get(
        cfg.hydrocron_url,
        params=params,
        timeout=cfg.hydrocron_timeout_seconds,
    )
    response.raise_for_status()
    rows = hydrocron_response_to_df(response.text)
    if rows.empty:
        raise ValueError("Hydrocron returned 0 rows for tight window.")

    rows["time_str"] = pd.to_datetime(rows["time_str"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["time_str"])
    if rows.empty:
        raise ValueError("Hydrocron rows had invalid time_str.")
    rows["dt_sec"] = (rows["time_str"] - target_time_utc).abs().dt.total_seconds()
    row = rows.sort_values("dt_sec").iloc[0]

    width = float(row["width"])
    try:
        reach_q = int(row["reach_q"])
    except Exception:
        reach_q = np.nan
    if not np.isfinite(reach_q) or int(reach_q) not in set(cfg.allowed_reach_q):
        raise ValueError(
            f"Hydrocron QA gate failed: reach_q={reach_q} "
            f"(allowed={cfg.allowed_reach_q})"
        )
    return {
        "width": width,
        "quality": reach_q,
        "time_str": row["time_str"],
        "dt_sec": float(row["dt_sec"]),
    }


def parse_pixc_start_time(path: Path) -> datetime:
    match = GRANULE_TIME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse times from filename: {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    )


def build_reach_line(cfg: RAQWConfig):
    geometry_wgs84 = get_reach_geometry(cfg)
    geometry = gpd.GeoSeries([geometry_wgs84], crs="EPSG:4326").to_crs(
        epsg=projected_epsg(cfg)
    ).iloc[0]
    if isinstance(geometry, MultiLineString):
        geometry = linemerge(geometry)
    if geometry.geom_type != "LineString":
        raise ValueError(
            "Reach geometry could not be merged into one continuous LineString; "
            f"received {geometry.geom_type}."
        )
    return geometry


def process_one_pixc_file(
    cfg: RAQWConfig,
    nc_path: Path,
    reach_line,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, Path]:
    target_time = parse_pixc_start_time(nc_path)
    hydrocron = hydrocron_width_and_quality(cfg, target_time)
    width_m = hydrocron["width"]
    reach_q = hydrocron["quality"]
    buffer_distance_m = (width_m / 2.0) * cfg.width_factor
    print(
        f"  Hydrocron width={width_m:.2f} m | reach_q={reach_q} | "
        f"buffer_dist={buffer_distance_m:.2f} m"
    )

    reach_buffer_geometry = reach_line.buffer(buffer_distance_m, cap_style=2)
    reach_buffer = gpd.GeoDataFrame(
        {"reach_id": [str(cfg.reach_id)]},
        geometry=[reach_buffer_geometry],
        crs=f"EPSG:{projected_epsg(cfg)}",
    )

    with nc.Dataset(str(nc_path), "r") as dataset:
        cloud = dataset.groups["pixel_cloud"]
        latitude = cloud.variables["latitude"][:]
        longitude = cloud.variables["longitude"][:]
        height = cloud.variables["height"][:]
        classification = cloud.variables["classification"][:]
        if cfg.correct_tides:
            height = (
                height
                - cloud.variables["solid_earth_tide"][:]
                - cloud.variables["load_tide_fes"][:]
                - cloud.variables["pole_tide"][:]
            )

    latitude = np.ma.filled(latitude, np.nan).astype("float64")
    longitude = np.ma.filled(longitude, np.nan).astype("float64")
    height = np.ma.filled(height, np.nan).astype("float64")
    classification = np.ma.filled(classification, np.nan).astype("float64")
    points = pd.DataFrame(
        {
            "lat": latitude,
            "lon": longitude,
            "height": height,
            "classification": classification,
        }
    ).dropna()
    geospatial = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=projected_epsg(cfg))
    geospatial = geospatial[
        (geospatial["classification"] > 2)
        & (geospatial["classification"] != 5)
    ].copy()
    if geospatial.empty:
        raise ValueError("No points after classification filter.")

    try:
        clipped = gpd.clip(geospatial, reach_buffer)
    except Exception:
        clipped = geospatial[geospatial.within(reach_buffer_geometry)].copy()
    if clipped.empty:
        raise ValueError("No points inside reach buffer.")

    clipped = clipped.copy()
    clipped["s_m"] = clipped.geometry.apply(lambda point: float(reach_line.project(point)))
    clipped = clipped.sort_values("s_m")
    point_data = clipped[["s_m", "height"]].dropna().copy()
    if point_data.empty:
        raise ValueError("No valid (s_m, height) points.")
    if cfg.min_points_per_granule and len(point_data) < cfg.min_points_per_granule:
        raise ValueError(
            f"Only {len(point_data)} spatially screened PIXC points remained; "
            f"at least {cfg.min_points_per_granule} are required."
        )

    reach_length_m = float(reach_line.length)
    max_s_m = float(point_data["s_m"].max())
    coverage_fraction = max_s_m / reach_length_m if reach_length_m > 0 else np.nan
    if (
        not np.isfinite(coverage_fraction)
        or coverage_fraction < cfg.minimum_reach_coverage
    ):
        raise ValueError(
            f"Reach-length coverage failed: max_s_m={max_s_m:.2f} m, "
            f"reach_length_m={reach_length_m:.2f} m, "
            f"coverage_frac={coverage_fraction:.3f} "
            f"(<{cfg.minimum_reach_coverage:.3f})"
        )

    cfg.points_dir.mkdir(parents=True, exist_ok=True)
    points_path = cfg.points_dir / f"{nc_path.stem}_pts.csv"
    point_data.to_csv(points_path, index=False)

    quantiles = np.round(
        np.arange(cfg.quantile_step, 1.0, cfg.quantile_step),
        10,
    )
    regression_data = point_data.rename(columns={"s_m": "x", "height": "y"}).copy()
    slopes = []
    for quantile in quantiles:
        result = smf.quantreg("y ~ x", regression_data).fit(
            q=quantile,
            max_iter=cfg.quantile_max_iterations,
        )
        slopes.append(float(result.params["x"]) * 1000.0)

    summary = {
        "file": nc_path.name,
        "t0_utc": target_time.isoformat(),
        "reach_id": str(cfg.reach_id),
        "n_points": int(len(point_data)),
        "reach_length_m": reach_length_m,
        "max_s_m": max_s_m,
        "reach_coverage_frac": coverage_fraction,
        "reach_coverage_threshold": cfg.minimum_reach_coverage,
        "width_m": float(width_m),
        "buffer_dist_m": float(buffer_distance_m),
        "hydrocron_time_str": hydrocron["time_str"].isoformat(),
        "hydrocron_dt_sec": hydrocron["dt_sec"],
        "reach_q": int(reach_q),
        "pts_csv": str(points_path),
    }
    return summary, quantiles, np.asarray(slopes, dtype="float64"), points_path


def preprocess_reach_year(
    cfg: RAQWConfig,
    login: bool = True,
    nc_paths: list[Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(nc_paths) if nc_paths is not None else download_pixc(cfg, login=login)
    if not paths:
        raise ValueError(f"No PIXC granules cover reach {cfg.reach_id} in {cfg.year}.")
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.points_dir.mkdir(parents=True, exist_ok=True)
    reach_line = build_reach_line(cfg)

    summaries = []
    slope_parts = []
    for index, path in enumerate(sorted(paths), start=1):
        print(f"[{index}/{len(paths)}] {path.name}")
        try:
            summary, quantiles, slopes, _ = process_one_pixc_file(
                cfg,
                path,
                reach_line,
            )
            summaries.append(summary)
            slope_parts.append(
                pd.DataFrame(
                    {
                        "file": path.name,
                        "t0_utc": summary["t0_utc"],
                        "reach_id": str(cfg.reach_id),
                        "quantile": quantiles,
                        "slope_m_per_km": slopes,
                    }
                )
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            summaries.append(
                {
                    "file": path.name,
                    "t0_utc": None,
                    "reach_id": str(cfg.reach_id),
                    "error": str(exc),
                }
            )

    summary = pd.DataFrame(summaries)
    slopes = pd.concat(slope_parts, ignore_index=True) if slope_parts else pd.DataFrame()
    summary.to_csv(cfg.processed_dir / "summary.csv", index=False)
    slopes.to_csv(cfg.processed_dir / "quantile_slopes_long.csv", index=False)
    if slopes.empty:
        raise ValueError("No PIXC granules passed preprocessing.")
    return summary, slopes
