from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString
from shapely.ops import linemerge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REACH_ID = "66405900761"
YEAR = 2024
FILE_NAME = "SWOT_L2_HR_PIXC_012_104_182L_20240310T103854_20240310T103905_PGD0_01.nc"
NC_PATH = PROJECT_ROOT / "data" / "key_figures" / "source_nc" / FILE_NAME
SUMMARY_CSV = (
    PROJECT_ROOT
    / "data"
    / "chile_reaches_with_valid_discharge_run"
    / "processed"
    / f"reach_{REACH_ID}"
    / str(YEAR)
    / "summary.csv"
)
SWORD_GPKG = PROJECT_ROOT / "data" / "Petrohue_SWORD_reaches" / "chile_reaches.gpkg"
OUT_CSV = PROJECT_ROOT / "data" / "key_figures" / "reach_66405900761_2024-03-10_clipped_geoid_points.csv"


def filled(var) -> np.ndarray:
    return np.ma.filled(var[:], np.nan).astype("float64")


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    summary_row = summary.loc[summary["file"] == FILE_NAME].iloc[0]
    width_m = float(summary_row["width_m"])
    buffer_dist_m = width_m / 2.0

    reaches = gpd.read_file(SWORD_GPKG)
    reaches["reach_id_str"] = reaches["reach_id"].astype("int64").astype(str)
    reach = reaches.loc[reaches["reach_id_str"] == REACH_ID].to_crs(epsg=32718)
    geom = reach.geometry.iloc[0]
    line = linemerge(geom) if isinstance(geom, MultiLineString) else geom
    reach_buffer = line.buffer(buffer_dist_m, cap_style=2)

    with nc.Dataset(NC_PATH) as ds:
        pc = ds.groups["pixel_cloud"]
        lat = filled(pc.variables["latitude"])
        lon = filled(pc.variables["longitude"])
        height = filled(pc.variables["height"])
        geoid = filled(pc.variables["geoid"])
        classification = filled(pc.variables["classification"])
        solid_earth_tide = filled(pc.variables["solid_earth_tide"])
        load_tide_fes = filled(pc.variables["load_tide_fes"])
        pole_tide = filled(pc.variables["pole_tide"])

    height_tide_corrected = height - solid_earth_tide - load_tide_fes - pole_tide
    height_geoid_tide_corrected = height_tide_corrected - geoid

    df = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "height_ellipsoid_tide_corrected": height_tide_corrected,
            "height_geoid_tide_corrected": height_geoid_tide_corrected,
            "geoid": geoid,
            "classification": classification,
        }
    ).dropna()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=32718)
    gdf = gdf[(gdf["classification"] > 2) & (gdf["classification"] != 5)].copy()
    clipped = gdf[gdf.within(reach_buffer)].copy()
    clipped["s_m"] = clipped.geometry.apply(lambda point: float(line.project(point)))
    clipped = clipped.sort_values("s_m")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    clipped[
        [
            "s_m",
            "height_ellipsoid_tide_corrected",
            "height_geoid_tide_corrected",
            "geoid",
            "classification",
        ]
    ].to_csv(OUT_CSV, index=False)

    print(f"width_m: {width_m:.6f}")
    print(f"buffer_dist_m: {buffer_dist_m:.6f}")
    print(f"clipped points: {len(clipped):,}")
    print(clipped[["geoid", "height_ellipsoid_tide_corrected", "height_geoid_tide_corrected"]].describe().to_string())
    print(f"median geoid: {clipped['geoid'].median():.6f}")
    print(f"mean geoid: {clipped['geoid'].mean():.6f}")
    print(f"wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()

