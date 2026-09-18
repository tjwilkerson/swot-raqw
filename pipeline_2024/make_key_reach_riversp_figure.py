from __future__ import annotations

from io import StringIO
from pathlib import Path
from time import sleep

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REACH_ID = "66405900761"
YEAR = 2024
FILE_NAME = "SWOT_L2_HR_PIXC_012_104_182L_20240310T103854_20240310T103905_PGD0_01.nc"
STEM = Path(FILE_NAME).stem

POINTS_CSV = (
    PROJECT_ROOT
    / "data"
    / "chile_reaches_with_valid_discharge_run"
    / "processed"
    / f"reach_{REACH_ID}"
    / str(YEAR)
    / "pixc_points"
    / f"{STEM}_pts.csv"
)
GEOID_POINTS_CSV = PROJECT_ROOT / "data" / "key_figures" / "reach_66405900761_2024-03-10_clipped_geoid_points.csv"
UPDATED_RESULTS_CSV = (
    PROJECT_ROOT
    / "data"
    / "tau_range_update_experiment"
    / f"reach_{REACH_ID}"
    / str(YEAR)
    / "updated_tau_window_results.csv"
)
MATCHED_RIVERSP_CSV = PROJECT_ROOT / "data" / "riversp_comparison_updated_filter" / "matched_pixc_riversp_slopes.csv"
RIVERSP_CACHE_CSV = PROJECT_ROOT / "data" / "riversp_comparison_updated_filter" / "riversp_daily_cache.csv"
SWORD_NODES_SHP = Path(r"C:\UNESCO\Code\data\shp\SA\sa_sword_nodes_hb66_v17b.shp")
OUT_DIR = PROJECT_ROOT / "data" / "key_figures"
OUT_PNG = OUT_DIR / "reach_66405900761_2024-03-10_filtered_ols_vs_riversp.png"
OUT_GEOD_PNG = OUT_DIR / "reach_66405900761_2024-03-10_filtered_ols_vs_riversp_geoid_corrected.png"
NODE_CACHE_CSV = OUT_DIR / "reach_66405900761_2024-03-10_hydrocron_nodes.csv"

HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
NODE_COLLECTION = "SWOT_L2_HR_RiverSP_node_D"
NODE_FIELDS = "reach_id,node_id,time_str,wse,node_q,p_dist_out,geoid_hght"

TITLE_FONTSIZE = 17
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12.5
ANNOTATION_FONTSIZE = 11


def load_rows() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    points = pd.read_csv(POINTS_CSV).dropna(subset=["s_m", "height"]).sort_values("s_m")

    updated = pd.read_csv(UPDATED_RESULTS_CSV)
    updated_row = updated.loc[updated["file"] == FILE_NAME].iloc[0]

    matched = pd.read_csv(MATCHED_RIVERSP_CSV)
    matched_row = matched.loc[
        (matched["reach_id"].astype(str) == REACH_ID)
        & (matched["year"].astype(int) == YEAR)
        & (matched["file"] == FILE_NAME)
    ].iloc[0]

    cache = pd.read_csv(RIVERSP_CACHE_CSV)
    riversp_row = cache.loc[
        (cache["reach_id"].astype(str) == REACH_ID)
        & (cache["time_utc"] == matched_row["riversp_time_utc"])
    ].iloc[0]
    return points, updated_row, matched_row, riversp_row


def filtered_points(points: pd.DataFrame, updated_row: pd.Series) -> pd.DataFrame:
    x = points["s_m"].to_numpy(dtype="float64")
    y = points["height"].to_numpy(dtype="float64")
    slope_det = float(updated_row["updated_detrend_slope_m_per_km"]) / 1000.0
    y_detr = y - slope_det * x
    lo = float(np.quantile(y_detr, float(updated_row["updated_tau_low"])))
    hi = float(np.quantile(y_detr, float(updated_row["updated_tau_high"])))
    return points.loc[(y_detr >= lo) & (y_detr <= hi)].copy()


def hydrocron_response_to_df(text: str) -> pd.DataFrame:
    payload = requests.models.complexjson.loads(text)
    csv_text = payload.get("results", {}).get("csv", "")
    if not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(csv_text))


def load_sword_nodes() -> pd.DataFrame:
    import geopandas as gpd

    nodes = gpd.read_file(SWORD_NODES_SHP, where=f"reach_id = {REACH_ID}")
    if nodes.empty:
        raise ValueError(f"No SWORD nodes found for reach {REACH_ID} in {SWORD_NODES_SHP}")

    out = pd.DataFrame(nodes.drop(columns="geometry"))
    out["node_id"] = out["node_id"].astype("int64").astype(str)
    out["reach_id"] = out["reach_id"].astype("int64").astype(str)
    out["s_m"] = out["dist_out"].astype(float) - float(out["dist_out"].min())
    return out.sort_values("s_m").reset_index(drop=True)


def fetch_hydrocron_node(node_id: str, start_time: str, end_time: str) -> pd.DataFrame:
    params = {
        "feature": "Node",
        "feature_id": str(node_id),
        "start_time": start_time,
        "end_time": end_time,
        "output": "csv",
        "collection_name": NODE_COLLECTION,
        "fields": NODE_FIELDS,
    }
    response = requests.get(HYDROCRON_URL, params=params, timeout=60)
    response.raise_for_status()
    return hydrocron_response_to_df(response.text)


def load_or_fetch_hydrocron_nodes(matched_row: pd.Series) -> pd.DataFrame:
    if NODE_CACHE_CSV.exists():
        nodes = pd.read_csv(NODE_CACHE_CSV)
        if "wse" in nodes.columns:
            nodes.loc[pd.to_numeric(nodes["wse"], errors="coerce") < -1.0e10, "wse"] = np.nan
        return nodes

    sword_nodes = load_sword_nodes()
    riversp_time = pd.to_datetime(matched_row["riversp_time_utc"], utc=True)
    start_time = (riversp_time - pd.Timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = (riversp_time + pd.Timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ")

    records: list[pd.DataFrame] = []
    for node_id in sword_nodes["node_id"]:
        node_df = fetch_hydrocron_node(node_id, start_time, end_time)
        if not node_df.empty:
            records.append(node_df)
        sleep(0.05)

    if records:
        hydro_nodes = pd.concat(records, ignore_index=True)
    else:
        hydro_nodes = pd.DataFrame(columns=NODE_FIELDS.split(","))

    hydro_nodes["node_id"] = hydro_nodes["node_id"].astype("int64").astype(str)
    hydro_nodes["reach_id"] = hydro_nodes["reach_id"].astype("int64").astype(str)
    hydro_nodes["wse"] = pd.to_numeric(hydro_nodes["wse"], errors="coerce")
    hydro_nodes.loc[hydro_nodes["wse"] < -1.0e10, "wse"] = np.nan
    hydro_nodes["node_q"] = pd.to_numeric(hydro_nodes["node_q"], errors="coerce")
    hydro_nodes["p_dist_out"] = pd.to_numeric(hydro_nodes["p_dist_out"], errors="coerce")
    hydro_nodes["geoid_hght"] = pd.to_numeric(hydro_nodes["geoid_hght"], errors="coerce")

    merged = sword_nodes[
        ["node_id", "reach_id", "dist_out", "s_m", "wse", "width", "node_len"]
    ].rename(columns={"wse": "sword_wse"}).merge(
        hydro_nodes,
        on=["node_id", "reach_id"],
        how="left",
        suffixes=("_sword", ""),
    )
    merged["node_s_m"] = merged["p_dist_out"].fillna(merged["dist_out"]) - float(sword_nodes["dist_out"].min())
    merged = merged.sort_values("node_s_m").reset_index(drop=True)
    NODE_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(NODE_CACHE_CSV, index=False)
    return merged


def make_plot() -> Path:
    points, updated_row, matched_row, riversp_row = load_rows()
    filt = filtered_points(points, updated_row)

    x = points["s_m"].to_numpy(dtype="float64")
    y = points["height"].to_numpy(dtype="float64")
    xf = filt["s_m"].to_numpy(dtype="float64")
    yf = filt["height"].to_numpy(dtype="float64")
    x_km = x / 1000.0
    xf_km = xf / 1000.0

    ols_slope_m_per_m, ols_intercept = np.polyfit(xf, yf, deg=1)
    ols_slope_m_per_km = ols_slope_m_per_m * 1000.0

    x_line = np.linspace(float(x.min()), float(x.max()), 300)
    x_line_km = x_line / 1000.0
    ols_line = ols_intercept + ols_slope_m_per_m * x_line

    riversp_wse = float(riversp_row["wse"])
    riversp_slope_native = float(riversp_row["riversp_slope"])
    riversp_slope_m_per_km = riversp_slope_native * 1000.0
    x_anchor = float(np.mean(x))
    riversp_line = riversp_wse + riversp_slope_native * (x_line - x_anchor)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(x_km, y, s=10, alpha=0.18, color="#888888", label=f"All PIXC points (n={len(points):,})")
    ax.scatter(xf_km, yf, s=14, alpha=0.62, color="#1F77B4", label=f"Filtered PIXC points (n={len(filt):,})")
    ax.plot(
        x_line_km,
        ols_line,
        color="#003A70",
        linewidth=3.0,
        label=f"Filtered OLS slope = {ols_slope_m_per_km:.3f} m/km",
    )
    ax.plot(
        x_line_km,
        riversp_line,
        color="#D62728",
        linewidth=2.4,
        linestyle="--",
        label=f"RiverSP WSE = {riversp_wse:.3f} m, slope = {riversp_slope_m_per_km:.3f} m/km",
    )
    ax.set_title("Reach 66405900761 | 2024-03-10 10:38:54 UTC", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Distance along reach (km)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Water surface elevation / height (m)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.92, fontsize=LEGEND_FONTSIZE)

    details = (
        f"PIXC file: {FILE_NAME}\n"
        f"Hydrocron RiverSP time: {matched_row['riversp_time_utc']} "
        f"(delta {float(matched_row['riversp_time_delta_s']):.0f} s, reach_q={int(matched_row['riversp_reach_q'])})\n"
        "RiverSP line is anchored at the mean PIXC along-reach distance."
    )
    ax.text(
        0.01,
        0.01,
        details,
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONTSIZE,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=250)
    plt.close(fig)
    return OUT_PNG


def make_geoid_corrected_plot() -> Path:
    geoid_points = pd.read_csv(GEOID_POINTS_CSV).dropna(
        subset=["s_m", "height_ellipsoid_tide_corrected", "height_geoid_tide_corrected", "geoid"]
    )
    _points, updated_row, matched_row, riversp_row = load_rows()
    nodes = load_or_fetch_hydrocron_nodes(matched_row)

    points = geoid_points.rename(
        columns={
            "height_ellipsoid_tide_corrected": "height",
            "height_geoid_tide_corrected": "height_geoid",
        }
    ).sort_values("s_m")

    x = points["s_m"].to_numpy(dtype="float64")
    y_ellipsoid = points["height"].to_numpy(dtype="float64")
    y_geoid = points["height_geoid"].to_numpy(dtype="float64")
    x_km = x / 1000.0

    slope_det = float(updated_row["updated_detrend_slope_m_per_km"]) / 1000.0
    y_detr = y_ellipsoid - slope_det * x
    lo = float(np.quantile(y_detr, float(updated_row["updated_tau_low"])))
    hi = float(np.quantile(y_detr, float(updated_row["updated_tau_high"])))
    keep = (y_detr >= lo) & (y_detr <= hi)

    xf = x[keep]
    yf_geoid = y_geoid[keep]
    xf_km = xf / 1000.0

    ols_slope_m_per_m, ols_intercept = np.polyfit(xf, yf_geoid, deg=1)
    ols_slope_m_per_km = ols_slope_m_per_m * 1000.0

    x_line = np.linspace(float(x.min()), float(x.max()), 300)
    x_line_km = x_line / 1000.0
    ols_line = ols_intercept + ols_slope_m_per_m * x_line

    riversp_wse = float(riversp_row["wse"])
    riversp_slope_native = float(riversp_row["riversp_slope"])
    riversp_slope_m_per_km = riversp_slope_native * 1000.0
    x_anchor = float(np.mean(x))
    riversp_line = riversp_wse + riversp_slope_native * (x_line - x_anchor)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(x_km, y_geoid, s=10, alpha=0.18, color="#888888", label=f"All PIXC points, height - geoid (n={len(points):,})")
    ax.scatter(xf_km, yf_geoid, s=14, alpha=0.62, color="#1F77B4", label=f"Filtered PIXC points (n={int(keep.sum()):,})")
    ax.plot(
        x_line_km,
        ols_line,
        color="#003A70",
        linewidth=3.0,
        label=f"Filtered OLS slope = {ols_slope_m_per_km:.3f} m/km",
    )
    ax.plot(
        x_line_km,
        riversp_line,
        color="#D62728",
        linewidth=2.4,
        linestyle="--",
        label=f"RiverSP WSE = {riversp_wse:.3f} m, slope = {riversp_slope_m_per_km:.3f} m/km",
    )
    valid_nodes = nodes[nodes["wse"].notna()].copy()
    if "node_q" in valid_nodes.columns:
        valid_nodes = valid_nodes[valid_nodes["node_q"].fillna(9) <= 1].copy()
    if not valid_nodes.empty:
        ax.scatter(
            valid_nodes["node_s_m"].to_numpy(dtype="float64") / 1000.0,
            valid_nodes["wse"].to_numpy(dtype="float64"),
            color="#D62728",
            edgecolor="white",
            linewidth=0.65,
            s=34,
            zorder=6,
            label=f"RiverSP node WSEs (n={len(valid_nodes):,})",
        )
    ax.set_title(
        "Reach 66405900761 | 2024-03-10 10:38:54 UTC | Geoid-corrected PIXC heights",
        fontsize=TITLE_FONTSIZE,
    )
    ax.set_xlabel("Distance along reach (km)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Water surface elevation, geoid-referenced (m)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.92, fontsize=LEGEND_FONTSIZE)

    median_geoid = float(points["geoid"].median())
    details = (
        f"PIXC height converted as height - geoid; median clipped geoid = {median_geoid:.3f} m\n"
        f"Hydrocron RiverSP time: {matched_row['riversp_time_utc']} "
        f"(delta {float(matched_row['riversp_time_delta_s']):.0f} s, reach_q={int(matched_row['riversp_reach_q'])})\n"
        "RiverSP reach and node WSEs are geoid-referenced Hydrocron values."
    )
    ax.text(
        0.99,
        0.01,
        details,
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONTSIZE,
        va="bottom",
        ha="right",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GEOD_PNG, dpi=250)
    plt.close(fig)
    return OUT_GEOD_PNG


if __name__ == "__main__":
    path = make_plot()
    print(f"Wrote {path}")
    if GEOID_POINTS_CSV.exists():
        corrected_path = make_geoid_corrected_plot()
        print(f"Wrote {corrected_path}")
