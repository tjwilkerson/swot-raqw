from __future__ import annotations

import argparse
from pathlib import Path

import earthaccess
import geopandas as gpd
import netCDF4 as nc
import pandas as pd

from pipeline_2024.paths import CHILE_REACHES_GPKG, VALID_DISCHARGE_REACHES_CSV


SHORT_NAME = "SWOT_L4_HR_DAWG_SOS_DISCHARGE_V3"
ALG_GROUP = "consensus"
Q_VARIABLE = "allq"
SWORD_REACH_COL = "reach_id"


def locate_or_download_sa_sos(preferred_path: Path | None, download_dir: Path) -> Path:
    if preferred_path is not None and preferred_path.exists():
        return preferred_path

    earthaccess.login(strategy="netrc")
    granules = earthaccess.search_data(short_name=SHORT_NAME, count=2000)
    sa_granules = [
        granule
        for granule in granules
        if str(granule["meta"]["native-id"]).lower().startswith("sa_")
    ]
    if not sa_granules:
        raise FileNotFoundError(f"No South America DAWG/SOS granule found for {SHORT_NAME}.")

    paths = earthaccess.download(sa_granules[:1], local_path=str(download_dir))
    if not paths:
        raise FileNotFoundError("earthaccess.download returned no local DAWG/SOS files.")
    return Path(paths[0])


def build_valid_discharge_reaches(
    sword_gpkg: Path,
    sos_nc_path: Path,
    out_csv: Path,
    reach_id_col: str = SWORD_REACH_COL,
) -> pd.DataFrame:
    gdf = gpd.read_file(sword_gpkg)
    gdf[reach_id_col] = pd.to_numeric(gdf[reach_id_col], errors="coerce")
    gdf = gdf.dropna(subset=[reach_id_col]).copy()
    gdf[reach_id_col] = gdf[reach_id_col].astype("int64")
    chile_reach_ids = set(gdf[reach_id_col].unique())

    records: list[dict[str, int | float]] = []
    with nc.Dataset(sos_nc_path, "r") as ds:
        reaches = ds.groups["reaches"]
        consensus = ds.groups[ALG_GROUP]
        reach_ids = reaches.variables[reach_id_col][:]
        q = consensus.variables[Q_VARIABLE][:]

        for idx, reach_id in enumerate(reach_ids):
            reach_id_int = int(reach_id)
            if reach_id_int not in chile_reach_ids:
                continue

            values = q[idx, :]
            valid = values.compressed() if hasattr(values, "compressed") else values[pd.notna(values)]
            n_valid = int(len(valid))
            if n_valid > 0:
                records.append(
                    {
                        "reach_id": reach_id_int,
                        "n_valid_obs": n_valid,
                        "mean_q": float(valid.mean()),
                    }
                )

    df = pd.DataFrame(records).sort_values(["n_valid_obs", "reach_id"], ascending=[False, True])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chile reach list with valid DAWG/SOS discharge.")
    parser.add_argument("--sword-gpkg", type=Path, default=CHILE_REACHES_GPKG)
    parser.add_argument("--sos-nc", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=Path(r"C:\UNESCO\Code\downloaded_files"))
    parser.add_argument("--out-csv", type=Path, default=VALID_DISCHARGE_REACHES_CSV)
    args = parser.parse_args()

    sos_nc = locate_or_download_sa_sos(args.sos_nc, args.download_dir)
    df = build_valid_discharge_reaches(args.sword_gpkg, sos_nc, args.out_csv)
    print(f"Wrote {args.out_csv}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()

