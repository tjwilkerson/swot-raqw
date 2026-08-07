"""Build a SWORD reach list constrained by available DAWG/SOS discharge."""

from __future__ import annotations

from pathlib import Path

import earthaccess
import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd


DEFAULT_SOS_SHORT_NAME = "SWOT_L4_HR_DAWG_SOS_DISCHARGE_V3"


def download_south_america_sos(
    output_dir: Path,
    short_name: str = DEFAULT_SOS_SHORT_NAME,
    login: bool = True,
) -> Path:
    """Download the first South America DAWG/SOS granule."""

    if login:
        earthaccess.login(strategy="netrc")
    granules = earthaccess.search_data(short_name=short_name, count=2000)
    matches = [
        granule
        for granule in granules
        if str(granule["meta"]["native-id"]).lower().startswith("sa_")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No South America DAWG/SOS granule was found for {short_name}."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = earthaccess.download(matches[:1], str(output_dir))
    if not downloaded:
        raise FileNotFoundError("earthaccess did not return a downloaded SOS file.")
    return Path(downloaded[0])


def build_valid_discharge_reaches(
    sword_file: Path,
    sos_file: Path,
    output_csv: Path | None = None,
    reach_layer: str | None = None,
    reach_id_field: str = "reach_id",
    algorithm_group: str = "consensus",
    discharge_variable: str = "allq",
) -> pd.DataFrame:
    """Return SWORD reaches having at least one finite SOS discharge value."""

    reaches = gpd.read_file(sword_file, layer=reach_layer)
    if reach_id_field not in reaches.columns:
        raise ValueError(
            f"{sword_file} does not contain reach identifier {reach_id_field!r}."
        )
    reach_ids = pd.to_numeric(reaches[reach_id_field], errors="coerce").dropna()
    sword_ids = set(reach_ids.astype("int64"))

    records: list[dict[str, int | float]] = []
    with nc.Dataset(sos_file, "r") as dataset:
        if "reaches" not in dataset.groups or algorithm_group not in dataset.groups:
            raise ValueError(
                f"SOS file must contain 'reaches' and {algorithm_group!r} groups."
            )
        source_ids = dataset.groups["reaches"].variables[reach_id_field][:]
        discharge = dataset.groups[algorithm_group].variables[discharge_variable][:]
        for index, source_id in enumerate(source_ids):
            reach_id = int(source_id)
            if reach_id not in sword_ids:
                continue
            values = np.ma.asarray(discharge[index]).compressed()
            values = np.asarray(values, dtype="float64")
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            records.append(
                {
                    "reach_id": str(reach_id),
                    "n_valid_obs": int(len(values)),
                    "mean_q": float(np.mean(values)),
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        result = pd.DataFrame(columns=["reach_id", "n_valid_obs", "mean_q"])
    else:
        result = result.sort_values(
            ["n_valid_obs", "reach_id"],
            ascending=[False, True],
        ).reset_index(drop=True)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result
