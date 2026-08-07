from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import earthaccess
import geopandas as gpd
import pandas as pd

from .config import RAQWConfig
from .filter import run_filter
from .preprocess import granule_footprint_from_umm, preprocess_reach_year


def run_reach_task(payload: dict[str, object]) -> dict[str, object]:
    reach_id = str(payload["reach_id"])
    year = int(payload["year"])
    cfg = RAQWConfig(
        reach_id=reach_id,
        year=year,
        reach_file=Path(payload["reach_file"]),
        run_root=Path(payload["run_root"]),
        reach_id_field=str(payload["reach_id_field"]),
        projected_crs=payload.get("projected_crs"),
    )
    paths = [Path(path) for path in payload["nc_paths"]]
    try:
        summary, slopes = preprocess_reach_year(cfg, login=False, nc_paths=paths)
        results = run_filter(cfg, make_plots=True)
        return {
            "reach_id": reach_id,
            "year": year,
            "status": "ok",
            "downloaded_granules": len(summary),
            "accepted_granules": slopes["file"].nunique(),
            "result_rows": len(results),
            "figure_count": len(list((cfg.results_dir / "plots").glob("*.png"))),
            "error": "",
        }
    except Exception as exc:
        return {
            "reach_id": reach_id,
            "year": year,
            "status": "failed",
            "downloaded_granules": len(paths),
            "accepted_granules": 0,
            "result_rows": 0,
            "figure_count": 0,
            "error": repr(exc),
        }


def download_shared_year(
    reach_ids: list[str],
    year: int,
    reach_file: Path,
    reach_id_field: str,
    shared_dir: Path,
) -> dict[str, list[Path]]:
    reaches = gpd.read_file(reach_file).to_crs(4326)
    reaches[reach_id_field] = reaches[reach_id_field].astype(str)
    reaches = reaches.loc[reaches[reach_id_field].isin(reach_ids)].copy()
    geometries = {
        str(row[reach_id_field]): row.geometry
        for _, row in reaches.iterrows()
    }
    if len(geometries) != len(reach_ids):
        missing = sorted(set(reach_ids) - set(geometries))
        raise ValueError(f"Reach IDs absent from {reach_file}: {missing}")

    earthaccess.login(strategy="netrc")
    bounds = reaches.total_bounds
    granules = earthaccess.search_data(
        short_name="SWOT_L2_HR_PIXC_D",
        bounding_box=tuple(bounds),
        temporal=(f"{year}-01-01T00:00:00", f"{year}-12-31T23:59:59"),
    )
    assignments: dict[str, list] = {reach_id: [] for reach_id in reach_ids}
    selected_by_name = {}
    for granule in granules:
        footprint = granule_footprint_from_umm(granule)
        if footprint is None:
            continue
        covered = [
            reach_id
            for reach_id, geometry in geometries.items()
            if footprint.covers(geometry)
        ]
        if not covered:
            continue
        name = granule["meta"]["native-id"]
        selected_by_name[name] = granule
        for reach_id in covered:
            assignments[reach_id].append(name)

    shared_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name: path for path in shared_dir.glob("*.nc")}
    missing_granules = [
        granule
        for name, granule in selected_by_name.items()
        if f"{name}.nc" not in existing and name not in existing
    ]
    if missing_granules:
        print(f"Year {year}: downloading {len(missing_granules)} shared PIXC granules")
        earthaccess.download(missing_granules, str(shared_dir), threads=2)
    paths_by_name = {path.name: path for path in shared_dir.glob("*.nc")}

    result: dict[str, list[Path]] = {}
    for reach_id, names in assignments.items():
        wanted = set(names)
        result[reach_id] = [
            path
            for file_name, path in paths_by_name.items()
            if file_name in wanted or Path(file_name).stem in wanted
        ]
    return result


def run_batch(
    reach_csv: Path,
    years: list[int],
    reach_file: Path,
    run_root: Path,
    reach_id_field: str = "reach_id",
    projected_crs: str | None = None,
    delete_raw_after_success: bool = True,
    max_workers: int = 4,
) -> pd.DataFrame:
    labels = pd.read_csv(reach_csv, dtype={reach_id_field: str})
    if reach_id_field not in labels.columns:
        raise ValueError(f"{reach_csv} does not contain {reach_id_field!r}.")
    reach_ids = labels[reach_id_field].dropna().drop_duplicates().tolist()
    status_path = run_root / "batch_status.csv"
    run_root.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(status_path, dtype={"reach_id": str}) if status_path.exists() else pd.DataFrame()
    completed = set(
        zip(
            existing.loc[existing.get("status", pd.Series(dtype=str)) == "ok", "reach_id"],
            existing.loc[existing.get("status", pd.Series(dtype=str)) == "ok", "year"].astype(int),
        )
        if not existing.empty
        else []
    )
    rows = existing.to_dict("records") if not existing.empty else []
    tasks = [(reach_id, year) for year in years for reach_id in reach_ids]
    task_index = 0
    for year in years:
        pending_ids = [reach_id for reach_id in reach_ids if (reach_id, year) not in completed]
        shared_dir = run_root / "raw" / "shared" / str(year)
        paths_by_reach = (
            download_shared_year(
                pending_ids,
                year,
                reach_file,
                reach_id_field,
                shared_dir,
            )
            if pending_ids
            else {}
        )
        payloads = [
            {
                "reach_id": reach_id,
                "year": year,
                "reach_file": str(reach_file.resolve()),
                "run_root": str(run_root.resolve()),
                "reach_id_field": reach_id_field,
                "projected_crs": projected_crs,
                "nc_paths": [str(path) for path in paths_by_reach.get(reach_id, [])],
            }
            for reach_id in pending_ids
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(run_reach_task, payload): payload
                for payload in payloads
            }
            for future in as_completed(future_map):
                payload = future_map[future]
                row = future.result()
                task_index += 1
                print(
                    f"[{task_index}/{len(tasks)}] reach {row['reach_id']}, "
                    f"{year}: {row['status']}"
                )
                rows = [
                    item
                    for item in rows
                    if not (
                        str(item.get("reach_id")) == row["reach_id"]
                        and int(item.get("year")) == year
                    )
                ]
                rows.append(row)
                pd.DataFrame(rows).sort_values(["reach_id", "year"]).to_csv(
                    status_path,
                    index=False,
                )
        task_index = sum(1 for _, task_year in tasks if task_year <= year)

        if delete_raw_after_success and shared_dir.exists():
            shutil.rmtree(shared_dir)

    return pd.DataFrame(rows).sort_values(["reach_id", "year"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAQW for reach IDs in a CSV.")
    parser.add_argument("--reach-csv", required=True, type=Path)
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--reach-file", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("data") / "raqw_batch")
    parser.add_argument("--reach-id-field", default="reach_id")
    parser.add_argument("--projected-crs")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    status = run_batch(
        reach_csv=args.reach_csv,
        years=list(range(args.start_year, args.end_year + 1)),
        reach_file=args.reach_file,
        run_root=args.run_root,
        reach_id_field=args.reach_id_field,
        projected_crs=args.projected_crs,
        delete_raw_after_success=not args.keep_raw,
        max_workers=args.max_workers,
    )
    print(status["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
