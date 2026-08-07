"""Resolve one SWORD reach from a local file or the Hydrocron API."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import shape

from .config import RAQWConfig


def _property_values(value: Any) -> set[str]:
    values = value if isinstance(value, list) else [value]
    return {str(item) for item in values if item is not None and str(item).strip()}


def _expected_sword_version(collection_name: str) -> str | None:
    if collection_name.endswith("_D"):
        return "17b"
    if collection_name.endswith("_2.0"):
        return "16"
    return None


def _normalized_sword_version(value: str) -> str:
    normalized = value.lower().removeprefix("v")
    # Hydrocron currently reports "7b" in Version D records while the PO.DAAC
    # version guide names that network release SWORD v17b.
    return "17b" if normalized == "7b" else normalized


def _geometry_from_payload(payload: dict[str, Any], cfg: RAQWConfig):
    metadata = payload.get("raqw_metadata", {})
    cached_collection = metadata.get("collection_name")
    if cached_collection and cached_collection != cfg.riversp_collection_name:
        raise ValueError(
            "Cached reach geometry collection does not match the configured "
            f"collection: {cached_collection!r} != {cfg.riversp_collection_name!r}."
        )

    features = payload.get("features", [])
    if not features:
        raise ValueError("Hydrocron GeoJSON contains no reach feature.")
    feature = features[0]
    geometry_data = feature.get("geometry")
    if not geometry_data:
        raise ValueError("Hydrocron reach feature has no geometry.")
    geometry = shape(geometry_data)
    if geometry.is_empty or geometry.geom_type not in {"LineString", "MultiLineString"}:
        raise ValueError(
            "Hydrocron reach geometry must be a LineString or MultiLineString; "
            f"received {geometry.geom_type}."
        )

    returned_ids = _property_values(feature.get("properties", {}).get("reach_id"))
    if returned_ids and returned_ids != {str(cfg.reach_id)}:
        raise ValueError(
            f"Hydrocron returned reach ID(s) {sorted(returned_ids)}, "
            f"not requested reach {cfg.reach_id}."
        )
    properties = feature.get("properties", {})
    returned_collections = _property_values(properties.get("collection_shortname"))
    if returned_collections and cfg.riversp_collection_name not in returned_collections:
        raise ValueError(
            f"Hydrocron returned collection(s) {sorted(returned_collections)}, not "
            f"configured collection {cfg.riversp_collection_name}."
        )
    expected_sword = _expected_sword_version(cfg.riversp_collection_name)
    returned_sword = _property_values(properties.get("sword_version"))
    normalized_sword = {_normalized_sword_version(value) for value in returned_sword}
    if expected_sword and normalized_sword and normalized_sword != {expected_sword}:
        raise ValueError(
            f"Hydrocron returned SWORD version(s) {sorted(returned_sword)} for "
            f"{cfg.riversp_collection_name}; expected SWORD v{expected_sword}."
        )
    return geometry


def _query_windows(cfg: RAQWConfig) -> list[tuple[str, str]]:
    if cfg.geometry_search_start and cfg.geometry_search_end:
        return [(cfg.geometry_search_start, cfg.geometry_search_end)]

    analysis_window = (
        f"{cfg.year}-01-01T00:00:00Z",
        f"{cfg.year}-12-31T23:59:59Z",
    )
    now = datetime.now(timezone.utc)
    broad_end_year = max(cfg.year, now.year)
    broad_window = (
        "2023-07-01T00:00:00Z",
        f"{broad_end_year}-12-31T23:59:59Z",
    )
    return [analysis_window] if analysis_window == broad_window else [analysis_window, broad_window]


def fetch_hydrocron_reach_geometry(cfg: RAQWConfig, refresh: bool = False):
    """Fetch and cache a single observed SWORD reach as GeoJSON.

    The configured analysis year is searched first. If no RiverSP observation
    is available in that year, the request expands to the SWOT science-orbit
    era. Supplying ``geometry_search_start`` and ``geometry_search_end`` disables
    that automatic fallback and pins the lookup interval explicitly.
    """

    cache_path = cfg.reach_geometry_cache_path
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return _geometry_from_payload(payload, cfg)

    errors: list[str] = []
    for start_time, end_time in _query_windows(cfg):
        params = {
            "feature": "Reach",
            "feature_id": str(cfg.reach_id),
            "start_time": start_time,
            "end_time": end_time,
            "collection_name": cfg.riversp_collection_name,
            "compact": "true",
            "fields": (
                "reach_id,time_str,geometry,sword_version,"
                "collection_shortname,collection_version"
            ),
        }
        try:
            response = requests.get(
                cfg.hydrocron_url,
                params=params,
                headers={"Accept": "application/geo+json"},
                timeout=cfg.hydrocron_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            geometry = _geometry_from_payload(payload, cfg)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{start_time} to {end_time}: {error}")
            continue

        feature = payload["features"][0]
        properties = feature.setdefault("properties", {})
        sword_versions = sorted(_property_values(properties.get("sword_version")))
        payload["raqw_metadata"] = {
            "source": "Hydrocron",
            "endpoint": cfg.hydrocron_url,
            "reach_id": str(cfg.reach_id),
            "collection_name": cfg.riversp_collection_name,
            "sword_versions": sword_versions,
            "expected_sword_release": (
                f"v{_expected_sword_version(cfg.riversp_collection_name)}"
                if _expected_sword_version(cfg.riversp_collection_name)
                else None
            ),
            "query_start": start_time,
            "query_end": end_time,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".geojson.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
        print(
            f"Cached Hydrocron reach geometry at {cache_path} "
            f"(SWORD={sword_versions or ['not reported']})"
        )
        return geometry

    detail = "; ".join(errors) or "no response details"
    raise RuntimeError(
        f"Could not retrieve Hydrocron geometry for reach {cfg.reach_id}. "
        "Hydrocron only returns reaches having a RiverSP observation in the "
        f"search interval. Attempts: {detail}"
    )


def read_local_reach_geometry(cfg: RAQWConfig):
    if cfg.reach_file is None:
        raise ValueError("No local reach_file was configured.")
    reaches = gpd.read_file(cfg.reach_file, layer=cfg.reach_layer)
    if cfg.reach_id_field not in reaches.columns:
        raise ValueError(
            f"Column {cfg.reach_id_field!r} is absent from {cfg.reach_file}. "
            f"Available columns: {list(reaches.columns)}"
        )
    reaches[cfg.reach_id_field] = reaches[cfg.reach_id_field].astype(str)
    reach = reaches.loc[reaches[cfg.reach_id_field] == str(cfg.reach_id)]
    if reach.empty:
        raise ValueError(f"Reach {cfg.reach_id} was not found in {cfg.reach_file}.")
    if reach.crs is None:
        raise ValueError(f"Local reach file {cfg.reach_file} has no CRS.")
    return reach.to_crs(4326).iloc[0].geometry


def get_reach_geometry(cfg: RAQWConfig):
    """Return the configured reach in WGS 84.

    An explicit local file is the reproducibility/offline override. Otherwise,
    one reach is retrieved from Hydrocron and cached below ``run_root/cache``.
    """

    if cfg.reach_file is not None:
        return read_local_reach_geometry(cfg)
    return fetch_hydrocron_reach_geometry(cfg)


def geometry_provenance_path(cfg: RAQWConfig) -> Path | None:
    """Return the local geometry input that should be added to a manifest."""

    if cfg.reach_file is not None:
        return cfg.reach_file
    return cfg.reach_geometry_cache_path if cfg.reach_geometry_cache_path.exists() else None
