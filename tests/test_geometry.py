import json
from pathlib import Path

import pytest
from shapely.geometry import LineString

from raqw.config import RAQWConfig
from raqw.geometry import fetch_hydrocron_reach_geometry


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def online_config(tmp_path: Path) -> RAQWConfig:
    return RAQWConfig(
        reach_id="12345678901",
        year=2024,
        run_root=tmp_path,
        projected_crs="EPSG:32718",
    )


def hydrocron_payload(reach_id: str = "12345678901") -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "reach_id": [reach_id, reach_id],
                    "time_str": ["2024-01-02T00:00:00Z"],
                    "sword_version": ["17b"],
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-72.0, -41.0], [-71.99, -41.0]],
                },
            }
        ],
    }


def test_fetches_and_caches_one_hydrocron_reach(tmp_path: Path, monkeypatch) -> None:
    cfg = online_config(tmp_path)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(hydrocron_payload())

    monkeypatch.setattr("raqw.geometry.requests.get", fake_get)
    geometry = fetch_hydrocron_reach_geometry(cfg)

    assert isinstance(geometry, LineString)
    assert len(calls) == 1
    assert calls[0][1]["headers"] == {"Accept": "application/geo+json"}
    assert calls[0][1]["params"]["collection_name"] == "SWOT_L2_HR_RiverSP_D"
    assert cfg.reach_geometry_cache_path.exists()
    cached = json.loads(cfg.reach_geometry_cache_path.read_text(encoding="utf-8"))
    assert cached["raqw_metadata"]["sword_versions"] == ["17b"]

    monkeypatch.setattr(
        "raqw.geometry.requests.get",
        lambda *args, **kwargs: pytest.fail("cached geometry should avoid the network"),
    )
    cached_geometry = fetch_hydrocron_reach_geometry(cfg)
    assert cached_geometry.equals_exact(geometry, tolerance=0.0)


def test_rejects_wrong_reach_id_from_hydrocron(tmp_path: Path, monkeypatch) -> None:
    cfg = online_config(tmp_path)
    monkeypatch.setattr(
        "raqw.geometry.requests.get",
        lambda *args, **kwargs: FakeResponse(hydrocron_payload("99999999999")),
    )

    with pytest.raises(RuntimeError, match="not requested reach"):
        fetch_hydrocron_reach_geometry(cfg)


def test_rejects_sword_version_inconsistent_with_collection(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = online_config(tmp_path)
    payload = hydrocron_payload()
    payload["features"][0]["properties"]["sword_version"] = ["16"]
    monkeypatch.setattr(
        "raqw.geometry.requests.get",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    with pytest.raises(RuntimeError, match="expected SWORD v17b"):
        fetch_hydrocron_reach_geometry(cfg)
