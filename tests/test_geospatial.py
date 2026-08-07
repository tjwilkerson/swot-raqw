from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

from raqw.config import RAQWConfig
from raqw.preprocess import build_reach_line, granule_footprint_from_umm


class FakeGranule:
    render_dict = {
        "umm": {
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "GPolygons": [
                            {
                                "Boundary": {
                                    "Points": [
                                        {"Longitude": -72.0, "Latitude": -41.0},
                                        {"Longitude": -71.9, "Latitude": -41.0},
                                        {"Longitude": -71.9, "Latitude": -40.9},
                                        {"Longitude": -72.0, "Latitude": -40.9},
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
    }


def test_closes_granule_polygon() -> None:
    footprint = granule_footprint_from_umm(FakeGranule())
    assert footprint is not None
    assert footprint.is_valid
    assert footprint.area > 0


def test_merges_multipart_reach_before_projection(tmp_path: Path) -> None:
    first = LineString([(-72.0, -41.0), (-71.99, -41.0)])
    second = LineString([(-71.99, -41.0), (-71.98, -41.0)])
    reaches = gpd.GeoDataFrame(
        {"reach_id": ["12345678901"]},
        geometry=[MultiLineString([first, second])],
        crs="EPSG:4326",
    )
    reach_file = tmp_path / "reaches.geojson"
    reaches.to_file(reach_file, driver="GeoJSON")
    cfg = RAQWConfig(
        reach_id="12345678901",
        year=2024,
        reach_file=reach_file,
        run_root=tmp_path,
        projected_crs="EPSG:32718",
    )

    line = build_reach_line(cfg)

    assert line.geom_type == "LineString"
    assert line.length > 1_000

