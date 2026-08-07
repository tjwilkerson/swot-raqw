from pathlib import Path

import pytest

from raqw.config import RAQWConfig


def test_loads_sectioned_toml_and_resolves_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[run]
reach_id = "12345678901"
year = 2024

[data]
reach_file = "inputs/reaches.geojson"
run_root = "outputs"
projected_crs = "EPSG:32718"

[preprocessing]
allowed_reach_q = [0, 1]
min_points_per_granule = 50

[filter]
core_tau_low = 0.10
core_tau_high = 0.90
""".strip(),
        encoding="utf-8",
    )

    cfg = RAQWConfig.from_toml(config_path)

    assert cfg.reach_id == "12345678901"
    assert cfg.year == 2024
    assert cfg.reach_file == (tmp_path / "inputs" / "reaches.geojson").resolve()
    assert cfg.run_root == (tmp_path / "outputs").resolve()
    assert cfg.allowed_reach_q == (0, 1)
    assert cfg.min_points_per_granule == 50


def test_rejects_unknown_configuration_key(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
[run]
reach_id = "123"
year = 2024
invented_parameter = true

[data]
reach_file = "reaches.geojson"
run_root = "outputs"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown configuration keys"):
        RAQWConfig.from_toml(config_path)


def test_loads_online_geometry_configuration_without_reach_file(tmp_path: Path) -> None:
    config_path = tmp_path / "online.toml"
    config_path.write_text(
        """
[run]
reach_id = "12345678901"
year = 2024

[data]
run_root = "outputs"
riversp_collection_name = "SWOT_L2_HR_RiverSP_D"
""".strip(),
        encoding="utf-8",
    )

    cfg = RAQWConfig.from_toml(config_path)

    assert cfg.reach_file is None
    assert cfg.run_root == (tmp_path / "outputs").resolve()
    assert cfg.reach_geometry_cache_path.name == "reach_12345678901.geojson"


def test_validates_tau_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="core_tau_low"):
        RAQWConfig(
            reach_id="123",
            year=2024,
            reach_file=tmp_path / "reaches.geojson",
            run_root=tmp_path,
            core_tau_low=0.9,
            core_tau_high=0.1,
        )
