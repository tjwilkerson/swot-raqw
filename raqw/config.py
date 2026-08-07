from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class RAQWConfig:
    reach_id: str
    year: int
    run_root: Path
    reach_file: Path | None = None
    reach_layer: str | None = None
    reach_id_field: str = "reach_id"
    projected_crs: str | None = None

    pixc_short_name: str = "SWOT_L2_HR_PIXC_D"
    riversp_collection_name: str = "SWOT_L2_HR_RiverSP_D"
    hydrocron_url: str = (
        "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
    )
    hydrocron_timeout_seconds: int = 60
    geometry_search_start: str | None = None
    geometry_search_end: str | None = None
    hydrocron_window_minutes: int = 30
    allowed_reach_q: tuple[int, ...] = (0, 1)
    width_factor: float = 1.0
    correct_tides: bool = True
    minimum_reach_coverage: float = 0.95
    quantile_step: float = 0.01
    quantile_max_iterations: int = 10_000
    target_coverage: float = 0.95
    min_files_per_quantile: int = 5
    band_trim_fraction: float = 0.05
    mad_epsilon: float = 1e-12
    universal_k_cap: float = 8.0
    # Zero reproduces the preserved publication run. Set a positive value to
    # reject spatially screened overpasses below an explicit point count.
    min_points_per_granule: int = 0
    min_points_after_trim: int = 10
    min_unique_coordinates: int = 2

    core_tau_low: float = 0.10
    core_tau_high: float = 0.90
    min_core_width: float = 0.15
    tail_guard: float = 0.05
    max_core_outside_fraction: float = 0.10
    max_expand_outside_fraction: float = 0.20
    max_expand_z_multiplier: float = 1.75
    max_expand_endpoint_jump: float = 10.0
    min_expand_score_gain: float = -0.05

    width_weight: float = 4.5
    closeness_weight: float = 2.0
    smoothness_weight: float = 2.0
    centrality_weight: float = 1.0
    outside_band_weight: float = 1.25
    tail_penalty_weight: float = 0.3
    endpoint_jump_weight: float = 0.2
    trim_spread_weight: float = 1.5
    trim_stability_weight: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.reach_id).strip():
            raise ValueError("reach_id must not be empty.")
        if self.year < 2000:
            raise ValueError("year must be a four-digit calendar year.")
        if bool(self.geometry_search_start) != bool(self.geometry_search_end):
            raise ValueError(
                "geometry_search_start and geometry_search_end must be supplied together."
            )
        if self.hydrocron_timeout_seconds < 1:
            raise ValueError("hydrocron_timeout_seconds must be positive.")
        if not self.allowed_reach_q:
            raise ValueError("allowed_reach_q must contain at least one quality value.")
        for name in (
            "minimum_reach_coverage",
            "target_coverage",
            "band_trim_fraction",
            "core_tau_low",
            "core_tau_high",
            "min_core_width",
            "tail_guard",
            "max_core_outside_fraction",
            "max_expand_outside_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1; received {value}.")
        if self.core_tau_low >= self.core_tau_high:
            raise ValueError("core_tau_low must be smaller than core_tau_high.")
        if self.quantile_step <= 0 or self.quantile_step >= 1:
            raise ValueError("quantile_step must be between 0 and 1.")
        if self.quantile_max_iterations < 1:
            raise ValueError("quantile_max_iterations must be positive.")
        if self.min_files_per_quantile < 1:
            raise ValueError("min_files_per_quantile must be at least 1.")
        if self.mad_epsilon <= 0:
            raise ValueError("mad_epsilon must be positive.")
        if self.universal_k_cap <= 0:
            raise ValueError("universal_k_cap must be positive.")
        if self.min_points_per_granule < 0:
            raise ValueError("min_points_per_granule cannot be negative.")
        if self.min_points_after_trim < 2:
            raise ValueError("min_points_after_trim must be at least 2.")
        if self.min_unique_coordinates < 2:
            raise ValueError("min_unique_coordinates must be at least 2.")

    @property
    def raw_dir(self) -> Path:
        return self.run_root / "raw" / "pixc_d" / f"reach_{self.reach_id}" / str(self.year)

    @property
    def processed_dir(self) -> Path:
        return self.run_root / "processed" / f"reach_{self.reach_id}" / str(self.year)

    @property
    def points_dir(self) -> Path:
        return self.processed_dir / "pixc_points"

    @property
    def results_dir(self) -> Path:
        return self.run_root / "results" / f"reach_{self.reach_id}" / str(self.year)

    @property
    def reach_geometry_cache_path(self) -> Path:
        collection = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in self.riversp_collection_name
        )
        return (
            self.run_root
            / "cache"
            / "sword"
            / collection
            / f"reach_{self.reach_id}.geojson"
        )

    def as_serializable_dict(self) -> dict[str, object]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}

    @classmethod
    def from_toml(
        cls,
        path: Path,
        **overrides: Any,
    ) -> "RAQWConfig":
        """Load a run configuration from a TOML file.

        Values may be grouped under ``run``, ``data``, ``preprocessing``, and
        ``filter`` sections. Relative paths are resolved from the TOML file's
        directory. Non-``None`` keyword overrides take precedence.
        """

        config_path = Path(path).resolve()
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)

        values: dict[str, Any] = {}
        for section in ("run", "data", "preprocessing", "filter"):
            section_values = document.get(section, {})
            if not isinstance(section_values, dict):
                raise ValueError(f"[{section}] must be a TOML table.")
            values.update(section_values)

        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")

        for key, value in overrides.items():
            if value is not None:
                if key not in allowed:
                    raise ValueError(f"Unknown configuration override: {key}")
                values[key] = value

        for key in ("reach_file", "run_root"):
            if key not in values or values[key] is None:
                continue
            candidate = Path(values[key]).expanduser()
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            values[key] = candidate.resolve()

        if "allowed_reach_q" in values:
            values["allowed_reach_q"] = tuple(int(value) for value in values["allowed_reach_q"])
        return cls(**values)
