import os

import pytest
from shapely.geometry import LineString

from raqw.gui import (
    EarthdataCredentialError,
    PUBLICATION_2024_PROJECTED_CRS,
    _default_config,
    _format_duration,
    _map_view,
    authenticate_earthdata_for_gui,
    temporary_earthdata_credentials,
    validate_earthdata_credential_fields,
)


def test_gui_defaults_match_executed_publication_parameters() -> None:
    cfg = _default_config()

    assert PUBLICATION_2024_PROJECTED_CRS == "EPSG:32718"
    assert cfg.projected_crs is None  # Worldwide-safe automatic projection.
    assert cfg.hydrocron_window_minutes == 30
    assert cfg.allowed_reach_q == (0, 1)
    assert cfg.width_factor == 1.0
    assert cfg.correct_tides is True
    assert cfg.minimum_reach_coverage == 0.95
    assert cfg.quantile_step == 0.01
    assert cfg.quantile_max_iterations == 10_000
    assert cfg.min_points_per_granule == 0
    assert cfg.target_coverage == 0.95
    assert cfg.min_files_per_quantile == 5
    assert cfg.band_trim_fraction == 0.05
    assert cfg.mad_epsilon == 1e-12
    assert cfg.universal_k_cap == 8.0
    assert cfg.min_points_after_trim == 10
    assert cfg.min_unique_coordinates == 2
    assert cfg.core_tau_low == 0.10
    assert cfg.core_tau_high == 0.90
    assert cfg.min_core_width == 0.15
    assert cfg.tail_guard == 0.05
    assert cfg.max_core_outside_fraction == 0.10
    assert cfg.max_expand_outside_fraction == 0.20
    assert cfg.max_expand_z_multiplier == 1.75
    assert cfg.max_expand_endpoint_jump == 10.0
    assert cfg.min_expand_score_gain == -0.05
    assert cfg.width_weight == 4.5
    assert cfg.closeness_weight == 2.0
    assert cfg.smoothness_weight == 2.0
    assert cfg.centrality_weight == 1.0
    assert cfg.outside_band_weight == 1.25
    assert cfg.tail_penalty_weight == 0.3
    assert cfg.endpoint_jump_weight == 0.2
    assert cfg.trim_spread_weight == 1.5
    assert cfg.trim_stability_weight == 1.0


def test_temporary_credentials_are_removed(monkeypatch) -> None:
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)

    with temporary_earthdata_credentials("scientist", "secret"):
        assert os.environ["EARTHDATA_USERNAME"] == "scientist"
        assert os.environ["EARTHDATA_PASSWORD"] == "secret"

    assert "EARTHDATA_USERNAME" not in os.environ
    assert "EARTHDATA_PASSWORD" not in os.environ


def test_temporary_credentials_restore_existing_environment(monkeypatch) -> None:
    monkeypatch.setenv("EARTHDATA_USERNAME", "existing-user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "existing-password")

    with pytest.raises(RuntimeError):
        with temporary_earthdata_credentials("temporary-user", "temporary-password"):
            raise RuntimeError("analysis failed")

    assert os.environ["EARTHDATA_USERNAME"] == "existing-user"
    assert os.environ["EARTHDATA_PASSWORD"] == "existing-password"


def test_map_view_centers_and_scales_reach() -> None:
    longitude, latitude, zoom = _map_view(
        LineString([(-72.0, -41.0), (-71.9, -40.9)])
    )

    assert longitude == pytest.approx(-71.95)
    assert latitude == pytest.approx(-40.95)
    assert 2.0 <= zoom <= 15.0


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(5, "5s"), (65, "1m 05s"), (3_665, "1h 01m")],
)
def test_formats_progress_duration(seconds: float, expected: str) -> None:
    assert _format_duration(seconds) == expected


def test_gui_authentication_uses_environment_without_terminal_prompt(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("earthaccess.login", lambda **kwargs: calls.append(kwargs) or object())

    with temporary_earthdata_credentials("scientist", "secret"):
        authenticate_earthdata_for_gui("scientist", "secret")

    assert calls == [{"strategy": "environment"}]


def test_gui_authentication_uses_netrc_when_environment_is_empty(monkeypatch) -> None:
    calls = []
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setattr("earthaccess.login", lambda **kwargs: calls.append(kwargs) or object())

    authenticate_earthdata_for_gui(None, None)

    assert calls == [{"strategy": "netrc"}]


def test_rejected_entered_credentials_have_actionable_message(monkeypatch) -> None:
    from earthaccess.exceptions import LoginAttemptFailure

    monkeypatch.setattr(
        "earthaccess.login",
        lambda **kwargs: (_ for _ in ()).throw(LoginAttemptFailure("HTTP 401")),
    )

    with pytest.raises(EarthdataCredentialError) as caught:
        authenticate_earthdata_for_gui("scientist", "wrong-password")

    assert "did not accept the entered credentials" in str(caught.value)
    assert any("urs.earthdata.nasa.gov" in item for item in caught.value.guidance)
    assert "wrong-password" not in str(caught.value)


def test_missing_netrc_has_actionable_message(monkeypatch) -> None:
    from earthaccess.exceptions import LoginStrategyUnavailable

    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setattr(
        "earthaccess.login",
        lambda **kwargs: (_ for _ in ()).throw(LoginStrategyUnavailable("missing")),
    )

    with pytest.raises(EarthdataCredentialError) as caught:
        authenticate_earthdata_for_gui(None, None)

    assert "No usable Earthdata credentials" in str(caught.value)
    assert any("_netrc" in item for item in caught.value.guidance)


@pytest.mark.parametrize(
    ("username", "password", "missing"),
    [("scientist", None, "password"), (None, "secret", "username")],
)
def test_incomplete_gui_credential_pair_identifies_missing_field(
    username: str | None,
    password: str | None,
    missing: str,
) -> None:
    with pytest.raises(EarthdataCredentialError, match=missing):
        validate_earthdata_credential_fields(username, password)
