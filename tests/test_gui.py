import os

import pytest
from shapely.geometry import LineString

from raqw.gui import (
    _format_duration,
    _map_view,
    authenticate_earthdata_for_gui,
    temporary_earthdata_credentials,
)


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
