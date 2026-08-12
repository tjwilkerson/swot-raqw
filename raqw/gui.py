"""Local Streamlit interface for the RAQW workflow."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterator

from shapely.geometry import mapping, shape

from .config import RAQWConfig


class EarthdataCredentialError(RuntimeError):
    """An Earthdata login problem that can be corrected in the GUI."""

    def __init__(self, message: str, guidance: tuple[str, ...]) -> None:
        super().__init__(message)
        self.guidance = guidance


@contextmanager
def temporary_earthdata_credentials(
    username: str | None,
    password: str | None,
) -> Iterator[None]:
    """Expose credentials only for the duration of an analysis call."""

    names = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    previous = {name: os.environ.get(name) for name in names}
    try:
        if username and password:
            os.environ[names[0]] = username
            os.environ[names[1]] = password
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _map_view(geometry) -> tuple[float, float, float]:
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    span = max(max_lon - min_lon, max_lat - min_lat, 1e-6)
    zoom = max(2.0, min(15.0, math.log2(360.0 / span) - 1.5))
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0, zoom


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def authenticate_earthdata_for_gui(username: str | None, password: str | None):
    """Authenticate without falling through to a terminal-only prompt."""

    import earthaccess
    from earthaccess.exceptions import LoginAttemptFailure, LoginStrategyUnavailable

    has_environment_credentials = bool(
        os.environ.get("EARTHDATA_TOKEN")
        or (
            os.environ.get("EARTHDATA_USERNAME")
            and os.environ.get("EARTHDATA_PASSWORD")
        )
    )
    strategy = "environment" if (username and password) or has_environment_credentials else "netrc"
    try:
        return earthaccess.login(strategy=strategy)
    except LoginAttemptFailure as error:
        source = "entered" if username and password else "saved"
        raise EarthdataCredentialError(
            f"Earthdata Login did not accept the {source} credentials.",
            (
                "Check the username and password, then try again.",
                "Confirm that the same credentials work at urs.earthdata.nasa.gov.",
                "If the account was recently created, confirm that its email verification is complete.",
                "RAQW did not save the password you entered.",
            ),
        ) from error
    except LoginStrategyUnavailable as error:
        if strategy == "netrc":
            raise EarthdataCredentialError(
                "No usable Earthdata credentials were found on this computer.",
                (
                    "Enter both the Earthdata username and password in the sidebar, then run again.",
                    "Alternatively, configure `_netrc` on Windows or `.netrc` on macOS/Linux.",
                    "Use an Earthdata Login account; Hydrocron access alone does not authenticate PIXC downloads.",
                ),
            ) from error
        raise EarthdataCredentialError(
            "The saved Earthdata environment credentials are incomplete.",
            (
                "Enter both the Earthdata username and password in the sidebar.",
                "Or correct the `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables.",
            ),
        ) from error


def validate_earthdata_credential_fields(
    username: str | None,
    password: str | None,
) -> None:
    """Require a complete credential pair when either GUI field is used."""

    if bool(username) == bool(password):
        return
    missing = "password" if username else "username"
    raise EarthdataCredentialError(
        f"The Earthdata {missing} is missing.",
        (
            "Enter both the Earthdata username and password, or leave both fields blank to use saved credentials.",
            "RAQW never writes credentials entered in the sidebar to its configuration or outputs.",
        ),
    )


def display_earthdata_credential_error(st, error: EarthdataCredentialError) -> None:
    """Render a concise correction path instead of a technical traceback."""

    st.error(f"Earthdata sign-in failed: {error}")
    st.markdown("**How to fix it:**")
    for instruction in error.guidance:
        st.markdown(f"- {instruction}")
    st.markdown(
        "[Open NASA Earthdata Login](https://urs.earthdata.nasa.gov/) "
        "to verify the account or reset the password."
    )


class _LiveLog(io.StringIO):
    def __init__(self, placeholder) -> None:
        super().__init__()
        self.placeholder = placeholder

    def write(self, text: str) -> int:
        result = super().write(text)
        lines = self.getvalue().splitlines()[-30:]
        if lines:
            self.placeholder.code("\n".join(lines), language=None)
        return result


def _configuration_widgets(st) -> dict[str, object]:
    values: dict[str, object] = {}
    with st.expander("Preprocessing parameters", expanded=True):
        first, second, third = st.columns(3)
        values["minimum_reach_coverage"] = first.number_input(
            "Minimum reach coverage", 0.0, 1.0, 0.95, 0.01
        )
        values["min_points_per_granule"] = second.number_input(
            "Minimum PIXC points", 0, 100_000, 0, 1
        )
        values["width_factor"] = third.number_input(
            "Width multiplier", 0.1, 10.0, 1.0, 0.1
        )
        values["correct_tides"] = first.checkbox("Apply tide corrections", value=True)
        values["hydrocron_window_minutes"] = second.number_input(
            "RiverSP match window (minutes)", 1, 1_440, 30, 1
        )
        values["quantile_step"] = third.number_input(
            "Quantile step", 0.001, 0.25, 0.01, 0.001, format="%.3f"
        )
        quality_text = st.text_input("Allowed RiverSP reach_q values", "0,1")
        try:
            values["allowed_reach_q"] = tuple(
                int(item.strip()) for item in quality_text.split(",") if item.strip()
            )
        except ValueError:
            st.error("RiverSP reach_q values must be comma-separated integers.")
            values["allowed_reach_q"] = (0, 1)

    with st.expander("Quantile-window parameters", expanded=True):
        first, second, third = st.columns(3)
        values["core_tau_low"] = first.number_input(
            "Core τ lower bound", 0.0, 1.0, 0.10, 0.01
        )
        values["core_tau_high"] = second.number_input(
            "Core τ upper bound", 0.0, 1.0, 0.90, 0.01
        )
        values["min_core_width"] = third.number_input(
            "Minimum core width", 0.0, 1.0, 0.15, 0.01
        )
        values["target_coverage"] = first.number_input(
            "Reference coverage", 0.0, 1.0, 0.95, 0.01
        )
        values["min_files_per_quantile"] = second.number_input(
            "Minimum files per quantile", 1, 10_000, 5, 1
        )
        values["band_trim_fraction"] = third.number_input(
            "Band trim fraction", 0.0, 1.0, 0.05, 0.01
        )
        values["tail_guard"] = first.number_input(
            "Tail guard", 0.0, 1.0, 0.05, 0.01
        )
        values["min_points_after_trim"] = second.number_input(
            "Minimum retained points", 2, 100_000, 10, 1
        )
        values["min_unique_coordinates"] = third.number_input(
            "Minimum unique coordinates", 2, 100_000, 2, 1
        )

    with st.expander("Advanced selection controls"):
        defaults = {
            "max_core_outside_fraction": 0.10,
            "max_expand_outside_fraction": 0.20,
            "max_expand_z_multiplier": 1.75,
            "max_expand_endpoint_jump": 10.0,
            "min_expand_score_gain": -0.05,
            "universal_k_cap": 8.0,
            "mad_epsilon": 1e-12,
            "width_weight": 4.5,
            "closeness_weight": 2.0,
            "smoothness_weight": 2.0,
            "centrality_weight": 1.0,
            "outside_band_weight": 1.25,
            "tail_penalty_weight": 0.3,
            "endpoint_jump_weight": 0.2,
            "trim_spread_weight": 1.5,
            "trim_stability_weight": 1.0,
        }
        columns = st.columns(3)
        for index, (name, default) in enumerate(defaults.items()):
            values[name] = columns[index % 3].number_input(
                name.replace("_", " ").title(),
                value=default,
                format="%.12g",
                key=f"advanced_{name}",
            )
    return values


def _display_map(st, pdk, geometry_data: dict, reach_id: str) -> None:
    geometry = shape(geometry_data)
    longitude, latitude, zoom = _map_view(geometry)
    feature = {
        "type": "Feature",
        "properties": {"reach_id": reach_id},
        "geometry": geometry_data,
    }
    layer = pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": [feature]},
        get_line_color=[0, 115, 170, 230],
        get_line_width=5,
        line_width_min_pixels=3,
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                longitude=longitude,
                latitude=latitude,
                zoom=zoom,
            ),
            tooltip={"text": "SWORD reach {reach_id}"},
        ),
        width="stretch",
    )


def _display_results(st, cfg: RAQWConfig) -> None:
    results_path = cfg.results_dir / "raqw_results.csv"
    if not results_path.exists():
        st.info("No completed results were found for this reach and year.")
        return

    import pandas as pd

    results = pd.read_csv(results_path)
    first, second, third = st.columns(3)
    first.metric("Accepted observations", len(results))
    slope_column = "raqw_filtered_slope_m_per_km"
    if slope_column in results:
        second.metric("Median filtered slope", f"{results[slope_column].median():.3f} m/km")
    if "raqw_keep_fraction" in results:
        third.metric("Median retained fraction", f"{results['raqw_keep_fraction'].median():.2f}")
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "Download results CSV",
        data=results_path.read_bytes(),
        file_name=results_path.name,
        mime="text/csv",
    )
    plot_paths = sorted((cfg.results_dir / "plots").glob("*.png"))
    if plot_paths:
        st.subheader("Generated figures")
        for plot_path in plot_paths:
            st.image(str(plot_path), caption=plot_path.name, width="stretch")


def render_app() -> None:
    try:
        import pydeck as pdk
        import streamlit as st
    except ImportError as error:  # pragma: no cover - launch guard
        raise RuntimeError('Install GUI dependencies with: pip install -e ".[gui]"') from error

    from .geometry import get_reach_geometry
    from .workflow import run_workflow

    st.set_page_config(page_title="SWOT RAQW", page_icon="🌊", layout="wide")
    st.title("SWOT Reach-Adaptive Quantile Window Filter")
    st.caption("Select one SWORD reach, configure the analysis, and inspect the results locally.")

    with st.sidebar:
        st.header("Run setup")
        reach_id = st.text_input("SWORD reach ID", "66220000061").strip()
        year = int(st.number_input("Analysis year", 2000, 2100, 2024, 1))
        run_root_text = st.text_input("Output directory", "outputs")
        collection = st.selectbox(
            "RiverSP collection",
            ("SWOT_L2_HR_RiverSP_D", "SWOT_L2_HR_RiverSP_2.0"),
        )
        projected_crs = st.text_input("Projected CRS (optional)", "").strip() or None
        with st.expander("Offline geometry override"):
            reach_file_text = st.text_input("Local SWORD file (optional)", "").strip()
            reach_layer = st.text_input("Layer name (optional)", "").strip() or None
            reach_id_field = st.text_input("Reach ID field", "reach_id").strip()
        with st.expander("Earthdata credentials"):
            st.caption("Leave blank to use `_netrc`, `.netrc`, or environment credentials.")
            earthdata_username = st.text_input("Username", "").strip()
            earthdata_password = st.text_input("Password", "", type="password")
            st.caption("Typed credentials remain in this local app session and are not saved.")

    parameters = _configuration_widgets(st)
    try:
        cfg = RAQWConfig(
            reach_id=reach_id,
            year=year,
            run_root=Path(run_root_text).expanduser().resolve(),
            reach_file=(
                Path(reach_file_text).expanduser().resolve() if reach_file_text else None
            ),
            reach_layer=reach_layer,
            reach_id_field=reach_id_field,
            projected_crs=projected_crs,
            riversp_collection_name=collection,
            **parameters,
        )
    except ValueError as error:
        st.error(str(error))
        return

    config_json = json.dumps(
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        indent=2,
    )
    st.download_button(
        "Download effective configuration",
        data=config_json,
        file_name=f"raqw_{reach_id}_{year}_config.json",
        mime="application/json",
    )

    map_tab, run_tab, results_tab = st.tabs(("Reach map", "Run analysis", "Results"))
    with map_tab:
        if st.button("Load reach geometry", type="primary"):
            try:
                with st.spinner("Requesting the reach from Hydrocron..."):
                    geometry = get_reach_geometry(cfg)
                st.session_state["raqw_geometry"] = mapping(geometry)
                st.session_state["raqw_geometry_key"] = (
                    reach_id,
                    collection,
                    str(cfg.reach_file),
                )
            except Exception as error:
                st.error(f"Reach lookup failed: {error}")
        geometry_key = (reach_id, collection, str(cfg.reach_file))
        if st.session_state.get("raqw_geometry_key") == geometry_key:
            _display_map(st, pdk, st.session_state["raqw_geometry"], reach_id)
            geometry_source = cfg.reach_file or cfg.reach_geometry_cache_path
            st.success(f"Reach geometry ready. Source: {geometry_source}")
        else:
            st.info("Load the reach to confirm its location before running the analysis.")

    with run_tab:
        use_local_pixc = st.checkbox("Use existing local PIXC NetCDF files")
        raw_dir_text = ""
        if use_local_pixc:
            raw_dir_text = st.text_input("Directory containing PIXC `.nc` files", "")
        make_plots = st.checkbox("Generate diagnostic plots", value=True)
        st.warning(
            "A full reach-year run can take time. Keep this tab open while it runs; "
            "completed downloads and stages are cached."
        )
        if st.button("Run RAQW analysis", type="primary"):
            try:
                validate_earthdata_credential_fields(
                    earthdata_username or None,
                    earthdata_password or None,
                )
                nc_paths = None
                login = True
                if use_local_pixc:
                    raw_dir = Path(raw_dir_text).expanduser().resolve()
                    nc_paths = sorted(raw_dir.glob("*.nc"))
                    if not nc_paths:
                        raise FileNotFoundError(f"No .nc files were found in {raw_dir}.")
                    login = False
                status = st.status("Starting analysis", expanded=True)
                log_placeholder = st.empty()
                progress_bar = st.progress(0.0, text="Waiting for filtering stage")
                progress_detail = st.empty()
                logger = _LiveLog(log_placeholder)

                def update_status(stage: str, message: str) -> None:
                    status.write(f"**{stage}:** {message}")

                def update_filter_progress(event: dict[str, object]) -> None:
                    file_number = int(event["file_number"])
                    file_total = int(event["file_total"])
                    elapsed = float(event.get("elapsed_seconds", 0.0))
                    event_name = str(event["event"])
                    if event_name == "candidate_progress":
                        candidates_complete = int(event["candidates_complete"])
                        candidates_total = int(event["candidates_total"])
                        within_file = candidates_complete / max(candidates_total, 1)
                        fraction = ((file_number - 1) + within_file) / file_total
                        detail = (
                            f"Overpass {file_number}/{file_total}: candidate windows "
                            f"{candidates_complete:,}/{candidates_total:,}"
                        )
                    elif event_name == "overpass_complete":
                        fraction = file_number / file_total
                        detail = f"Completed overpass {file_number}/{file_total}"
                    else:
                        fraction = (file_number - 1) / file_total
                        detail = f"Starting overpass {file_number}/{file_total}"
                    computed_eta = (
                        elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
                    )
                    eta = float(event.get("eta_seconds", computed_eta or 0.0))
                    progress_bar.progress(
                        min(max(fraction, 0.0), 1.0),
                        text=(
                            f"Filtering {file_number}/{file_total} overpasses — "
                            f"elapsed {_format_duration(elapsed)} — "
                            f"ETA {_format_duration(eta)}"
                        ),
                    )
                    progress_detail.caption(detail)

                with temporary_earthdata_credentials(
                    earthdata_username or None,
                    earthdata_password or None,
                ), redirect_stdout(logger):
                    if login:
                        status.write("**authentication:** Verifying Earthdata Login")
                        authenticate_earthdata_for_gui(
                            earthdata_username or None,
                            earthdata_password or None,
                        )
                    summary, slopes, results = run_workflow(
                        cfg,
                        # Authentication was completed explicitly above so the
                        # browser process can never fall through to stdin.
                        login=False,
                        nc_paths=nc_paths,
                        make_plots=make_plots,
                        callback=update_status,
                        progress_callback=update_filter_progress,
                    )
                progress_bar.progress(1.0, text="Filtering complete")
                status.update(
                    label=f"Complete: {len(results):,} filtered observations",
                    state="complete",
                    expanded=False,
                )
                st.session_state["raqw_last_results"] = str(cfg.results_dir)
                st.success(f"Results written to {cfg.results_dir}")
            except EarthdataCredentialError as error:
                display_earthdata_credential_error(st, error)
            except Exception as error:
                st.exception(error)

    with results_tab:
        _display_results(st, cfg)


def launch() -> None:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as error:  # pragma: no cover - installation guard
        raise SystemExit('Install GUI dependencies with: pip install -e ".[gui]"') from error
    script_path = Path(__file__).with_name("gui_app.py").resolve()
    sys.argv = ["streamlit", "run", str(script_path)]
    raise SystemExit(streamlit_cli.main())
