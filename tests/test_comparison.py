import numpy as np
import pandas as pd

from raqw.comparison import (
    agreement_summary,
    attach_riversp_matches,
    hydrocron_response_to_table,
    normalize_riversp_table,
)


def test_matches_nearest_allowed_riversp_observation() -> None:
    results = pd.DataFrame(
        {
            "reach_id": ["123", "123"],
            "file": [
                "SWOT_L2_HR_PIXC_001_001_001L_20240101T120000_20240101T120010_PGD0_01.nc",
                "SWOT_L2_HR_PIXC_001_001_001L_20240102T120000_20240102T120010_PGD0_01.nc",
            ],
            "raqw_filtered_slope_m_per_km": [1.1, 1.3],
            "raqw_raw_slope_m_per_km": [1.8, 1.7],
        }
    )
    riversp = pd.DataFrame(
        {
            "reach_id": ["123", "123", "123"],
            "riversp_time_utc": [
                "2024-01-01T12:01:00Z",
                "2024-01-01T12:03:00Z",
                "2024-01-02T13:00:00Z",
            ],
            "riversp_slope_m_per_km": [9.0, 1.0, 1.2],
            "reach_q": [2, 0, 0],
        }
    )
    riversp = normalize_riversp_table(riversp)

    matched = attach_riversp_matches(results, riversp)
    summary = agreement_summary(matched)

    assert matched.loc[0, "riversp_slope_m_per_km"] == 1.0
    assert matched.loc[0, "riversp_time_delta_s"] == 180.0
    assert np.isnan(matched.loc[1, "riversp_slope_m_per_km"])
    assert summary.loc[summary["method"] == "filtered_pixc", "n"].iloc[0] == 1


def test_converts_native_riversp_slope_units() -> None:
    table = pd.DataFrame(
        {
            "reach_id": ["123"],
            "time_str": ["2024-01-01T00:00:00Z"],
            "slope": [0.0015],
        }
    )
    normalized = normalize_riversp_table(
        table,
        slope_column="slope",
        slope_scale=1000.0,
    )
    assert normalized["riversp_slope_m_per_km"].iloc[0] == 1.5


def test_reads_hydrocron_json_csv_payload() -> None:
    payload = (
        '{"results":{"csv":"reach_id,time_str,slope\\n'
        '123,2024-01-01T00:00:00Z,0.0015\\n"}}'
    )
    table = hydrocron_response_to_table(payload)
    assert table.loc[0, "reach_id"] == 123
    assert table.loc[0, "slope"] == 0.0015
