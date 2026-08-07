import pandas as pd

from raqw.apply import apply_raqw_window


def test_exported_retained_mask_matches_saved_window() -> None:
    points = pd.DataFrame(
        {
            "s_m": [0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0],
            "height": [10.0, 11.0, 12.0, 13.0, 30.0],
        }
    )
    result = pd.Series(
        {
            "raqw_detrend_slope_m_per_km": 1.0,
            "raqw_tau_low": 0.0,
            "raqw_tau_high": 0.8,
        }
    )

    output = apply_raqw_window(points, result)

    assert output["raqw_retained"].sum() == 4
    assert not bool(output["raqw_retained"].iloc[-1])

