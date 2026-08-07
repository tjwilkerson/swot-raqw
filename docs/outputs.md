# Output schemas

## Preprocessing summary

`processed/reach_<id>/<year>/summary.csv` contains one row per candidate PIXC
granule. Successful rows record the observation time, point count, reach length,
endpoint coverage, RiverSP width and quality, buffer distance, and point-table
path. Rejected rows contain an `error` value describing the first failed gate.

## Quantile slopes

`quantile_slopes_long.csv` is a long table with:

- `file`: PIXC granule filename;
- `t0_utc`: PIXC start time;
- `reach_id`: SWORD reach identifier;
- `quantile`: fitted quantile;
- `slope_m_per_km`: linear quantile-regression slope.

## Reference envelope

`raqw_envelope.csv` contains the quantile grid, median and MAD slope, quantile
multiplier, lower and upper reference bounds, universal multiplier, reach ID,
and reference year. This file is the frozen reference accepted by
`raqw apply-reference`.

## Final results

`raqw_results.csv` contains one row per processed overpass. Important fields
include:

- `raqw_tau_low`, `raqw_tau_high`, and `raqw_window_width`;
- `raqw_detrend_slope_m_per_km`;
- `raqw_n_raw`, `raqw_n_kept`, and `raqw_keep_fraction`;
- `raqw_raw_slope_m_per_km`;
- `raqw_filtered_slope_m_per_km`, the reported retained-point OLS slope;
- `raqw_trimmed_residual_iqr_m`;
- `reference_year` and `k_universal`.

## Provenance

Stage manifests contain relative paths, byte sizes, and SHA-256 checksums for
input files plus the Git revision, operating system, Python version, and key
library versions. `run_config.json` records every numerical and data-selection
parameter used by the filter.

