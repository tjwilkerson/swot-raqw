# SWOT Reach-Adaptive Quantile Window Filter

RAQW is a reproducible Python workflow for estimating reach-scale river slopes
from SWOT Level 2 High-Rate Pixel Cloud (PIXC) observations. It performs PIXC
discovery and pre-screening, geospatial projection, reach-year quantile-slope
reference fitting, adaptive tau-window selection, retained-point ordinary least
squares fitting, and RiverSP comparison.

The package implements the workflow accompanying the manuscript *An Open-Source
SWOT Pixel-Cloud Workflow Improves River Slope Usability at Scale*. It is designed
to keep acquisition, preprocessing, reference fitting, and application as
separate, auditable stages.

## Workflow

```text
SWORD reach (Hydrocron or local file) + DAWG/SOS
        |
        v
qualified reach list
        |
        v
PIXC discovery -> classification and spatial screening -> s coordinate
        |
        v
0.01--0.99 quantile-slope curves -> reach-year reference envelope
        |
        v
adaptive tau selection -> detrend/trim -> retained-point OLS slope
        |
        v
RiverSP matching, comparison tables, figures, and provenance manifests
```

## Installation

For exact reproduction, create the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate swot-raqw
python -m pip install --no-deps -e .
```

For ordinary use in an existing Python 3.11 or newer environment:

```bash
python -m pip install -e .
```

Downloading SWOT data requires an Earthdata Login account and credentials
available to `earthaccess`, normally through a `.netrc` file. Existing PIXC
NetCDF files can be processed without an Earthdata login.

Hydrocron reach lookup does not require an Earthdata login or a full SWORD
download. Do not put usernames or passwords in a RAQW configuration file.

For unattended downloads, place the following in `~/.netrc` (or `_netrc` in
the Windows user profile), replacing the two placeholders:

```text
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
```

## Quick start

Copy [the online-reach configuration](configs/online_reach.toml), change the
reach identifier, year, and any analysis parameters, and run:

```bash
raqw run --config configs/my_run.toml --plots
```

With default analysis parameters, a configuration file is optional:

```bash
raqw run --reach-id 66220000061 --year 2024 --plots
```

When `reach_file` is omitted, RAQW requests that one reach from Hydrocron as
GeoJSON and caches it under
`outputs/cache/sword/<collection>/reach_<id>.geojson`. The cache records the
Hydrocron collection, reported SWORD version, query interval, and retrieval
time. Later runs reuse it. Version D is pinned in the example because it uses
SWORD v17b.

Hydrocron derives reach geometry from archived RiverSP observations. The reach
must therefore have at least one RiverSP observation in the requested or
automatic fallback interval. To pin that lookup interval, add both of these to
`[data]`:

```toml
geometry_search_start = "2024-01-01T00:00:00Z"
geometry_search_end = "2024-12-31T23:59:59Z"
```

For offline use or exact reproduction from an archived SWORD file, add:

```toml
reach_file = "../inputs/chile_reaches.gpkg"
reach_id_field = "reach_id"
```

An explicit `reach_file` always takes precedence over Hydrocron.

To process locally available PIXC files:

```bash
raqw run \
  --config configs/my_run.toml \
  --raw-dir /path/to/pixc_netcdf \
  --no-login \
  --plots
```

The stages can also be executed independently.

## Graphical interface

Install the optional GUI dependencies and launch the local browser application:

```bash
python -m pip install -e ".[gui]"
raqw-gui
```

The interface provides:

- Hydrocron reach lookup and an interactive centerline map;
- Earthdata credentials entered for the current session or loaded from the
  standard `_netrc`, `.netrc`, or environment variables;
- preprocessing, tau-window, and advanced scoring controls;
- either automatic PIXC discovery/download or a local PIXC directory;
- live per-overpass and candidate-window progress with elapsed time and ETA;
- cached restart behavior, result tables, downloadable configuration/results,
  and generated diagnostic figures.

Credentials typed into the GUI are placed in the process environment only
during the analysis call and are then removed or restored. They are never added
to the configuration, cache, provenance manifest, or output tables.

The first GUI release intentionally selects reaches by SWORD ID. Hydrocron is a
feature-ID service rather than a global spatial search service. A future
click-any-river map will require a separately hosted, lightweight SWORD vector
tile index; the analytical reach geometry will still be retrieved from
Hydrocron after selection.

### 1. Build the discharge-qualified reach list

```bash
raqw reach-list \
  --sword-file inputs/chile_reaches.gpkg \
  --sos-file inputs/sa_sword_v17_SOS_priors.nc \
  --output inputs/chile_reaches_with_valid_discharge.csv
```

If `--sos-file` is omitted, provide `--download-dir` and RAQW will retrieve the
South America DAWG/SOS product through Earthdata.

### 2. Preprocess PIXC and fit quantile slopes

```bash
raqw preprocess --config configs/my_run.toml
```

This stage applies the PIXC class filter, subtracts the configured tide
corrections, clips points to a flat-ended width-based reach buffer, projects
points onto the merged reach centerline, checks endpoint coverage, and fits 99
linear quantile regressions per accepted overpass.

### 3. Fit the reach-year reference and select tau windows

```bash
raqw fit --config configs/my_run.toml --plots
```

### 4. Apply a frozen reference to another year

Use a configuration whose `year` points to the already-preprocessed application
year, then supply the fitted reference envelope:

```bash
raqw apply-reference \
  --config configs/application_2025.toml \
  --reference outputs/results/reach_66220000061/2024/raqw_envelope.csv \
  --plots
```

### 5. Export retained-point flags

```bash
raqw export \
  --reach-id 66220000061 \
  --date 2024-04-23 \
  --run-root outputs
```

### 6. Retrieve RiverSP observations

```bash
raqw fetch-riversp \
  --filter-results outputs/results/reach_66220000061/2024/raqw_results.csv \
  --output outputs/riversp_2024.csv
```

This queries Hydrocron over the PIXC observation window for each reach and
stores both native RiverSP slopes and slopes converted to m km\(^{-1}\).

### 7. Compare with a RiverSP table

The RiverSP input must contain a reach identifier, observation time, slope, and
optionally `reach_q`. Native RiverSP slopes in m/m can be converted with
`--slope-scale 1000`.

```bash
raqw compare \
  --filter-results outputs/results/reach_66220000061/2024/raqw_results.csv \
  --riversp-table inputs/riversp_slopes.csv \
  --slope-column slope \
  --slope-scale 1000 \
  --output-dir outputs/comparison
```

### 8. Create an independent file manifest

```bash
raqw manifest inputs/file1.nc inputs/file2.csv \
  --output outputs/input_manifest.json \
  --root .
```

Preprocessing and filtering automatically write stage-specific manifests with
SHA-256 checksums, the Git revision, Python version, and installed package
versions.

## Output structure

```text
outputs/
  cache/sword/<collection>/reach_<id>.geojson
  raw/pixc_d/reach_<id>/<year>/
  processed/reach_<id>/<year>/
    summary.csv
    quantile_slopes_long.csv
    pixc_points/*_pts.csv
    preprocessing_input_manifest.json
  results/reach_<id>/<year>/
    raqw_results.csv
    raqw_envelope.csv
    run_config.json
    filter_input_manifest.json
    plots/                         # optional
```

The final slope column is `raqw_filtered_slope_m_per_km`.

The complete calculation is documented in
[docs/algorithm.md](docs/algorithm.md), and field-level output descriptions are
provided in [docs/outputs.md](docs/outputs.md).

## Publication profile and the 50-point decision

The tau-selection defaults reproduce the preserved 2024 results exactly. A
read-only parity audit of representative observations matched saved tau bounds,
scores, retained fractions, detrending slopes, and final slopes to floating-point
precision.

The current archived output includes successful records with fewer than 50
spatially screened pixels. Therefore the publication configuration leaves
`min_points_per_granule = 0`, meaning no additional point-count rejection is
applied. Setting it to `50` is supported, but doing so defines a revised analysis
that should be rerun and documented consistently in the manuscript and archived
results.

## Testing

```bash
python -m pytest
```

Tests use synthetic data and do not require Earthdata credentials or download
SWOT products. GitHub Actions runs the same suite on supported Python versions.

## Data and reproducibility

SWOT PIXC, RiverSP, and DAWG/SOS products are obtained from NASA PO.DAAC and are
not redistributed in this source repository. The cached or locally supplied
SWORD geometry, exact reach list, input-granule manifest, processed observation
table, figure-source tables, and archived software release should be deposited
with the publication. Do not commit Earthdata credentials, `.netrc`, raw PIXC
granules, or local output trees.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The source
repository is [tjwilkerson/swot-raqw](https://github.com/tjwilkerson/swot-raqw).
The software DOI and associated article DOI will be added after the archival
release is created.

## License

Released under the [MIT License](LICENSE).
