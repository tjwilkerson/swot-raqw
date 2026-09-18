# Reproduce the 2024 manuscript analysis

Use this workflow to check the published analysis. Use the README quick start
or GUI to analyze a new reach. The manuscript study covers Chile and adjacent
Peru/Bolivia; it is not a global validation of the method.

## 1. Recreate the main comparison figures

Install the package as described in the README. Obtain the preserved
`matched_pixc_riversp_slopes.csv` table, then run from this repository root:

```bash
python scripts/reproduce_2024_figures.py --matched-table /path/to/matched_pixc_riversp_slopes.csv --output-dir outputs/manuscript_2024
```

No download, account, notebook server, or parameter editing is needed. This
executes the official calculation and plotting cells from the preserved Figure
3 and Figure 4 notebooks, substitutes explicit input/output paths, and skips
exploratory cells. It creates PNG/PDF figures, `annual_reach_rmse.csv`, a text
summary, and a manifest with input and notebook SHA-256 hashes.

**Data availability:** the table exists in the author's working analysis under
`data/riversp_comparison_updated_filter/`. Its public data archive DOI is still
pending. Cloning the software alone does not supply the processed inputs.

The verified local table yields:

| Quantity | Result |
| --- | --- |
| Common valid observations | 7,149 |
| Reaches | 347 |
| Population-median annual RMSE: RiverSP slope | 0.114 m/km |
| Population-median annual RMSE: RiverSP slope2 | 0.098 m/km |
| Population-median annual RMSE: filtered PIXC | 0.086 m/km |
| Lower filtered RMSE than RiverSP slope | 55.3% of reaches |
| Lower filtered RMSE than RiverSP slope2 | 45.0% of reaches |
| Negative slope to positive filtered slope | 642/705 (91.1%) |
| Negative slope2 to positive filtered slope | 638/707 (90.2%) |

RMSE here is temporal scatter about **each product's annual reach median** on
the common observation set. `raqw compare` instead measures pairwise agreement
against RiverSP. These statistics are not interchangeable or independent
ground-truth accuracy measurements.

## 2. Verify filtering against the preserved points

The processed-data tree must contain:

```text
DATA_ROOT/
  chile_reaches_with_valid_discharge_run/processed/reach_<id>/2024/
    quantile_slopes_long.csv
    pixc_points/*_pts.csv
  tau_range_update_experiment/run_config.json
  riversp_comparison_updated_filter/all_reach_updated_tau_window_results.csv
```

```bash
python scripts/audit_2024.py --data-root /path/to/DATA_ROOT --output outputs/audit_2024.json --selection-samples 5
```

The audit compares the saved configuration to `configs/publication_2024.toml`,
refits all 7,214 saved windows using preserved points, and completely reselects
windows for five deterministic, evenly spaced observations. It records input
hashes, source hashes, software versions, discrepancies, and scope. Use
`--selection-samples 7214` to reselect every observation, allowing substantially
more runtime. Source inputs are not modified.

Of the 7,214 candidate observations, 7,149 yield slopes and 65 fail fitting.
Legacy failed fits encode retained fraction as zero; RAQW encodes it as missing.
The audit reports this representation difference separately from numerical
discrepancies. It does not treat a missing slope as a successful observation.

The completed local audit passed all 7,214 saved-window refits with maximum
absolute numerical difference `4.55e-13`; all five sampled full selections
matched. See [the audit report](validation/audit_2024.json) and
[figure reproduction hashes/results](validation/figure_reproduction.json).
The report describes the tested source snapshot through checksums and its base
commit. It is not a claim that a later untested source revision is equivalent.

These checks start from preserved processed points, not raw satellite data.
Full acquisition/preprocessing reproduction and a clean-environment run remain
release checks. The package's 31 existing tests pass in the local Anaconda
environment; synthetic tests alone do not establish manuscript reproduction.

## 3. Full source and figure inventory

The original regional notebooks, `tau_range_update_experiment.py`, and
`pipeline_2024/` are included for provenance. Some are experimental or later
refactors. Their presence does not prove that every current default was used
in the historical run. The saved run configuration is the reference for 2024
parameters; use `configs/publication_2024.toml` for RAQW.

`docs/source_inventory_2024.json` records hashes of source copies from the
research workspace. Notebook output and execution counters were cleared;
source cells were preserved. Figure notebooks 1–4 are in `publication_figs/`;
the supporting-figure script is under `manuscript/supporting_information/`.
Figures 1 and 2 need additional geometry/point inputs and path configuration.
The original regional pipeline also has local input assumptions; its run order
is documented in `pipeline_2024/RUN_ORDER.py`.

Do not replace preserved input geometry with a fresh Hydrocron lookup when
checking historical equivalence. Do not introduce a 50-point gate: the original
executed run retained 320 successful records below that threshold. The saved
2024 reference includes the evaluated observations and is not leave-one-out.

## Known limitations retained in v1.0.0

This release preserves the existing 2024 analysis without changing scientific
parameters, slope signs, or figure calculations. The author elected to release
this snapshot with the following limitations documented:

- The authoritative manuscript is *Improving River Slope Usability at Scale:
  An Open-Source SWOT Pixel-Cloud Workflow*. Its Supporting Information explicitly
  distinguishes 7,214 candidates, 7,149 successful slopes, and 65 failures, and
  the main Results section uses 7,149. The abstract and introduction still call
  7,214 observations valid estimates; that wording needs editorial correction.
  No manuscript text was changed by the software release.
- The Figure 4 exploratory cell raises a coordinate/sign-convention concern
  and flips RiverSP signs for display. Reproducing the transition counts does
  not resolve their physical interpretation. The concern remains unresolved;
  v1.0.0 must not be cited as an independent validation of physical recovery.
- Full raw-PIXC acquisition/preprocessing equivalence, full-population window
  reselection, and a clean installation of the pinned Conda environment were
  not established by the reported audit. The audit's scope is stated above.
- The separate processed-data archive remains pending. The source archive
  includes the saved run configuration, candidate reach list, input hashes,
  analysis/figure code, and validation reports, but not the full point tables.

The source tag identifies the immutable software snapshot. A software DOI is
added to the development documentation after Zenodo archives that tag. See
[the release checklist](release_checklist.md) for the archival procedure.

Author: Trevor Wilkerson. Software license: MIT. The software citation has one
author; AI assistance is not listed as a software coauthor.
