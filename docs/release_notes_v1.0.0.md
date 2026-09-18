# SWOT Reach-Adaptive Quantile Window Filter v1.0.0

Frozen software and preserved analysis sources for the 2024 analysis in
*Improving River Slope Usability at Scale: An Open-Source SWOT Pixel-Cloud
Workflow*. Sole software author: Trevor Wilkerson. License: MIT.

## Contents

- RAQW acquisition, preprocessing, quantile-window filtering, retained-point
  ordinary least squares slopes, RiverSP comparison, CLI, and GUI.
- Original regional analysis sources, four manuscript figure notebooks, and
  the supporting-figure script. Experimental/legacy sources are identified in
  the reproduction guide; the saved run configuration defines the 2024 settings.
- Saved 2024 configuration and candidate reach list, source/input SHA-256
  manifests, validation reports, installation and environment specifications.
- Offline commands for Figures 3 and 4 and verification of saved-point refits.

## Validation

The study covers 347 reaches in Chile and adjacent Peru/Bolivia. All 7,214
candidate saved-window refits were checked: 7,149 successful slopes agree to
floating-point precision (maximum audited numerical difference 4.55e-13), and
65 failed fits remain missing. Legacy retained-fraction failure sentinels are
zero whereas RAQW uses missing values; this difference is recorded separately.
Five deterministic full window reselections matched. Figures 3 and 4 and their
reported summary statistics were regenerated. The 31 package tests passed
locally and the release-preparation GitHub checks passed on Python 3.11/3.12.

Figure 3 RMSE measures annual temporal scatter about each product's own reach
median, not independent ground-truth accuracy. Full acquisition/preprocessing
from raw PIXC, full-population window reselection, and a fresh pinned Conda
installation are outside the completed validation scope.

## Known limitations

The existing scientific analysis and figures are preserved. Figure 4's
exploratory code raises an unresolved coordinate/sign-convention concern;
matching its sign-transition counts does not establish their physical
interpretation. This limitation is retained explicitly in v1.0.0.

The authoritative manuscript's Supporting Information and Results distinguish
7,214 candidates from 7,149 successful slopes; abstract/introduction wording
still needs editorial alignment. Manuscript text is not modified by this release.
The separate processed-data archive is pending; raw PIXC and full processed-point
tables are not bundled in this software source archive.

## Reproduction and citation

Start with `README.md` and `docs/reproduce_2024.md`. Configuration:
`configs/publication_2024.toml`; historical settings:
`configs/historical_2024_run.json`. Reports and manuscript-snapshot hashes are
under `docs/validation/`. The annotated `v1.0.0` tag identifies the frozen commit;
the GitHub release description records its full SHA. Cite the version-specific
Zenodo DOI once archival publication is confirmed. Do not move the release tag
when adding the DOI to development documentation.
