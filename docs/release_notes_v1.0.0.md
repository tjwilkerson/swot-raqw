# SWOT Reach-Adaptive Quantile Window Filter v1.0.0

**Draft: complete the evidence fields before publishing.**

Frozen software for the submitted GRL manuscript, including PIXC acquisition
and filtering, retained-point OLS slope estimation, RiverSP matching and
comparison, annual reach-median RMSE analysis, and manuscript figure sources.

## Frozen version

- Manuscript title and submission/version identifier: **pending confirmation**
- Full source commit SHA: **pending scientific freeze**
- Configuration and input/processed-data manifest: **pending archive identifiers**
- Reproduction report and environment: **pending verification**
- Software authors and MIT license: Trevor Wilkerson; MIT

## Scientific scope

The local manuscript evaluates 2024 observations across 347 reaches in Chile
and adjacent Peru/Bolivia. Figure 3 measures annual temporal scatter around each
product's reach median; this is distinct from pairwise agreement with RiverSP
and does not establish absolute accuracy against independent ground truth.

The release must retain the actual preprocessing and tau-selection settings
used for the submitted results. See `docs/reproduce_2024.md` for source
coverage, the 50-point screening decision, and unresolved observation counts.

## Reproduction and citation

Installation and command-line usage are in `README.md`; the regional pipeline
run order is in `pipeline_2024/RUN_ORDER.py`. Manuscript figure notebooks are in
`publication_figs/`. Input observations and processed tables are obtained from
the separately cited data sources/archive, not bundled raw into this source
release. Cite the version-specific Zenodo software DOI once verified.
