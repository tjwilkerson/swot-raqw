# Publication release checklist

## Scientific freeze

- [ ] Decide whether the publication analysis retains the archived behavior
      (`min_points_per_granule = 0`) or is regenerated with the proposed
      50-point gate.
- [ ] If any scientific parameter changes, document the complete change list,
      regenerate all affected result tables and figures once, and rerun the
      archived-result parity audit against the new frozen outputs.
- [ ] Confirm that `configs/publication_2024.toml` and Supporting Information
      Table S1 describe the same frozen run.

## Repository metadata

- [ ] Confirm the software title and repository name.
- [ ] Confirm the MIT license or replace `LICENSE` and the license declarations
      in `pyproject.toml` and `CITATION.cff`.
- [x] Add the public repository URL to `pyproject.toml` after the repository is
      created.
- [ ] Add repository and article identifiers to `CITATION.cff` when available.
- [ ] Replace submission placeholders in the manuscript Open Research section
      and Supporting Information Text S6.

## Validation

- [ ] Create the pinned environment from `environment.yml` on a clean machine.
- [ ] Run `python -m pip install --no-deps -e .`.
- [ ] Run `python -m pytest` and confirm all tests pass.
- [ ] Run one documented example from local PIXC data or Earthdata acquisition.
- [ ] Confirm `raqw apply-reference` reproduces the expected frozen-reference
      behavior for a held-out year.
- [ ] Confirm RiverSP unit conversion and the 30-minute matching tolerance.

## Data archive

- [ ] Archive the SWORD reach list and geometry version identifier.
- [ ] Archive PIXC and RiverSP input manifests with product identifiers, times,
      and checksums or stable source identifiers.
- [ ] Archive the complete processed observation table and figure-source tables.
- [ ] Archive the frozen TOML configuration and software-environment manifest.
- [ ] Record PO.DAAC, SWORD, software, and processed-data citations.

## GitHub and DOI release

- [ ] Review `git status` so unrelated notebooks, local outputs, credentials,
      raw NetCDF files, and manuscript working files are not staged accidentally.
- [ ] Create a signed or annotated `v0.1.0` tag from the validated commit.
- [ ] Create the GitHub release and connect the repository to an archival service
      such as Zenodo.
- [ ] Add the resulting software DOI to `CITATION.cff`, the README, and the
      manuscript, then tag the final archival release if metadata changed.
