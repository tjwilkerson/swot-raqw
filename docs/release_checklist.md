# Publication release checklist

Target release: **v1.0.0**, preserving the full 2024 analysis.

Confirmed repository: `https://github.com/tjwilkerson/swot-raqw`.
Confirmed sole author: Trevor Wilkerson. License: MIT.
Use [the reproduction guide](reproduce_2024.md) to review analysis and
figure coverage. Do not tag the current branch tip merely because it is latest.

## Scientific freeze

- [x] Preserve the executed 2024 behavior (`min_points_per_granule = 0`).
- [ ] If any scientific parameter changes, document the complete change list,
      regenerate all affected result tables and figures once, and rerun the
      archived-result parity audit against the new frozen outputs.
- [ ] Confirm that `configs/publication_2024.toml` and Supporting Information
      Table S1 describe the same frozen run.

## Repository metadata

- [ ] Confirm the software title and repository name.
- [x] Retain MIT; identify Trevor Wilkerson as the sole software author.
- [x] Use `tjwilkerson/swot-raqw`; add manuscript analysis and figure sources.
- [ ] Add the chosen public repository URL to `pyproject.toml` and `CITATION.cff`.
- [ ] Set `pyproject.toml`, `raqw/__init__.py`, and `CITATION.cff` to `1.0.0`
      only after the scientific freeze is ready; set the actual release date.
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
- [ ] Review and commit the complete software/analysis/figure source inventory.
      Record the full validated commit SHA and the frozen-data verification report.
- [ ] Sign into Zenodo, connect the intended GitHub account, and enable the
      chosen repository **before publishing the GitHub release**. Organization
      repositories may require organization authorization.
- [ ] Validate `CITATION.cff` (for example with `cffconvert --validate`) and confirm
      the software title, authors, license, version, and date. Use the software
      name, `SWOT Reach-Adaptive Quantile Window Filter`, as the record title.
- [ ] Create an annotated `v1.0.0` tag at the explicit validated SHA. Inspect
      the tag contents and push that tag. Never force-move a published tag.
- [ ] Publish the GitHub release from the existing tag, using
      `docs/release_notes_v1.0.0.md` after completing its evidence fields.
- [ ] Wait for Zenodo processing; inspect the archived files, metadata, version,
      and DOI landing page. Check the archival status shown on the record.
- [ ] Record the **version-specific DOI** in the manuscript and formal software
      reference. Use it to identify the frozen submission, rather than only a
      DOI that resolves to the latest version.
- [ ] Add the resulting DOI to README/CITATION on the development branch as a
      follow-up metadata commit. Do not move `v1.0.0` or create a second release
      solely to put its own DOI inside the already archived source snapshot.

## Explicit-commit release commands

After all checks above pass, replace the SHA below with the recorded full hash.
These commands are instructions, not evidence that a release has been created.

```powershell
$manuscriptCommit = '<full validated commit SHA>'
git show --stat $manuscriptCommit
git tag -a v1.0.0 $manuscriptCommit -m 'Frozen GRL manuscript software v1.0.0'
git rev-parse 'v1.0.0^{commit}'
git push origin v1.0.0
gh release create v1.0.0 --verify-tag --title 'SWOT Reach-Adaptive Quantile Window Filter v1.0.0' --notes-file docs/release_notes_v1.0.0.md
```

If the tag already exists, inspect it and stop on any SHA mismatch. Authenticate
the GitHub CLI with `gh auth login -h github.com` if needed. Never paste tokens
into repository files or release notes.

## Manuscript availability statement template

Complete the bracketed fields only after verifying the public records:

> Version 1.0.0 of SWOT Reach-Adaptive Quantile Window Filter, including the
> analysis and figure-generation code used in this study, is preserved on Zenodo
> at [version DOI] ([author-year software citation]) under the MIT License and
> developed at [chosen GitHub URL]. Processed observations and figure-source
> tables are preserved at [data DOI] ([data citation]). Input SWOT and SWORD
> products, their versions, and access conditions are identified in [manifest
> location and input-data citations].

Add formal software and data references as well as the availability statement.
Confirm the license and actual archive contents before using this wording.

## Authoritative guidance

- [AGU data and software requirements](https://www.agu.org/publications/authors/journals/data-software-for-authors)
- [Zenodo: archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/)
