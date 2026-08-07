# Algorithm specification

This document describes the implementation in the `raqw` package. Parameter
names correspond directly to fields in `RAQWConfig` and
`configs/publication_2024.toml`.

## PIXC preprocessing

For each SWORD reach and year, RAQW:

1. Searches the configured PIXC collection using the reach bounding box and
   keeps granules whose footprint covers the reach geometry.
2. Retrieves the closest RiverSP reach record within
   `hydrocron_window_minutes` and requires `reach_q` to be in
   `allowed_reach_q`.
3. Corrects PIXC heights by subtracting solid-Earth tide, FES load tide, and
   pole tide when `correct_tides = true`.
4. Retains classifications greater than 2 except classification 5.
5. Reprojects the reach and PIXC points to `projected_crs`, or an estimated
   local UTM CRS when none is supplied.
6. Merges multipart reach geometry with `shapely.ops.linemerge` and requires
   the result to be one continuous `LineString`.
7. Constructs a flat-ended centerline buffer with half-width
   `width_factor * RiverSP width / 2` and retains PIXC points inside it.
8. Projects each retained point onto the centerline to obtain along-channel
   coordinate `s_m`.
9. Optionally applies `min_points_per_granule`, then requires
   `max(s_m) / reach_length >= minimum_reach_coverage`.
10. Fits linear quantile regressions at the configured quantile grid and saves
    slope in m km\(^{-1}\).

The coverage test is an endpoint test. It does not measure the largest internal
gap or total occupied fraction.

## Reach-year reference envelope

For every quantile \(\tau\), the reference uses the slopes from all valid
overpasses for the reach-year. Let \(b_j(\tau)\) be the slope for overpass
\(j\). RAQW calculates

\[
m(\tau)=\operatorname{median}_j b_j(\tau)
\]

and

\[
d(\tau)=\max\left[\operatorname{median}_j
|b_j(\tau)-m(\tau)|,\epsilon\right],
\]

where \(\epsilon\) is `mad_epsilon`. Standardized deviations are

\[
z_j(\tau)=\frac{|b_j(\tau)-m(\tau)|}{d(\tau)}.
\]

The `target_coverage` quantile of \(z_j(\tau)\) gives \(k_\tau\). After trimming
the lowest and highest `band_trim_fraction` of the \(k_\tau\) values, their mean
is capped by `universal_k_cap` to obtain \(K\). The reference envelope is

\[
m(\tau)-K d(\tau) \leq b(\tau) \leq m(\tau)+K d(\tau).
\]

At least `min_files_per_quantile` overpasses must contribute at a quantile.

## Tau-window selection

For an individual overpass, every contiguous interval with endpoints inside
`core_tau_low` to `core_tau_high` and width at least `min_core_width` is an
initial candidate. A candidate receives terms rewarding width, closeness to the
reference, curve smoothness, centrality, small retained residual spread, and
raw-to-filtered slope stability. Penalties account for reference-envelope
exceedance, tail extension, and endpoint jumps. The weights are all explicit
configuration fields.

Candidates satisfying `max_core_outside_fraction` are preferred. Starting from
the highest-scoring preferred core, the algorithm tests one-quantile expansion
to the left and right. An expansion must satisfy:

- the expanded outside fraction is no larger than
  `max_expand_outside_fraction`;
- the newly included standardized deviation is no larger than
  `max_expand_z_multiplier * K`;
- endpoint jump is no larger than `max_expand_endpoint_jump`; and
- score change is at least `min_expand_score_gain`.

When both directions pass, the higher-scoring expansion is accepted. Expansion
continues until neither direction passes. Consequently, the final interval may
extend from 0.01 to 0.99 even though the initial core endpoints are restricted.

## Retained points and final slope

The median quantile-regression slope inside the selected interval is used only
for detrending. Quantile cutoffs at \(\tau_L\) and \(\tau_H\) are calculated from
the detrended heights. Pixels inside those cutoffs retain their original
corrected heights and coordinates. The reported slope is a new ordinary least
squares fit through those retained original values, expressed in m km\(^{-1}\).

A final result requires at least `min_points_after_trim` pixels and
`min_unique_coordinates` distinct along-channel coordinates.

