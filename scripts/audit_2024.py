"""Check saved 2024 results against RAQW without modifying scientific inputs.

Run from the repository root. This validates filtering of preserved processed
points; it does not validate acquisition or preprocessing from raw PIXC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from raqw.config import RAQWConfig
from raqw.filter import build_envelope, prepare_file_curves, select_window, trim_and_refit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--selection-samples', type=int, default=0,
                        help='Evenly spaced observations for full window reselection; 0 skips it.')
    args = parser.parse_args()
    if args.selection_samples < 0:
        parser.error('--selection-samples must be nonnegative')
    root = args.data_root.resolve()
    result_path = root / 'riversp_comparison_updated_filter/all_reach_updated_tau_window_results.csv'
    config_path = root / 'tau_range_update_experiment/run_config.json'
    if args.output.resolve() in {result_path.resolve(), config_path.resolve()}:
        parser.error('The audit output must not overwrite a scientific input.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=args.output.parent):
        pass
    saved = pd.read_csv(result_path, dtype={'reach_id': str}).sort_values(['reach_id', 'file']).reset_index(drop=True)
    if saved.duplicated(['reach_id', 'year', 'file']).any():
        raise ValueError('Duplicate observation keys in saved results')
    if saved.empty or not saved.year.eq(2024).all():
        raise ValueError('Expected nonempty 2024 results')
    cfg = RAQWConfig.from_toml(Path(__file__).resolve().parents[1] / 'configs/publication_2024.toml')
    aliases = {'trim_frac': 'band_trim_fraction', 'max_core_outside_frac': 'max_core_outside_fraction',
               'max_expand_outside_frac': 'max_expand_outside_fraction', 'max_expand_z': 'max_expand_z_multiplier',
               'min_expand_gain': 'min_expand_score_gain'}
    stored_config = json.loads(config_path.read_text())
    errors = []
    representation_differences = []
    for key, value in stored_config.items():
        if key in {'project_root', 'processed_root', 'output_root', 'reaches'}:
            continue
        target = aliases.get(key, key)
        if not hasattr(cfg, target) or getattr(cfg, target) != value:
            errors.append({'configuration': key, 'saved': value, 'current': getattr(cfg, target, None)})
    manifest = {}

    def record(path):
        relative = path.relative_to(root).as_posix()
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    record(result_path)
    record(config_path)
    selected = set(np.linspace(0, len(saved) - 1, min(args.selection_samples, len(saved)), dtype=int)) if len(saved) else set()
    checked = selected_checked = 0
    maximum_error = 0.0
    for reach, observations in saved.groupby('reach_id', sort=True):
        folder = root / 'chile_reaches_with_valid_discharge_run/processed' / f'reach_{reach}' / '2024'
        curves = None
        if selected.intersection(observations.index):
            slope_path = folder / 'quantile_slopes_long.csv'
            record(slope_path)
            slopes = pd.read_csv(slope_path)
            envelope, _ = build_envelope(slopes, cfg)
            curves = prepare_file_curves(slopes, envelope, reach)
        for index, row in observations.iterrows():
            path = folder / 'pixc_points' / f'{Path(row["file"]).stem}_pts.csv'
            if not path.exists():
                errors.append({'reach_id': reach, 'file': row['file'], 'error': 'missing points'})
                continue
            record(path)
            points = pd.read_csv(path)
            result = trim_and_refit(points, row.updated_tau_low, row.updated_tau_high,
                                    row.updated_detrend_slope_m_per_km, cfg.min_points_after_trim,
                                    cfg.min_unique_coordinates)
            for current, old in [('filtered_slope_m_per_km', 'updated_postfilter_slope_m_per_km'),
                                 ('keep_fraction', 'updated_keep_frac'),
                                 ('trimmed_residual_iqr_m', 'updated_trim_resid_iqr_m')]:
                value = result[current] if result else np.nan
                expected = row[old]
                if result is None and current == 'keep_fraction' and expected == 0:
                    representation_differences.append({'reach_id': reach, 'file': row['file'],
                        'field': current, 'saved': 0.0, 'current': None,
                        'reason': 'Legacy failure sentinel is zero; RAQW uses missing on failed fits.'})
                    continue
                if np.isfinite(value) and np.isfinite(expected):
                    maximum_error = max(maximum_error, float(abs(value - expected)))
                if not np.isclose(value, expected, rtol=1e-9, atol=1e-9, equal_nan=True):
                    errors.append({'reach_id': reach, 'file': row['file'], 'field': current,
                                   'saved': float(expected), 'current': float(value)})
            checked += 1
            if index in selected:
                curve = curves.loc[curves.file == row['file']].sort_values('quantile').reset_index(drop=True)
                chosen = select_window(curve, points, cfg)
                for current, old in [('tau_low', 'updated_tau_low'), ('tau_high', 'updated_tau_high'),
                                     ('score', 'updated_score'), ('detrend_slope_m_per_km', 'updated_detrend_slope_m_per_km'),
                                     ('filtered_slope_m_per_km', 'updated_postfilter_slope_m_per_km')]:
                    if not np.isclose(chosen[current], row[old], rtol=1e-9, atol=1e-9, equal_nan=True):
                        errors.append({'reach_id': reach, 'file': row['file'], 'selection_field': current,
                                       'saved': float(row[old]), 'current': float(chosen[current])})
                selected_checked += 1
        print(f'{reach}: {checked}/{len(saved)} refits; {selected_checked} selections; {len(errors)} discrepancies', flush=True)
    repo = Path(__file__).resolve().parents[1]
    source_hashes = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [repo / 'scripts/audit_2024.py', repo / 'raqw/filter.py',
                               repo / 'raqw/config.py', repo / 'configs/publication_2024.toml']}
    report = {'scope': 'Saved-window refits of processed points; optional sampled full window selection. Not raw-PIXC reproduction.',
              'package_base_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': source_hashes, 'python': platform.python_version(),
              'packages': {name: version(name) for name in ('numpy', 'pandas', 'scipy')},
              'observations': len(saved), 'reaches': int(saved.reach_id.nunique()),
              'saved_successful_slopes': int(saved.updated_postfilter_slope_m_per_km.notna().sum()),
              'checked_refits': checked, 'checked_selections': selected_checked,
              'max_absolute_refit_error': maximum_error, 'passed': not errors and checked == len(saved),
              'errors': errors, 'representation_differences': representation_differences,
              'input_sha256': manifest}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
