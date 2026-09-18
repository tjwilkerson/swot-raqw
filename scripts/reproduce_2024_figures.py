"""Reproduce manuscript Figures 3 and 4 from the frozen matched table, offline."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched-table', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/manuscript_2024'))
    args = parser.parse_args()
    table = args.matched_table.resolve()
    if not table.is_file():
        parser.error(f'Matched table not found: {table}')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    report = {'matched_table_sha256': hashlib.sha256(table.read_bytes()).hexdigest(), 'notebooks': {}}
    log = io.StringIO()
    # Execute the original imports, calculation, and official figure cells.
    # Replace only the machine-specific path configuration. Exploratory cells
    # (including Hydrocron requests) are intentionally not part of this runner.
    specifications = [('figure_03_slope_repeatability_comparison', (1, 3, 4)),
                      ('figure_04_negative_slope_recovery', (2, 4, 5))]
    with contextlib.redirect_stdout(log):
        for stem, indices in specifications:
            path = repo / 'publication_figs' / f'{stem}.ipynb'
            document = json.loads(path.read_text(encoding='utf-8'))
            namespace = {'__name__': '__main__', 'MATCHED_CSV': table, 'FIGURE_DIR': output,
                         'OUTPUT_STEM': stem, 'EXPORT_FILES': True}
            for index in indices:
                cell = document['cells'][index]
                if cell['cell_type'] != 'code':
                    raise ValueError(f'Unexpected notebook structure at {path}:{index}')
                exec(compile(''.join(cell['source']), f'{path}:cell{index}', 'exec'), namespace)
            report['notebooks'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                            'executed_cell_indices': indices}
            if 'reach_metrics' in namespace:
                namespace['reach_metrics'].to_csv(output / 'annual_reach_rmse.csv', index=False)
                report['common_observations'] = len(namespace['paired'])
                report['reaches'] = namespace['n_reaches']
                report['median_rmse_m_per_km'] = namespace['median_rmse']
            if 'transition_results' in namespace:
                report['sign_transition_counts'] = {label: counts.tolist() for label, (counts, _) in namespace['transition_results'].items()}
            plt.close('all')
    (output / 'reproduction_summary.txt').write_text(log.getvalue(), encoding='utf-8')
    (output / 'reproduction_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(log.getvalue())


if __name__ == '__main__':
    main()
