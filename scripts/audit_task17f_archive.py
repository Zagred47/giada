"""Independently validate and retain the completed Task17f Kaggle archive."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np


def audit(archive, output, repo):
    with zipfile.ZipFile(archive) as zipped:
        bad = zipped.testzip()
        if bad is not None:
            raise ValueError(f'ZIP CRC failure: {bad}')
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        def read(name):
            return json.loads(zipped.read(name))
        report = read('final_report.json')
        pairs = read('paired_metrics.json')
        effects = read('paired_effects.json')
        freeze = read('protocol_freeze.json')
        provenance = read('code_provenance.json')
        status = read('process_status.json')
        pilot = read('pilot_progress.json')
        repeat = read('native_repeat.json')
        claimed = freeze['sha256']
        actual = hashlib.sha256(json.dumps({k: v for k, v in freeze.items() if k != 'sha256'},
                                         sort_keys=True, allow_nan=False).encode()).hexdigest()
        assert claimed == actual == report['protocol_freeze_sha256']
        assert not freeze['confirmation_accessed'] and not freeze['candidate_outcomes_accessed']
        assert set(freeze['pilot_seeds']).isdisjoint(freeze['confirmation_seeds'])
        assert report['code_revision'] == provenance['code_revision'] == '1b87bdd083f550f655c043d0fcbc731de972c9cd'
        assert not provenance['working_tree_dirty']
        for name, digest in provenance['source_sha256'].items():
            source = subprocess.check_output(['git', '-C', str(repo), 'show', report['code_revision'] + ':' + name])
            assert hashlib.sha256(source).hexdigest() == digest, name
        assert report['valid'] and report['decision'] == 'GATE_C_PASS_BOUNDED_CAHVA'
        assert report['gate_c_authorized'] and report['task18_authorized'] and all(report['gates'].values())
        assert report['confirmation_accessed'] and not report['development_smoke'] and not report['pilot_only']
        assert report['completed_trial_count'] == 144 and len(pairs) == report['comparison_count'] == 108
        assert len(pilot) == report['pilot_trial_count'] == 27 and repeat['passed']
        assert status['returncode'] == 0 and status['launch_error'] is None
        arms = ('native_super', 'formula', 'm_pair_2049_f64')
        expected = {(p, s, g) for p in range(4) for s in freeze['confirmation_seeds'] for g in (0.5, 1.0, 1.5)}
        for arm in arms:
            rows = [r for r in pairs if r['arm'] == arm]
            assert len(rows) == 36 and {(r['protocol_index'], r['seed'], r['gbar_multiplier']) for r in rows} == expected
            assert all(r['passed'] and all(r['gates'].values()) for r in rows)
        summary = {}
        for arm in arms:
            rows = [r for r in pairs if r['arm'] == arm]
            values = [v for r in rows for v in r['metrics'].values()]
            summary[arm] = {
                'worst_voltage_rmse_mv': max([v['voltage_rmse_mv'] for v in values] +
                    [v for r in rows for v in r['event_probe_voltage_rmse_mv'].values()]),
                'worst_gate_abs_error': max(max(v['m_max_error'], v['h_max_error']) for v in values),
                'worst_current_rmse_ma_cm2': max(v['current_rmse_ma_cm2'] for v in values),
                'worst_release_quantity_difference': max(r['release']['max_released_quantity_difference'] for r in rows),
            }
        identified = [v for row in effects for v in row['candidate'].values() if v['effect_identifiable']]
        assert len(identified) == report['identifiable_effect_count'] and identified
        assert max(v['relative_error'] for v in identified) <= 0.2
        dense = [n for n in names if n.endswith('.npz')]
        assert len(dense) == 171
        arrays = 0
        for name in dense:
            with np.load(io.BytesIO(zipped.read(name)), allow_pickle=False) as data:
                for key in data.files:
                    value = data[key]
                    assert np.isfinite(value).all(), (name, key, 'nonfinite')
                    if key.startswith('site_') and key.endswith(('_m', '_h')):
                        assert ((value >= 0) & (value <= 1)).all(), (name, key, 'occupancy')
                    arrays += 1
        support = report['support']
        assert support['valid'] and support['quiet_control_silent']
        assert all(set(rows) == set(map(str, freeze['confirmation_seeds'])) and all(v > 0 for v in rows.values())
                   for rows in support['uncensored_events_at_gbar1_by_seed'].values())
        result = {'valid': True, 'archive_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                  'archive_size_bytes': Path(archive).stat().st_size,
                  'code_provenance_verified': True, 'protocol_freeze_verified': True,
                  'dense_trial_count': len(dense), 'finite_dense_array_count': arrays,
                  'all_dense_gate_occupancies_valid': True, 'summary_by_arm': summary,
                  'support': support, 'identifiable_effect_count': len(identified),
                  'worst_paired_effect_relative_error': max(v['relative_error'] for v in identified),
                  'process_status': status, 'decision': report['decision'], 'task18_authorized': True}
        output.mkdir(parents=True, exist_ok=True)
        for name in ('final_report.json', 'paired_metrics.json', 'paired_effects.json', 'protocol_freeze.json',
                     'code_provenance.json', 'process_status.json', 'pilot_progress.json', 'native_repeat.json',
                     'confirmation_support.json', 'runtime_seconds.json'):
            target = output / name
            if target.exists():
                raise FileExistsError(target)
            target.write_bytes(zipped.read(name))
        (output / 'archive_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('.'))
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.output, args.repo), indent=2))
