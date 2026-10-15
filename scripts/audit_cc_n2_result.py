"""Independent saved-array and selection audit; no training or model selection."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/results/cc_n2_kaggle_469661b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read = lambda name: json.loads((SOURCE / name).read_text(encoding='utf-8'))
    report, freeze, spec = map(read, ('final_report.json', 'selection_freeze.json', 'preregistration.json'))
    manifest, curves = map(read, ('dataset_manifest.json', 'learning_curves.json'))
    assert report['valid'] and report['member_count'] == 27
    assert not any(report[k] for k in ('selection_used_diagnostic', 'sealed_confirmation_performed',
                                     'synapses_tested', 'new_candidate_promoted', 'inference_speed_claim'))
    assert report['code_revision'] == freeze['code_revision'] == '469661b570ba56825309f917fdf25a4a886a83f8'
    claimed = freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest() == claimed
    assert not freeze['diagnostic_opened']
    assert freeze['checkpoint_sha256'] == sha(SOURCE / 'frozen_checkpoint.pt')
    assert freeze['preregistration_sha256'] == sha(SOURCE / 'preregistration.json')
    assert sha(SOURCE / 'preregistration.json') == sha(ROOT / 'experiments/cc_n2_unified_cell_preregistration_v1.json')
    assert manifest['dataset_sha256'] == sha(SOURCE / 'native_dataset.npz')
    assert read('evaluation.json') == report['results']
    choices = [(a['name'], s) for a in spec['arms'] for s in spec['seeds']]
    for i, (arm, seed) in enumerate(choices):
        rows = [r for r in curves if r['arm'] == arm and r['seed'] == seed]
        winner = min(rows, key=lambda r: r['score'])
        assert winner['update'] == freeze['selected_updates'][i]
        assert winner['score'] == freeze['development_scores'][i]
    dataset = np.load(SOURCE / 'native_dataset.npz')
    start = round(spec['score_start_ms'] / spec['dt_ms'])
    maximum_relative = 0.
    for role in ('development', 'diagnostic'):
        pred = np.load(SOURCE / f'{role}_frozen_rollout.npz')
        ids = pred['episode_indices']
        assert len(ids) == spec['role_counts'][role]
        assert [manifest['rows'][int(i)]['role'] for i in ids] == [role] * len(ids)
        for steps in (80, 300):
            measured = {}
            for variable, prefix, unit in [('voltage', 'voltage', 'mv'), ('calcium', 'calcium', 'mM')]:
                target = dataset[variable][start+1:start+steps+1, ids]
                error = pred[variable][:steps, :, :, 0].astype(np.float64) - target[:, None, :]
                # Reproduce the published float32 arithmetic independently.
                err32 = error.astype(np.float32)
                measured[f'{prefix}_pooled_rmse_{unit}'] = np.sqrt(np.mean(err32**2, axis=(0, 2)))
                measured[f'{prefix}_worst_episode_rmse_{unit}'] = np.sqrt(np.mean(err32**2, axis=0)).max(axis=1)
            for i, (arm, seed) in enumerate(choices):
                row = next(r for r in report['results'] if r['role'] == role and
                           r['horizon_ms'] == steps / 10 and r['arm'] == arm and r['seed'] == seed)
                for key, values in measured.items():
                    value, expected = float(values[i]), row[key]
                    relative = abs(value-expected) / max(abs(expected), 1e-12)
                    maximum_relative = max(maximum_relative, relative)
                    assert relative < 2e-5, (role, arm, seed, key, value, expected)
                passed = row['finite'] and row['physical_violations'] == 0 and all(
                    row[k] <= limit for k, limit in spec['diagnostic_gate'].items())
                assert passed == row['passed']
    summary = []
    for arm, _ in choices[::3]:
        for horizon in (8., 30.):
            rows = [r for r in report['results'] if r['role'] == 'diagnostic' and
                    r['arm'] == arm and r['horizon_ms'] == horizon]
            summary.append({'arm': arm, 'horizon_ms': horizon,
                            'mean_seed_voltage_rmse_mv': float(np.mean([r['voltage_pooled_rmse_mv'] for r in rows])),
                            'worst_episode_voltage_rmse_mv': max(r['voltage_worst_episode_rmse_mv'] for r in rows),
                            'maximum_seed_calcium_worst_rmse_mM': max(r['calcium_worst_episode_rmse_mM'] for r in rows),
                            'passing_seeds': sum(r['passed'] for r in rows)})
    audit = {'valid': True, 'audit_scope': 'saved-array recomputation, selection minima, freeze and artifact hashes; no GPU checkpoint replay',
             'code_revision': report['code_revision'], 'maximum_metric_relative_difference': maximum_relative,
             'selection_reproduced': True, 'candidate_promoted': False, 'diagnostic_summary': summary,
             'limitations': ['No independent test; all opened data now historical/development.',
                            'Probe decodes initial gates from initial state; it does not certify evolved latent dynamics.',
                            'Most selections use the final budget; this alone does not establish a training-budget bottleneck.',
                            'No time comparison against Task35: domains, synapses and thresholds differ.'],
             'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(SOURCE.iterdir())
                       if p.is_file() and p.name != 'result_audit.json'}}
    (SOURCE / 'result_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'valid': True, 'evaluations_recomputed': len(report['results']),
                      'max_relative_error': maximum_relative, 'promoted': False}))


if __name__ == '__main__':
    main()
