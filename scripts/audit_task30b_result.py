"""Independently verify active exposure and every Task30b comparison."""
import hashlib
import json
import math
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'experiments/results/task30b_kaggle_ddbffb0'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = FOLDER / 'artifact_bundle.zip'
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        names = ('final_report.json', 'run_contract.json', 'code_provenance.json',
                 'process_status.json', 'native_passive_solver_calibration.json')
        docs = {name: json.loads(zipped.read(name)) for name in names}
    for name, value in docs.items():
        assert json.loads((FOLDER / name).read_text(encoding='utf-8')) == value
    cfg = docs['run_contract.json']
    report = docs['final_report.json']
    provenance = docs['code_provenance.json']
    assert cfg == json.loads((ROOT / 'experiments/task30b_active_exposure_confirmation.json').read_text(encoding='utf-8'))
    assert report['code_revision'] == provenance['code_revision'] == 'ddbffb0f68a39c39a5e1ee3e47f08e3f8c44bcba'
    assert not provenance['dirty_runtime'] and docs['process_status.json']['returncode'] == 0
    for name, expected in provenance['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert report['valid'] and report['scientific_primary_passed']
    assert report['native_passive_solver_calibration'] == docs['native_passive_solver_calibration.json']
    assert report['native_passive_solver_calibration']['valid']
    assert report['primary_exposed_episode_count'] == 24
    assert report['primary_nonzero_injection_sample_count'] == 1440
    assert report['reference_active_episode_count_at_primary'] >= 1
    assert report['gate_d_performance_status'] == 'NO_GO_UNCHANGED'
    assert not report['task31_authorized'] and not report['full_active_neuron_native_validated']
    assert report['autonomous_voltage'] and not report['autonomous_calcium']
    assert not report['training_performed'] and not report['fresh_used_for_selection']
    expected = {(f,a,s,e,h) for f in cfg['families'] for a in cfg['arms']
                for s in cfg['model_seeds'] for e in range(32)
                for h in cfg['horizons_ms']}
    rows = report['rows']
    observed = {(r['family'],r['arm'],r['seed'],r['episode_index'],r['horizon_ms']) for r in rows}
    assert len(rows) == len(observed) == len(expected) == 1920 and observed == expected
    assert all(r['finite'] and math.isfinite(r['voltage_rmse_mv'])
               and r['occupancy_violations'] == 0 and r['physical_voltage_violations'] == 0
               for r in rows)
    summary = {}
    for family in cfg['families']:
        subset = [r for r in rows if r['family'] == family and r['arm'] == 'both'
                  and r['horizon_ms'] == cfg['primary_horizon_ms']]
        assert len(subset) == 96
        per_seed = {}
        for seed in cfg['model_seeds']:
            selected = [r for r in subset if r['seed'] == seed]
            assert len(selected) == 32
            pooled = math.sqrt(sum(r['voltage_rmse_mv']**2 for r in selected)/len(selected))
            worst = max(r['voltage_rmse_mv'] for r in selected)
            stored = report['family_primary'][family][str(seed)]
            assert abs(pooled-stored['pooled_voltage_rmse_mv']) < 1e-12
            assert abs(worst-stored['worst_episode_voltage_rmse_mv']) < 1e-12
            assert pooled <= cfg['primary_voltage_rmse_limit_mv']
            assert worst <= cfg['primary_worst_episode_voltage_rmse_limit_mv']
            assert stored['passed']
            per_seed[str(seed)] = {'pooled_rmse_mv': pooled, 'worst_episode_rmse_mv': worst}
        summary[family] = per_seed
    audit = {'schema_version': 'giada-task30b-active-exposure-audit-v1',
             'valid': True, 'decision_grade_within_formula_map_scope': True,
             'full_active_neuron_native_validated': False,
             'task31_authorized': False,
             'active_primary_episode_count': 24,
             'reference_active_threshold_crossing_episode_count': report['reference_active_episode_count_at_primary'],
             'row_count': len(rows), 'summary': summary,
             'artifact_sha256': sha(archive),
             'report_sha256': sha(FOLDER / 'final_report.json')}
    (FOLDER / 'result_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
