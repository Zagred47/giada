"""Independently audit the native-active Task30c Kaggle artifact."""
import hashlib
import json
import math
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'experiments/results/task30c_kaggle_df82f42'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pooled(rows):
    return math.sqrt(sum(r['voltage_rmse_mv']**2 for r in rows)/len(rows))


def main():
    archive = FOLDER / 'artifact_bundle.zip'
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        names = ('final_report.json', 'run_contract.json', 'code_provenance.json', 'process_status.json')
        docs = {name: json.loads(zipped.read(name)) for name in names}
        compiled = {name[:-4]: zipped.read(f'compiled_native_eleven/{name}')
                    for name in (channel+'.mod' for channel in docs['run_contract.json']['channels'])}
    for name, value in docs.items():
        assert json.loads((FOLDER / name).read_text(encoding='utf-8')) == value, name
    cfg = docs['run_contract.json']
    report = docs['final_report.json']
    provenance = docs['code_provenance.json']
    assert cfg == json.loads((ROOT / 'experiments/task30c_native_active_confirmation.json').read_text(encoding='utf-8'))
    assert provenance['code_revision'] == report['code_revision'] == 'df82f42d8808a5f31018e443fce72da7028e5eed'
    assert not provenance['dirty_runtime'] and docs['process_status.json']['returncode'] == 0
    for name, digest in provenance['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert report['valid'] and report['native_floor_admissible'] and report['scientific_primary_passed']
    assert report['primary_exposed_nonrest_episode_count'] == 24
    assert report['primary_nonzero_injection_sample_count'] == 1440
    assert not report['task31_authorized'] and report['gate_d_performance_status'] == 'NO_GO_UNCHANGED'
    assert not report['training_performed'] and not report['fresh_used_for_selection']
    assert report['native_metadata']['calcium_imposed'] and report['native_metadata']['reversals_imposed']
    assert report['native_metadata']['neuron_version'].split('+')[0] == cfg['native_neuron_version']
    assert set(compiled) == set(cfg['channels']) == set(report['canonical_channel_sha256'])
    for channel, raw in compiled.items():
        assert hashlib.sha256(raw).hexdigest() == report['canonical_channel_sha256'][channel]['checkout_sha256']
    old = json.loads((ROOT / cfg['parent_config']).read_text(encoding='utf-8'))
    rows = report['rows']
    horizons = [cfg['primary_horizon_ms'], *cfg['diagnostic_horizons_ms']]
    expected = {(e,h,'formula_vs_native',None,None)
                for e in range(32) for h in horizons}
    expected |= {(e,h,arm,family,seed)
                 for e in range(32) for h in horizons
                 for arm in old['arms'] for family in old['families'] for seed in old['model_seeds']}
    observed = {(r['episode_index'],r['horizon_ms'],r['arm'],r.get('family'),r.get('seed')) for r in rows}
    assert len(rows) == len(observed) == len(expected) and observed == expected
    assert len(rows) == 1664
    assert all(math.isfinite(r['voltage_rmse_mv']) and r['voltage_rmse_mv'] >= 0 for r in rows)
    primary = [r for r in rows if r['horizon_ms'] == cfg['primary_horizon_ms']]
    floor_rows = [r for r in primary if r['arm'] == 'formula_vs_native']
    assert len(floor_rows) == 32
    floor = report['native_floor']
    assert abs(pooled(floor_rows)-floor['pooled_rmse_mv']) < 1e-12
    assert abs(max(r['voltage_rmse_mv'] for r in floor_rows)-floor['worst_episode_rmse_mv']) < 1e-12
    assert floor['pooled_rmse_mv'] <= cfg['native_floor_primary_pooled_limit_mv']
    assert floor['worst_episode_rmse_mv'] <= cfg['native_floor_primary_worst_episode_limit_mv']
    assert floor['passed']
    assert sum(r['native_crossings'] > 0 for r in floor_rows) == 6
    models = {}
    for family in old['families']:
        models[family] = {}
        for seed in old['model_seeds']:
            selected = [r for r in primary if r['arm'] == 'both' and r.get('family') == family and r.get('seed') == seed]
            assert len(selected) == 32 and all(r['finite'] for r in selected)
            result = report['models'][family][str(seed)]
            value = pooled(selected)
            worst = max(r['voltage_rmse_mv'] for r in selected)
            assert abs(value-result['pooled_rmse_mv']) < 1e-12
            assert abs(worst-result['worst_episode_rmse_mv']) < 1e-12
            assert result['passed_if_floor_valid']
            assert value <= cfg['candidate_primary_pooled_limit_mv']
            assert worst <= cfg['candidate_primary_worst_episode_limit_mv']
            models[family][str(seed)] = {'pooled_rmse_mv': value, 'worst_episode_rmse_mv': worst}
    audit = {'schema_version': 'giada-task30c-native-active-audit-v1',
             'valid': True, 'native_floor_admissible': True,
             'scientific_primary_passed': True, 'task31_authorized': False,
             'full_multicompartment_validated': False, 'native_active_single_compartment_validated': True,
             'episode_count': 32, 'active_primary_episode_count': 24,
             'native_primary_crossing_episode_count': 6, 'row_count': len(rows),
             'native_floor': floor, 'models': models,
             'artifact_sha256': sha(archive), 'report_sha256': sha(FOLDER / 'final_report.json')}
    (FOLDER / 'result_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
