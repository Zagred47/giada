"""Independently check the archived Task29 execution and decision scope."""
import hashlib
import json
import math
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'experiments/results/task29_kaggle_1245b70'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = FOLDER / 'artifact_bundle.zip'
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        names = ('final_report.json', 'run_contract.json', 'code_provenance.json',
                 'process_status.json', 'native_passive_audit.json')
        documents = {name: json.loads(zipped.read(name)) for name in names}
    for name, value in documents.items():
        assert json.loads((FOLDER / name).read_text(encoding='utf-8')) == value, name
    report = documents['final_report.json']
    contract = documents['run_contract.json']
    provenance = documents['code_provenance.json']
    assert contract == json.loads((ROOT / 'experiments/task29_external_clamp_ionic_passive.json').read_text(encoding='utf-8'))
    assert provenance['code_revision'] == report['code_revision'] == '1245b70c021e7c67270144c11f26addba0b1c4e4'
    assert not provenance['dirty_runtime'] and documents['process_status.json']['returncode'] == 0
    for name, expected in provenance['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert report['valid'] and report['diagnostic_scientific_passed']
    assert report['gate_d_performance_status'] == 'NO_GO_UNCHANGED'
    assert not report['gate_d_completed'] and not report['task29_performance_promotion_authorized']
    assert not report['task30_autonomous_voltage_authorized']
    assert not report['voltage_autonomous'] and not report['calcium_autonomous']
    assert not report['training_performed'] and not report['fresh_used_for_selection']
    native = documents['native_passive_audit.json']
    assert native == report['native_passive_audit'] and native['valid']
    assert len(native['rows']) == len(contract['native_passive_test_voltages_mv'])
    assert native['maximum_error_ma_cm2'] <= contract['native_passive_current_atol_ma_cm2']
    rows = report['rows']
    expected = {(p, f, a, s, panel) for p in contract['path_names']
                for f in contract['families'] for a in contract['arms']
                for s in contract['model_seeds'] for panel in contract['panel_names']}
    observed = {(r['path'], r['family'], r['arm'], r['seed'], r['panel']) for r in rows}
    assert len(rows) == len(expected) == len(observed) and observed == expected
    for row in rows:
        assert row['finite'] and row['occupancy_violations'] == 0
        assert row['passed']
        assert all(math.isfinite(row[k]) for k in (
            'gate_rmse', 'worst_individual_current_normalized_rmse',
            'clamp_demand_normalized_rmse', 'clamp_demand_absolute_rmse_ma_cm2'))
        assert row['gate_rmse'] <= contract['gate_rmse_limit']
        assert row['worst_individual_current_normalized_rmse'] <= contract['worst_individual_current_normalized_rmse_limit']
        assert row['clamp_demand_normalized_rmse'] <= contract['clamp_demand_normalized_rmse_limit']
    summary = {}
    for family in contract['families']:
        for arm in contract['arms']:
            group = [r for r in rows if r['family'] == family and r['arm'] == arm]
            summary[family + '/' + arm] = {
                'passed': sum(r['passed'] for r in group), 'total': len(group),
                'worst_gate_rmse': max(r['gate_rmse'] for r in group),
                'worst_individual_current_normalized_rmse': max(r['worst_individual_current_normalized_rmse'] for r in group),
                'worst_clamp_demand_normalized_rmse': max(r['clamp_demand_normalized_rmse'] for r in group),
                'worst_clamp_demand_absolute_rmse_ma_cm2': max(r['clamp_demand_absolute_rmse_ma_cm2'] for r in group),
            }
    audit = {'schema_version': 'giada-task29-independent-audit-v1', 'valid': True,
             'diagnostic_scientific_passed': True, 'performance_promotion': False,
             'task30_authorized': False, 'artifact_sha256': sha(archive),
             'report_sha256': sha(FOLDER / 'final_report.json'),
             'row_count': len(rows), 'summary': summary}
    (FOLDER / 'result_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
