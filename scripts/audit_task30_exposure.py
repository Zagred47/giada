"""Audit Task30 artifact and detect absent stimulus at the primary horizon."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FOLDER = ROOT / 'experiments/results/task30_kaggle_fb08789'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = FOLDER / 'artifact_bundle.zip'
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        names = ('final_report.json', 'run_contract.json', 'code_provenance.json',
                 'process_status.json', 'native_passive_solver_calibration.json')
        documents = {name: json.loads(zipped.read(name)) for name in names}
    for name, value in documents.items():
        assert json.loads((FOLDER / name).read_text(encoding='utf-8')) == value
    report = documents['final_report.json']
    cfg = documents['run_contract.json']
    provenance = documents['code_provenance.json']
    assert cfg == json.loads((ROOT / 'experiments/task30_autonomous_voltage_microcanary.json').read_text(encoding='utf-8'))
    assert report['valid'] and report['scientific_primary_passed']
    assert report['code_revision'] == provenance['code_revision'] == 'fb08789e760940f4b8c34511abbc42cbfa64c73c'
    assert not provenance['dirty_runtime'] and documents['process_status.json']['returncode'] == 0
    for name, expected in provenance['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert report['native_passive_solver_calibration']['valid']
    assert not report['task31_authorized'] and not report['full_active_neuron_native_validated']
    assert len(report['rows']) == 4 * 3 * 32 * 5
    from src.giada_teacher.task30_autonomous_voltage import episodes
    design = episodes(cfg)
    primary_steps = int(round(cfg['primary_horizon_ms'] / cfg['dt_ms']))
    active_before_primary = int((design['injection_ma_cm2'][:primary_steps] != 0).sum())
    assert active_before_primary == 0
    audit = {'schema_version': 'giada-task30-exposure-audit-v1',
             'technical_valid': True, 'decision_grade': False,
             'registered_primary_passed_but_uninformative': True,
             'active_injection_samples_before_primary_horizon': active_before_primary,
             'reason': 'All registered stimulus windows begin at or after 8ms; the 8ms primary metric had no injected-current exposure. Longer horizons are diagnostic only.',
             'task31_authorized': False,
             'artifact_sha256': sha(archive),
             'report_sha256': sha(FOLDER / 'final_report.json')}
    (FOLDER / 'result_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
