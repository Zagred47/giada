"""Audit the archived Kaggle Gate D bottleneck diagnostic."""
import hashlib
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'experiments/results/gate_d_forensic_kaggle_b19aa65'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = FOLDER / 'artifact_bundle.zip'
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        reports = {name: json.loads(zipped.read(name)) for name in (
            'final_report.json', 'run_contract.json', 'code_provenance.json',
            'process_status.json')}
    report, contract, provenance, status = (reports[name] for name in (
        'final_report.json', 'run_contract.json', 'code_provenance.json',
        'process_status.json'))
    assert reports['final_report.json'] == json.loads((FOLDER / 'final_report.json').read_text())
    assert contract == json.loads((ROOT / 'experiments/task28d_gate_d_bottleneck_forensic.json').read_text())
    assert provenance['code_revision'] == report['code_revision'] == 'b19aa65fec6d6d1ce200f6c590e5d2cd376fbfb0'
    assert not provenance['dirty_runtime'] and status['returncode'] == 0
    assert report['valid'] and report['diagnostic_only'] and not report['gate_d_completed']
    assert not report['task29_authorized'] and not report['training_performed']
    assert report['device_name'] == 'Tesla T4' and len(report['rows']) == 2
    for name, expected in provenance['sources'].items():
        # This audit is run before any source edits after the pinned commit.
        assert sha(ROOT / name) == expected, name
    summary = []
    for row in report['rows']:
        timings = row['timing']
        assert row['batch_size'] == 642
        assert row['family'] in contract['families']
        assert row['decomposition_max_gate_difference'] <= contract['numeric_tolerance_gate']
        assert row['decomposition_max_current_difference_ma_cm2'] <= contract['numeric_tolerance_current_ma_cm2']
        for item in timings.values():
            samples = item['samples_ms']
            assert len(samples) == contract['benchmark_repetitions']
            assert all(math.isfinite(x) and x > 0 for x in samples)
            assert item['median_ms'] == statistics.median(samples)
        assert timings['neural_core_five']['median_ms'] > timings['exact_core_five']['median_ms']
        assert timings['hybrid_full']['median_ms'] > timings['exact_full']['median_ms']
        assert row['profiles']['exact_full']['valid'] and row['profiles']['hybrid_full']['valid']
        summary.append({
            'family': row['family'],
            'exact_core_ms': timings['exact_core_five']['median_ms'],
            'neural_core_ms': timings['neural_core_five']['median_ms'],
            'exact_full_ms': timings['exact_full']['median_ms'],
            'hybrid_full_ms': timings['hybrid_full']['median_ms'],
            'exact_full_kernel_events_per_4_calls': row['profiles']['exact_full']['kernel_event_count'],
            'hybrid_full_kernel_events_per_4_calls': row['profiles']['hybrid_full']['kernel_event_count'],
        })
    audit = {'schema_version': 'giada-gate-d-bottleneck-audit-v1', 'valid': True,
             'scientific_status': 'diagnostic_only_gate_d_no_go_unchanged',
             'artifact_sha256': sha(archive), 'report_sha256': sha(FOLDER / 'final_report.json'),
             'summary': summary}
    (FOLDER / 'result_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
