"""Independently check the archived IV-B v2 decision against its preregistration."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'experiments/results/iv_b_v2_kaggle_8b3a4e5'
ARCHIVE = FOLDER / 'artifact_bundle.zip'
CONFIG = json.loads((ROOT / 'experiments/iv_b1_b2_calcium_prerequisite_v2.json').read_text())


def main():
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None, 'ZIP CRC failure'
        archived = archive.read('final_report.json')
        report = json.loads(archived)
        assert archived == (FOLDER / 'final_report.json').read_bytes()
        provenance = json.loads(archive.read('code_provenance.json'))
        process = json.loads(archive.read('process_status.json'))
        with np.load(io.BytesIO(archive.read('b2_native_traces.npz'))) as traces:
            trace_keys = set(traces.files)
            finite_traces = all(np.isfinite(traces[key]).all() for key in traces.files)
    assert report['schema_version'] == CONFIG['schema_version']
    assert report['code_revision'].startswith('8b3a4e5')
    assert report['native_neuron_version'] == CONFIG['neuron_version']
    assert len(report['b1_rows']) == report['b1_episode_count'] == 72
    assert len(report['b2_rows']) == report['b2_episode_count'] == 12
    assert all(key in trace_keys for row in report['b2_rows'] for key in
               (f"{row['cai0']}_{row['protocol']}_cai", f"{row['cai0']}_{row['protocol']}_z"))
    assert finite_traces
    b1 = all(row['rmse'] <= CONFIG['b1_cai_rmse_limit_mM']
             and row['max_abs'] <= CONFIG['b1_cai_max_error_limit_mM']
             and row['native_min_cai_mM'] >= 0 for row in report['b1_rows'])
    b2_metrics = ('cai', 'sk_oracle_cai', 'sk_composed', 'sk_current')
    limits = (CONFIG['b2_cai_rmse_limit_mM'], CONFIG['b2_sk_gate_rmse_limit'],
              CONFIG['b2_sk_gate_rmse_limit'], CONFIG['b2_sk_current_rmse_limit_ma_cm2'])
    b2 = all(all(row[key]['rmse'] <= limit for key, limit in zip(b2_metrics, limits))
             and row['voltage_clamp_max_error_mv'] <= CONFIG['b2_voltage_clamp_max_error_mv']
             for row in report['b2_rows'])
    contrasts = report['pulse_vs_rest_negative_control']
    expected_contrasts = {f'{cai0}_{protocol}' for cai0 in CONFIG['b2_initial_cai_mM']
                          for protocol in CONFIG['b2_protocols'] if protocol != 'rest'}
    informative = set(contrasts) == expected_contrasts and all(
        row['peak_cai_difference_mM'] > 1e-5 and row['peak_sk_difference'] > 1e-3
        for row in contrasts.values())
    assert report['iv_b1_valid'] == b1
    assert report['iv_b2_valid'] == (b1 and b2 and informative)
    assert report['task32_feedback_authorized'] == report['iv_b2_valid']
    assert report['valid'] == (b1 and b2 and informative)
    audit = {
        'valid': True,
        'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        'preregistration_sha256': hashlib.sha256((ROOT / 'experiments/iv_b1_b2_calcium_prerequisite_v2.json').read_bytes()).hexdigest(),
        'native_trace_finite': finite_traces,
        'b1_episode_count': len(report['b1_rows']),
        'b2_episode_count': len(report['b2_rows']),
        'b1_worst_cai_rmse_mM': max(row['rmse'] for row in report['b1_rows']),
        'b2_worst_composed_sk_gate_rmse': max(row['sk_composed']['rmse'] for row in report['b2_rows']),
        'b2_worst_current_rmse_ma_cm2': max(row['sk_current']['rmse'] for row in report['b2_rows']),
        'b2_worst_clamp_error_mv': max(row['voltage_clamp_max_error_mv'] for row in report['b2_rows']),
        'perturbation_informative': informative,
        'iv_b1_valid': b1,
        'iv_b2_valid': b2,
        'task32_feedback_authorized': b1 and b2 and informative,
        'code_revision': report['code_revision'],
        'provenance': provenance,
        'process_status': process,
    }
    (FOLDER / 'result_audit.json').write_bytes((json.dumps(audit, indent=2) + '\n').encode())
    print(json.dumps({key: value for key, value in audit.items()
                      if key not in ('provenance', 'process_status')}, indent=2))


if __name__ == '__main__':
    main()
