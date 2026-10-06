"""Audit the first Task32 floor failure without changing its registered result."""

import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.giada_teacher import task32_dynamic_calcium_feedback as task

FOLDER = ROOT / 'experiments/results/task32_kaggle_f8c5d25'


def main():
    cfg = json.loads((ROOT / 'experiments/task32_dynamic_calcium_feedback_preregistration.json').read_text())
    base = task.t30.config(ROOT, Path(cfg['base_config']).name)
    d = task.design(cfg, base)
    with zipfile.ZipFile(FOLDER / 'artifact_bundle.zip') as archive:
        assert archive.testzip() is None
        report = json.loads(archive.read('final_report.json'))
        assert report == json.loads((FOLDER / 'final_report.json').read_text())
        with np.load(io.BytesIO(archive.read('native_traces.npz'))) as arrays:
            native = {key: arrays[key] for key in arrays.files}
        process = json.loads(archive.read('process_status.json'))
        provenance = json.loads(archive.read('code_provenance.json'))
    assert process['returncode'] == 0 and not provenance['dirty_runtime']
    assert not report['native_floor_admissible'] and not report['models']
    roles = {
        'calibration': [j for j, row in enumerate(d['rows']) if row['protocol'] in cfg['calibration_protocols']],
        'confirmation': [j for j, row in enumerate(d['rows']) if row['protocol'] in cfg['confirmation_protocols']],
    }
    comparison = {}
    for scheme in cfg['source_hypotheses']:
        predicted = task.rollout(d, cfg, base, scheme)
        comparison[scheme] = {role: task.aggregate(task.episode_metrics(
            predicted, native, d, cfg, indices, cfg['primary_horizon_ms']))
            for role, indices in roles.items()}
    dt = cfg['dt_ms']
    v0, v1 = native['voltage'][:-1], native['voltage'][1:]
    ca0, ca1 = native['calcium'][:-1], native['calcium'][1:]
    gate0, gate1 = native['gates'][:-1], native['gates'][1:]
    next_gate = task.ionic.exact_step(v0, ca0, gate0, dt)
    task._sk_update(next_gate, gate0, ca1, dt)
    next_voltage = task.t30.voltage_step(v0, gate1, d['multipliers'],
                                         d['injection_ma_cm2'], base)
    local = {
        'voltage_given_native_next_gates_max_abs_mv': float(np.max(abs(next_voltage-v1))),
        'gate_given_native_next_cai_max_abs': float(np.max(abs(next_gate-gate1))),
        'calcium_given_native_ica_max_abs_mM': float(np.max(abs(
            task.calcium_step(ca0, native['ica'][:-1], cfg)-ca1))),
        'calcium_old_gate_ica_max_abs_mM': float(np.max(abs(
            task.calcium_step(ca0, task._ca_current(v0, gate0, d['multipliers']), cfg)-ca1))),
        'calcium_updated_gate_ica_max_abs_mM': float(np.max(abs(
            task.calcium_step(ca0, task._ca_current(v0, gate1, d['multipliers']), cfg)-ca1))),
    }
    gca = d['multipliers'][None, :, 0]*task.ionic.GBAR[0]*native['gates'][..., 0]**2*native['gates'][..., 1]
    gca += d['multipliers'][None, :, 1]*task.ionic.GBAR[1]*native['gates'][..., 2]**2*native['gates'][..., 3]
    informative_g = gca > 1e-9
    inferred_eca = native['voltage'][informative_g] - native['ica'][informative_g]/gca[informative_g]
    local['native_ica_vs_fixed_eca120_rmse_ma_cm2'] = task.rmse(
        native['ica'], gca*(native['voltage']-120.))
    local['inferred_eca_range_mv'] = [float(np.min(inferred_eca)), float(np.max(inferred_eca))]
    gate_residual = abs(next_gate-gate1)
    gate_index = np.unravel_index(np.argmax(gate_residual), gate_residual.shape)
    local['worst_gate_one_step'] = {'step': int(gate_index[0]), 'episode': int(gate_index[1]),
                                    'state_index': int(gate_index[2]),
                                    'error': float(gate_residual[gate_index])}
    selected_rollout = task.rollout(d, cfg, base, report['source_hypothesis_selected_on_calibration_only'])
    onset = {}
    for j in roles['confirmation']:
        difference = abs(selected_rollout['voltage'][0, :, j]-native['voltage'][:, j])
        crossings = np.flatnonzero(difference > 0.1)
        onset[str(j)] = {'episode': d['rows'][j],
                         'first_error_gt_0p1_mv_ms': (float(crossings[0]*dt) if len(crossings) else None),
                         'max_error_mv': float(difference.max())}
    result = {
        'valid': True,
        'first_run_decision_unchanged': True,
        'models_not_judged': True,
        'registered_source_selection': report['source_hypothesis_selected_on_calibration_only'],
        'source_comparison_posthoc': comparison,
        'native_trace_finite': all(np.isfinite(value).all() for value in native.values()),
        'local_one_step_audit': local,
        'confirmation_error_onset': onset,
        'calibration_margin_worst_voltage_mv': abs(
            comparison['old_state_ica']['calibration']['worst_episode_voltage_rmse_mv']
            - comparison['updated_gate_ica']['calibration']['worst_episode_voltage_rmse_mv']),
        'interpretation': 'Both source hypotheses fail the registered 40ms confirmation floor in the active high-current arms. The one-step identities and error onsets separate a true calcium-source mismatch from amplified active-trajectory divergence. The first-run decision is unchanged.'
    }
    (FOLDER / 'result_audit.json').write_bytes((json.dumps(result, indent=2)+'\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
