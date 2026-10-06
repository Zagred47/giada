"""Post-hoc source-timing attribution; does not promote the Task32 v2 result."""

import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.giada_teacher import task32_dynamic_calcium_feedback as task

FOLDER = ROOT / 'experiments/results/task32_v2_kaggle_e22b669'


def main():
    cfg = json.loads((ROOT / 'experiments/task32_dynamic_calcium_feedback_v2.json').read_text())
    base = task.t30.config(ROOT, Path(cfg['base_config']).name)
    d = task.design(cfg, base)
    with zipfile.ZipFile(FOLDER / 'artifact_bundle.zip') as archive:
        assert archive.testzip() is None
        report = json.loads(archive.read('final_report.json'))
        assert report == json.loads((FOLDER / 'final_report.json').read_text())
        with np.load(io.BytesIO(archive.read('native_traces.npz'))) as arrays:
            native = {key: arrays[key] for key in arrays.files}
        assert json.loads(archive.read('process_status.json'))['returncode'] == 0
        assert not json.loads(archive.read('code_provenance.json'))['dirty_runtime']
    assert report['native_eca_max_deviation_mv'] == 0
    assert not report['native_floor_admissible'] and not report['models']
    v0 = native['voltage'][:-1]
    ca0, ca1 = native['calcium'][:-1], native['calcium'][1:]
    gate0, gate1 = native['gates'][:-1], native['gates'][1:]
    source_errors = {}
    for scheme, gates in [('old_state_ica', gate0), ('updated_gate_ica', gate1)]:
        source = task._ca_current(v0, gates, d['multipliers'])
        residual = task.calcium_step(ca0, source, cfg)-ca1
        source_errors[scheme] = {'rmse_mM': task.rmse(residual, np.zeros_like(residual)),
                                 'max_abs_mM': float(np.max(abs(residual)))}
    gate_step = {}
    for voltage_label, voltage in [('old_voltage', v0), ('new_voltage', native['voltage'][1:])]:
        predicted_gates = task.ionic.exact_step(voltage, ca0, gate0, cfg['dt_ms'])
        task._sk_update(predicted_gates, gate0, ca1, cfg['dt_ms'])
        gate_step[voltage_label] = {
            'rmse': task.rmse(predicted_gates, gate1),
            'max_abs': float(np.max(abs(predicted_gates-gate1)))}
    confirmation = [j for j, row in enumerate(d['rows'])
                    if row['protocol'] in cfg['confirmation_protocols']]
    counterfactual = task.rollout(d, cfg, base, 'updated_gate_ica')
    metrics = task.aggregate(task.episode_metrics(counterfactual, native, d,
                                                  cfg, confirmation, cfg['primary_horizon_ms']))
    audit = {'valid': True, 'v2_decision_unchanged': True, 'model_not_judged': True,
             'native_eca_max_deviation_mv': report['native_eca_max_deviation_mv'],
             'one_step_calcium_source_errors': source_errors,
             'one_step_gate_voltage_alignment': gate_step,
             'updated_gate_source_confirmation_posthoc': metrics,
             'interpretation': 'The old-state source pre-registered in v2 failed its cai gate. Native one-step calcium is reconstructed from recorded old gates, but the free-running formula with updated-gate source matches held-out trajectories post-hoc. This is a phase-alignment issue, not proof that NEURON uses new gates as its calcium source. V3 must prospectively test the effective discrete interface.'}
    (FOLDER / 'result_audit.json').write_bytes((json.dumps(audit, indent=2)+'\n').encode())
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
