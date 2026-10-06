"""Post-hoc Task33 v2 floor attribution; diagnostic only, never selection."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.giada_teacher import ionic_block_teacher_forced as ionic
from src.giada_teacher import task30_autonomous_voltage as t30
from src.giada_teacher import task32_dynamic_calcium_feedback as t32
from src.giada_teacher.task33_observable_synaptic_feedback import (
    contract, coupled_rollout, metrics, voltage_step_with_synapses,
)


def main():
    cfg, t32cfg, _ = contract(ROOT)
    base = t30.config(ROOT, Path(t32cfg['base_config']).name)
    with np.load(ROOT / 'experiments/results/task33_v2_native_shadow_traces.npz') as bundle:
        natives = [{'voltage': bundle['voltage'][j], 'calcium': bundle['calcium'][j],
                    'gates': bundle['gates'][j], 'injection': bundle['injection'][j],
                    'multipliers': ionic.panel_multipliers()['canonical'],
                    'area_um2': float(bundle['area_um2'])}
                   for j in range(len(bundle['voltage']))]
        shadows = [{'base_g_us': bundle['base_g_us'][j]} for j in range(len(natives))]
    predicted = coupled_rollout(natives, shadows, cfg, t32cfg, base,
                                synaptic_state_phase='next')
    old_phase = coupled_rollout(natives, shadows, cfg, t32cfg, base,
                                synaptic_state_phase='old')
    area = natives[0]['area_um2']
    one_step_v = []
    one_step_ca = []
    one_step_v_old_g = []
    fed_ca = {label: np.full(len(natives), cfg['initial_cai_mM'])
              for label in ('old_v', 'mid_v', 'next_v')}
    fed_ca_errors = {label: [] for label in fed_ca}
    for k in range(len(natives[0]['voltage'])-1):
        old_v = np.array([row['voltage'][k] for row in natives])
        next_v = np.array([row['voltage'][k+1] for row in natives])
        old_cai = np.array([row['calcium'][k] for row in natives])
        next_cai = np.array([row['calcium'][k+1] for row in natives])
        next_state = np.stack([row['gates'][k+1] for row in natives])
        conductance = np.stack([row['base_g_us'][k+1] for row in shadows])
        injection = np.array([row['injection'][k] for row in natives])
        multipliers = np.stack([row['multipliers'] for row in natives])
        proposal = voltage_step_with_synapses(old_v, next_state, multipliers,
                         injection, conductance, area, base, cfg['dt_ms'])
        one_step_v.append(proposal-next_v)
        old_conductance = np.stack([row['base_g_us'][k] for row in shadows])
        proposal_old_g = voltage_step_with_synapses(old_v, next_state, multipliers,
                         injection, old_conductance, area, base, cfg['dt_ms'])
        one_step_v_old_g.append(proposal_old_g-next_v)
        ica = t32._ca_current(old_v, next_state, multipliers)
        one_step_ca.append(t32.calcium_step(old_cai, ica, t32cfg)-next_cai)
        for label, source_v in (('old_v', old_v), ('mid_v', (old_v+next_v)/2),
                                ('next_v', next_v)):
            source = t32._ca_current(source_v, next_state, multipliers)
            fed_ca[label] = t32.calcium_step(fed_ca[label], source, t32cfg)
            fed_ca_errors[label].append(fed_ca[label]-next_cai)
    one_step_v = np.asarray(one_step_v)
    one_step_v_old_g = np.asarray(one_step_v_old_g)
    one_step_ca = np.asarray(one_step_ca)
    fed_ca_errors = {label: np.asarray(values) for label, values in fed_ca_errors.items()}
    actual_ca_error = predicted['calcium'][0] - np.stack([row['calcium'] for row in natives], axis=-1)
    actual_v_error = predicted['voltage'][0] - np.stack([row['voltage'] for row in natives], axis=-1)
    report = {'formula_floor': metrics(predicted, natives),
        'posthoc_old_synaptic_state_floor_diagnostic_only': metrics(old_phase, natives),
        'one_step_native_state_voltage_max_mv': float(np.max(np.abs(one_step_v))),
        'one_step_native_state_voltage_rmse_mv': float(np.sqrt(np.mean(one_step_v**2))),
        'one_step_native_state_voltage_old_g_rmse_mv': float(np.sqrt(np.mean(one_step_v_old_g**2))),
        'one_step_native_state_voltage_old_g_max_mv': float(np.max(np.abs(one_step_v_old_g))),
        'one_step_native_state_calcium_max_mM': float(np.max(np.abs(one_step_ca))),
        'teacher_voltage_calcium_rollout_max_mM': {label: float(np.max(np.abs(values)))
                                                   for label, values in fed_ca_errors.items()},
        'own_voltage_calcium_rollout_max_mM': float(np.max(np.abs(actual_ca_error))),
        'peak_voltage_error_mv': float(np.max(np.abs(actual_v_error))),
        'worst_calcium_case': int(np.argmax(np.max(np.abs(actual_ca_error), axis=0))),
        'worst_voltage_case': int(np.argmax(np.max(np.abs(actual_v_error), axis=0)))}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
