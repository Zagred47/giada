"""Post-hoc audit of the Task34 privileged teacher-state voltage arm."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.giada_teacher import ionic_block_teacher_forced as ionic
from src.giada_teacher import task30_autonomous_voltage as t30
from src.giada_teacher import task32_dynamic_calcium_feedback as t32
from src.giada_teacher import task33_observable_synaptic_feedback as t33
from src.giada_teacher.task34_privileged_current_state_probes import contract, load_traces, rmse


def main():
    spec, _, cfg, t32cfg = contract(ROOT)
    native, shadows = load_traces(ROOT, spec)
    base = t30.config(ROOT, Path(t32cfg['base_config']).name)
    v = np.stack([row['voltage'] for row in native], axis=-1)
    cai = np.stack([row['calcium'] for row in native], axis=-1)
    gates = np.stack([row['gates'] for row in native], axis=1)
    mult = np.broadcast_to(native[0]['multipliers'], (len(native), 11))
    errors = {key: [] for key in ('native_next_state', 'native_old_state',
                                 'formula_from_native_old_state')}
    gate_errors = []
    for k in range(len(v)-1):
        old = gates[k]
        formula_next = ionic.exact_step(v[k], cai[k], old, cfg['dt_ms'])
        source = t32._ca_current(v[k], formula_next, mult)
        next_cai = t32.calcium_step(cai[k], source, t32cfg)
        t32._sk_update(formula_next, old, next_cai, cfg['dt_ms'])
        gate_errors.append(formula_next-gates[k+1])
        conductance = np.stack([shadow['base_g_us'][k] for shadow in shadows])
        injection = np.array([row['injection'][k] for row in native])
        for key, state in (('native_next_state', gates[k+1]),
                           ('native_old_state', gates[k]),
                           ('formula_from_native_old_state', formula_next)):
            proposal = t33.voltage_step_with_synapses(
                v[k], state, mult, injection, conductance,
                native[0]['area_um2'], base, cfg['dt_ms'])
            errors[key].append(proposal-v[k+1])
    report = {'one_step_voltage_rmse_mv': {key: rmse(value) for key, value in errors.items()},
              'formula_vs_native_next_gate_rmse': rmse(gate_errors),
              'diagnostic_only': True, 'model_selection_performed': False}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
