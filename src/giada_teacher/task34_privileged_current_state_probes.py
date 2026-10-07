"""Task34: frozen, non-selective privileged probes on confirmed Task33 traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import ionic_block_teacher_forced as ionic
from . import task30_autonomous_voltage as t30
from . import task32_dynamic_calcium_feedback as t32
from . import task33_observable_synaptic_feedback as t33
from .hh_family_transfer import write


def contract(root: Path) -> tuple[dict, dict, dict, dict]:
    spec = json.loads((root / 'experiments/task34_privileged_current_state_probes_preregistration.json').read_text(encoding='utf-8'))
    for name, digest in (('parent_report', 'parent_report_sha256'),
                         ('parent_traces', 'parent_traces_sha256')):
        if hashlib.sha256((root / spec[name]).read_bytes()).hexdigest() != spec[digest]:
            raise RuntimeError(f'Task34 parent {name} bytes changed')
    parent = json.loads((root / spec['parent_report']).read_text(encoding='utf-8'))
    if not (parent['valid'] and parent['task33_passed'] and parent['model_judged']
            and parent['native_floor_admissible'] and parent['shadow_passed']
            and parent['code_revision'] == spec['parent_code_revision']):
        raise RuntimeError('Task34 parent scientific contract not confirmed')
    cfg, t32cfg, _ = t33.contract(root)
    if spec['frozen_families'] != t32cfg['families'] or spec['frozen_seeds'] != t32cfg['model_seeds']:
        raise RuntimeError('Task34 frozen candidate inventory changed')
    return spec, parent, cfg, t32cfg


def load_traces(root: Path, spec: dict) -> tuple[list[dict], list[dict]]:
    with np.load(root / spec['parent_traces'], allow_pickle=False) as bundle:
        count = len(bundle['voltage'])
        if count != 16 or bundle['gates'].shape[-1] != 18 or bundle['base_g_us'].shape[-1] != 4:
            raise RuntimeError('Task34 parent trace shape changed')
        natives = [{'voltage': bundle['voltage'][j], 'calcium': bundle['calcium'][j],
                    'gates': bundle['gates'][j], 'injection': bundle['injection'][j],
                    'multipliers': ionic.panel_multipliers()['canonical'],
                    'area_um2': float(bundle['area_um2'])} for j in range(count)]
        shadows = [{'base_g_us': bundle['base_g_us'][j]} for j in range(count)]
    return natives, shadows


def teacher_voltage_model_state(native: list[dict], cfg: dict, t32cfg: dict,
                                model, torch) -> dict:
    """Advance candidate gates/calcium with teacher V clamped; diagnostic oracle."""
    teacher_v = np.stack([row['voltage'] for row in native], axis=-1)
    count = len(native)
    seeds = len(t32cfg['model_seeds'])
    steps = teacher_v.shape[0]
    calcium = np.empty((seeds, steps, count))
    gates = np.empty((seeds, steps, count, 18))
    calcium[:, 0] = cfg['initial_cai_mM']
    gates[:, 0] = t30.initial_states(teacher_v[0], calcium[0, 0])
    mult = np.broadcast_to(native[0]['multipliers'], (seeds, count, 11))
    for k in range(steps-1):
        old_v = np.broadcast_to(teacher_v[k], (seeds, count))
        next_state = t30._candidate_gate_step(model, torch, old_v,
                                               calcium[:, k], gates[:, k], cfg['dt_ms'])
        source = t32._ca_current(old_v, next_state, mult)
        calcium[:, k+1] = t32.calcium_step(calcium[:, k], source, t32cfg)
        t32._sk_update(next_state, gates[:, k], calcium[:, k+1], cfg['dt_ms'])
        gates[:, k+1] = next_state
    return {'voltage': np.broadcast_to(teacher_v, (seeds, steps, count)).copy(),
            'calcium': calcium, 'gates': gates}


def model_voltage_teacher_state(native: list[dict], shadows: list[dict],
                                cfg: dict, base: dict) -> dict:
    """Advance analytic voltage using the teacher's next gate states; oracle."""
    teacher_v = np.stack([row['voltage'] for row in native], axis=-1)
    teacher_gates = np.stack([row['gates'] for row in native], axis=1)
    teacher_cai = np.stack([row['calcium'] for row in native], axis=-1)
    count = len(native)
    steps = teacher_v.shape[0]
    voltage = np.empty((1, steps, count))
    voltage[:, 0] = teacher_v[0]
    mult = np.broadcast_to(native[0]['multipliers'], (1, count, 11))
    for k in range(steps-1):
        conductance = np.stack([s['base_g_us'][k] for s in shadows])
        injected = np.array([row['injection'][k] for row in native])
        voltage[:, k+1] = t33.voltage_step_with_synapses(
            voltage[:, k], teacher_gates[k+1][None], mult, injected,
            conductance, native[0]['area_um2'], base, cfg['dt_ms'])
    return {'voltage': voltage, 'calcium': teacher_cai[None],
            'gates': teacher_gates[None]}


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def current_factorial(native_v: np.ndarray, native_s: np.ndarray,
                      model_v: np.ndarray, model_s: np.ndarray,
                      atol: float) -> dict:
    """Exact algebraic decomposition at four matched V/state endpoints."""
    i00 = ionic.currents(native_v, native_s)
    i10 = ionic.currents(native_v, model_s)
    i01 = ionic.currents(model_v, native_s)
    i11 = ionic.currents(model_v, model_s)
    state = i10-i00
    voltage = i01-i00
    interaction = i11-i10-i01+i00
    combined = i11-i00
    residual = float(np.max(np.abs(combined-state-voltage-interaction)))
    if residual > atol:
        raise RuntimeError('Task34 current decomposition identity failed')
    channel = {}
    for k, name in enumerate(ionic.CHANNELS):
        channel[name] = {label: rmse(value[..., k]) for label, value in (
            ('state_only_ma_cm2', state), ('voltage_only_ma_cm2', voltage),
            ('interaction_ma_cm2', interaction), ('combined_ma_cm2', combined))}
    net = {label: rmse(value.sum(-1)) for label, value in (
        ('state_only_ma_cm2', state), ('voltage_only_ma_cm2', voltage),
        ('interaction_ma_cm2', interaction), ('combined_ma_cm2', combined))}
    combined_channel_sum = sum(row['combined_ma_cm2'] for row in channel.values())
    return {'per_channel': channel, 'net': net,
            'combined_channel_rmse_sum_ma_cm2': combined_channel_sum,
            'cancellation_ratio': net['combined_ma_cm2']/max(combined_channel_sum, 1e-30),
            'identity_max_abs_error_ma_cm2': residual}


def _gate_probe(pred: np.ndarray, native: np.ndarray) -> dict:
    result = {}
    for k, name in enumerate(ionic.CHANNELS):
        a, b = ionic.OFFSETS[k:k+2]
        result[name] = rmse(pred[..., a:b] - native[..., a:b])
    return result


def _synaptic_current_probe(native_v: np.ndarray, model_v: np.ndarray,
                            base_g: np.ndarray) -> dict:
    native_i = t33.receptor_conductance(base_g, native_v) * (native_v[..., None]-t33.REVERSALS)
    model_i = t33.receptor_conductance(base_g, model_v) * (model_v[..., None]-t33.REVERSALS)
    return {name: rmse(model_i[..., k]-native_i[..., k])
            for k, name in enumerate(t33.RECEPTORS)}


def run(root: Path, output: Path, *, code_revision: str) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Task34 frozen probe requires CUDA')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec, parent, cfg, t32cfg = contract(root)
    native, shadows = load_traces(root, spec)
    base = t30.config(root, Path(t32cfg['base_config']).name)
    native_v = np.stack([row['voltage'] for row in native], axis=-1)
    native_s = np.stack([row['gates'] for row in native], axis=1)
    native_cai = np.stack([row['calcium'] for row in native], axis=-1)
    base_g = np.stack([row['base_g_us'] for row in shadows], axis=1)
    oracle_voltage = model_voltage_teacher_state(native, shadows, cfg, base)
    formula = t33.coupled_rollout(native, shadows, cfg, t32cfg, base)
    formula_metrics = t33.metrics(formula, native)
    floor_delta = max(abs(formula_metrics[key]-parent['formula_floor'][key]) for key in (
        'pooled_voltage_rmse_mv', 'worst_voltage_rmse_mv', 'worst_cai_rmse_mM'))
    if floor_delta > spec['reproduction_atol']:
        raise RuntimeError('Task34 Task33 formula floor did not reproduce')
    loaded, seeds = ionic.load_frozen(root, ionic.config(root), torch, 'cuda')
    if list(seeds) != spec['frozen_seeds']:
        raise RuntimeError('Task34 candidate seeds changed')
    families = {}
    max_parent_delta = 0.
    max_identity = 0.
    for family in spec['frozen_families']:
        model = loaded[family, t32cfg['frozen_arm']]
        closed = t33.coupled_rollout(native, shadows, cfg, t32cfg, base, model, torch)
        clamped = teacher_voltage_model_state(native, cfg, t32cfg, model, torch)
        families[family] = {}
        for j, seed in enumerate(seeds):
            baseline = t33.metrics(closed, native, j)
            parent_metrics = parent['models'][family][str(seed)]
            delta = max(abs(baseline[key]-parent_metrics[key]) for key in (
                'pooled_voltage_rmse_mv', 'worst_voltage_rmse_mv', 'worst_cai_rmse_mM'))
            max_parent_delta = max(max_parent_delta, delta)
            if delta > spec['reproduction_atol']:
                raise RuntimeError(f'Task34 Task33 baseline mismatch: {family}/{seed}')
            state_at_teacher_v = clamped['gates'][j]
            state_closed = closed['gates'][j]
            voltage_closed = closed['voltage'][j]
            factorial = current_factorial(native_v, native_s, voltage_closed,
                                          state_closed, spec['decomposition_atol'])
            max_identity = max(max_identity, factorial['identity_max_abs_error_ma_cm2'])
            teacher_v_state_error = rmse(state_at_teacher_v-native_s)
            closed_state_error = rmse(state_closed-native_s)
            teacher_state_voltage_rmse = rmse(oracle_voltage['voltage'][0]-native_v)
            ratio = spec['interpretation_ratio']
            families[family][str(seed)] = {
                'deployable_baseline': baseline,
                'teacher_voltage_model_state_privileged_probe': {
                    'state_rmse': teacher_v_state_error,
                    'cai_rmse_mM': rmse(clamped['calcium'][j]-native_cai),
                    'per_channel_gate_rmse': _gate_probe(state_at_teacher_v, native_s)},
                'model_voltage_teacher_state_privileged_probe': {
                    'voltage_rmse_mv': teacher_state_voltage_rmse},
                'model_voltage_model_state': {
                    'state_rmse': closed_state_error,
                    'per_channel_gate_rmse': _gate_probe(state_closed, native_s),
                    'synaptic_current_rmse_na': _synaptic_current_probe(
                        native_v, voltage_closed, base_g)},
                'ionic_current_factorial': factorial,
                'diagnostic_flags': {
                    'voltage_feedback_amplifies_state_error_2x': bool(
                        closed_state_error >= ratio*teacher_v_state_error
                        and closed_state_error > 0),
                    'teacher_state_reduces_voltage_error_2x': bool(
                        baseline['pooled_voltage_rmse_mv'] >=
                        ratio*teacher_state_voltage_rmse)}}
        print(f'[GIADA Task34] frozen probes {family}: {len(seeds)} seeds', flush=True)
    report = {'schema_version': spec['schema_version'], 'valid': True,
              'diagnostic_complete': True, 'code_revision': code_revision,
              'parent_code_revision': spec['parent_code_revision'],
              'parent_baseline_max_metric_delta': max_parent_delta,
              'parent_formula_floor_max_metric_delta': floor_delta,
              'current_identity_max_abs_error_ma_cm2': max_identity,
              'case_count': len(native), 'families': families,
              'formula_floor': formula_metrics,
              'teacher_state_oracle_voltage_rmse_mv': rmse(oracle_voltage['voltage'][0]-native_v),
              'teacher_privileged_arms_selection_eligible': False,
              'training_performed': False, 'model_selection_performed': False,
              'independent_confirmation_claimed': False,
              'learned_voltage_updater_tested': False,
              'full_cell_claim': False, 'speedup_claim': False}
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'final_report.json', report)
    return report
