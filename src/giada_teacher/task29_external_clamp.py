"""Task29: diagnostic ionic/passive current balance at imposed voltage.

This deliberately does not integrate voltage. Task30 owns autonomous voltage.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import ionic_block_teacher_forced as ionic
from .hh_family_transfer import write


def config(root: Path) -> dict:
    return json.loads((root / 'experiments/task29_external_clamp_ionic_passive.json').read_text(encoding='utf-8'))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_prerequisites(root: Path, cfg: dict) -> None:
    amendment = json.loads((root / cfg['amendment_path']).read_text(encoding='utf-8'))
    if not (amendment['task29_diagnostic_authorized']
            and not amendment['task29_performance_promotion_authorized']
            and not amendment['task30_autonomous_voltage_authorized']
            and amendment['gate_d_performance_status'] == 'NO_GO_UNCHANGED'):
        raise RuntimeError('Task29 scientific-track amendment is not in force')
    for prefix in ('parent_task28', 'parent_gate_d_compiled'):
        folder = root / cfg[prefix + '_dir']
        expected = cfg[prefix + '_sha256']
        if {name: _sha(folder / name) for name in expected} != expected:
            raise RuntimeError(prefix + ' parent hash mismatch')
    task28 = json.loads((root / cfg['parent_task28_dir'] / 'final_report.json').read_text(encoding='utf-8'))
    compiled = json.loads((root / cfg['parent_gate_d_compiled_dir'] / 'final_report.json').read_text(encoding='utf-8'))
    if not (task28['valid'] and task28['teacher_forced_ionic_block_passed']
            and compiled['valid'] and not compiled['gate_d_robustly_supported']):
        raise RuntimeError('Task29 scientific prerequisite or Gate D status changed')


def _path(name: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(cfg['path_steps'], dtype=np.float64) * cfg['dt_ms']
    if name == 'rest':
        voltage = np.full_like(t, -76.)
        calcium = np.full_like(t, 1e-5)
    elif name == 'slow_ramp':
        voltage = -78. + 75. * np.minimum(t / 50., 1.) - 45. * np.maximum((t - 50.) / 30., 0.)
        calcium = 1e-5 * (1. + 4. * np.exp(-((t - 48.) / 12.) ** 2))
    elif name == 'paired_steps':
        voltage = -76. + 62. * (((t >= 12.) & (t < 22.)) | ((t >= 42.) & (t < 54.)))
        calcium = 1e-5 * (1. + 3. * (((t >= 12.) & (t < 27.)) | ((t >= 42.) & (t < 59.))))
    elif name == 'oscillation':
        voltage = -60. + 32. * np.sin(2. * np.pi * t / 17.) + 10. * np.sin(2. * np.pi * t / 5.)
        calcium = 10 ** (-5. + .7 * np.sin(2. * np.pi * t / 31.))
    else:
        raise ValueError(name)
    return np.asarray(voltage), np.asarray(calcium)


def _native_passive_audit(cfg: dict) -> dict:
    try:
        import neuron
        from neuron import h
    except ImportError as exc:
        raise RuntimeError('Task29 requires NEURON for independent passive audit') from exc
    if neuron.__version__.split('+')[0] != '8.2.7':
        raise RuntimeError('Task29 requires NEURON 8.2.7')
    sec = h.Section(name='giada_task29_passive_oracle')
    sec.L = sec.diam = 10.
    sec.nseg = 1
    sec.cm = cfg['cm_uf_cm2']
    sec.insert('pas')
    sec.g_pas = cfg['g_pas_s_cm2']
    sec.e_pas = cfg['e_pas_mv']
    segment = sec(.5)
    rows = []
    try:
        for v in cfg['native_passive_test_voltages_mv']:
            h.finitialize(v)
            h.fcurrent()
            observed = float(segment.pas.i)
            expected = cfg['g_pas_s_cm2'] * (v - cfg['e_pas_mv'])
            rows.append({'voltage_mv': v, 'native_ma_cm2': observed,
                         'formula_ma_cm2': expected, 'absolute_error_ma_cm2': abs(observed - expected)})
    finally:
        sec = None
    error = max(row['absolute_error_ma_cm2'] for row in rows)
    return {'valid': error <= cfg['native_passive_current_atol_ma_cm2'],
            'neuron_version': neuron.__version__, 'maximum_error_ma_cm2': error, 'rows': rows}


def _exact_states(v: np.ndarray, ca: np.ndarray, initial: np.ndarray, dt: float) -> np.ndarray:
    states = np.empty((len(v) - 1, 18), dtype=np.float64)
    state = initial.copy()
    for k in range(len(states)):
        states[k] = state
        state = ionic.exact_step(v[k:k+1], ca[k:k+1], state[None], np.array([dt]))[0]
    return states


def _candidate_states(model, torch, device: str, v: np.ndarray, ca: np.ndarray,
                      initial: np.ndarray, dt: float, seed_count: int) -> np.ndarray:
    steps = len(v) - 1
    # The frozen physical-tau heads depend on imposed V, not current occupancy.
    # As in Task28, a fixed placeholder is supplied only to satisfy the model interface.
    packed = np.c_[v[:-1], np.full((steps, 10), .5), np.full(steps, dt)]
    with torch.inference_mode():
        x = torch.tensor(packed, device=device, dtype=torch.float32)[None].expand(seed_count, -1, -1)
        _, inf, tau = model(x)
        inf = inf.detach().cpu().double().numpy().transpose(0, 2, 1, 3).reshape(seed_count, steps, 10)
        tau = tau.detach().cpu().double().numpy().transpose(0, 2, 1, 3).reshape(seed_count, steps, 10)
    if not np.isfinite(inf).all() or not np.isfinite(tau).all() or np.any(tau <= 0):
        raise RuntimeError('Nonfinite or nonpositive frozen learned rates')
    tail_inf = np.empty((steps, 8), dtype=np.float64)
    tail_tau = np.empty((steps, 8), dtype=np.float64)
    for index, name in enumerate(ionic.CHANNELS[5:], start=5):
        a, b = ionic.OFFSETS[index:index + 2]
        i, t = ionic.rates(name, v[:-1], ca[:-1])
        tail_inf[:, a-10:b-10] = np.asarray(i).reshape(steps, b-a)
        tail_tau[:, a-10:b-10] = np.asarray(t).reshape(steps, b-a)
    result = np.empty((seed_count, steps, 18), dtype=np.float64)
    state = np.repeat(initial[None], seed_count, axis=0)
    for k in range(steps):
        result[:, k] = state
        z = -np.expm1(-dt / tau[:, k])
        state[:, :10] = (1-z) * state[:, :10] + z * inf[:, k]
        tail_z = -np.expm1(-dt / tail_tau[k])
        state[:, 10:] = (1-tail_z) * state[:, 10:] + tail_z * tail_inf[k]
    return result


def _metrics(v: np.ndarray, truth: np.ndarray, candidate: np.ndarray,
             multiplier: np.ndarray, cfg: dict) -> dict:
    voltage = v[:-1]
    ion_true = ionic.currents(voltage, truth, multiplier)
    ion_pred = ionic.currents(voltage, candidate, multiplier)
    passive = cfg['g_pas_s_cm2'] * (voltage - cfg['e_pas_mv'])
    capacitive = cfg['cm_uf_cm2'] * 1e-3 * np.diff(v) / cfg['dt_ms']
    clamp_true = capacitive + passive + ion_true.sum(-1)
    clamp_pred = capacitive + passive + ion_pred.sum(-1)
    drive = ionic.GBAR[None] * multiplier[None] * (voltage[:, None] - ionic.REVERSALS[None])
    denom = np.maximum(abs(capacitive) + abs(passive) + abs(drive).sum(-1), 1e-12)
    normalized_clamp = (clamp_pred - clamp_true) / denom
    individual = []
    for k in range(len(ionic.CHANNELS)):
        mask = abs(drive[:, k]) > 1e-12
        value = ((ion_pred[:, k] - ion_true[:, k])[mask] / drive[:, k][mask]) if mask.any() else np.array([0.])
        individual.append(float(np.sqrt(np.mean(value**2))))
    return {'gate_rmse': float(np.sqrt(np.mean((candidate-truth)**2))),
            'worst_individual_current_normalized_rmse': max(individual),
            'per_channel_current_normalized_rmse': dict(zip(ionic.CHANNELS, individual)),
            'clamp_demand_normalized_rmse': float(np.sqrt(np.mean(normalized_clamp**2))),
            'clamp_demand_absolute_rmse_ma_cm2': float(np.sqrt(np.mean((clamp_pred-clamp_true)**2))),
            'clamp_demand_peak_absolute_error_ma_cm2': float(np.max(abs(clamp_pred-clamp_true))),
            'occupancy_violations': int(np.count_nonzero((candidate < 0) | (candidate > 1))),
            'inward_ionic_current_sample_count': int(np.count_nonzero(ion_true < 0)),
            'capacitive_peak_absolute_ma_cm2': float(np.max(abs(capacitive))),
            'passive_peak_absolute_ma_cm2': float(np.max(abs(passive))),
            'finite': bool(np.isfinite(candidate).all() and np.isfinite(clamp_pred).all())}


def run(root: Path, output: Path, cfg: dict, revision: str) -> dict:
    import torch
    verify_prerequisites(root, cfg)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('Task29 registered frozen GPU execution requires CUDA')
    native = _native_passive_audit(cfg)
    write(output / 'native_passive_audit.json', native)
    if not native['valid']:
        raise RuntimeError('Task29 native passive preflight failed')
    frozen_cfg = ionic.config(root)
    models, seeds = ionic.load_frozen(root, frozen_cfg, torch, 'cuda')
    if list(seeds) != cfg['model_seeds']:
        raise RuntimeError('Frozen model seed contract changed')
    panels = ionic.panel_multipliers()
    if tuple(panels) != tuple(cfg['panel_names']):
        raise RuntimeError('Task29 current panel contract changed')
    rng = np.random.default_rng(cfg['initial_state_seed'])
    rows = []
    for path_name in cfg['path_names']:
        v, ca = _path(path_name, cfg)
        initial = rng.uniform(.05, .95, 18)
        truth = _exact_states(v, ca, initial, cfg['dt_ms'])
        for family in cfg['families']:
            for arm in cfg['arms']:
                model = models[family, arm]
                candidates = _candidate_states(model, torch, 'cuda', v, ca, initial, cfg['dt_ms'], len(seeds))
                for seed_index, seed in enumerate(seeds):
                    for panel_name, multipliers in panels.items():
                        m = _metrics(v, truth, candidates[seed_index], multipliers, cfg)
                        passed = bool(m['finite'] and m['occupancy_violations'] == 0
                                      and m['gate_rmse'] <= cfg['gate_rmse_limit']
                                      and m['worst_individual_current_normalized_rmse'] <= cfg['worst_individual_current_normalized_rmse_limit']
                                      and m['clamp_demand_normalized_rmse'] <= cfg['clamp_demand_normalized_rmse_limit'])
                        rows.append({'path': path_name, 'family': family, 'arm': arm,
                                     'seed': seed, 'panel': panel_name, 'passed': passed, **m})
        print(f'[GIADA Task29] path {path_name}: {len(rows)} cumulative rows', flush=True)
    both = [r for r in rows if r['arm'] == 'both']
    report = {'schema_version': 'giada-roadmap-task29-external-clamp-v1', 'valid': True,
              'diagnostic_scientific_passed': bool(native['valid'] and all(r['passed'] for r in both)),
              'gate_d_performance_status': 'NO_GO_UNCHANGED', 'gate_d_completed': False,
              'task29_diagnostic_authorized': True, 'task29_performance_promotion_authorized': False,
              'task30_autonomous_voltage_authorized': False, 'voltage_autonomous': False,
              'calcium_autonomous': False, 'training_performed': False,
              'fresh_used_for_selection': False, 'native_passive_audit': native,
              'rows': rows, 'code_revision': revision, 'limits': cfg['limits']}
    write(output / 'final_report.json', report)
    return report
