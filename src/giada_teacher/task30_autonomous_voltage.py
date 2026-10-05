"""Task30 single-compartment autonomous-voltage microcanary.

The formula arm is an internal discrete reference, not an active NEURON oracle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import ionic_block_teacher_forced as ionic
from .hh_family_transfer import write


def config(root: Path) -> dict:
    return json.loads((root / 'experiments/task30_autonomous_voltage_microcanary.json').read_text(encoding='utf-8'))


def verify_parent(root: Path, cfg: dict) -> None:
    folder = root / cfg['scientific_parent']
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
              for name in cfg['scientific_parent_sha256']}
    if hashes != cfg['scientific_parent_sha256']:
        raise RuntimeError('Task29 scientific parent hash mismatch')
    report = json.loads((folder / 'final_report.json').read_text(encoding='utf-8'))
    audit = json.loads((folder / 'result_audit.json').read_text(encoding='utf-8'))
    if not (report['diagnostic_scientific_passed'] and audit['valid']
            and not report['gate_d_completed'] and not report['task30_autonomous_voltage_authorized']):
        raise RuntimeError('Task29 diagnostic evidence or scope changed')
    performance = json.loads((root / cfg['performance_parent'] / 'final_report.json').read_text(encoding='utf-8'))
    if not (performance['valid'] and not performance['gate_d_robustly_supported']
            and cfg['performance_status'] == 'NO_GO_UNCHANGED'):
        raise RuntimeError('Gate D performance status changed')


def _injection(name: str, t: np.ndarray) -> np.ndarray:
    if name == 'rest': return np.zeros_like(t)
    if name == 'low': return .002 * ((t >= 10.) & (t < 22.))
    if name == 'high': return .008 * ((t >= 10.) & (t < 22.))
    if name == 'paired':
        return .006 * (((t >= 8.) & (t < 16.)) | ((t >= 33.) & (t < 41.)))
    raise ValueError(name)


def episodes(cfg: dict) -> dict:
    rows = []
    for initial in cfg['initial_voltage_mv']:
        for protocol in cfg['protocols']:
            for ca in cfg['calcium_levels_mM']:
                for panel in cfg['conductance_panels']:
                    rows.append({'initial_voltage_mv': initial, 'protocol': protocol,
                                 'calcium_mM': ca, 'panel': panel})
    t = np.arange(cfg['steps'], dtype=np.float64) * cfg['dt_ms']
    return {'rows': rows, 'time_ms': t,
            'injection_ma_cm2': np.stack([_injection(r['protocol'], t[:-1]) for r in rows], axis=1),
            'calcium_mM': np.array([r['calcium_mM'] for r in rows]),
            'multipliers': np.stack([ionic.panel_multipliers()[r['panel']] for r in rows])}


def initial_states(v: np.ndarray, ca: np.ndarray) -> np.ndarray:
    state = np.empty((len(v), 18), dtype=np.float64)
    for k, name in enumerate(ionic.CHANNELS):
        a, b = ionic.OFFSETS[k:k+2]
        inf, _ = ionic.rates(name, v, ca)
        state[:, a:b] = np.asarray(inf).reshape(len(v), b-a)
    return state


def conductances(state: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    opening = []
    for k, names in enumerate(ionic.STATE_NAMES):
        a = ionic.OFFSETS[k]
        value = state[..., a] ** ionic.POWERS[k]
        if len(names) == 2: value = value * state[..., a+1]
        opening.append(value)
    return np.stack(opening, axis=-1) * ionic.GBAR * multipliers


def voltage_step(v: np.ndarray, state_next: np.ndarray, multipliers: np.ndarray,
                 injected: np.ndarray, cfg: dict) -> np.ndarray:
    g = conductances(state_next, multipliers)
    cap = 1e-3 * cfg['cm_uf_cm2'] / cfg['dt_ms']
    denominator = cap + cfg['g_pas_s_cm2'] + g.sum(-1)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise RuntimeError('Nonpositive or nonfinite semi-implicit voltage denominator')
    return ((cap * v + cfg['g_pas_s_cm2'] * cfg['e_pas_mv']
             + (g * ionic.REVERSALS).sum(-1) + injected) / denominator)


def _formula_gate_step(v: np.ndarray, ca: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    return ionic.exact_step(v, ca, state, np.full(len(v), dt))


def formula_reference(design: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    count = len(design['rows'])
    voltage = np.empty((cfg['steps'], count), dtype=np.float64)
    state = np.empty((cfg['steps'], count, 18), dtype=np.float64)
    voltage[0] = [row['initial_voltage_mv'] for row in design['rows']]
    state[0] = initial_states(voltage[0], design['calcium_mM'])
    for k in range(cfg['steps'] - 1):
        state[k+1] = _formula_gate_step(voltage[k], design['calcium_mM'], state[k], cfg['dt_ms'])
        voltage[k+1] = voltage_step(voltage[k], state[k+1], design['multipliers'],
                                     design['injection_ma_cm2'][k], cfg)
    return voltage, state


def _neural_rates(model, torch, voltage: np.ndarray, state: np.ndarray,
                  dt: float) -> tuple[np.ndarray, np.ndarray]:
    seeds, count = voltage.shape
    packed = np.concatenate((voltage[..., None], state[..., :10],
                             np.full((seeds, count, 1), dt)), axis=-1)
    with torch.inference_mode():
        tensor = torch.tensor(packed, dtype=torch.float32, device='cuda')
        _, inf, tau = model(tensor)
        inf = inf.detach().cpu().double().numpy().transpose(0, 2, 1, 3).reshape(seeds, count, 10)
        tau = tau.detach().cpu().double().numpy().transpose(0, 2, 1, 3).reshape(seeds, count, 10)
    if not (np.isfinite(inf).all() and np.isfinite(tau).all() and np.all(tau > 0)):
        raise RuntimeError('Frozen model produced nonfinite or nonpositive rate')
    return inf, tau


def _candidate_gate_step(model, torch, voltage: np.ndarray, ca: np.ndarray,
                         state: np.ndarray, dt: float) -> np.ndarray:
    seeds, count = voltage.shape
    inf, tau = _neural_rates(model, torch, voltage, state, dt)
    result = np.empty_like(state)
    z = -np.expm1(-dt / tau)
    result[..., :10] = (1-z) * state[..., :10] + z * inf
    for k, name in enumerate(ionic.CHANNELS[5:], start=5):
        a, b = ionic.OFFSETS[k:k+2]
        i, t = ionic.rates(name, voltage, ca)
        i = np.asarray(i).reshape(seeds, count, b-a)
        t = np.asarray(t).reshape(seeds, count, b-a)
        zz = -np.expm1(-dt / t)
        result[..., a:b] = (1-zz)*state[..., a:b] + zz*i
    return result


def candidate_rollout(model, torch, design: dict, cfg: dict,
                      reference_v: np.ndarray, reference_state: np.ndarray,
                      seed_count: int) -> dict:
    steps, count = reference_v.shape
    own_v = np.empty((seed_count, steps, count), dtype=np.float64)
    own_state = np.empty((seed_count, steps, count, 18), dtype=np.float64)
    forced_state = np.empty_like(own_state)
    forced_voltage_proposal = np.empty((seed_count, steps-1, count), dtype=np.float64)
    own_v[:, 0] = reference_v[0]
    own_state[:, 0] = reference_state[0]
    forced_state[:, 0] = reference_state[0]
    calcium = np.broadcast_to(design['calcium_mM'], (seed_count, count))
    multipliers = np.broadcast_to(design['multipliers'], (seed_count, count, 11))
    for k in range(steps - 1):
        forced_v = np.broadcast_to(reference_v[k], (seed_count, count))
        forced_state[:, k+1] = _candidate_gate_step(
            model, torch, forced_v, calcium, forced_state[:, k], cfg['dt_ms'])
        forced_voltage_proposal[:, k] = voltage_step(
            forced_v, forced_state[:, k+1], multipliers,
            design['injection_ma_cm2'][k], cfg)
        own_state[:, k+1] = _candidate_gate_step(
            model, torch, own_v[:, k], calcium, own_state[:, k], cfg['dt_ms'])
        own_v[:, k+1] = voltage_step(
            own_v[:, k], own_state[:, k+1], multipliers,
            design['injection_ma_cm2'][k], cfg)
    return {'autonomous_voltage': own_v, 'autonomous_state': own_state,
            'teacher_voltage_state': forced_state,
            'teacher_voltage_one_step_proposal': forced_voltage_proposal}


def native_passive_calibration(cfg: dict) -> dict:
    try:
        import neuron
        from neuron import h
    except ImportError as exc:
        raise RuntimeError('Task30 requires NEURON passive-only solver calibration') from exc
    if neuron.__version__.split('+')[0] != '8.2.7':
        raise RuntimeError('Task30 requires NEURON 8.2.7')
    sec = h.Section(name='giada_task30_passive_solver')
    sec.L = sec.diam = 10.
    sec.nseg = 1
    sec.cm = cfg['cm_uf_cm2']
    sec.insert('pas')
    sec.g_pas = cfg['g_pas_s_cm2']
    sec.e_pas = cfg['e_pas_mv']
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = cfg['dt_ms']
    segment = sec(.5)
    clamp = h.IClamp(segment)
    clamp.delay = 0.
    clamp.dur = 10.
    density = .002
    area = float(h.area(.5, sec=sec))
    clamp.amp = density * area * .01
    h.finitialize(-60.)
    voltage = -60.
    errors = []
    for k in range(200):
        t = k * cfg['dt_ms']
        injection = density if t < 10. else 0.
        voltage = voltage_step(np.array([voltage]), np.zeros((1,18)),
                               np.zeros((1,11)), np.array([injection]), cfg)[0]
        h.fadvance()
        errors.append(abs(float(segment.v)-voltage))
    limit = cfg['native_passive_only_calibration_limit_mv']
    return {'valid': bool(np.isfinite(errors).all() and max(errors) <= limit),
            'maximum_voltage_disagreement_mv': float(max(errors)),
            'limit_mv': limit, 'neuron_version': neuron.__version__,
            'scope': 'passive-only fixed-step 20ms, not active 11-channel native calibration'}


def summarize(reference_v: np.ndarray, reference_state: np.ndarray, design: dict,
              candidates: dict, cfg: dict, family: str, arm: str, seeds: list[int]) -> list[dict]:
    rows = []
    for seed_index, seed in enumerate(seeds):
        for episode_index, episode in enumerate(design['rows']):
            for horizon in cfg['horizons_ms']:
                k = int(round(horizon / cfg['dt_ms']))
                truth = reference_v[:k+1, episode_index]
                candidate = candidates['autonomous_voltage'][seed_index, :k+1, episode_index]
                own_state = candidates['autonomous_state'][seed_index, :k+1, episode_index]
                forced_state = candidates['teacher_voltage_state'][seed_index, :k+1, episode_index]
                direct_proposal = candidates['teacher_voltage_one_step_proposal'][seed_index, :k, episode_index]
                error = candidate-truth
                forced_error = direct_proposal-reference_v[1:k+1, episode_index]
                physical = cfg['physical_voltage_range_mv']
                row = {'family': family, 'arm': arm, 'seed': seed,
                       'episode_index': episode_index, **episode, 'horizon_ms': horizon,
                       'voltage_rmse_mv': float(np.sqrt(np.mean(error**2))),
                       'voltage_endpoint_error_mv': float(error[-1]),
                       'voltage_max_abs_error_mv': float(np.max(abs(error))),
                       'teacher_voltage_one_step_rmse_mv': float(np.sqrt(np.mean(forced_error**2))),
                       'gate_rmse': float(np.sqrt(np.mean((own_state-reference_state[:k+1, episode_index])**2))),
                       'teacher_voltage_gate_rmse': float(np.sqrt(np.mean((forced_state-reference_state[:k+1, episode_index])**2))),
                       'occupancy_violations': int(np.count_nonzero((own_state < 0) | (own_state > 1))),
                       'physical_voltage_violations': int(np.count_nonzero((candidate < physical[0]) | (candidate > physical[1]))),
                       'reference_threshold_crossings': int(np.count_nonzero((truth[:-1] < cfg['event_threshold_mv']) & (truth[1:] >= cfg['event_threshold_mv']))),
                       'candidate_threshold_crossings': int(np.count_nonzero((candidate[:-1] < cfg['event_threshold_mv']) & (candidate[1:] >= cfg['event_threshold_mv']))),
                       'finite': bool(np.isfinite(candidate).all() and np.isfinite(own_state).all())}
                rows.append(row)
    return rows


def run(root: Path, output: Path, cfg: dict, revision: str) -> dict:
    import torch
    verify_parent(root, cfg)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('Task30 frozen model evaluation requires CUDA')
    passive = native_passive_calibration(cfg)
    write(output / 'native_passive_solver_calibration.json', passive)
    if not passive['valid']:
        raise RuntimeError('Passive-only native solver calibration failed; no scientific inference')
    design = episodes(cfg)
    reference_v, reference_state = formula_reference(design, cfg)
    if not (np.isfinite(reference_v).all() and np.isfinite(reference_state).all()):
        raise RuntimeError('Formula control nonfinite')
    bounds = cfg['physical_voltage_range_mv']
    if np.any((reference_v < bounds[0]) | (reference_v > bounds[1])):
        raise RuntimeError('Formula control outside preregistered physical range')
    models, seeds = ionic.load_frozen(root, ionic.config(root), torch, 'cuda')
    if list(seeds) != cfg['model_seeds']:
        raise RuntimeError('Frozen model seeds changed')
    rows = []
    for family in cfg['families']:
        for arm in cfg['arms']:
            result = candidate_rollout(models[family, arm], torch, design, cfg,
                                       reference_v, reference_state, len(seeds))
            rows.extend(summarize(reference_v, reference_state, design, result,
                                  cfg, family, arm, seeds))
            print(f'[GIADA Task30] {family}/{arm}: {len(rows)} cumulative summaries', flush=True)
    primary = [r for r in rows if r['arm'] == 'both' and r['horizon_ms'] == cfg['primary_horizon_ms']]
    family_primary = {}
    for family in cfg['families']:
        family_rows = [r for r in primary if r['family'] == family]
        per_seed = {}
        for seed in seeds:
            subset = [r for r in family_rows if r['seed'] == seed]
            pooled = float(np.sqrt(np.mean([r['voltage_rmse_mv']**2 for r in subset])))
            worst = max(r['voltage_rmse_mv'] for r in subset)
            passed = bool(pooled <= cfg['primary_voltage_rmse_limit_mv']
                          and worst <= cfg['primary_worst_episode_voltage_rmse_limit_mv']
                          and all(r['finite'] and r['occupancy_violations'] == 0
                                  and r['physical_voltage_violations'] == 0 for r in subset))
            per_seed[str(seed)] = {'passed': passed, 'pooled_voltage_rmse_mv': pooled,
                                  'worst_episode_voltage_rmse_mv': worst}
        family_primary[family] = per_seed
    passed = all(row['passed'] for seeds_result in family_primary.values() for row in seeds_result.values())
    report = {'schema_version': 'giada-roadmap-task30-autonomous-voltage-v1',
              'valid': True, 'scientific_primary_passed': passed,
              'native_passive_solver_calibration': passive,
              'family_primary': family_primary, 'rows': rows,
              'formula_reference_voltage_range_mv': [float(reference_v.min()),float(reference_v.max())],
              'gate_d_performance_status': 'NO_GO_UNCHANGED',
              'task31_authorized': False, 'training_performed': False,
              'fresh_used_for_selection': False, 'autonomous_voltage': True,
              'autonomous_calcium': False, 'full_active_neuron_native_validated': False,
              'code_revision': revision, 'limits': cfg['limits']}
    write(output / 'final_report.json', report)
    return report
