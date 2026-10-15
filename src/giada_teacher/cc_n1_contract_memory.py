"""CC-01..04: cost, causal state collisions and controlled memory tails.

No learned model is trained. Native NEURON and torch run in isolated processes.
Clamp-conditioned states are interventions; recent-history collisions are exact
input identities, not nearest-neighbour distances or decoder failures.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from . import ionic_block_teacher_forced as ionic
from . import task30_autonomous_voltage as t30
from . import task32_dynamic_calcium_feedback as t32
from . import task33_observable_synaptic_feedback as t33
from . import task35_full_local_compartment as t35
from .hh_family_transfer import write


CONFIG = 'experiments/cc_n1_contract_memory_preregistration_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(root):
    spec = json.loads((root / CONFIG).read_text(encoding='utf-8'))
    if spec['roadmap_ids'] != ['CC-01', 'CC-02', 'CC-03', 'CC-04'] or spec['dt_ms'] != .1:
        raise ValueError('CC-N1 identity or step changed')
    for path, expected in spec['sources'].items():
        if sha(root / path) != expected:
            raise ValueError(f'CC-N1 immutable source mismatch: {path}')
    old = json.loads((root / 'experiments/results/task35_final_report.json').read_text())
    if not old['valid'] or not old['task35_passed']:
        raise ValueError('Task35 prerequisite not confirmed')
    _, cfg, calcium, c3 = t35.contract(root)
    base = t30.config(root, Path(calcium['base_config']).name)
    return spec, cfg, calcium, c3, base


def numpy_step(v, ca, gates, mult, injection, syn_g, area, cfg, calcium, base):
    next_g = ionic.exact_step(v, ca, gates, cfg['dt_ms'])
    next_ca = t32.calcium_step(ca, t32._ca_current(v, next_g, mult), calcium)
    t32._sk_update(next_g, gates, next_ca, cfg['dt_ms'])
    next_v = t33.voltage_step_with_synapses(v, next_g, mult, injection,
                                          syn_g, area, base, cfg['dt_ms'])
    return next_v, next_ca, next_g


def conditioned_state(prep_v, duration, tail, memory):
    ca = np.array([memory['initial_cai_mM']])
    common_v = np.array([memory['initial_voltage_mv']])
    gates = t30.initial_states(common_v, ca)
    gates = ionic.exact_step(np.array([prep_v]), ca, gates, duration)
    return ionic.exact_step(common_v, ca, gates, tail)[0]


def memory_design(spec):
    memory = spec['memory']
    rows, gates = [], []
    for tail, duration, prep_v, panel, current in itertools.product(
        memory['common_tail_ms'], memory['preparation_durations_ms'],
        memory['preparation_voltages_mv'], memory['panels'],
        memory['future_current_density_ma_cm2']):
        rows.append(dict(tail_ms=tail, preparation_ms=duration,
                         preparation_voltage_mv=prep_v, panel=panel,
                         future_current_ma_cm2=current))
        gates.append(conditioned_state(prep_v, duration, tail, memory))
    return rows, np.stack(gates)


def memory_rollout(rows, initial, spec, cfg, calcium, base):
    memory = spec['memory']
    count, ticks = len(rows), round(memory['rollout_ms'] / spec['dt_ms']) + 1
    v = np.full((ticks, count), memory['initial_voltage_mv'])
    ca = np.full_like(v, memory['initial_cai_mM'])
    gates = np.empty((ticks, count, 18)); gates[0] = initial
    mult = np.stack([ionic.panel_multipliers()[r['panel']] for r in rows])
    current = np.array([r['future_current_ma_cm2'] for r in rows])
    for k in range(ticks - 1):
        v[k+1], ca[k+1], gates[k+1] = numpy_step(
            v[k], ca[k], gates[k], mult, current, np.zeros((count, 4)),
            100 * np.pi, cfg, calcium, base)
    if not all(np.isfinite(x).all() for x in (v, ca, gates)):
        raise RuntimeError('Nonfinite memory continuation')
    return {'voltage': v, 'calcium': ca, 'gates': gates}


def collision_report(rows, rollout, spec):
    """Only compare pairs with exactly identical visible inputs and future forcing."""
    memory = spec['memory']; records = []
    for tail, panel, current in itertools.product(memory['common_tail_ms'],
                                                memory['panels'],
                                                memory['future_current_density_ma_cm2']):
        ids = [j for j, r in enumerate(rows) if r['tail_ms'] == tail
               and r['panel'] == panel and r['future_current_ma_cm2'] == current]
        assert len(ids) == 12
        assert np.ptp(rollout['voltage'][0, ids]) == 0
        assert np.ptp(rollout['calcium'][0, ids]) == 0
        for horizon in memory['horizons_ms']:
            tick = round(horizon / spec['dt_ms'])
            values = rollout['voltage'][tick, ids]
            low, high = ids[int(np.argmin(values))], ids[int(np.argmax(values))]
            v_spread = float(np.ptp(values))
            ca_spread = float(np.ptp(rollout['calcium'][tick, ids]))
            records.append({'tail_ms': tail, 'panel': panel,
                'future_current_ma_cm2': current, 'horizon_ms': horizon,
                'matching_recent_windows_ms': [w for w in memory['visible_history_windows_ms'] if w <= tail],
                'instantaneous_view_identical': True,
                'matched_recent_history_max_abs_difference': 0.,
                'voltage_endpoint_separation_mv': v_spread,
                'calcium_endpoint_separation_mM': ca_spread,
                'voltage_minimax_lower_bound_mv': v_spread / 2,
                'calcium_minimax_lower_bound_mM': ca_spread / 2,
                'voltage_extreme_indices': [low, high],
                'voltage_extreme_histories': [rows[low], rows[high]],
                'detectable': v_spread >= memory['detectable_endpoint_separation_mv']
                              or ca_spread >= memory['detectable_endpoint_separation_cai_mM']})
    return records


def analyze_memory(root, output):
    spec, cfg, calcium, _, base = contract(root)
    rows, initial = memory_design(spec)
    prediction = memory_rollout(rows, initial, spec, cfg, calcium, base)
    duplicate = memory_rollout(rows[:2], initial[:2].copy(), spec, cfg, calcium, base)
    duplicate_error = max(float(np.max(np.abs(duplicate[k] - prediction[k][:, :2])))
                          for k in prediction)
    reconstructed = np.stack([conditioned_state(r['preparation_voltage_mv'],
        r['preparation_ms'], r['tail_ms'], spec['memory']) for r in rows])
    summary_error = float(np.max(np.abs(reconstructed - initial)))
    # Swap only hidden gates. Every other initial value/input is constant in group.
    probe = next(j for j, r in enumerate(rows) if r['tail_ms'] == 0
                 and r['preparation_voltage_mv'] == -25 and r['preparation_ms'] == 200
                 and r['panel'] == 'canonical' and r['future_current_ma_cm2'] == 0)
    reference = next(j for j, r in enumerate(rows) if r['tail_ms'] == 0
                     and r['preparation_voltage_mv'] == -85 and r['preparation_ms'] == 200
                     and r['panel'] == 'canonical' and r['future_current_ma_cm2'] == 0)
    transplanted = memory_rollout([rows[probe]], initial[reference:reference+1],
                                  spec, cfg, calcium, base)
    transplant_error = max(float(np.max(np.abs(transplanted[k][:, 0] - prediction[k][:, reference])))
                           for k in ('voltage', 'calcium'))
    collision = collision_report(rows, prediction, spec)
    by_window = {}
    for window in [0.] + spec['memory']['visible_history_windows_ms']:
        same = [r for r in collision if r['tail_ms'] >= window]
        by_window[str(window)] = {
            'matched_groups': len(same), 'detected_groups': sum(r['detectable'] for r in same),
            'max_voltage_endpoint_separation_mv': max(r['voltage_endpoint_separation_mv'] for r in same),
            'max_calcium_endpoint_separation_mM': max(r['calcium_endpoint_separation_mM'] for r in same),
            'interpretation': 'Failure of this fixed recent V/Ca window only; absence of collision does not prove sufficiency.'}
    report = {'valid': max(duplicate_error, summary_error, transplant_error)
                         <= spec['memory']['technical_duplicate_atol'],
        'episode_count': len(rows), 'rows': rows, 'collisions': collision,
        'history_windows': by_window,
        'controls': {'duplicate_max_abs_error': duplicate_error,
                     'causal_summary_oracle_state_max_abs_error': summary_error,
                     'hidden_state_transplant_max_abs_error': transplant_error},
        'recent_history_collision_detected': any(r['detectable'] and r['matching_recent_windows_ms'] for r in collision),
        'instantaneous_collision_detected': any(r['detectable'] for r in collision),
        'latent_dimension_identified': False,
        'operational_memory_candidate': 'Persistent recurrent z initialized from physical state once or a separately validated causal warmup; keep synaptic states explicit initially.',
        'summary_oracle_is_deployable_learned_model': False,
        'limits': spec['memory']['absence_policy']}
    np.savez_compressed(output / 'memory_traces.npz', initial_gates=initial, **prediction)
    write(output / 'memory_analysis.json', report)
    print(f'[GIADA CC-N1] memory: {len(rows)} continuations, '
          f'{sum(r["detectable"] for r in collision)}/{len(collision)} detectable groups', flush=True)
    return report


def native_memory_case(h, row, initial, spec, cfg, calcium, base):
    """Canonical native continuation from a declared physical state intervention."""
    h.CVode().active(0); h.secondorder = 0; h.dt = spec['dt_ms']; h.celsius = cfg['temperature_c']
    sec = h.Section(name='giada_cc_memory'); sec.L = sec.diam = 10.; sec.nseg = 1
    sec.cm = base['cm_uf_cm2']; sec.insert('pas')
    sec.g_pas, sec.e_pas = base['g_pas_s_cm2'], base['e_pas_mv']
    for name in (*ionic.CHANNELS, 'CaDynamics_E2'): sec.insert(name)
    h.ion_style('ca_ion', 3, 1, 0, 0, 0, sec=sec)
    seg = sec(.5); area = float(h.area(.5, sec=sec))
    clamp = h.IClamp(seg); clamp.delay = 0.; clamp.dur = spec['memory']['rollout_ms'] + 1
    clamp.amp = row['future_current_ma_cm2'] * area * .01
    h.finitialize(spec['memory']['initial_voltage_mv'])
    seg.eca, seg.ena, seg.ek = cfg['eca_mv'], 55., -85.
    seg.cai = spec['memory']['initial_cai_mM']
    try:
        seg.cai_CaDynamics_E2 = spec['memory']['initial_cai_mM']
    except (AttributeError, LookupError):
        pass  # USEION cai is already set above; this matches the audited Task32 interface.
    cad = seg.CaDynamics_E2
    cad.decay, cad.gamma, cad.depth, cad.minCai = calcium['decay_ms'], calcium['gamma'], calcium['depth_um'], 1e-4
    mult = ionic.panel_multipliers()[row['panel']]
    for k, name in enumerate(ionic.CHANNELS):
        mech = getattr(seg, name); setattr(mech, 'g'+name+'bar', float(ionic.GBAR[k] * mult[k]))
        for q, state_name in enumerate(ionic.STATE_NAMES[k]):
            setattr(mech, state_name, float(initial[int(ionic.OFFSETS[k])+q]))
    h.fcurrent()
    ticks = round(spec['memory']['rollout_ms'] / spec['dt_ms']) + 1
    v, ca, gates = np.empty(ticks), np.empty(ticks), np.empty((ticks, 18))
    for n in range(ticks):
        v[n], ca[n] = float(seg.v), float(seg.cai)
        for k, name in enumerate(ionic.CHANNELS):
            for q, state_name in enumerate(ionic.STATE_NAMES[k]):
                gates[n, int(ionic.OFFSETS[k])+q] = float(getattr(getattr(seg, name), state_name))
        if n < ticks - 1: h.fadvance()
    h.delete_section(sec=sec)
    return {'voltage': v, 'calcium': ca, 'gates': gates}


def floor_pass(metrics, spec):
    g = spec['technical_gates']
    return bool(metrics['finite'] and metrics['physical_voltage_violations'] == 0
        and metrics['pooled_voltage_rmse_mv'] <= g['native_formula_pooled_voltage_rmse_mv']
        and metrics['worst_voltage_rmse_mv'] <= g['native_formula_worst_voltage_rmse_mv']
        and metrics['worst_cai_rmse_mM'] <= g['native_formula_worst_cai_rmse_mM'])


def native_phase(root, teacher, output, revision):
    import subprocess
    import neuron
    from neuron import h, load_mechanisms
    spec, cfg, calcium, c3, base = contract(root)
    if neuron.__version__.split('+')[0] != spec['neuron_version']:
        raise RuntimeError('Pinned native version mismatch')
    actual = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != spec['teacher_revision']: raise RuntimeError('Canonical teacher revision changed')
    t32.verify(root, teacher, calcium)
    build = t33.compile_combined(root, teacher, output)
    load_mechanisms(str(build.resolve()))
    memory = analyze_memory(root, output)
    rows, initial = memory_design(spec)
    with np.load(output / 'memory_traces.npz') as traces:
        memory_arrays = {k: traces[k] for k in ('voltage', 'calcium', 'gates')}
    anchors = []
    for j, row in enumerate(rows):
        if row['preparation_voltage_mv'] not in (-85., -25.) or row['preparation_ms'] != 200. \
                or row['tail_ms'] not in (0., 40.) or row['panel'] != 'canonical': continue
        native = native_memory_case(h, row, initial[j], spec, cfg, calcium, base)
        error_v = t32.rmse(native['voltage'], memory_arrays['voltage'][:, j])
        error_ca = t32.rmse(native['calcium'], memory_arrays['calcium'][:, j])
        anchors.append({'row_index': j, 'row': row, 'voltage_rmse_mv': error_v,
                        'cai_rmse_mM': error_ca,
                        'initial_gate_max_abs_error': float(np.max(np.abs(native['gates'][0]-initial[j]))),
                        'finite': all(np.isfinite(a).all() for a in native.values())})
    g = spec['technical_gates']
    anchor_valid = len(anchors) == 8 and all(r['finite'] and r['initial_gate_max_abs_error'] <= 1e-12
        and r['voltage_rmse_mv'] <= g['native_formula_worst_voltage_rmse_mv']
        and r['cai_rmse_mM'] <= g['native_formula_worst_cai_rmse_mM'] for r in anchors)
    write(output / 'native_memory_anchors.json', {'valid': anchor_valid, 'rows': anchors})
    if not anchor_valid or not memory['valid']:
        raise RuntimeError('Native memory anchor or duplicate control failed; no architecture verdict')

    cfg = {**cfg, 'schedules_ms': {'cc_n1': spec['native']['schedule_ms']}}
    fresh, shadows, summaries = [], [], []
    for panel, protocol in itertools.product(spec['native']['panels'], spec['native']['protocols']):
        row = {'panel': panel, 'protocol': protocol, 'schedule': 'cc_n1',
               'seed': spec['native']['seed'], 'initial_voltage_mv': spec['native']['initial_voltage_mv']}
        start = time.perf_counter(); native = t33._native_case(h, cfg, base, calcium, c3, row)
        native_seconds = time.perf_counter() - start
        start = time.perf_counter(); shadow = t33.shadow_case(h, cfg, c3, row, native)
        shadow_seconds = time.perf_counter() - start
        fresh.append(native); shadows.append(shadow)
        summaries.append({'row': row, 'native_wall_seconds': native_seconds,
                          'shadow_wall_seconds': shadow_seconds,
                          'shadow_passed': t33._shadow_pass(shadow, cfg)})
        print(f'[GIADA CC-N1] native full compartment {len(fresh)}/8', flush=True)
    predicted = t33.coupled_rollout(fresh, shadows, cfg, calcium, base)
    floors = {panel: t33.metrics({k: value[:, :, [i for i, r in enumerate(summaries)
              if r['row']['panel'] == panel], ...] for k, value in predicted.items()},
              [n for n, r in zip(fresh, summaries) if r['row']['panel'] == panel])
              for panel in spec['native']['panels']}
    zero = t33._native_case(h, cfg, base, calcium, c3, summaries[1]['row'], zero_weight=True)
    zero_shadow = t33.shadow_case(h, cfg, c3, summaries[1]['row'], zero, zero_weight=True)
    effect = float(np.max(np.abs(zero['voltage']-fresh[1]['voltage'])))
    controls = {'zero_weight_no_current': float(np.max(np.abs(zero['native_i']))) == 0,
        'zero_weight_no_release': zero_shadow['release_count'] == 0,
        'zero_weight_no_rng_advance': bool(np.array_equal(zero['rng_seq'][0], zero['rng_seq'][-1])),
        'synaptic_effect_present': effect >= g['negative_control_voltage_min_mv']}
    timed = []
    for _ in range(spec['native']['wall_repetitions']):
        start = time.perf_counter(); n = t33._native_case(h, cfg, base, calcium, c3, summaries[1]['row'])
        native_time = time.perf_counter()-start
        start = time.perf_counter(); t33.shadow_case(h, cfg, c3, summaries[1]['row'], n)
        timed.append({'native_wall_seconds': native_time, 'shadow_wall_seconds': time.perf_counter()-start})
    native_valid = all(r['shadow_passed'] for r in summaries) and all(controls.values()) \
                   and all(floor_pass(v, spec) for v in floors.values())
    np.savez_compressed(output / 'new_native_traces.npz',
        **{k: np.stack([r[k] for r in fresh]) for k in ('voltage', 'calcium', 'gates', 'injection', 'multipliers')},
        base_g_us=np.stack([r['base_g_us'] for r in shadows]), area_um2=np.array(fresh[0]['area_um2']))
    report = {'valid': bool(native_valid), 'code_revision': revision,
              'teacher_revision': actual, 'neuron_version': neuron.__version__,
              'native_memory_anchor_valid': bool(anchor_valid), 'floors_by_panel': floors,
              'native_rows': summaries, 'negative_controls': controls,
              'synaptic_voltage_effect_mv': effect, 'cpu_timings': timed,
              'cpu_timing_scope': spec['native']['timing_scope'],
              'gpu_authorized_by_technical_floor': bool(native_valid)}
    write(output / 'native_report.json', report)
    if not native_valid: raise RuntimeError('New native floor/interface failed; GPU phase blocked')
    return report


def torch_step(torch, v, ca, state, mult, injection, syn_g, area, cfg, calcium, base, model=None):
    """Resident complete membrane core; fixed iteration count for symmetric compile."""
    from .gate_d_compute_audit import exact_gpu, hybrid_gpu, GBAR_VALUES, OFFSETS, POWERS
    dt = torch.full_like(v, cfg['dt_ms'])
    next_g, _ = exact_gpu(v, ca, state, dt, torch) if model is None else hybrid_gpu(model, v, ca, state, dt, torch)
    gbar = torch.tensor(GBAR_VALUES, dtype=v.dtype, device=v.device) * mult
    ica = sum(gbar[..., k] * next_g[..., OFFSETS[k]].square() * next_g[..., OFFSETS[k]+1] * (v-120.) for k in (0, 1))
    persistence = np.exp(-cfg['dt_ms'] / calcium['decay_ms'])
    source = -10000 * ica * calcium['gamma'] / (2 * 96485.33212 * calcium['depth_um'])
    next_ca = 1e-4 + (ca-1e-4)*persistence + source*calcium['decay_ms']*(1-persistence)
    sk = int(ionic.OFFSETS[9])
    sk_inf = 1/(1+(.00043/next_ca.clamp_min(1e-7)).pow(4.8))
    sk_next = sk_inf + (state[..., sk]-sk_inf)*np.exp(-cfg['dt_ms'])
    next_g = torch.cat((next_g[..., :sk], sk_next[..., None], next_g[..., sk+1:]), -1)
    openings = []
    for k, names in enumerate(ionic.STATE_NAMES):
        opening = next_g[..., OFFSETS[k]].pow(POWERS[k])
        if len(names) == 2: opening = opening * next_g[..., OFFSETS[k]+1]
        openings.append(opening)
    ion_g = torch.stack(openings, -1) * gbar
    reversal = torch.tensor(tuple(float(x) for x in ionic.REVERSALS), dtype=v.dtype, device=v.device)
    syn_rev = torch.tensor((0., 0., -80., -97.), dtype=v.dtype, device=v.device)
    density = syn_g * (100./area)
    fixed = torch.cat((density[..., :1], torch.zeros_like(density[..., 1:2]), density[..., 2:]), -1)
    cap = 1e-3 * base['cm_uf_cm2'] / cfg['dt_ms']
    denom = cap + base['g_pas_s_cm2'] + ion_g.sum(-1) + fixed.sum(-1)
    numer = cap*v + base['g_pas_s_cm2']*base['e_pas_mv'] + (ion_g*reversal).sum(-1) \
            + (fixed*syn_rev).sum(-1) + injection
    next_v = numer / denom
    for _ in range(12):
        exponent = torch.exp((-.08*next_v).clamp(-80, 80))/3.57
        nmda = density[..., 1]/(1+exponent)
        derivative = .08*density[..., 1]*exponent/(1+exponent).square()
        next_v = next_v - (denom*next_v-numer+nmda*next_v)/(denom+nmda+derivative*next_v)
    return next_v, next_ca, next_g


def metric_arrays(prediction, arrays):
    result = {}
    for seed in range(prediction['voltage'].shape[0]):
        v_error = prediction['voltage'][seed] - arrays['voltage'].T
        ca_error = prediction['calcium'][seed] - arrays['calcium'].T
        result[str(seed)] = {'pooled_voltage_rmse_mv': float(np.sqrt(np.mean(v_error**2))),
            'per_episode_voltage_rmse_mv': np.sqrt(np.mean(v_error**2, axis=0)).tolist(),
            'worst_episode_voltage_rmse_mv': float(np.max(np.sqrt(np.mean(v_error**2, axis=0)))),
            'worst_episode_cai_rmse_mM': float(np.max(np.sqrt(np.mean(ca_error**2, axis=0))))}
    return result


def gpu_phase(root, output, revision):
    import torch
    spec, cfg, calcium, _, base = contract(root)
    native = json.loads((output / 'native_report.json').read_text())
    if not native['valid']: raise RuntimeError('Native gate not passed')
    if not torch.cuda.is_available(): raise RuntimeError('CC-N1 CUDA required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Three arms x three batch shapes share one Python code object. Preserve
    # all specializations instead of exhausting Dynamo's default cache of 8.
    torch._dynamo.config.cache_size_limit = max(32, torch._dynamo.config.cache_size_limit)
    models, seeds = ionic.load_frozen(root, ionic.config(root), torch, 'cuda')
    if list(seeds) != spec['gpu']['model_seeds']: raise RuntimeError('Frozen seed contract changed')
    with np.load(root / 'experiments/results/task35_native_shadow_traces.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    arrays = {k: torch.tensor(v, dtype=torch.float32, device='cuda') for k, v in data.items()}
    area = float(data['area_um2'])
    arms = {'formula': None, **{f: models[f, spec['gpu']['frozen_arm']] for f in spec['gpu']['families']}}
    historical_rows = [{k: data[k][j] for k in ('voltage', 'calcium', 'gates', 'injection', 'multipliers')}
                       | {'area_um2': area} for j in range(len(data['voltage']))]
    historical_shadows = [{'base_g_us': g} for g in data['base_g_us']]
    cpu_formula = t33.coupled_rollout(historical_rows, historical_shadows, cfg, calcium, base)
    accuracy, rows, numerical, compile_failures = {}, [], [], []
    # Same 32 historical episodes, all 3 frozen seeds, every tick autonomous.
    with torch.inference_mode():
        for name, model in arms.items():
            v = arrays['voltage'][:, 0][None].repeat(3, 1)
            ca = arrays['calcium'][:, 0][None].repeat(3, 1)
            gates = arrays['gates'][:, 0][None].repeat(3, 1, 1)
            mult = arrays['multipliers'][None].repeat(3, 1, 1)
            vv, cc = [v], [ca]
            for k in range(data['voltage'].shape[1]-1):
                v, ca, gates = torch_step(torch, v, ca, gates, mult,
                    arrays['injection'][:, k][None].expand(3, -1),
                    arrays['base_g_us'][:, k][None].expand(3, -1, -1),
                    area, cfg, calcium, base, model)
                vv.append(v); cc.append(ca)
            predicted = {'voltage': torch.stack(vv, 1).cpu().double().numpy(),
                         'calcium': torch.stack(cc, 1).cpu().double().numpy()}
            accuracy[name] = metric_arrays(predicted, data)
            start = time.perf_counter()
            legacy = cpu_formula if model is None else t33.coupled_rollout(
                historical_rows, historical_shadows, cfg, calcium, base, model, torch)
            legacy_seconds = time.perf_counter()-start
            error_v = float(np.max(np.abs(predicted['voltage'] - legacy['voltage'])))
            error_ca = float(np.max(np.abs(predicted['calcium'] - legacy['calcium'])))
            if error_v > spec['gpu']['torch_rollout_vs_legacy_voltage_atol_mv'] \
                    or error_ca > spec['gpu']['torch_rollout_vs_legacy_cai_atol_mM']:
                raise RuntimeError(f'GPU/legacy {name} rollout disagreement {error_v}, {error_ca}')
            numerical.append({'comparison': 'resident_vs_legacy_rollout', 'arm': name,
                              'voltage_max_abs_mv': error_v, 'calcium_max_abs_mM': error_ca,
                              'legacy_python_seconds_excluding_formula_precompute': legacy_seconds,
                              'timing_is_diagnostic_not_symmetric_benchmark': True})
            print(f'[GIADA CC-N1] resident accuracy {name}, all 32 historical cases', flush=True)
        write(output / 'historical_baseline_accuracy.json', accuracy)

        for batch in spec['gpu']['batch_sizes']:
            # Deterministic sampling from observed physical states, not arbitrary occupancy.
            rng = np.random.default_rng(90600 + batch)
            ep = rng.integers(0, len(data['voltage']), batch)
            tick = rng.integers(0, data['injection'].shape[1], batch)
            selected = [data['voltage'][ep, tick], data['calcium'][ep, tick], data['gates'][ep, tick],
                        data['multipliers'][ep], data['injection'][ep, tick], data['base_g_us'][ep, tick]]
            tensors = [torch.tensor(a, dtype=torch.float32, device='cuda')[None].repeat(3, *([1]*a.ndim)) for a in selected]
            tv, tc, ts, tm, ti, tg = tensors
            formula_np = numpy_step(*selected[:3], *selected[3:], area, cfg, calcium, base)
            for name, model in arms.items():
                def core(v, ca, state, m, inj, syn):
                    return torch_step(torch, v, ca, state, m, inj, syn, area, cfg, calcium, base, model)
                eager = core(*tensors)
                if not all(torch.isfinite(a).all().item() for a in eager):
                    raise RuntimeError(f'Nonfinite GPU output {name}')
                if name == 'formula':
                    diffs = [float(np.max(np.abs(a[0].cpu().double().numpy()-b))) for a, b in zip(eager, formula_np)]
                    limits = [spec['gpu']['formula_gpu_vs_numpy_voltage_atol_mv'],
                              spec['gpu']['formula_gpu_vs_numpy_cai_atol_mM'], spec['gpu']['formula_gpu_vs_numpy_gate_atol']]
                    if any(a > b for a, b in zip(diffs, limits)):
                        raise RuntimeError(f'GPU formula one-step mismatch: {diffs}')
                    numerical.append({'batch': batch, 'comparison': 'formula_gpu_numpy', 'max_abs_v_ca_gates': diffs})
                compile_seconds = None
                try:
                    start = time.perf_counter()
                    compiled = torch.compile(core, backend='inductor', fullgraph=True)
                    candidate = compiled(*tensors); torch.cuda.synchronize()
                    compile_seconds = time.perf_counter()-start
                except Exception as error:
                    compile_failures.append({'arm': name, 'batch': batch, 'error': str(error)[:2000]})
                    compiled = None
                if compiled is not None:
                    diffs = [float((a-b).abs().max().item()) for a, b in zip(eager, candidate)]
                    limits = [spec['gpu']['compiled_vs_eager_voltage_atol_mv'],
                              spec['gpu']['compiled_vs_eager_cai_atol_mM'], spec['gpu']['compiled_vs_eager_gate_atol']]
                    if not np.isfinite(diffs).all() or any(a > b for a, b in zip(diffs, limits)):
                        raise RuntimeError(f'Compiled/eager mismatch {name}: {diffs}')
                    numerical.append({'batch': batch, 'arm': name, 'comparison': 'compiled_eager', 'max_abs_v_ca_gates': diffs})
                for mode, fn in [('eager', core)] + ([('compiled', compiled)] if compiled is not None else []):
                    for _ in range(spec['gpu']['warmup']): fn(*tensors)
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(spec['gpu']['repetitions']):
                        # Block-wall latency includes launches/synchronization. Inputs and outputs resident.
                        torch.cuda.synchronize(); start = time.perf_counter()
                        v, ca, state = tv, tc, ts
                        for _ in range(spec['gpu']['rollout_steps_per_timing']):
                            v, ca, state = fn(v, ca, state, tm, ti, tg)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter()-start)*1000/spec['gpu']['rollout_steps_per_timing'])
                    count = sum(p.numel() for p in model.parameters())//3 if model is not None else 0
                    weight_macs = sum(p.numel() for n, p in model.named_parameters() if n.startswith('w'))//3 if model is not None else 0
                    profile = None
                    if batch == 642 and mode == 'compiled':
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                              torch.profiler.ProfilerActivity.CUDA]) as prof:
                            fn(*tensors); torch.cuda.synchronize()
                        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                        counts = {}
                        for event in events: counts[event.name] = counts.get(event.name, 0)+1
                        profile = {'cuda_event_count': len(events),
                                   'events_by_name': counts, 'selection_eligible': False,
                                   'scope': 'one compiled core step; profiler overhead excluded from latency'}
                    rows.append({'arm': name, 'batch_per_seed': batch, 'seed_count': 3, 'mode': mode,
                        'parameter_count_per_seed': count,
                        'neural_dense_macs_per_compartment_step': weight_macs,
                        'macs_exclude_activations_analytic_rates_and_solver': True,
                        'profile': profile,
                        'persistent_core_state_values_per_compartment': 20,
                        'persistent_core_state_bytes_float32': 80,
                        'synaptic_state_excluded_from_80_bytes': True,
                        'median_wall_ms_per_step_ensemble': float(np.median(times)),
                        'p95_wall_ms_per_step_ensemble': float(np.quantile(times, .95)),
                        'ns_per_compartment_step': float(np.median(times)*1e6/(3*batch)),
                        'ms_wall_per_ms_biological_for_ensemble': float(np.median(times)/spec['dt_ms']),
                        'compile_first_call_seconds': compile_seconds if mode == 'compiled' else None,
                        'timings_ms': times})
                write(output / 'gpu_cost_partial.json', {'rows': rows, 'compile_failures': compile_failures})
                print(f'[GIADA CC-N1] cost {name} batch={batch} compiled={compiled is not None}', flush=True)
    report = {'valid': not compile_failures, 'device': torch.cuda.get_device_name(),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'code_revision': revision, 'rows': rows, 'numerical_checks': numerical,
        'compile_failures': compile_failures, 'accuracy': accuracy,
        'core_timing_scope': spec['gpu']['scope'], 'speedup_claim': False,
        'no_parameter_count_speed_inference': True}
    write(output / 'gpu_report.json', report)
    return report


def finalize(root, output, revision):
    spec, *_ = contract(root)
    native = json.loads((output / 'native_report.json').read_text())
    gpu = json.loads((output / 'gpu_report.json').read_text())
    memory = json.loads((output / 'memory_analysis.json').read_text())
    valid = native['valid'] and gpu['valid'] and memory['valid']
    report = {'schema_version': spec['schema_version'], 'roadmap_ids': spec['roadmap_ids'],
        'valid': bool(valid), 'code_revision': revision, 'preregistration_sha256': sha(root / CONFIG),
        'source_sha256': spec['sources'], 'training_performed': False, 'fresh_used_for_selection': False,
        'cc01': {'contract_recorded': True, 'path': 'contract.json'},
        'cc02': {'native_anchor_valid': native['valid'], 'cost_map_valid': gpu['valid'],
                 'scope': spec['gpu']['scope']},
        'cc03': {'instantaneous_collision_detected': memory['instantaneous_collision_detected']},
        'cc04': {'recent_history_collision_detected': memory['recent_history_collision_detected'],
                 'history_windows': memory['history_windows'],
                 'operational_memory_candidate': memory['operational_memory_candidate'],
                 'minimum_latent_dimension_proved': False},
        'diagnosis': 'N1_DIAGNOSTIC_COMPLETE' if valid else 'N1_COST_OR_TECHNICAL_GATE_INCOMPLETE',
        'n2_preparation_recommended': bool(valid), 'n2_run_authorized': False,
        'task36_authorized_by_this_run': False, 'new_model_promoted': False, 'speedup_claim': False}
    write(output / 'final_report.json', report)
    return report
