"""Task33: causal stochastic synapses in an autonomous active compartment.

The native teacher and the analytic synaptic shadow run in one process. Frozen
Task32 neural checkpoints are evaluated later in an isolated CUDA process.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from . import ionic_block_teacher_forced as ionic
from . import task30_autonomous_voltage as t30
from . import task30c_native_active_confirmation as t30c
from . import task32_dynamic_calcium_feedback as t32
from .hh_family_transfer import write
from .iv_c1_deterministic_synapse import normalization
from .iv_c2_stochastic_release import _rng, shadow_step


RECEPTORS = ('AMPA', 'NMDA', 'GABAA', 'GABAB')
REVERSALS = np.array([0., 0., -80., -97.])
AB_NAMES = tuple(f'{prefix}_{receptor}' for receptor in RECEPTORS for prefix in ('A', 'B'))


def contract(root: Path) -> tuple[dict, dict, dict]:
    base_path = root / 'experiments/task33_observable_synaptic_feedback_preregistration.json'
    base = json.loads(base_path.read_text())
    amendment = json.loads((root / 'experiments/task33_observable_synaptic_feedback_v2.json').read_text())
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != amendment['base_contract_sha256']:
        raise RuntimeError('Task33 v1 preregistration changed')
    for file_key, hash_key in (('prior_calibration', 'prior_calibration_sha256'),
                               ('prior_final_report', 'prior_final_sha256')):
        path = root / amendment[file_key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != amendment[hash_key]:
            raise RuntimeError(f'Task33 v1 diagnostic changed: {file_key}')
    previous = json.loads((root / amendment['prior_final_report']).read_text())
    if previous['diagnosis'] != 'CALIBRATION_IMPLEMENTATION_FAILURE' or previous['confirmation_opened'] or previous['model_judged']:
        raise RuntimeError('Task33 v1 decision status changed')
    v2 = {**base, **amendment['overrides'], 'schema_version': amendment['schema_version']}
    v2_path = root / 'experiments/task33_observable_synaptic_feedback_v2.json'
    v3 = json.loads((root / 'experiments/task33_observable_synaptic_feedback_v3.json').read_text())
    if hashlib.sha256(v2_path.read_bytes()).hexdigest() != v3['parent_v2_contract_sha256']:
        raise RuntimeError('Task33 v2 preregistration changed')
    for path_key, hash_key in (('parent_v2_report', 'parent_v2_report_sha256'),
                               ('parent_v2_traces', 'parent_v2_traces_sha256')):
        if hashlib.sha256((root / v3[path_key]).read_bytes()).hexdigest() != v3[hash_key]:
            raise RuntimeError(f'Task33 v2 evidence changed: {path_key}')
    previous_v2 = json.loads((root / v3['parent_v2_report']).read_text())
    if not (previous_v2['valid'] and previous_v2['confirmation_opened']
            and previous_v2['shadow_passed'] and not previous_v2['native_floor_admissible']
            and not previous_v2['model_judged']):
        raise RuntimeError('Task33 v2 floor diagnosis changed')
    cfg = {**v2, **v3['overrides'], 'schema_version': v3['schema_version']}
    if cfg['synaptic_state_phase'] != 'old':
        raise RuntimeError('Task33 v3 causal synaptic phase changed')
    for path_key, hash_key, required in (
        ('parent_task32_report', 'parent_task32_sha256', 'scientific_primary_passed'),
        ('parent_iv_c3_report', 'parent_iv_c3_sha256', 'iv_c3_passed')):
        path = root / cfg[path_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cfg[hash_key]
        assert json.loads(path.read_text())[required]
    if cfg['roadmap_id'] != 'Task33' or cfg['dt_ms'] != .1 or cfg['duration_ms'] != 40.:
        raise RuntimeError('Task33 registered domain changed')
    t32cfg = t32.load_v3_contract(root)
    ivc3 = json.loads((root / cfg['synapse_parameters_source']).read_text())
    if len(ivc3['synapses']) != 4 or len(cfg['random123_streams']) != 4:
        raise RuntimeError('Task33 synapse inventory changed')
    cal = cfg['schedules_ms'][cfg['calibration']['schedule']]
    opened = set(time for events in cal.values() for time in events)
    for label in cfg['confirmation']['schedules']:
        schedule = cfg['schedules_ms'][label]
        if set(time for events in schedule.values() for time in events) & opened:
            raise RuntimeError('Task33 calibration and confirmation event times overlap')
    return cfg, t32cfg, ivc3


def compile_combined(root: Path, teacher: Path, output: Path) -> Path:
    source = teacher / 'L5PC_NEURON_simulation/mods'
    build = output / 'compiled_active_synapses'
    build.mkdir(parents=True, exist_ok=False)
    inventory = json.loads((root / 'experiments/teacher_mechanism_inventory_v1.json').read_text())
    channel_hashes = {row['mechanism']['name']: row['sha256'] for row in inventory['mechanisms']}
    synapse_hashes = json.loads((root / 'experiments/iv_c1_deterministic_synapse_preregistration.json').read_text())['mechanisms']
    for name in (*ionic.CHANNELS, 'CaDynamics_E2', *synapse_hashes):
        path = source / f'{name}.mod'
        raw = path.read_bytes()
        if name in synapse_hashes:
            canonical = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            expected = synapse_hashes[name]
        else:
            canonical = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            expected = channel_hashes[name]
        if hashlib.sha256(canonical).hexdigest() != expected:
            raise RuntimeError(f'Canonical NMODL hash mismatch: {name}')
        shutil.copy2(path, build / path.name)
    compiler = shutil.which('nrnivmodl') or str(Path(sys.executable).parent / 'nrnivmodl')
    if not Path(compiler).is_file():
        raise RuntimeError('nrnivmodl unavailable')
    with (build / 'compile.log').open('w', encoding='utf-8') as log:
        subprocess.run([compiler, str(build.resolve())], cwd=build,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return build


def cases(cfg: dict) -> list[dict]:
    return [{'schedule': schedule, 'seed': seed, 'initial_voltage_mv': v, 'protocol': protocol}
            for schedule in cfg['confirmation']['schedules']
            for seed in cfg['confirmation']['seeds']
            for v in cfg['confirmation']['initial_voltages_mv']
            for protocol in cfg['confirmation']['protocols']]


def _native_case(h, cfg: dict, base: dict, t32cfg: dict, c3: dict,
                 row: dict, zero_weight: bool = False) -> dict:
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = cfg['dt_ms']
    h.celsius = cfg['temperature_c']
    sec = h.Section(name='giada_task33')
    sec.L = sec.diam = 10.
    sec.nseg = 1
    sec.cm = base['cm_uf_cm2']
    sec.insert('pas')
    sec.g_pas = base['g_pas_s_cm2']
    sec.e_pas = base['e_pas_mv']
    for name in (*ionic.CHANNELS, 'CaDynamics_E2'):
        sec.insert(name)
    h.ion_style('ca_ion', 3, 1, 0, 0, 0, sec=sec)
    seg = sec(.5)
    area = float(h.area(.5, sec=sec))
    multipliers = ionic.panel_multipliers()[row.get('panel', 'canonical')].copy()
    initial_v = float(row['initial_voltage_mv'])
    initial_gate = t30.initial_states(np.array([initial_v]),
                                      np.array([cfg['initial_cai_mM']]))[0]
    time = np.arange(round(cfg['duration_ms']/cfg['dt_ms'])+1) * cfg['dt_ms']
    injection = t30._injection(row['protocol'], time[:-1])
    clamps = []
    for start, duration, density in t30c._pulses(injection, cfg['dt_ms']):
        clamp = h.IClamp(seg)
        clamp.delay, clamp.dur, clamp.amp = start, duration, density * area * .01
        clamps.append(clamp)
    synapses = []
    schedule = cfg['schedules_ms'][row['schedule']]
    for index, params in enumerate(c3['synapses']):
        syn = getattr(h, params['mechanism'])(seg)
        for field in ('Use', 'Fac', 'Dep'):
            setattr(syn, field, params[field])
        if params['mechanism'] == 'ProbAMPANMDA_EMS':
            syn.NMDA_ratio = c3['nmda_ratio']
        else:
            syn.GABAB_ratio = c3['gabab_ratio']
        rng = _rng(h, row['seed'], cfg['random123_streams'][index])
        syn.setRNG(rng)
        con = h.NetCon(None, syn)
        con.delay = 0
        con.weight[0] = 0. if zero_weight else cfg['synapse_weight_ns']
        synapses.append((params, syn, rng, con))
    h.finitialize(initial_v)
    seg.eca, seg.ena, seg.ek = cfg['eca_mv'], 55., -85.
    seg.cai = cfg['initial_cai_mM']
    try:
        seg.cai_CaDynamics_E2 = cfg['initial_cai_mM']
    except (AttributeError, LookupError):
        pass
    cad = seg.CaDynamics_E2
    cad.decay, cad.gamma, cad.depth = t32cfg['decay_ms'], t32cfg['gamma'], t32cfg['depth_um']
    cad.minCai = 1e-4
    for k, name in enumerate(ionic.CHANNELS):
        mech = getattr(seg, name)
        setattr(mech, 'g'+name+'bar', float(ionic.GBAR[k] * multipliers[k]))
        for q, state_name in enumerate(ionic.STATE_NAMES[k]):
            setattr(mech, state_name, float(initial_gate[ionic.OFFSETS[k]+q]))
    h.fcurrent()
    initial_plastic = [{'u': float(syn.u), 'Rstate': int(syn.Rstate),
                        'tsyn_fac': float(syn.tsyn_fac), 'tsyn': float(con.weight[4]),
                        'released': False} for _, syn, _, con in synapses]
    initial_rng = [int(rng.seq()) for _, _, rng, _ in synapses]
    for params, _, _, con in synapses:
        for event in schedule[params['id']]:
            con.event(float(event))
    parameters = {}
    for receptor in RECEPTORS:
        sample = next(syn for params, syn, _, _ in synapses
                      if (receptor in ('AMPA', 'NMDA')) == (params['mechanism'] == 'ProbAMPANMDA_EMS'))
        rise = float(getattr(sample, f'tau_r_{receptor}'))
        decay = float(getattr(sample, f'tau_d_{receptor}'))
        parameters[receptor] = (rise, decay, normalization(rise, decay))
    steps = len(time)
    voltage = np.empty(steps)
    calcium = np.empty(steps)
    gates = np.empty((steps, 18))
    clock = np.empty(steps)
    native_ab = np.zeros((steps, len(synapses), len(AB_NAMES)))
    native_g = np.zeros((steps, len(RECEPTORS)))
    native_i = np.zeros_like(native_g)
    plastic = np.empty((steps, len(synapses), 4))
    rng_seq = np.empty((steps, len(synapses)), dtype=np.int64)
    for n in range(steps):
        h.fcurrent()
        clock[n] = float(h.t)
        voltage[n], calcium[n] = float(seg.v), float(seg.cai)
        for k, name in enumerate(ionic.CHANNELS):
            mech = getattr(seg, name)
            for q, state_name in enumerate(ionic.STATE_NAMES[k]):
                gates[n, ionic.OFFSETS[k]+q] = float(getattr(mech, state_name))
        for j, (params, syn, rng, con) in enumerate(synapses):
            plastic[n, j] = [float(syn.u), float(syn.Rstate),
                             float(syn.tsyn_fac), float(con.weight[4])]
            rng_seq[n, j] = int(rng.seq())
            active = ('AMPA', 'NMDA') if params['mechanism'] == 'ProbAMPANMDA_EMS' else ('GABAA', 'GABAB')
            for receptor in active:
                r = RECEPTORS.index(receptor)
                native_g[n, r] += float(getattr(syn, f'g_{receptor}'))
                native_i[n, r] += float(getattr(syn, f'i_{receptor}'))
                for prefix in ('A', 'B'):
                    native_ab[n, j, AB_NAMES.index(f'{prefix}_{receptor}')] = float(getattr(syn, f'{prefix}_{receptor}'))
        if n < steps - 1:
            h.fadvance()
    h.delete_section(sec=sec)
    return {'clock': clock, 'voltage': voltage, 'calcium': calcium, 'gates': gates,
            'native_ab': native_ab, 'native_g': native_g, 'native_i': native_i,
            'plastic': plastic, 'rng_seq': rng_seq, 'initial_plastic': initial_plastic,
            'initial_rng': initial_rng, 'parameters': parameters, 'area_um2': area,
            'injection': injection, 'multipliers': multipliers}


def shadow_case(h, cfg: dict, c3: dict, row: dict, native: dict,
                zero_weight: bool = False) -> dict:
    """Independent online plasticity/receptor replay; native V is audit-only."""
    synapses = c3['synapses']
    shadow_rng = [_rng(h, row['seed'], cfg['random123_streams'][j])
                  for j in range(len(synapses))]
    states = [dict(item) for item in native['initial_plastic']]
    if any(int(rng.seq()) != native['initial_rng'][j]
           for j, rng in enumerate(shadow_rng)):
        raise RuntimeError('Initial native and shadow RNG positions differ')
    schedule = cfg['schedules_ms'][row['schedule']]
    cursors = [0] * len(synapses)
    previous = np.zeros((len(synapses), len(AB_NAMES)))
    prior_native_a = np.zeros(len(synapses))
    base_g = np.zeros((len(native['clock']), len(RECEPTORS)))
    predicted_ab = np.zeros_like(native['native_ab'])
    predicted_i = np.zeros_like(native['native_i'])
    max_ab = max_plastic = max_current = max_rng = 0.
    releases = mismatches = 0
    scheduled = sum(len(events) for events in schedule.values())
    previous_clock = 0.
    for n, clock in enumerate(native['clock']):
        elapsed = clock - previous_clock
        for j, params in enumerate(synapses):
            active = ('AMPA', 'NMDA') if params['mechanism'] == 'ProbAMPANMDA_EMS' else ('GABAA', 'GABAB')
            for receptor in active:
                rise, decay, _ = native['parameters'][receptor]
                previous[j, AB_NAMES.index(f'A_{receptor}')] *= math.exp(-elapsed / rise)
                previous[j, AB_NAMES.index(f'B_{receptor}')] *= math.exp(-elapsed / decay)
            released_now = 0
            times = schedule[params['id']]
            while cursors[j] < len(times) and times[cursors[j]] <= clock:
                event = times[cursors[j]]
                cursors[j] += 1
                if zero_weight:
                    continue
                states[j], _ = shadow_step(states[j], event, params, shadow_rng[j].repick)
                if states[j]['released']:
                    releases += 1
                    released_now += 1
                    for receptor in active:
                        rise, decay, factor = native['parameters'][receptor]
                        ratio = c3['nmda_ratio'] if receptor == 'NMDA' else (
                            c3['gabab_ratio'] if receptor == 'GABAB' else 1.)
                        amount = cfg['synapse_weight_ns'] * ratio * factor
                        previous[j, AB_NAMES.index(f'A_{receptor}')] += amount * math.exp(-(clock-event)/rise)
                        previous[j, AB_NAMES.index(f'B_{receptor}')] += amount * math.exp(-(clock-event)/decay)
            primary = active[0]
            rise, _, factor = native['parameters'][primary]
            actual_a = native['native_ab'][n, j, AB_NAMES.index(f'A_{primary}')]
            witnessed = int(actual_a - prior_native_a[j] * math.exp(-elapsed/rise) >
                            cfg['synapse_weight_ns'] * factor * .25) if not zero_weight else 0
            mismatches += int(released_now != witnessed)
            prior_native_a[j] = actual_a
            max_ab = max(max_ab, float(np.max(np.abs(previous[j]-native['native_ab'][n, j]))))
            actual_state = native['plastic'][n, j]
            max_plastic = max(max_plastic, *(abs(actual_state[q]-states[j][key]) for q, key in
                                            enumerate(('u', 'Rstate', 'tsyn_fac', 'tsyn'))))
            max_rng = max(max_rng, abs(native['rng_seq'][n, j]-int(shadow_rng[j].seq())))
            predicted_ab[n, j] = previous[j]
        for r, receptor in enumerate(RECEPTORS):
            base_g[n, r] = c3['gmax_us_per_ns'] * sum(
                previous[j, AB_NAMES.index(f'B_{receptor}')]
                - previous[j, AB_NAMES.index(f'A_{receptor}')]
                for j in range(len(synapses)))
        voltage = float(native['voltage'][n])
        g = receptor_conductance(base_g[n], voltage)
        predicted_i[n] = g * (voltage-REVERSALS)
        max_current = max(max_current, float(np.max(np.abs(predicted_i[n]-native['native_i'][n]))))
        previous_clock = clock
    return {'base_g_us': base_g, 'predicted_ab': predicted_ab,
            'predicted_i': predicted_i, 'release_count': releases,
            'scheduled_count': scheduled, 'release_mismatch_count': mismatches,
            'max_shadow_receptor_state_error': max_ab,
            'max_plastic_state_error': max_plastic,
            'max_rng_sequence_difference': max_rng,
            'max_shadow_current_error_na': max_current}


def receptor_conductance(base_g_us: np.ndarray, voltage_mv: np.ndarray) -> np.ndarray:
    g = np.broadcast_to(base_g_us, np.shape(voltage_mv) + (len(RECEPTORS),)).copy()
    g[..., 1] /= 1 + np.exp(np.clip(-.08 * voltage_mv, -700, 700))/3.57
    return g


def voltage_step_with_synapses(old_v: np.ndarray, next_state: np.ndarray,
                               multipliers: np.ndarray, injected: np.ndarray,
                               base_g_us: np.ndarray, area_um2: float,
                               base: dict, dt_ms: float) -> np.ndarray:
    """Semi-implicit voltage solve with an implicit NMDA magnesium block."""
    ion_g = t30.conductances(next_state, multipliers)
    cap = 1e-3 * base['cm_uf_cm2'] / dt_ms
    ion_sum = np.sum(ion_g, axis=-1)
    ion_rhs = np.sum(ion_g * ionic.REVERSALS, axis=-1)
    density = np.broadcast_to(base_g_us, old_v.shape + (len(RECEPTORS),)) * (100./area_um2)
    fixed_density = density.copy()
    fixed_density[..., 1] = 0.
    fixed_sum = np.sum(fixed_density, axis=-1)
    fixed_rhs = np.sum(fixed_density * REVERSALS, axis=-1)
    denominator = cap + base['g_pas_s_cm2'] + ion_sum + fixed_sum
    numerator = cap*old_v + base['g_pas_s_cm2']*base['e_pas_mv'] + ion_rhs + fixed_rhs + injected
    v = numerator / denominator
    nmda_base = density[..., 1]
    for _ in range(12):
        exponent = np.exp(np.clip(-.08*v, -700, 700))/3.57
        g_nmda = nmda_base/(1+exponent)
        dg_nmda = .08*nmda_base*exponent/(1+exponent)**2
        residual = denominator*v - numerator + g_nmda*v
        delta = residual/(denominator + g_nmda + dg_nmda*v)
        v -= delta
        if np.max(np.abs(delta)) < 1e-11:
            break
    if not np.isfinite(v).all():
        raise RuntimeError('Task33 voltage solve became nonfinite')
    return v


def coupled_rollout(native_rows: list[dict], shadow_rows: list[dict],
                    cfg: dict, t32cfg: dict, base: dict,
                    model=None, torch=None,
                    synaptic_state_phase: str | None = None) -> dict:
    count = len(native_rows)
    steps = len(native_rows[0]['voltage'])
    seeds = len(t32cfg['model_seeds']) if model is not None else 1
    voltage = np.empty((seeds, steps, count))
    calcium = np.empty_like(voltage)
    states = np.empty((seeds, steps, count, 18))
    voltage[:, 0] = [row['voltage'][0] for row in native_rows]
    calcium[:, 0] = cfg['initial_cai_mM']
    states[:, 0] = t30.initial_states(voltage[0, 0], calcium[0, 0])
    multipliers = np.broadcast_to(
        np.stack([row['multipliers'] for row in native_rows])[None, :, :],
        (seeds, count, len(ionic.CHANNELS)))
    area = native_rows[0]['area_um2']
    phase = synaptic_state_phase or cfg.get('synaptic_state_phase', 'next')
    if phase not in ('old', 'next'):
        raise ValueError(f'Unknown synaptic state phase: {phase}')
    for k in range(steps-1):
        if model is None:
            next_state = ionic.exact_step(voltage[:, k], calcium[:, k], states[:, k], cfg['dt_ms'])
        else:
            next_state = t30._candidate_gate_step(model, torch, voltage[:, k], calcium[:, k],
                                                   states[:, k], cfg['dt_ms'])
        source = t32._ca_current(voltage[:, k], next_state, multipliers)
        calcium[:, k+1] = t32.calcium_step(calcium[:, k], source, t32cfg)
        t32._sk_update(next_state, states[:, k], calcium[:, k+1], cfg['dt_ms'])
        states[:, k+1] = next_state
        conductance = np.stack([shadow['base_g_us'][k if phase == 'old' else k+1]
                                for shadow in shadow_rows])
        injected = np.array([row['injection'][k] for row in native_rows])
        voltage[:, k+1] = voltage_step_with_synapses(
            voltage[:, k], next_state, multipliers, injected,
            conductance, area, base, cfg['dt_ms'])
        if not (np.isfinite(voltage[:, k+1]).all() and np.isfinite(calcium[:, k+1]).all()):
            raise RuntimeError(f'Task33 nonfinite rollout at step {k+1}')
    return {'voltage': voltage, 'calcium': calcium, 'gates': states}


def metrics(prediction: dict, native_rows: list[dict], seed_index: int = 0) -> dict:
    voltage_errors = []
    calcium_errors = []
    for j, row in enumerate(native_rows):
        voltage_errors.append(t32.rmse(prediction['voltage'][seed_index, :, j], row['voltage']))
        calcium_errors.append(t32.rmse(prediction['calcium'][seed_index, :, j], row['calcium']))
    state = prediction['gates'][seed_index]
    voltage = prediction['voltage'][seed_index]
    return {'pooled_voltage_rmse_mv': t32.rmse(voltage_errors, np.zeros(len(voltage_errors))),
            'worst_voltage_rmse_mv': max(voltage_errors),
            'worst_cai_rmse_mM': max(calcium_errors),
            'finite': bool(np.isfinite(voltage).all() and np.isfinite(state).all()),
            'occupancy_violations': int(np.count_nonzero((state < 0) | (state > 1))),
            'physical_voltage_violations': int(np.count_nonzero((voltage < -140) | (voltage > 100))),
            'per_episode_voltage_rmse_mv': voltage_errors}


def _shadow_pass(shadow: dict, cfg: dict) -> bool:
    return bool(shadow['release_mismatch_count'] <= cfg['gates']['max_release_mismatch_count']
                and shadow['max_rng_sequence_difference'] <= cfg['gates']['max_rng_sequence_difference']
                and shadow['max_shadow_receptor_state_error'] <= cfg['gates']['max_shadow_receptor_state_error']
                and shadow['max_shadow_current_error_na'] <= cfg['gates']['max_shadow_current_error_na'])


def _floor_pass(result: dict, cfg: dict) -> bool:
    return bool(result['finite'] and result['occupancy_violations'] == 0
                and result['physical_voltage_violations'] == 0
                and result['pooled_voltage_rmse_mv'] <= cfg['gates']['max_formula_floor_pooled_voltage_rmse_mv']
                and result['worst_voltage_rmse_mv'] <= cfg['gates']['max_formula_floor_worst_voltage_rmse_mv']
                and result['worst_cai_rmse_mM'] <= cfg['gates']['max_formula_floor_worst_cai_rmse_mM'])


def _candidate_pass(result: dict, cfg: dict) -> bool:
    return bool(result['finite'] and result['occupancy_violations'] == 0
                and result['physical_voltage_violations'] == 0
                and result['pooled_voltage_rmse_mv'] <= cfg['gates']['max_candidate_pooled_voltage_rmse_mv']
                and result['worst_voltage_rmse_mv'] <= cfg['gates']['max_candidate_worst_voltage_rmse_mv']
                and result['worst_cai_rmse_mM'] <= cfg['gates']['max_candidate_worst_cai_rmse_mM'])


def run_native(root: Path, teacher: Path, output: Path, revision: str) -> dict:
    import neuron
    from neuron import h, load_mechanisms
    cfg, t32cfg, c3 = contract(root)
    if neuron.__version__.split('+')[0] != cfg['neuron_version']:
        raise RuntimeError(f"Task33 NEURON version mismatch: {neuron.__version__}")
    actual_teacher = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_teacher != cfg['teacher_revision']:
        raise RuntimeError('Task33 canonical teacher revision changed')
    base = t32.verify(root, teacher, t32cfg)
    build = compile_combined(root, teacher, output)
    load_mechanisms(str(build.resolve()))
    cal_row = {key: cfg['calibration'][key] for key in ('seed', 'initial_voltage_mv', 'protocol')}
    cal_row['schedule'] = cfg['calibration']['schedule']
    cal_native = _native_case(h, cfg, base, t32cfg, c3, cal_row)
    cal_shadow = shadow_case(h, cfg, c3, cal_row, cal_native)
    cal_formula = coupled_rollout([cal_native], [cal_shadow], cfg, t32cfg, base)
    cal_floor = metrics(cal_formula, [cal_native])
    np.savez_compressed(output / 'calibration_diagnostic.npz',
        clock=cal_native['clock'], voltage=cal_native['voltage'],
        native_ab=cal_native['native_ab'], predicted_ab=cal_shadow['predicted_ab'],
        native_i=cal_native['native_i'], predicted_i=cal_shadow['predicted_i'])
    calibration_passed = _shadow_pass(cal_shadow, cfg) and _floor_pass(cal_floor, cfg)
    calibration = {'passed': calibration_passed,
                   'shadow': {key: value for key, value in cal_shadow.items()
                              if key not in ('base_g_us', 'predicted_ab', 'predicted_i')},
                   'formula_floor': cal_floor}
    write(output / 'calibration_report.json', calibration)
    if not calibration_passed:
        report = {'schema_version': cfg['schema_version'], 'valid': False,
                  'diagnosis': 'CALIBRATION_IMPLEMENTATION_FAILURE', 'calibration': calibration,
                  'confirmation_opened': False, 'model_judged': False,
                  'task33_passed': False, 'code_revision': revision}
        write(output / 'final_report.json', report)
        return report
    rows = cases(cfg)
    natives, shadows, shadow_summaries, voltage_effects = [], [], [], []
    zero_control_pass = True
    for index, row in enumerate(rows, start=1):
        native = _native_case(h, cfg, base, t32cfg, c3, row)
        shadow = shadow_case(h, cfg, c3, row, native)
        zero = _native_case(h, cfg, base, t32cfg, c3, row, zero_weight=True)
        zero_shadow = shadow_case(h, cfg, c3, row, zero, zero_weight=True)
        effect = float(np.max(np.abs(native['voltage']-zero['voltage'])))
        voltage_effects.append(effect)
        zero_control_pass &= bool(_shadow_pass(zero_shadow, cfg)
            and zero_shadow['release_count'] == 0
            and np.max(np.abs(zero['native_i'])) == 0
            and np.array_equal(zero['rng_seq'][0], zero['rng_seq'][-1]))
        natives.append(native)
        shadows.append(shadow)
        shadow_summaries.append({'row': row, 'release_count': shadow['release_count'],
            'scheduled_count': shadow['scheduled_count'],
            'max_shadow_current_error_na': shadow['max_shadow_current_error_na'],
            'max_shadow_receptor_state_error': shadow['max_shadow_receptor_state_error'],
            'max_rng_sequence_difference': shadow['max_rng_sequence_difference'],
            'release_mismatch_count': shadow['release_mismatch_count'],
            'max_plastic_state_error': shadow['max_plastic_state_error'],
            'synaptic_voltage_effect_mv': effect, 'passed': _shadow_pass(shadow, cfg)})
        if index % 4 == 0:
            print(f'[GIADA Task33] native {index}/{len(rows)}', flush=True)
    negative = {
        'zero_weight_zero_current_no_rng_draw': zero_control_pass,
        'scheduled_events_not_equated_to_releases': any(s['release_count'] < s['scheduled_count'] for s in shadow_summaries),
        'synaptic_voltage_effect_present': max(voltage_effects) >= cfg['gates']['min_synaptic_voltage_effect_mv'],
        'teacher_current_probe_not_selection_eligible': True}
    prediction = coupled_rollout(natives, shadows, cfg, t32cfg, base)
    floor = metrics(prediction, natives)
    shadow_pass = all(row['passed'] for row in shadow_summaries)
    floor_pass = _floor_pass(floor, cfg)
    support = {'scheduled': sum(s['scheduled_count'] for s in shadow_summaries),
               'realized': sum(s['release_count'] for s in shadow_summaries),
               'max_synaptic_voltage_effect_mv': max(voltage_effects)}
    np.savez_compressed(output / 'native_shadow_traces.npz',
        voltage=np.stack([row['voltage'] for row in natives]),
        calcium=np.stack([row['calcium'] for row in natives]),
        gates=np.stack([row['gates'] for row in natives]),
        base_g_us=np.stack([row['base_g_us'] for row in shadows]),
        injection=np.stack([row['injection'] for row in natives]),
        area_um2=np.array(natives[0]['area_um2']))
    report = {'schema_version': cfg['schema_version'], 'valid': True,
              'code_revision': revision, 'teacher_revision': actual_teacher,
              'calibration': calibration, 'confirmation_opened': True,
              'confirmation_case_count': len(rows), 'confirmation': shadow_summaries,
              'negative_controls': negative, 'support': support,
              'formula_floor': floor, 'shadow_passed': shadow_pass,
              'native_floor_admissible': floor_pass,
              'model_judged': False, 'task33_passed': False,
              'diagnosis': 'READY_FOR_FROZEN_MODEL_EVALUATION' if
                shadow_pass and floor_pass and all(negative.values()) else
                'NATIVE_OR_INTERFACE_GATE_FAILED'}
    write(output / 'premodel_report.json', report)
    if report['diagnosis'] != 'READY_FOR_FROZEN_MODEL_EVALUATION':
        write(output / 'final_report.json', report)
    return report


def evaluate_frozen(root: Path, teacher: Path, output: Path) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Task33 frozen model evaluation requires CUDA')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg, t32cfg, _ = contract(root)
    base = t32.verify(root, teacher, t32cfg)
    premodel = json.loads((output / 'premodel_report.json').read_text())
    if premodel['diagnosis'] != 'READY_FOR_FROZEN_MODEL_EVALUATION':
        raise RuntimeError('Task33 native floor does not authorize frozen model judgment')
    with np.load(output / 'native_shadow_traces.npz') as bundle:
        natives = [{'voltage': bundle['voltage'][j], 'calcium': bundle['calcium'][j],
                    'gates': bundle['gates'][j], 'injection': bundle['injection'][j],
                    'multipliers': ionic.panel_multipliers()['canonical'],
                    'area_um2': float(bundle['area_um2'])}
                   for j in range(len(bundle['voltage']))]
        shadows = [{'base_g_us': bundle['base_g_us'][j]} for j in range(len(natives))]
    loaded, seeds = ionic.load_frozen(root, ionic.config(root), torch, 'cuda')
    if list(seeds) != t32cfg['model_seeds']:
        raise RuntimeError('Task33 frozen checkpoint seeds changed')
    model_results = {}
    for family in t32cfg['families']:
        predicted = coupled_rollout(natives, shadows, cfg, t32cfg, base,
                                    loaded[family, t32cfg['frozen_arm']], torch)
        model_results[family] = {}
        for seed_index, seed in enumerate(seeds):
            result = metrics(predicted, natives, seed_index)
            result['passed'] = _candidate_pass(result, cfg)
            model_results[family][str(seed)] = result
        write(output / 'model_results_partial.json', model_results)
        print(f'[GIADA Task33] frozen {family}: {len(seeds)} seeds', flush=True)
    passed = (len(model_results) == 2 and
              all(row['passed'] for family in model_results.values() for row in family.values()))
    report = {**premodel, 'model_judged': True, 'models': model_results,
              'task33_passed': passed,
              'diagnosis': 'CONFIRMED_WITHIN_ONE_COMPARTMENT_SCOPE' if passed else
                           'FROZEN_CANDIDATE_NO_GO_WITH_ADMISSIBLE_FLOOR',
              'training_performed': False, 'teacher_future_release_used_as_input': False,
              'full_cell_claim': False, 'speedup_claim': False}
    write(output / 'final_report.json', report)
    return report
