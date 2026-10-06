"""IV-C3: causal online shadow of four mixed canonical stochastic synapses."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .hh_family_transfer import write
from .iv_c1_deterministic_synapse import compile_native, normalization
from .iv_c2_stochastic_release import _rng, shadow_step


RECEPTORS = ('AMPA', 'NMDA', 'GABAA', 'GABAB')
STATE_NAMES = tuple(f'{prefix}_{receptor}' for receptor in RECEPTORS
                    for prefix in ('A', 'B'))
OUTPUT_NAMES = tuple(f'{prefix}_{receptor}' for receptor in RECEPTORS
                     for prefix in ('g', 'i'))


def load_contract(repo: Path) -> dict:
    cfg = json.loads((repo / 'experiments/iv_c3_integrated_synaptic_interface_preregistration.json').read_text())
    if cfg['roadmap_id'] != 'IV-C3' or len(cfg['synapses']) != 4:
        raise RuntimeError('IV-C3 contract changed')
    for key, item in cfg['parent_reports'].items():
        path = repo / item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'{key} parent hash changed')
        report = json.loads(path.read_text())
        if not report['valid'] or not report[f'{key.lower().replace("-", "_")}_passed']:
            raise RuntimeError(f'{key} prerequisite did not pass')
        if report['task33_authorized']:
            raise RuntimeError(f'{key} incorrectly authorizes Task33')
    ids = [row['id'] for row in cfg['synapses']]
    if len(set(ids)) != 4 or set(cfg['calibration_schedule_ms']) != set(ids):
        raise RuntimeError('IV-C3 synapse IDs changed')
    for schedule in cfg['confirmation_schedules_ms'].values():
        if set(schedule) != set(ids):
            raise RuntimeError('IV-C3 schedule mapping changed')
    return cfg


def _receptors(mechanism: str) -> tuple[str, str]:
    return ('AMPA', 'NMDA') if mechanism == 'ProbAMPANMDA_EMS' else ('GABAA', 'GABAB')


def _observed(synapses: dict, h, sec) -> dict:
    h.fcurrent()
    totals = {name: 0. for name in (*STATE_NAMES, *OUTPUT_NAMES)}
    per_synapse = {}
    for identity, row in synapses.items():
        syn, con, native_rng = row['syn'], row['con'], row['native_rng']
        item = {'u': float(syn.u), 'Rstate': int(syn.Rstate),
                'tsyn_fac': float(syn.tsyn_fac), 'tsyn': float(con.weight[4]),
                'rng_seq': int(native_rng.seq())}
        item['states'] = {}
        for receptor in _receptors(row['config']['mechanism']):
            for prefix in ('A', 'B', 'g', 'i'):
                name = f'{prefix}_{receptor}'
                value = float(getattr(syn, name))
                totals[name] += value
                if prefix in ('A', 'B'):
                    item['states'][name] = value
        per_synapse[identity] = item
    return {'clock': float(h.t), 'voltage': float(sec(.5).v),
            'totals': totals, 'synapses': per_synapse}


def _case(h, cfg: dict, schedule: dict, seed: int, hold: float,
          zero_weight: bool = False) -> dict:
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = cfg['dt_ms']
    h.celsius = 6.3
    sec = h.Section(name='giada_iv_c3')
    sec.L = sec.diam = 10
    sec.nseg = 1
    sec.insert('pas')
    sec.g_pas = 1e-5
    sec.e_pas = hold
    clamp = h.SEClamp(sec(.5))
    clamp.dur1 = cfg['duration_ms'] + cfg['dt_ms']
    clamp.amp1 = hold
    clamp.rs = .001
    synapses = {}
    for index, config in enumerate(cfg['synapses']):
        identity = config['id']
        syn = getattr(h, config['mechanism'])(sec(.5))
        for field in ('Use', 'Fac', 'Dep'):
            setattr(syn, field, config[field])
        if config['mechanism'] == 'ProbAMPANMDA_EMS':
            syn.NMDA_ratio = cfg['nmda_ratio']
        else:
            syn.GABAB_ratio = cfg['gabab_ratio']
        native_rng = _rng(h, seed, cfg['random123_streams'][index])
        shadow_rng = _rng(h, seed, cfg['random123_streams'][index])
        syn.setRNG(native_rng)
        con = h.NetCon(None, syn)
        con.delay = 0
        con.weight[0] = 0. if zero_weight else config['weight_ns']
        synapses[identity] = {'syn': syn, 'con': con,
            'native_rng': native_rng, 'shadow_rng': shadow_rng,
            'config': config, 'times': schedule[identity], 'next_event': 0}
    h.finitialize(hold)
    for row in synapses.values():
        for event in row['times']:
            row['con'].event(float(event))
        syn, con = row['syn'], row['con']
        row['shadow'] = {'u': float(syn.u), 'Rstate': int(syn.Rstate),
                         'tsyn_fac': float(syn.tsyn_fac),
                         'tsyn': float(con.weight[4]), 'released': False}
        row['previous_a'] = 0.
    parameters = {}
    for receptor in RECEPTORS:
        sample = next(row['syn'] for row in synapses.values()
                      if receptor in _receptors(row['config']['mechanism']))
        tau_r = float(getattr(sample, f'tau_r_{receptor}'))
        tau_d = float(getattr(sample, f'tau_d_{receptor}'))
        parameters[receptor] = (tau_r, tau_d, normalization(tau_r, tau_d))
    predicted = {name: 0. for name in STATE_NAMES}
    state_errors = {name: 0. for name in STATE_NAMES}
    output_errors = {name: 0. for name in OUTPUT_NAMES}
    native_values = {name: [] for name in OUTPUT_NAMES}
    predicted_values = {name: [] for name in OUTPUT_NAMES}
    native_min = {name: float('inf') for name in OUTPUT_NAMES}
    native_max = {name: float('-inf') for name in OUTPUT_NAMES}
    release_mismatches = 0
    scheduled_count = sum(len(row['times']) for row in synapses.values())
    native_release_count = shadow_release_count = 0
    release_by_receptor = {name: 0 for name in RECEPTORS}
    max_plastic_error = max_rng_difference = max_holding_error = 0.
    snapshots = []
    checkpoint = None
    previous_clock = 0.
    n_samples = round(cfg['duration_ms'] / cfg['dt_ms']) + 1
    for n in range(n_samples):
        observed = _observed(synapses, h, sec)
        clock = observed['clock']
        elapsed = clock - previous_clock
        for receptor in RECEPTORS:
            tau_r, tau_d, _ = parameters[receptor]
            for prefix, tau in (('A', tau_r), ('B', tau_d)):
                predicted[f'{prefix}_{receptor}'] *= math.exp(-elapsed / tau)
        release_this_sample = {identity: 0 for identity in synapses}
        for identity, row in synapses.items():
            config = row['config']
            while (row['next_event'] < len(row['times'])
                   and row['times'][row['next_event']] <= clock):
                event = row['times'][row['next_event']]
                row['next_event'] += 1
                if zero_weight:
                    continue  # Canonical NET_RECEIVE returns before all draws.
                row['shadow'], _ = shadow_step(row['shadow'], event, config,
                                                row['shadow_rng'].repick)
                if row['shadow']['released']:
                    release_this_sample[identity] += 1
                    shadow_release_count += 1
                    for receptor in _receptors(config['mechanism']):
                        release_by_receptor[receptor] += 1
                        tau_r, tau_d, factor = parameters[receptor]
                        ratio = (cfg['nmda_ratio'] if receptor == 'NMDA' else
                                 cfg['gabab_ratio'] if receptor == 'GABAB' else 1.)
                        amount = config['weight_ns'] * ratio * factor
                        predicted[f'A_{receptor}'] += amount * math.exp(-(clock-event)/tau_r)
                        predicted[f'B_{receptor}'] += amount * math.exp(-(clock-event)/tau_d)
            native = observed['synapses'][identity]
            shadow = row['shadow']
            max_plastic_error = max(max_plastic_error,
                *(abs(native[key] - shadow[key]) for key in ('u', 'Rstate', 'tsyn_fac', 'tsyn')))
            max_rng_difference = max(max_rng_difference,
                abs(native['rng_seq'] - int(row['shadow_rng'].seq())))
            primary = 'AMPA' if config['mechanism'] == 'ProbAMPANMDA_EMS' else 'GABAA'
            tau_r, _, factor = parameters[primary]
            a = native['states'][f'A_{primary}']
            remaining = row['previous_a'] * math.exp(-elapsed / tau_r)
            witnessed = int(a - remaining > config['weight_ns'] * factor * .25)
            native_release_count += witnessed
            release_mismatches += int(witnessed != release_this_sample[identity])
            row['previous_a'] = a
        for name in STATE_NAMES:
            state_errors[name] = max(state_errors[name],
                                     abs(observed['totals'][name] - predicted[name]))
        voltage = observed['voltage']
        for receptor in RECEPTORS:
            g = cfg['gmax_us_per_ns'] * (predicted[f'B_{receptor}'] - predicted[f'A_{receptor}'])
            if receptor == 'NMDA':
                g /= 1 + math.exp(-.08 * voltage) / 3.57
            reversal = 0. if receptor in ('AMPA', 'NMDA') else (-80. if receptor == 'GABAA' else -97.)
            values = {f'g_{receptor}': g, f'i_{receptor}': g * (voltage - reversal)}
            for name, value in values.items():
                native_value = observed['totals'][name]
                output_errors[name] = max(output_errors[name], abs(native_value - value))
                native_values[name].append(native_value)
                predicted_values[name].append(value)
                native_min[name] = min(native_min[name], native_value)
                native_max[name] = max(native_max[name], native_value)
        max_holding_error = max(max_holding_error, abs(voltage-hold))
        snapshots.append(observed)
        if checkpoint is None and clock >= cfg['checkpoint_time_ms']:
            checkpoint = {'index': n, 'clock': clock,
                'synapses': {identity: {
                    **observed['synapses'][identity],
                    'next_event': row['next_event']}
                    for identity, row in synapses.items()}}
        previous_clock = clock
        if n < n_samples - 1:
            h.fadvance()
    charge_errors = {receptor: float(abs(np.trapezoid(
        np.asarray(native_values[f'i_{receptor}'])-
        np.asarray(predicted_values[f'i_{receptor}']), dx=cfg['dt_ms'])))
        for receptor in RECEPTORS}
    # Native restart from the entire mixed synapse state and all four RNG
    # positions. Only future prescribed input events are requeued.
    h.finitialize(hold)
    for _ in range(checkpoint['index']):
        h.fadvance()
    restart_clock_error = abs(float(h.t) - checkpoint['clock'])
    sec(.5).v = snapshots[checkpoint['index']]['voltage']
    for identity, row in synapses.items():
        state = checkpoint['synapses'][identity]
        syn, con = row['syn'], row['con']
        for name in ('u', 'Rstate', 'tsyn_fac'):
            setattr(syn, name, state[name])
        for name, value in state['states'].items():
            setattr(syn, name, value)
        con.weight[4] = state['tsyn']
        row['native_rng'].seq(state['rng_seq'])
        for event in row['times'][state['next_event']:]:
            con.event(float(event))
    restart_state_error = restart_current_error = restart_rng_difference = 0.
    for n in range(checkpoint['index']+1, n_samples):
        h.fadvance()
        actual = _observed(synapses, h, sec)
        reference = snapshots[n]
        restart_clock_error = max(restart_clock_error, abs(actual['clock']-reference['clock']))
        for identity in synapses:
            a, b = actual['synapses'][identity], reference['synapses'][identity]
            restart_state_error = max(restart_state_error,
                *(abs(a[key]-b[key]) for key in ('u', 'Rstate', 'tsyn_fac', 'tsyn')),
                *(abs(a['states'][key]-b['states'][key]) for key in a['states']))
            restart_rng_difference = max(restart_rng_difference, abs(a['rng_seq']-b['rng_seq']))
        restart_current_error = max(restart_current_error,
            *(abs(actual['totals'][name]-reference['totals'][name]) for name in OUTPUT_NAMES))
    report = {'seed': seed, 'hold_mv': hold, 'zero_weight': zero_weight,
        'scheduled_count': scheduled_count, 'native_release_count': native_release_count,
        'shadow_release_count': shadow_release_count,
        'release_mismatch_count': release_mismatches,
        'release_by_receptor': release_by_receptor,
        'max_plastic_state_error': max_plastic_error,
        'max_rng_sequence_difference': max_rng_difference,
        'max_receptor_state_error': max(state_errors.values()),
        'receptor_state_errors': state_errors,
        'max_conductance_error_us': max(output_errors[f'g_{name}'] for name in RECEPTORS),
        'max_current_error_na': max(output_errors[f'i_{name}'] for name in RECEPTORS),
        'output_errors': output_errors,
        'max_integrated_charge_error_na_ms': max(charge_errors.values()),
        'charge_errors': charge_errors,
        'max_holding_error_mv': max_holding_error,
        'max_native_restart_state_error': restart_state_error,
        'max_native_restart_current_error_na': restart_current_error,
        'max_native_restart_rng_difference': restart_rng_difference,
        'max_native_restart_clock_error_ms': restart_clock_error,
        'native_min': native_min, 'native_max': native_max}
    h.delete_section(sec=sec)
    return report


def _passes(row: dict, gates: dict) -> bool:
    return (row['max_plastic_state_error'] <= gates['max_plastic_state_error']
        and row['max_rng_sequence_difference'] <= gates['max_rng_sequence_difference']
        and row['release_mismatch_count'] <= gates['max_release_mismatch_count']
        and row['max_receptor_state_error'] <= gates['max_receptor_state_error']
        and row['max_conductance_error_us'] <= gates['max_conductance_error_us']
        and row['max_current_error_na'] <= gates['max_current_error_na']
        and row['max_integrated_charge_error_na_ms'] <= gates['max_integrated_charge_error_na_ms']
        and row['max_native_restart_state_error'] <= gates['max_native_restart_state_error']
        and row['max_native_restart_current_error_na'] <= gates['max_native_restart_current_error_na']
        and row['max_native_restart_rng_difference'] == 0
        and row['max_native_restart_clock_error_ms'] <= 1e-8
        and row['max_holding_error_mv'] <= gates['max_holding_error_mv'])


def run(repo: Path, teacher: Path, output: Path, revision: str) -> dict:
    from neuron import h, load_mechanisms
    cfg = load_contract(repo)
    actual = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != cfg['teacher_revision']:
        raise RuntimeError('IV-C3 teacher revision changed')
    c1 = json.loads((repo / 'experiments/iv_c1_deterministic_synapse_preregistration.json').read_text())
    build = compile_native(repo, teacher, output, c1)
    load_mechanisms(str(build.resolve()))
    cal = _case(h, cfg, cfg['calibration_schedule_ms'], cfg['calibration_seed'], -65.)
    cal['passed'] = _passes(cal, cfg['gates'])
    write(output / 'calibration_report.json', cal)
    cases = {}
    if cal['passed']:
        for label, schedule in cfg['confirmation_schedules_ms'].items():
            for seed in cfg['confirmation_seeds']:
                for hold in cfg['holding_voltages_mv']:
                    row = _case(h, cfg, schedule, seed, hold)
                    row['passed'] = _passes(row, cfg['gates'])
                    cases[f'{label}:{seed}:{hold}'] = row
            print(f'[GIADA IV-C3] {label}: 12 cells complete', flush=True)
    controls = {'all_zero_weight_has_zero_current_and_no_rng_draws': False,
                'scheduled_events_are_not_assumed_released': False,
                'opposite_voltage_current_signs_and_nmda_block': False}
    zero = None
    if cal['passed']:
        zero = _case(h, cfg, cfg['confirmation_schedules_ms']['coincident_burst'],
                     cfg['confirmation_seeds'][0], -75., zero_weight=True)
        zero['passed'] = _passes(zero, cfg['gates'])
        controls['all_zero_weight_has_zero_current_and_no_rng_draws'] = bool(
            zero['passed'] and zero['native_release_count'] == 0
            and zero['shadow_release_count'] == 0
            and all(zero['native_max'][f'g_{name}'] == 0 and
                    zero['native_min'][f'i_{name}'] == 0 for name in RECEPTORS))
        controls['scheduled_events_are_not_assumed_released'] = any(
            row['native_release_count'] < row['scheduled_count']
            for row in cases.values())
        minus = [row for row in cases.values() if row['hold_mv'] == -75.]
        plus = [row for row in cases.values() if row['hold_mv'] == 20.]
        controls['opposite_voltage_current_signs_and_nmda_block'] = bool(
            any(row['native_min']['i_AMPA'] < 0 and
                row['native_max']['i_GABAA'] > 0 for row in minus)
            and any(row['native_max']['i_AMPA'] > 0 for row in plus)
            and max(row['native_max']['g_NMDA'] for row in plus) >
                max(row['native_max']['g_NMDA'] for row in minus))
    support = {name: sum(row['release_by_receptor'][name] for row in cases.values())
               for name in RECEPTORS}
    passed = bool(cal['passed'] and len(cases) == 36 and
        all(row['passed'] for row in cases.values()) and
        all(value > 0 for value in support.values()) and all(controls.values()))
    report = {'schema_version': cfg['schema_version'], 'valid': True,
        'code_revision': revision, 'teacher_revision': actual,
        'calibration': cal, 'confirmation': cases, 'confirmation_case_count': len(cases),
        'zero_weight_case': zero, 'release_support': support,
        'negative_controls': controls,
        'iv_c1_passed': True, 'iv_c2_passed': True, 'iv_c3_passed': passed,
        'task33_authorized': passed,
        'diagnosis': 'CONFIRMED_WITHIN_ISOLATED_VOLTAGE_CLAMP_SCOPE' if passed else
                     ('CALIBRATION_IMPLEMENTATION_FAILURE' if not cal['passed'] else
                      'CONFIRMATION_GATE_FAILED'),
        'scope': cfg['scope']}
    write(output / 'final_report.json', report)
    return report
