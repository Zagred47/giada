"""IV-C2 prospective audit of the canonical stochastic EMS synapses.

The shadow implements the public NET_RECEIVE equations, but owns an independent
Random123 stream.  Native outcomes are never supplied to the shadow.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .iv_c1_deterministic_synapse import compile_native, normalization
from .hh_family_transfer import write


def load_contract(root: Path) -> dict:
    cfg = json.loads((root / 'experiments/iv_c2_stochastic_release_preregistration.json').read_text())
    if cfg['roadmap_id'] != 'IV-C2' or cfg['random123_distribution'] != 'negexp(1.0)':
        raise RuntimeError('IV-C2 contract changed')
    parent = json.loads((root / cfg['parent_report']).read_text())
    if not parent.get('iv_c1_passed') or parent.get('task33_authorized'):
        raise RuntimeError('IV-C1 prerequisite not satisfied')
    return cfg


def shadow_step(state: dict, event_time: float, arm: dict, draw) -> tuple[dict, int]:
    """Exact event-ordered NET_RECEIVE logic; draw is an independent stream."""
    state = dict(state)
    if arm['Fac'] > 0:
        state['u'] *= math.exp(-(event_time - state['tsyn_fac']) / arm['Fac'])
        state['u'] += arm['Use'] * (1 - state['u'])
    else:
        state['u'] = arm['Use']
    state['tsyn_fac'] = event_time
    draws = 0
    if state['Rstate'] == 0:
        survival = math.exp(-(event_time - state['tsyn']) / arm['Dep'])
        recovered = draw() > survival
        draws += 1
        if recovered:
            state['Rstate'] = 1
        else:
            state['tsyn'] = event_time
    state['released'] = False
    if state['Rstate'] == 1:
        state['released'] = draw() < state['u']
        draws += 1
        if state['released']:
            state['tsyn'] = event_time
            state['Rstate'] = 0
    return state, draws


def _rng(h, seed: int, stream: int):
    rng = h.Random()
    rng.Random123(seed, stream, 0)
    rng.negexp(1.0)
    return rng


def _case(h, mechanism: str, arm: dict, times: list[float], seed: int,
          stream: int, cfg: dict) -> dict:
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = cfg['dt_ms']
    h.celsius = 6.3
    sec = h.Section(name='giada_iv_c2')
    sec.L = sec.diam = 10
    sec.nseg = 1
    sec.insert('pas')
    sec.e_pas = cfg['hold_mv']
    clamp = h.SEClamp(sec(.5))
    clamp.dur1 = cfg['duration_ms'] + 1
    clamp.amp1 = cfg['hold_mv']
    clamp.rs = .001
    syn = getattr(h, mechanism)(sec(.5))
    for field in ('Use', 'Fac', 'Dep'):
        setattr(syn, field, arm[field])
    native_rng = _rng(h, seed, stream)
    shadow_rng = _rng(h, seed, stream)
    syn.setRNG(native_rng)
    con = h.NetCon(None, syn)
    con.delay = 0
    con.weight[0] = cfg['weight_ns']
    h.finitialize(cfg['hold_mv'])
    for event in times:
        con.event(float(event))
    state = {'u': float(syn.u), 'Rstate': int(syn.Rstate),
             'tsyn_fac': float(syn.tsyn_fac), 'tsyn': float(con.weight[4]),
             'released': False}
    receptor = 'AMPA' if mechanism == 'ProbAMPANMDA_EMS' else 'GABAA'
    factor = normalization(float(getattr(syn, f'tau_r_{receptor}')),
                           float(getattr(syn, f'tau_d_{receptor}')))
    native_events = []
    shadow_events = []
    errors = []
    sequence_errors = []
    checkpoint_suffix_errors = []
    native_checkpoint_errors = []
    initial_shadow_seq = int(shadow_rng.seq())
    previous_a = 0.0
    previous_clock = float(h.t)
    for i, event in enumerate(times):
        # Observe one solver tick after the event, clear of the exact-time
        # pre/post boundary ambiguity found in IV-C1.
        while float(h.t) <= event + cfg['dt_ms'] * .5:
            h.fadvance()
        before_shadow_seq = int(shadow_rng.seq())
        state, draw_count = shadow_step(state, event, arm, shadow_rng.repick)
        after_shadow_seq = int(shadow_rng.seq())
        native_seq = int(native_rng.seq())
        observed = {'u': float(syn.u), 'Rstate': int(syn.Rstate),
                    'tsyn_fac': float(syn.tsyn_fac),
                    'tsyn': float(con.weight[4])}
        # Receptor A state is an independent release witness, corrected for
        # its decay from the preceding sample.
        a = float(getattr(syn, f'A_{receptor}'))
        tau_r = float(getattr(syn, f'tau_r_{receptor}'))
        expected_no_release_a = previous_a * math.exp(-(float(h.t) - previous_clock) / tau_r)
        native_released = bool(a - expected_no_release_a > cfg['weight_ns'] * factor * .5)
        previous_a, previous_clock = a, float(h.t)
        native_events.append({**observed, 'released': native_released,
                              'A': a, 'rng_seq': native_seq,
                              'receptors': {name: float(getattr(syn, name))
                                  for name in (('A_AMPA', 'B_AMPA', 'A_NMDA', 'B_NMDA')
                                  if mechanism == 'ProbAMPANMDA_EMS' else
                                  ('A_GABAA', 'B_GABAA', 'A_GABAB', 'B_GABAB'))}})
        shadow_events.append({**state, 'draw_count': draw_count,
                              'rng_seq': after_shadow_seq})
        errors.append(max(abs(observed['u'] - state['u']),
                          abs(observed['Rstate'] - state['Rstate']),
                          abs(observed['tsyn_fac'] - state['tsyn_fac']),
                          abs(observed['tsyn'] - state['tsyn'])))
        sequence_errors.append(abs(native_seq - after_shadow_seq))
        # Restoration is tested without consulting native future outcomes:
        # clone the exact shadow sequence and plastic state at a middle event,
        # then replay the suffix from the same checkpoint.
        if i == len(times) // 2:
            checkpoint = (dict(state), after_shadow_seq)
            native_checkpoint = {
                'clock': float(h.t), 'u': float(syn.u),
                'Rstate': float(syn.Rstate), 'tsyn_fac': float(syn.tsyn_fac),
                'tsyn': float(con.weight[4]), 'rng_seq': native_seq,
                'receptors': {name: float(getattr(syn, name))
                    for name in (('A_AMPA', 'B_AMPA', 'A_NMDA', 'B_NMDA')
                    if mechanism == 'ProbAMPANMDA_EMS' else
                    ('A_GABAA', 'B_GABAA', 'A_GABAB', 'B_GABAB'))}}
    restored = _rng(h, seed, stream)
    restored.seq(checkpoint[1])
    replay_state = dict(checkpoint[0])
    for i in range(len(times) // 2 + 1, len(times)):
        replay_state, _ = shadow_step(replay_state, times[i], arm, restored.repick)
        target = shadow_events[i]
        checkpoint_suffix_errors.append(max(
            abs(replay_state[key] - target[key]) for key in ('u', 'Rstate', 'tsyn_fac', 'tsyn')))
        checkpoint_suffix_errors.append(abs(int(restored.seq()) - target['rng_seq']))
    # Independently restore native point-process state and RNG at the same
    # public solver clock.  Re-queue only suffix events, then compare with the
    # uninterrupted native run. This is not just a shadow-only checkpoint.
    h.finitialize(cfg['hold_mv'])
    while float(h.t) < native_checkpoint['clock'] - cfg['dt_ms'] * .25:
        h.fadvance()
    for name in ('u', 'Rstate', 'tsyn_fac'):
        setattr(syn, name, native_checkpoint[name])
    con.weight[4] = native_checkpoint['tsyn']
    for name, value in native_checkpoint['receptors'].items():
        setattr(syn, name, value)
    native_rng.seq(native_checkpoint['rng_seq'])
    for event in times[len(times) // 2 + 1:]:
        con.event(float(event))
    for i in range(len(times) // 2 + 1, len(times)):
        while float(h.t) <= times[i] + cfg['dt_ms'] * .5:
            h.fadvance()
        target = native_events[i]
        native_checkpoint_errors.append(max(
            abs(float(getattr(syn, key)) - target[key])
            for key in ('u', 'Rstate', 'tsyn_fac')))
        native_checkpoint_errors.append(abs(float(con.weight[4]) - target['tsyn']))
        native_checkpoint_errors.append(abs(int(native_rng.seq()) - target['rng_seq']))
        native_checkpoint_errors.extend(abs(float(getattr(syn, name)) - value)
                                        for name, value in target['receptors'].items())
    result = {'mechanism': mechanism, 'arm': arm['name'], 'seed': seed,
              'stream': stream, 'scheduled_count': len(times),
              'native_release_count': sum(row['released'] for row in native_events),
              'shadow_release_count': sum(row['released'] for row in shadow_events),
              'release_mismatch_count': sum(a['released'] != b['released']
                  for a, b in zip(native_events, shadow_events)),
              'max_state_error': max(errors, default=0.),
              'max_rng_sequence_difference': max(sequence_errors, default=0),
              'max_checkpoint_suffix_mismatch': max(checkpoint_suffix_errors, default=0),
              'max_native_checkpoint_suffix_mismatch': max(native_checkpoint_errors, default=0),
              'first_shadow_rng_seq': initial_shadow_seq,
              'last_shadow_rng_seq': after_shadow_seq,
              'trace': {'native': native_events, 'shadow': shadow_events}}
    h.delete_section(sec=sec)
    return result


def _distribution(h, cfg: dict) -> dict:
    rng = _rng(h, 99173, 61231)
    n = cfg['gates']['minimum_distribution_draws']
    values = np.asarray([float(rng.repick()) for _ in range(n)])
    empirical = {str(u): float(np.mean(values < u)) for u in (.2, .5, .8)}
    expected = {str(u): 1 - math.exp(-u) for u in (.2, .5, .8)}
    error = max(abs(empirical[key] - expected[key]) for key in expected)
    uniform_error = max(abs(empirical[str(u)] - u) for u in (.2, .5, .8))
    return {'n': n, 'empirical': empirical, 'negexp_expected': expected,
            'max_absolute_error': error, 'uniform_shadow_disagreement': uniform_error}


def _zero_weight_control(h, cfg: dict) -> dict:
    h.CVode().active(0)
    h.dt = cfg['dt_ms']
    results = {}
    for mechanism in cfg['mechanisms']:
        sec = h.Section(name='giada_iv_c2_zero')
        syn = getattr(h, mechanism)(sec(.5))
        rng = _rng(h, 113, 509)
        syn.setRNG(rng)
        con = h.NetCon(None, syn)
        con.weight[0] = 0.
        h.finitialize(cfg['hold_mv'])
        before = int(rng.seq())
        for event in cfg['event_schedules_ms']['burst']:
            con.event(float(event))
        while float(h.t) < cfg['duration_ms']:
            h.fadvance()
        after = int(rng.seq())
        results[mechanism] = {'before': before, 'after': after,
                              'passed': before == after}
        h.delete_section(sec=sec)
    return results


def run(root: Path, teacher: Path, output: Path, revision: str) -> dict:
    from neuron import h, load_mechanisms
    cfg = load_contract(root)
    actual = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != cfg['teacher_revision']:
        raise RuntimeError('IV-C2 teacher revision changed')
    c1 = json.loads((root / 'experiments/iv_c1_deterministic_synapse_preregistration.json').read_text())
    build = compile_native(root, teacher, output, c1)
    load_mechanisms(str(build.resolve()))
    distribution = _distribution(h, cfg)
    zero_weight = _zero_weight_control(h, cfg)
    cases = []
    for mechanism in cfg['mechanisms']:
        for arm in cfg['arms']:
            for schedule, times in cfg['event_schedules_ms'].items():
                for seed, stream in zip(cfg['replicate_seeds'], cfg['random123_streams']):
                    row = _case(h, mechanism, arm, times, seed, stream, cfg)
                    row['schedule'] = schedule
                    cases.append(row)
            print(f'[GIADA IV-C2] {mechanism} {arm["name"]} complete', flush=True)
    gates = cfg['gates']
    case_pass = all(row['max_state_error'] <= gates['max_state_error']
        and row['max_rng_sequence_difference'] <= gates['max_rng_sequence_difference']
        and row['release_mismatch_count'] <= gates['max_release_mismatch']
        and row['max_checkpoint_suffix_mismatch'] <= gates['max_checkpoint_suffix_mismatch']
        and row['max_native_checkpoint_suffix_mismatch'] <= gates['max_state_error']
        for row in cases)
    distribution_pass = distribution['max_absolute_error'] <= gates['max_distribution_absolute_error']
    negative_pass = distribution['uniform_shadow_disagreement'] > gates['max_distribution_absolute_error']
    zero_weight_pass = all(row['passed'] for row in zero_weight.values())
    scheduled_distinct = any(row['native_release_count'] < row['scheduled_count'] for row in cases)
    report = {'schema_version': cfg['schema_version'], 'valid': True,
              'code_revision': revision, 'teacher_revision': actual,
              'case_count': len(cases), 'cases': cases,
              'distribution': distribution, 'case_gate_passed': case_pass,
              'distribution_gate_passed': distribution_pass,
              'uniform_negative_control_passed': negative_pass,
              'zero_weight_control': zero_weight,
              'scheduled_vs_realized_distinct': scheduled_distinct,
              'iv_c1_passed': True,
              'iv_c2_passed': bool(case_pass and distribution_pass and negative_pass
                   and zero_weight_pass and scheduled_distinct),
              'iv_c3_passed': False, 'task33_authorized': False,
              'scope': cfg['scope']}
    write(output / 'final_report.json', report)
    return report
