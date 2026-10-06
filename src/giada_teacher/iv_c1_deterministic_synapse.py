"""IV-C1: isolated deterministic receptor-kernel audit of canonical synapses.

Each native point process receives at most one presynaptic event.  This removes
short-term plasticity and release-history confounds without replacing the
canonical teacher mechanisms.  IV-C2 will test those histories separately.
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

from .hh_family_transfer import write


COMPONENTS = ('AMPA', 'NMDA', 'GABAA', 'GABAB')
STATES = ('A_AMPA', 'B_AMPA', 'A_NMDA', 'B_NMDA',
          'A_GABAA', 'B_GABAA', 'A_GABAB', 'B_GABAB')
FIELD_NAMES = ('g_AMPA', 'g_NMDA', 'g_GABAA', 'g_GABAB',
               'i_AMPA', 'i_NMDA', 'i_GABAA', 'i_GABAB')


def load_contract(root: Path) -> dict:
    cfg = json.loads((root / 'experiments/iv_c1_deterministic_synapse_preregistration.json').read_text())
    if cfg['roadmap_id'] != 'IV-C1' or cfg['phase_candidates_steps'] != [0, 1]:
        raise RuntimeError('IV-C1 contract changed')
    if not cfg['confirmation_schedules'] or cfg['calibration_schedule'] in cfg['confirmation_schedules'].values():
        raise RuntimeError('IV-C1 calibration/confirmation schedule collision')
    return cfg


def load_v2_contract(root: Path) -> dict:
    spec = json.loads((root / 'experiments/iv_c1_deterministic_synapse_v2.json').read_text())
    base_path = root / spec['base_contract']
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != spec['base_contract_sha256']:
        raise RuntimeError('IV-C1 v2 base preregistration changed')
    parent = root / spec['parent_v1']
    for name, key in (('final_report.json', 'parent_v1_report_sha256'),
                      ('artifact_bundle.zip', 'parent_v1_artifact_sha256')):
        if hashlib.sha256((parent / name).read_bytes()).hexdigest() != spec[key]:
            raise RuntimeError('IV-C1 v1 parent artifact changed')
    previous = json.loads((parent / 'final_report.json').read_text())
    if not (previous['valid'] and previous['calibration_passed']
            and not previous['iv_c1_passed'] and not previous['task33_authorized']
            and len(previous['confirmation']) == 12):
        raise RuntimeError('IV-C1 v1 diagnostic status changed')
    cfg = {**json.loads(base_path.read_text()), **spec['overrides'],
           'schema_version': spec['schema_version']}
    if cfg['state_observation_phase'] != 'pre_event_at_exact_time':
        raise RuntimeError('IV-C1 v2 state phase changed')
    return cfg


def compile_native(root: Path, teacher: Path, output: Path, cfg: dict) -> Path:
    source = teacher / 'L5PC_NEURON_simulation/mods'
    build = output / 'compiled_synapses'
    build.mkdir(parents=True, exist_ok=False)
    for name, expected in cfg['mechanisms'].items():
        path = source / f'{name}.mod'
        raw = path.read_bytes()
        canonical = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if hashlib.sha256(canonical).hexdigest() != expected:
            raise RuntimeError(f'Canonical synapse source mismatch: {name}')
        shutil.copy2(path, build / path.name)
    nrnivmodl = shutil.which('nrnivmodl') or str(Path(sys.executable).parent / 'nrnivmodl')
    if not Path(nrnivmodl).is_file():
        raise RuntimeError('nrnivmodl unavailable')
    with (build / 'compile.log').open('w', encoding='utf-8') as log:
        subprocess.run([nrnivmodl, str(build.resolve())], cwd=build,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return build


def normalization(tau_r: float, tau_d: float) -> float:
    if not 0 < tau_r < tau_d:
        raise ValueError('Invalid dual-exponential time constants')
    peak = tau_r * tau_d / (tau_d - tau_r) * math.log(tau_d / tau_r)
    return 1.0 / (math.exp(-peak / tau_d) - math.exp(-peak / tau_r))


def analytic_trace(time: np.ndarray, voltage: np.ndarray, schedule: dict,
                   cfg: dict, phase_steps: int, parameters: dict) -> dict:
    """Independent closed-form receptor state and current for realized releases."""
    values = {name: np.zeros_like(time, dtype=np.float64)
              for name in (*STATES, *FIELD_NAMES)}
    mg = parameters['mg']
    ratios = {'AMPA': 1.0, 'NMDA': cfg['nmda_ratio'],
              'GABAA': 1.0, 'GABAB': cfg['gabab_ratio']}
    for receptor in COMPONENTS:
        times = schedule['exc_ms'] if receptor in ('AMPA', 'NMDA') else schedule['inh_ms']
        tau_r = parameters[f'tau_r_{receptor}']
        tau_d = parameters[f'tau_d_{receptor}']
        factor = normalization(tau_r, tau_d)
        amplitude = cfg['synapse_weight_ns'] * ratios[receptor] * factor
        for event in times:
            age = time - (event + phase_steps * cfg['dt_ms'])
            active = (age > 1e-10 if cfg.get('state_observation_phase') ==
                      'pre_event_at_exact_time' else age >= -1e-10)
            values[f'A_{receptor}'][active] += amplitude * np.exp(-np.maximum(age[active], 0) / tau_r)
            values[f'B_{receptor}'][active] += amplitude * np.exp(-np.maximum(age[active], 0) / tau_d)
        g = cfg['gmax_us_per_ns'] * (values[f'B_{receptor}'] - values[f'A_{receptor}'])
        if receptor == 'NMDA':
            g *= 1 / (1 + np.exp(-0.08 * voltage) * mg / 3.57)
        values[f'g_{receptor}'] = g
        reversal = 0.0 if receptor in ('AMPA', 'NMDA') else (
            -80.0 if receptor == 'GABAA' else -97.0)
        values[f'i_{receptor}'] = g * (voltage - reversal)
    return values


def native_episode(schedule: dict, hold_mv: float, cfg: dict) -> tuple[dict, dict]:
    from neuron import h

    h.CVode().active(0)
    h.secondorder = 0
    h.dt = cfg['dt_ms']
    h.celsius = 6.3
    sec = h.Section(name='giada_iv_c1')
    sec.L = sec.diam = 10
    sec.nseg = 1
    sec.cm = 1
    sec.insert('pas')
    sec.g_pas = 1e-5
    sec.e_pas = hold_mv
    clamp = h.SEClamp(sec(.5))
    clamp.dur1 = cfg['duration_ms'] + cfg['dt_ms']
    clamp.amp1 = hold_mv
    clamp.rs = 0.001
    synapses = []
    connections = []
    for kind, times in (('exc', schedule['exc_ms']), ('inh', schedule['inh_ms'])):
        for event in times:
            syn = (h.ProbAMPANMDA_EMS(sec(.5)) if kind == 'exc'
                   else h.ProbGABAAB_EMS(sec(.5)))
            syn.Use = 1.0
            syn.Fac = 0.0
            syn.Dep = 100.0
            # gmax is a GLOBAL PARAMETER in both canonical MOD files, not a
            # RANGE property. The pinned sources fix its default to .001 uS.
            if kind == 'exc':
                syn.NMDA_ratio = cfg['nmda_ratio']
            else:
                syn.GABAB_ratio = cfg['gabab_ratio']
            connection = h.NetCon(None, syn)
            connection.delay = 0
            connection.weight[0] = cfg['synapse_weight_ns']
            synapses.append((kind, syn, event))
            connections.append(connection)
    time = np.arange(round(cfg['duration_ms'] / cfg['dt_ms']) + 1) * cfg['dt_ms']
    values = {name: np.zeros(len(time), dtype=np.float64)
              for name in (*STATES, *FIELD_NAMES)}
    voltage = np.empty_like(time)
    h.finitialize(hold_mv)
    for (_, _, event), connection in zip(synapses, connections):
        connection.event(float(event))
    # mg is likewise a GLOBAL PARAMETER with canonical default 1 mM.
    parameters = {'mg': 1.0}
    for receptor in COMPONENTS:
        kind = 'exc' if receptor in ('AMPA', 'NMDA') else 'inh'
        attr = 'tau_r_' + receptor
        sample = next((syn for typ, syn, _ in synapses if typ == kind), None)
        if sample is None:
            sample = h.ProbAMPANMDA_EMS(sec(.5)) if kind == 'exc' else h.ProbGABAAB_EMS(sec(.5))
        parameters[attr] = float(getattr(sample, attr))
        parameters['tau_d_' + receptor] = float(getattr(sample, 'tau_d_' + receptor))
    for n in range(len(time)):
        h.fcurrent()
        voltage[n] = float(sec(.5).v)
        for kind, syn, _ in synapses:
            receptors = ('AMPA', 'NMDA') if kind == 'exc' else ('GABAA', 'GABAB')
            for receptor in receptors:
                for prefix in ('A_', 'B_', 'g_', 'i_'):
                    name = prefix + receptor
                    values[name][n] += float(getattr(syn, name))
        if n < len(time) - 1:
            h.fadvance()
    release_count = sum(float(max(values['A_AMPA' if kind == 'exc' else 'A_GABAA'])) > 0
                        for kind, _, _ in synapses)
    # The total-state check above is intentionally complemented by per-synapse
    # checks: each point process must actually receive its own unique event.
    per_synapse_released = [bool(float(getattr(syn, 'B_AMPA' if kind == 'exc'
                                              else 'B_GABAA')) > 0)
                            for kind, syn, _ in synapses]
    result = {'time_ms': time, 'voltage_mv': voltage, **values}
    metadata = {'parameters': parameters,
                'expected_release_count': len(synapses),
                'released_synapse_count': sum(per_synapse_released),
                'aggregate_release_indicator': release_count}
    h.delete_section(sec=sec)
    return result, metadata


def compare(native: dict, predicted: dict, cfg: dict, hold: float) -> dict:
    g_error = max(float(np.max(abs(native['g_' + name] - predicted['g_' + name])))
                  for name in COMPONENTS)
    i_error = max(float(np.max(abs(native['i_' + name] - predicted['i_' + name])))
                  for name in COMPONENTS)
    state_error = max(float(np.max(abs(native[name] - predicted[name])))
                      for name in STATES)
    charge_error = max(abs(float(np.trapezoid(native['i_' + name] - predicted['i_' + name],
                                               dx=cfg['dt_ms']))) for name in COMPONENTS)
    return {'hold_mv': hold, 'max_g_error_us': g_error,
            'max_i_error_na': i_error, 'max_state_error': state_error,
            'max_charge_error_na_ms': charge_error,
            'max_holding_error_mv': float(np.max(abs(native['voltage_mv'] - hold)))}


def run(root: Path, teacher: Path, output: Path, cfg: dict, revision: str) -> dict:
    from neuron import load_mechanisms
    actual = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != cfg['teacher_revision']:
        raise RuntimeError('IV-C1 teacher revision changed')
    build = compile_native(root, teacher, output, cfg)
    load_mechanisms(str(build.resolve()))
    cal = {}
    for hold in cfg['holding_voltages_mv']:
        native, metadata = native_episode(cfg['calibration_schedule'], hold, cfg)
        cal[str(hold)] = {'native': native, 'metadata': metadata,
                          'scores': {str(phase): compare(native,
                              analytic_trace(native['time_ms'], native['voltage_mv'],
                                  cfg['calibration_schedule'], cfg, phase,
                                  metadata['parameters']), cfg, hold)
                              for phase in cfg['phase_candidates_steps']}}
    scores = {phase: max(row['scores'][str(phase)]['max_g_error_us']
                         for row in cal.values())
              for phase in cfg['phase_candidates_steps']}
    selected = min(scores, key=lambda phase: (scores[phase], phase))
    calibration_pass = scores[selected] <= cfg['calibration_max_conductance_error_us']
    write(output / 'calibration_decision.json', {'phase_scores_us': scores,
        'selected_phase_steps': selected, 'passed': calibration_pass,
        'confirmation_accessed': False})
    confirmation = {}
    if calibration_pass:
        for label, schedule in cfg['confirmation_schedules'].items():
            for hold in cfg['holding_voltages_mv']:
                native, metadata = native_episode(schedule, hold, cfg)
                predicted = analytic_trace(native['time_ms'], native['voltage_mv'],
                    schedule, cfg, selected, metadata['parameters'])
                metrics = compare(native, predicted, cfg, hold)
                metrics['expected_release_count'] = metadata['expected_release_count']
                metrics['released_synapse_count'] = metadata['released_synapse_count']
                metrics['passed'] = bool(
                    metrics['max_g_error_us'] <= cfg['confirmation_max_conductance_error_us']
                    and metrics['max_state_error'] <= cfg['confirmation_max_state_error']
                    and metrics['max_i_error_na'] <= cfg['confirmation_max_current_error_na']
                    and metrics['max_charge_error_na_ms'] <= cfg['confirmation_max_charge_error_na_ms']
                    and metrics['max_holding_error_mv'] <= cfg['holding_voltage_max_error_mv']
                    and metadata['released_synapse_count'] == metadata['expected_release_count'])
                confirmation[f'{label}:{hold}'] = metrics
            print(f'[GIADA IV-C1] confirmation {label} complete', flush=True)
    zero, zero_meta = native_episode({'exc_ms': [], 'inh_ms': []}, -75., cfg)
    zero_pass = all(np.max(abs(zero['g_' + name])) == 0 for name in COMPONENTS)
    block_pass = False
    sign_pass = False
    if calibration_pass:
        minus, _ = native_episode(cfg['calibration_schedule'], -75., cfg)
        plus, _ = native_episode(cfg['calibration_schedule'], 20., cfg)
        block_pass = bool(np.max(plus['g_NMDA']) > np.max(minus['g_NMDA']))
        sign_pass = bool(np.min(minus['i_AMPA']) < 0 and
                         np.max(minus['i_GABAA']) > 0)
    report = {'schema_version': cfg['schema_version'], 'valid': True,
              'teacher_revision': actual, 'code_revision': revision,
              'selected_phase_steps': selected, 'calibration_scores_us': scores,
              'calibration_passed': calibration_pass,
              'confirmation': confirmation,
              'negative_controls': {'zero_event_zero_conductance': zero_pass,
                  'NMDA_Mg_block_voltage_contrast': block_pass,
                  'AMPA_and_GABA_reversal_sign': sign_pass},
              'iv_c1_passed': bool(calibration_pass and len(confirmation) == 12
                  and all(row['passed'] for row in confirmation.values())
                  and zero_pass and block_pass and sign_pass),
              'iv_c2_passed': False, 'iv_c3_passed': False,
              'task33_authorized': False,
              'scope': cfg['scope']}
    write(output / 'final_report.json', report)
    return report
