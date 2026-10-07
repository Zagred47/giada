"""Independent RC, passive current-interface and two-section axial preflight."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .hh_family_transfer import write


def config(root: Path, config_name: str = 'iv_ad1_passive_axial_preregistration.json') -> dict:
    if Path(config_name).name != config_name:
        raise ValueError('config_name must be a file name under experiments')
    spec = json.loads((root / 'experiments' / config_name).read_text())
    if spec['roadmap_ids'] != ['IV-A1', 'IV-A2', 'IV-D1'] or spec['dt_ms'] != .1:
        raise RuntimeError('IV-A1/A2/D1 prospective domain changed')
    if len(spec['single_cases']) != 5 or len(spec['geometries']) != 3:
        raise RuntimeError('IV-A1/A2/D1 case matrix changed')
    return spec


def geometry_area_um2(length_um: float, diam_um: float) -> float:
    return math.pi * length_um * diam_um


def membrane_coefficients(length_um: float, diam_um: float,
                          cm_uf_cm2: float, g_pas_s_cm2: float) -> tuple[float, float]:
    area = geometry_area_um2(length_um, diam_um)
    # uF/cm2 * um2 -> nF; S/cm2 * um2 -> uS.
    return cm_uf_cm2 * area * 1e-5, g_pas_s_cm2 * area * 1e-2


def axial_conductance_us(geometry: dict) -> float:
    ra = geometry['ra_ohm_cm']
    parent_half_mohm = .01 * ra * (geometry['parent_length_um'] / 2) / (
        math.pi * (geometry['parent_diam_um'] / 2) ** 2)
    child_half_mohm = .01 * ra * (geometry['child_length_um'] / 2) / (
        math.pi * (geometry['child_diam_um'] / 2) ** 2)
    return 1 / (parent_half_mohm + child_half_mohm)


def coefficients(geometry: dict, membrane: dict, connected: bool) -> tuple[np.ndarray, np.ndarray, float]:
    lengths = [geometry['parent_length_um'], geometry['child_length_um']]
    diams = [geometry['parent_diam_um'], geometry['child_diam_um']]
    values = [membrane_coefficients(lengths[j], diams[j],
              membrane['cm_uf_cm2'], membrane['g_pas_s_cm2']) for j in range(2)]
    cap = np.array([row[0] for row in values])
    leak = np.array([row[1] for row in values])
    return cap, leak, axial_conductance_us(geometry) if connected else 0.


def passive_step(old: np.ndarray, cap: np.ndarray, leak: np.ndarray,
                 axial: float, e_pas: float, injected: np.ndarray, dt: float) -> np.ndarray:
    system = np.array([[cap[0]/dt + leak[0] + axial, -axial],
                       [-axial, cap[1]/dt + leak[1] + axial]])
    rhs = cap * old / dt + leak * e_pas + injected
    return np.linalg.solve(system, rhs)


def exact_piecewise_endpoint(initial: np.ndarray, cap: np.ndarray,
                             leak: np.ndarray, axial: float, e_pas: float,
                             pulse: np.ndarray, window: tuple[float, float],
                             duration: float) -> np.ndarray:
    conductance = np.array([[leak[0]+axial, -axial], [-axial, leak[1]+axial]])
    generator = conductance / cap[:, None]
    eigenvalues, eigenvectors = np.linalg.eig(generator)
    inverse = np.linalg.inv(eigenvectors)
    voltage = np.asarray(initial, dtype=float)
    for interval, injected in ((window[0], np.zeros(2)),
                               (window[1]-window[0], pulse),
                               (duration-window[1], np.zeros(2))):
        equilibrium = np.linalg.solve(conductance, leak*e_pas + injected)
        voltage = equilibrium + np.real(eigenvectors @
            (np.exp(-eigenvalues*interval) * (inverse @ (voltage-equilibrium))))
    return voltage


def current_for_case(kind: str, amplitude: float) -> np.ndarray:
    return {'parent_only': np.array([amplitude, 0.]),
            'child_only': np.array([0., amplitude]),
            'symmetric': np.array([amplitude, amplitude]),
            'disconnected': np.array([amplitude, 0.])}[kind]


def simulate_discrete(initial: np.ndarray, cap: np.ndarray, leak: np.ndarray,
                      axial: float, e_pas: float, pulse: np.ndarray,
                      window: tuple[float, float], duration: float,
                      dt: float) -> np.ndarray:
    steps = round(duration/dt)
    voltage = np.empty((steps+1, 2))
    voltage[0] = initial
    for k in range(steps):
        t = k*dt
        injected = pulse if window[0]-1e-10 <= t < window[1]-1e-10 else np.zeros(2)
        voltage[k+1] = passive_step(voltage[k], cap, leak, axial, e_pas, injected, dt)
    return voltage


def native_single(h, case: dict, spec: dict) -> np.ndarray:
    sec = h.Section(name='giada_iv_a_single')
    sec.L, sec.diam, sec.nseg = case['length_um'], case['diam_um'], 1
    sec.cm = case['cm_uf_cm2']
    sec.insert('pas')
    sec.g_pas, sec.e_pas = case['g_pas_s_cm2'], case['e_pas_mv']
    h.CVode().active(0); h.secondorder = 0; h.dt = spec['dt_ms']
    clamp = h.IClamp(sec(.5))
    clamp.delay = spec['pulse_window_ms'][0]
    clamp.dur = spec['pulse_window_ms'][1]-spec['pulse_window_ms'][0]
    clamp.amp = case['pulse_na']
    h.finitialize(case['initial_mv'])
    values = [float(sec(.5).v)]
    for _ in range(round(spec['duration_ms']/spec['dt_ms'])):
        h.fadvance(); values.append(float(sec(.5).v))
    h.delete_section(sec=sec)
    return np.asarray(values)


def native_paired(h, geometry: dict, membrane: dict, initial: np.ndarray,
                  pulse: np.ndarray, connected: bool, window: tuple[float, float],
                  duration: float, dt: float) -> np.ndarray:
    parent = h.Section(name='giada_iv_d_parent')
    child = h.Section(name='giada_iv_d_child')
    for sec, prefix in ((parent, 'parent'), (child, 'child')):
        sec.L = geometry[f'{prefix}_length_um']
        sec.diam = geometry[f'{prefix}_diam_um']
        sec.nseg = 1
        sec.Ra = geometry['ra_ohm_cm']
        sec.cm = membrane['cm_uf_cm2']
        sec.insert('pas')
        sec.g_pas = membrane['g_pas_s_cm2']
        sec.e_pas = membrane['e_pas_mv']
    if connected:
        child.connect(parent(1), 0)
    h.CVode().active(0); h.secondorder = 0; h.dt = dt
    clamps = []
    for sec, amp in zip((parent, child), pulse):
        clamp = h.IClamp(sec(.5))
        clamp.delay, clamp.dur, clamp.amp = window[0], window[1]-window[0], float(amp)
        clamps.append(clamp)
    h.finitialize(float(initial[0]))
    child(.5).v = float(initial[1])
    h.fcurrent()
    voltage = np.empty((round(duration/dt)+1, 2))
    voltage[0] = [parent(.5).v, child(.5).v]
    for k in range(len(voltage)-1):
        h.fadvance()
        voltage[k+1] = [parent(.5).v, child(.5).v]
    h.delete_section(sec=child)
    h.delete_section(sec=parent)
    return voltage


def run(root: Path, output: Path, revision: str,
        config_name: str = 'iv_ad1_passive_axial_preregistration.json') -> dict:
    import neuron
    from neuron import h
    spec = config(root, config_name)
    if neuron.__version__.split('+')[0] != spec['teacher_version']:
        raise RuntimeError('IV-A/D1 registered NEURON version changed')
    output.mkdir(parents=True, exist_ok=False)
    limits = spec['gates']
    dt, duration = spec['dt_ms'], spec['duration_ms']
    window = tuple(spec['pulse_window_ms'])
    singles = []
    for case in spec['single_cases']:
        cap, leak = membrane_coefficients(case['length_um'], case['diam_um'],
                                           case['cm_uf_cm2'], case['g_pas_s_cm2'])
        native = native_single(h, case, spec)
        prediction = simulate_discrete(np.array([case['initial_mv'], case['initial_mv']]),
            np.array([cap, cap]), np.array([leak, leak]), 0., case['e_pas_mv'],
            np.array([case['pulse_na'], 0.]), window, duration, dt)[:, 0]
        if leak:
            equilibrium = case['e_pas_mv']
            exact = equilibrium + (case['initial_mv']-equilibrium)*np.exp(-window[0]*leak/cap)
            active_equilibrium = equilibrium + case['pulse_na']/leak
            exact = active_equilibrium + (exact-active_equilibrium)*np.exp(-(window[1]-window[0])*leak/cap)
            exact = equilibrium + (exact-equilibrium)*np.exp(-(duration-window[1])*leak/cap)
        else:
            exact = case['initial_mv'] + case['pulse_na']*(window[1]-window[0])/cap
        max_error = float(np.max(np.abs(prediction-native)))
        analytic_error = float(abs(native[-1]-exact))
        convergence = None
        if spec['schema_version'].endswith(('-v2', '-v3')):
            errors = []
            for spacing in spec['convergence_dt_ms']:
                refined = simulate_discrete(
                    np.array([case['initial_mv'], case['initial_mv']]),
                    np.array([cap, cap]), np.array([leak, leak]), 0.,
                    case['e_pas_mv'], np.array([case['pulse_na'], 0.]),
                    window, duration, spacing)[-1, 0]
                errors.append(float(abs(refined-exact)))
            convergence = {'dt_ms': spec['convergence_dt_ms'],
                           'endpoint_error_mv': errors,
                           'passed': (errors[2] <= limits['max_single_fine_endpoint_error_mv']
                               and (errors[0] > errors[1] > errors[2]
                                    if leak else max(errors) <= 1e-9))}
        singles.append({'case': case['id'], 'max_native_discrete_error_mv': max_error,
            'analytic_endpoint_error_mv': analytic_error,
            'discrete_continuous_convergence': convergence,
            'passed': max_error <= limits['max_single_native_discrete_error_mv']
                and (convergence['passed'] if convergence else
                     analytic_error <= limits['max_single_analytic_endpoint_error_mv'])})
    a_pass = all(row['passed'] for row in singles)
    write(output/'iv_a1_a2_report.json', {'passed': a_pass, 'cases': singles})
    if not a_pass:
        report = {'schema_version': spec['schema_version'], 'valid': False,
                  'iv_a1_a2_passed': False, 'iv_d1_opened': False, 'iv_d1_passed': False,
                  'task36_authorized': False, 'diagnosis': 'PASSIVE_SINGLE_INTERFACE_NO_GO',
                  'code_revision': revision, 'single_cases': singles}
        write(output/'final_report.json', report)
        return report

    paired = []
    for geometry in spec['geometries']:
        for kind in spec['coupling_cases']:
            connected = kind != 'disconnected'
            cap, leak, axial = coefficients(geometry, spec['passive_membrane'], connected)
            pulse = current_for_case(kind, spec['paired_pulse_na'])
            if kind == 'symmetric' and spec['schema_version'].endswith('-v3'):
                # Equal current per area (and equal I/C), not equal absolute nA.
                pulse[1] = pulse[0] * cap[1] / cap[0]
            initial = np.array([-70., -70.]) if kind == 'symmetric' else np.asarray(spec['paired_initial_mv'])
            native = native_paired(h, geometry, spec['passive_membrane'], initial,
                                   pulse, connected, window, duration, dt)
            predicted = simulate_discrete(initial, cap, leak, axial,
                spec['passive_membrane']['e_pas_mv'], pulse, window, duration, dt)
            error = float(np.max(np.abs(predicted-native)))
            balance = axial_balance = axial_ohm_error = 0.
            for k in range(len(predicted)-1):
                t = k*dt
                injected = pulse if window[0]-1e-10 <= t < window[1]-1e-10 else np.zeros(2)
                axial_out = axial*(native[k+1, 0]-native[k+1, 1])
                residual = cap*(native[k+1]-native[k])/dt + leak*(native[k+1]-spec['passive_membrane']['e_pas_mv']) + np.array([axial_out, -axial_out])-injected
                balance = max(balance, float(np.max(np.abs(residual))))
                inferred_axial = injected-cap*(native[k+1]-native[k])/dt-leak*(native[k+1]-spec['passive_membrane']['e_pas_mv'])
                axial_balance = max(axial_balance, float(abs(inferred_axial.sum())))
                axial_ohm_error = max(axial_ohm_error, float(np.max(np.abs(inferred_axial-np.array([axial_out, -axial_out])))))
            if kind == 'disconnected':
                control = float(np.max(np.abs(native[:, 1]-native[0, 1] if leak[1] == 0 else
                    native[:, 1]-simulate_discrete(initial, cap, leak, 0.,
                    spec['passive_membrane']['e_pas_mv'], np.zeros(2), window, duration, dt)[:, 1])))
            elif kind == 'symmetric':
                control = float(np.max(np.abs(native[:, 0]-native[:, 1])))
            else:
                control = 0.
            paired.append({'geometry': geometry['id'], 'protocol': kind,
                'axial_conductance_us': axial, 'max_native_discrete_error_mv': error,
                'max_capacitive_balance_error_na': balance,
                'max_axial_antisymmetry_error_na': axial_balance,
                'max_native_axial_ohm_error_na': axial_ohm_error,
                'negative_control_error_mv': control,
                'passed': error <= limits['max_paired_native_discrete_error_mv']
                    and balance <= limits['max_capacitive_balance_error_na']
                    and axial_balance <= limits['max_axial_antisymmetry_error_na']
                    and axial_ohm_error <= limits['max_native_axial_ohm_error_na']
                    and (control <= (limits['max_disconnected_cross_effect_mv'] if kind == 'disconnected'
                                    else limits['max_symmetric_voltage_difference_mv']) if kind in ('disconnected','symmetric') else True)})
        print(f'[GIADA IV-D1] geometry {geometry["id"]}: 4 protocols', flush=True)

    geometry = spec['geometries'][0]
    membrane = spec['passive_membrane']
    cap, leak, axial = coefficients(geometry, membrane, True)
    pulse = current_for_case('parent_only', spec['paired_pulse_na'])
    initial = np.asarray(spec['paired_initial_mv'])
    convergence = []
    exact = exact_piecewise_endpoint(initial, cap, leak, axial,
                                     membrane['e_pas_mv'], pulse, window, duration)
    for spacing in spec['convergence_dt_ms']:
        prediction = simulate_discrete(initial, cap, leak, axial,
            membrane['e_pas_mv'], pulse, window, duration, spacing)
        convergence.append({'dt_ms': spacing,
            'endpoint_error_mv': float(np.max(np.abs(prediction[-1]-exact)))})
    convergence_pass = (convergence[0]['endpoint_error_mv'] > convergence[1]['endpoint_error_mv']
        > convergence[2]['endpoint_error_mv']
        and convergence[2]['endpoint_error_mv'] <= limits['max_convergence_fine_endpoint_error_mv'])
    equilibrium_current = pulse
    system = np.array([[leak[0]+axial, -axial], [-axial, leak[1]+axial]])
    equilibrium = np.linalg.solve(system, leak*membrane['e_pas_mv']+equilibrium_current)
    equilibrium_residual = float(np.max(np.abs(system@equilibrium-(leak*membrane['e_pas_mv']+equilibrium_current))))
    equilibrium_native = native_paired(h, geometry, membrane, initial, pulse,
                                      True, (0., 200.), 200., dt)[-1]
    equilibrium_native_error = float(np.max(np.abs(equilibrium_native-equilibrium)))
    d_pass = all(row['passed'] for row in paired) and convergence_pass and equilibrium_residual <= limits['max_equilibrium_residual_na'] and equilibrium_native_error <= limits['max_paired_native_discrete_error_mv']
    report = {'schema_version': spec['schema_version'], 'valid': True,
              'code_revision': revision, 'neuron_version': neuron.__version__,
              'iv_a1_a2_passed': True, 'iv_d1_opened': True, 'iv_d1_passed': d_pass,
              'task36_authorized': d_pass,
              'diagnosis': 'PASSIVE_AXIAL_PREREQUISITE_CONFIRMED' if d_pass else 'PASSIVE_AXIAL_NO_GO',
              'single_cases': singles, 'paired_cases': paired,
              'convergence': convergence, 'convergence_passed': convergence_pass,
              'equilibrium_residual_na': equilibrium_residual,
              'equilibrium_native_error_mv': equilibrium_native_error,
              'active_compartments_tested': False,
              'training_performed': False, 'full_cell_claim': False, 'speedup_claim': False}
    write(output/'final_report.json', report)
    return report
