"""IV-B1/B2: canonical calcium concentration and SK composition checks.

The diagnostic current writer supplies an imposed ionic ica; it is not a
learned model or a substitute for the teacher's calcium channels.
"""

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


FARADAY = 96485.33212
INJECTOR = """NEURON {
    SUFFIX giada_ca_inject
    USEION ca WRITE ica
    RANGE amp
}
UNITS { (mA) = (milliamp) }
PARAMETER { amp = 0 (mA/cm2) }
ASSIGNED { ica (mA/cm2) }
BREAKPOINT { ica = amp }
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calcium_exact(cai, ica, dt, gamma, depth, decay, min_cai=1e-4):
    """Exact step for constant ica over dt; source units match CaDynamics_E2.mod."""
    if depth <= 0 or decay <= 0 or gamma < 0 or dt <= 0:
        raise ValueError("nonphysical CaDynamics parameter")
    source = -10000.0 * ica * gamma / (2.0 * FARADAY * depth)
    persistence = math.exp(-dt / decay)
    return min_cai + (cai - min_cai) * persistence + source * decay * (1.0 - persistence)


def sk_inf(cai):
    adjusted = cai + 1e-7 if cai < 1e-7 else cai
    return 1.0 / (1.0 + (0.00043 / adjusted) ** 4.8)


def current(protocol, t, config):
    for start, end, amplitude in config['current_protocol_ma_cm2'][protocol]:
        if start <= t < end:
            return float(amplitude)
    return 0.0


def compile_mechanisms(teacher, output, config):
    # Reuse the canonical CRLF-normalized source inventory check.
    if subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'],
                               text=True).strip() != config['teacher_revision']:
        raise RuntimeError('Canonical teacher revision mismatch')
    source = Path(teacher) / 'L5PC_NEURON_simulation/mods'
    inventory = json.loads((Path(__file__).resolve().parents[2] /
                            'experiments/teacher_mechanism_inventory_v1.json').read_text())
    canonical = {row['mechanism']['name']: row['sha256'] for row in inventory['mechanisms']}
    build = Path(output) / 'compiled_calcium'
    build.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ('CaDynamics_E2', 'SK_E2'):
        path = source / (name + '.mod')
        raw = path.read_bytes()
        normalized = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if hashlib.sha256(normalized).hexdigest() != canonical[name]:
            raise RuntimeError(f'Canonical source mismatch: {name}')
        shutil.copy2(path, build / path.name)
        hashes[name] = {'checkout_sha256': sha(path),
                        'inventory_crlf_sha256': canonical[name]}
    (build / 'giada_ca_inject.mod').write_bytes(INJECTOR.encode('ascii'))
    hashes['giada_ca_inject'] = {'sha256': sha(build / 'giada_ca_inject.mod'),
                                'diagnostic_only': True}
    command = shutil.which('nrnivmodl') or str(Path(sys.executable).parent / 'nrnivmodl')
    if not Path(command).is_file():
        raise RuntimeError('nrnivmodl unavailable')
    with (build / 'compile.log').open('w', encoding='utf-8') as log:
        subprocess.run([command, str(build.resolve())], cwd=build, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return build, hashes


def simulate_native(cai0, decay, gamma, depth, protocol, config, with_sk):
    import neuron
    from neuron import h
    if neuron.__version__.split('+')[0] != config['neuron_version']:
        raise RuntimeError('NEURON version mismatch')
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = config['dt_ms']
    h.celsius = config['temperature_c']
    sec = h.Section(name='giada_iv_b')
    sec.L = sec.diam = 10.0
    sec.nseg = 1
    sec.insert('CaDynamics_E2')
    sec.insert('giada_ca_inject')
    if with_sk:
        sec.insert('SK_E2')
    seg = sec(.5)
    cad = seg.CaDynamics_E2
    cad.gamma = gamma
    cad.depth = depth
    cad.decay = decay
    cad.minCai = 1e-4
    injector = seg.giada_ca_inject
    injector.amp = 0.0
    clamp = None
    if with_sk:
        seg.SK_E2.gSK_E2bar = config['b2_gbar_s_cm2']
        seg.ek = config['b2_ek_mv']
        clamp = h.SEClamp(seg)
        clamp.dur1 = 1e9
        clamp.amp1 = config['b2_imposed_voltage_mv']
        clamp.rs = 0.001
    h.finitialize(config['b2_imposed_voltage_mv'])
    # The concentration is a mechanism STATE and an ion concentration.
    seg.cai = float(cai0)
    try:
        seg.cai_CaDynamics_E2 = float(cai0)
    except (AttributeError, LookupError):
        pass
    if with_sk:
        seg.z_SK_E2 = sk_inf(cai0)
    h.fcurrent()
    if abs(float(seg.cai) - cai0) > 1e-12:
        raise RuntimeError('CaDynamics initial concentration was not installed')
    if with_sk and abs(float(seg.z_SK_E2) - sk_inf(cai0)) > 1e-12:
        raise RuntimeError('SK initial gate was not installed')
    dt = config['dt_ms']
    count = int(round(config['duration_ms'] / dt))
    arrays = {name: np.empty(count+1, dtype=np.float64)
              for name in ('cai', 'z', 'ik', 'v', 'ica')}
    arrays['cai'][0] = float(seg.cai)
    arrays['z'][0] = float(seg.z_SK_E2) if with_sk else np.nan
    arrays['ik'][0] = float(seg.ik) if with_sk else np.nan
    arrays['v'][0] = float(seg.v)
    arrays['ica'][0] = float(seg.ica)
    for index in range(count):
        imposed = current(protocol, index*dt, config)
        injector.amp = imposed
        h.fcurrent()
        if abs(float(seg.ica) - imposed) > 1e-10:
            raise RuntimeError(f'imposed ica not realized: {seg.ica} vs {imposed}')
        h.fadvance()
        h.fcurrent()
        arrays['cai'][index+1] = float(seg.cai)
        arrays['z'][index+1] = float(seg.z_SK_E2) if with_sk else np.nan
        arrays['ik'][index+1] = float(seg.ik) if with_sk else np.nan
        arrays['v'][index+1] = float(seg.v)
        arrays['ica'][index+1] = imposed
    h.delete_section(sec=sec)
    return arrays


def analytic(cai0, decay, gamma, depth, protocol, config, with_sk):
    dt = config['dt_ms']
    count = int(round(config['duration_ms'] / dt))
    cai = np.empty(count+1)
    z = np.empty(count+1)
    cai[0] = cai0
    z[0] = sk_inf(cai0)
    for index in range(count):
        amp = current(protocol, index*dt, config)
        cai[index+1] = calcium_exact(cai[index], amp, dt, gamma, depth, decay)
        if with_sk:
            # NEURON fixed-step orders CaDynamics before SK: updated cai is a
            # causal intermediate within the step, not a future teacher input.
            inf = sk_inf(cai[index+1])
            z[index+1] = inf + (z[index]-inf) * math.exp(-dt)
    return cai, z


def error(a, b):
    d = np.asarray(a) - np.asarray(b)
    return {'rmse': float(np.sqrt(np.mean(d*d))), 'max_abs': float(np.max(np.abs(d)))}


def run(teacher, output, config, revision):
    output = Path(output)
    if config['schema_version'] != 'giada-iv-b1-b2-calcium-prerequisite-v2':
        raise RuntimeError('IV-B confirmation requires the v2 semantic contract')
    build, hashes = compile_mechanisms(teacher, output, config)
    from neuron import load_mechanisms
    load_mechanisms(str(build.resolve()))
    b1_rows = []
    b2_rows = []
    raw = {}
    for cai0 in config['b1_initial_cai_mM']:
        for decay in config['b1_decay_ms']:
            for gamma, depth in config['b1_gamma_depth']:
                for protocol in config['b1_protocols']:
                    native = simulate_native(cai0, decay, gamma, depth, protocol, config, False)
                    calculated, _ = analytic(cai0, decay, gamma, depth, protocol, config, False)
                    metrics = error(native['cai'], calculated)
                    b1_rows.append({'cai0': cai0, 'decay_ms': decay, 'gamma': gamma,
                                    'depth_um': depth, 'protocol': protocol, **metrics,
                                    'native_min_cai_mM': float(native['cai'].min()),
                                    'native_max_cai_mM': float(native['cai'].max())})
    b1_pass = all(row['rmse'] <= config['b1_cai_rmse_limit_mM']
                  and row['max_abs'] <= config['b1_cai_max_error_limit_mM']
                  and row['native_min_cai_mM'] >= 0 for row in b1_rows)
    if b1_pass:
        gamma, depth = config['b1_gamma_depth'][0]
        decay = 80.0
        for cai0 in config['b2_initial_cai_mM']:
            for protocol in config['b2_protocols']:
                native = simulate_native(cai0, decay, gamma, depth, protocol, config, True)
                calculated_cai, calculated_z = analytic(cai0, decay, gamma, depth,
                                                        protocol, config, True)
                native_cai_gate = np.empty_like(calculated_z)
                native_cai_gate[0] = sk_inf(cai0)
                for index in range(len(native_cai_gate)-1):
                    inf = sk_inf(native['cai'][index+1])
                    native_cai_gate[index+1] = inf + (native_cai_gate[index]-inf)*math.exp(-config['dt_ms'])
                expected_ik = config['b2_gbar_s_cm2'] * calculated_z * (
                    native['v'] - config['b2_ek_mv'])
                row = {'cai0': cai0, 'protocol': protocol,
                       'cai': error(native['cai'], calculated_cai),
                       'sk_oracle_cai': error(native['z'], native_cai_gate),
                       'sk_composed': error(native['z'], calculated_z),
                       'sk_current': error(native['ik'], expected_ik),
                       'voltage_clamp_max_error_mv': float(np.max(np.abs(
                           native['v']-config['b2_imposed_voltage_mv']))),
                       'native_cai_max_mM': float(native['cai'].max()),
                       'native_sk_max': float(native['z'].max())}
                b2_rows.append(row)
                raw[f'{cai0}_{protocol}_cai'] = native['cai']
                raw[f'{cai0}_{protocol}_z'] = native['z']
        np.savez_compressed(output / 'b2_native_traces.npz', **raw)
    pulse_contrast = {}
    if b1_pass:
        for cai0 in config['b2_initial_cai_mM']:
            rest = next(row for row in b2_rows if row['cai0'] == cai0 and row['protocol'] == 'rest')
            for protocol in config['b2_protocols']:
                if protocol == 'rest':
                    continue
                pulse = next(row for row in b2_rows if row['cai0'] == cai0 and row['protocol'] == protocol)
                pulse_contrast[f'{cai0}_{protocol}'] = {
                    'peak_cai_difference_mM': pulse['native_cai_max_mM'] - rest['native_cai_max_mM'],
                    'peak_sk_difference': pulse['native_sk_max'] - rest['native_sk_max']}
    perturbation_informative = bool(pulse_contrast and all(
        row['peak_cai_difference_mM'] > 1e-5 and row['peak_sk_difference'] > 1e-3
        for row in pulse_contrast.values()))
    b2_pass = bool(b1_pass and perturbation_informative and all(
        row['cai']['rmse'] <= config['b2_cai_rmse_limit_mM']
        and row['sk_oracle_cai']['rmse'] <= config['b2_sk_gate_rmse_limit']
        and row['sk_composed']['rmse'] <= config['b2_sk_gate_rmse_limit']
        and row['sk_current']['rmse'] <= config['b2_sk_current_rmse_limit_ma_cm2']
        and row['voltage_clamp_max_error_mv'] <= config['b2_voltage_clamp_max_error_mv']
        for row in b2_rows))
    report = {'schema_version': config['schema_version'], 'valid': bool(b1_pass and b2_pass),
              'iv_b1_valid': bool(b1_pass), 'iv_b2_valid': b2_pass,
              'task32_feedback_authorized': b2_pass,
              'b1_episode_count': len(b1_rows), 'b2_episode_count': len(b2_rows),
              'b1_rows': b1_rows, 'b2_rows': b2_rows, 'canonical_source_hashes': hashes,
              'pulse_vs_rest_negative_control': pulse_contrast,
              'perturbation_informative': perturbation_informative,
              'native_neuron_version': config['neuron_version'], 'code_revision': revision,
              'training_performed': False, 'fresh_used_for_selection': False,
              'scope': config['scope']}
    (output / 'final_report.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    return report
