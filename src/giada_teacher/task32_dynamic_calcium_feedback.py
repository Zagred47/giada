"""Task32: prospective autonomous voltage--calcium--SK feedback canary.

The formula/native floor is a separate gate. A floor failure is not charged to
the frozen learned rates. Calcium and SK use the validated causal within-step
ordering, while the calcium-current source timing is calibrated separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from . import ionic_block_teacher_forced as ionic
from . import task30_autonomous_voltage as t30
from . import task30c_native_active_confirmation as t30c
from .iv_b_calcium_prerequisite import FARADAY, sk_inf
from .hh_family_transfer import write


def verify(root: Path, teacher: Path, cfg: dict) -> dict:
    for key, digest_key in [('parent_task30c_report', 'parent_task30c_sha256'),
                            ('parent_iv_b_report', 'parent_iv_b_sha256')]:
        if hashlib.sha256((root / cfg[key]).read_bytes()).hexdigest() != cfg[digest_key]:
            raise RuntimeError(f'Task32 parent changed: {key}')
    archive = root / 'experiments/results/iv_b_v2_kaggle_8b3a4e5/artifact_bundle.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != cfg['parent_iv_b_archive_sha256']:
        raise RuntimeError('IV-B2 archived native artifact changed')
    p30 = json.loads((root / cfg['parent_task30c_report']).read_text())
    pb = json.loads((root / cfg['parent_iv_b_report']).read_text())
    if not (p30['scientific_primary_passed'] and p30['native_floor_admissible']
            and pb['iv_b1_valid'] and pb['iv_b2_valid'] and pb['task32_feedback_authorized']):
        raise RuntimeError('Task32 prerequisites not passed')
    revision = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != cfg['teacher_revision']:
        raise RuntimeError('Canonical teacher revision changed')
    base = t30.config(root, Path(cfg['base_config']).name)
    if cfg['dt_ms'] != base['dt_ms'] or cfg['model_seeds'] != base['model_seeds']:
        raise RuntimeError('Task32 sampling or frozen seeds disagree with Task30b')
    if cfg['temperature_c'] != 6.3:
        raise RuntimeError('Task32 must explicitly preserve Task30c default temperature')
    if set(cfg['calibration_protocols']) & set(cfg['confirmation_protocols']):
        raise RuntimeError('Calibrating and confirmation protocols overlap')
    return base


def design(cfg: dict, base: dict) -> dict:
    rows = [{'initial_voltage_mv': initial, 'protocol': protocol, 'panel': panel}
            for initial in cfg['initial_voltage_mv']
            for protocol in cfg['protocols'] for panel in cfg['panels']]
    time = np.arange(int(round(cfg['duration_ms']/cfg['dt_ms']))+1) * cfg['dt_ms']
    multipliers = []
    for row in rows:
        value = ionic.panel_multipliers()['canonical'].copy()
        if row['panel'] == 'calcium_x16':
            value[:2] *= 16
        elif row['panel'] != 'canonical':
            raise ValueError(row['panel'])
        multipliers.append(value)
    return {'rows': rows, 'time_ms': time,
            'injection_ma_cm2': np.stack([t30._injection(row['protocol'], time[:-1])
                                          for row in rows], axis=1),
            'multipliers': np.stack(multipliers)}


def compile_native(teacher: Path, output: Path, cfg: dict) -> tuple[Path, dict]:
    source = teacher / 'L5PC_NEURON_simulation/mods'
    inventory = json.loads((Path(__file__).resolve().parents[2] /
                            'experiments/teacher_mechanism_inventory_v1.json').read_text())
    canonical = {row['mechanism']['name']: row['sha256'] for row in inventory['mechanisms']}
    build = output / 'compiled_native_dynamic_calcium'
    build.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in (*ionic.CHANNELS, 'CaDynamics_E2'):
        path = source / (name + '.mod')
        raw = path.read_bytes()
        normalized = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if hashlib.sha256(normalized).hexdigest() != canonical[name]:
            raise RuntimeError(f'Canonical NMODL source mismatch: {name}')
        shutil.copy2(path, build / path.name)
        hashes[name] = {'checkout_sha256': hashlib.sha256(raw).hexdigest(),
                        'inventory_crlf_sha256': canonical[name]}
    command = shutil.which('nrnivmodl') or str(Path(sys.executable).parent / 'nrnivmodl')
    if not Path(command).is_file():
        raise RuntimeError('nrnivmodl unavailable')
    with (build / 'compile.log').open('w', encoding='utf-8') as log:
        subprocess.run([command, str(build.resolve())], cwd=build,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return build, hashes


def native_reference(d: dict, cfg: dict, base: dict) -> dict:
    import neuron
    from neuron import h
    if neuron.__version__.split('+')[0] != cfg['neuron_version']:
        raise RuntimeError('Native NEURON version changed')
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = cfg['dt_ms']
    h.celsius = cfg['temperature_c']
    steps, count = len(d['time_ms']), len(d['rows'])
    voltage = np.empty((steps, count))
    calcium = np.empty_like(voltage)
    ica = np.empty_like(voltage)
    gates = np.empty((steps, count, 18))
    for j, row in enumerate(d['rows']):
        sec = h.Section(name=f'giada_task32_{j}')
        sec.L = sec.diam = 10.
        sec.nseg = 1
        sec.cm = base['cm_uf_cm2']
        sec.insert('pas')
        sec.g_pas = base['g_pas_s_cm2']
        sec.e_pas = base['e_pas_mv']
        for name in (*ionic.CHANNELS, 'CaDynamics_E2'):
            sec.insert(name)
        seg = sec(.5)
        area = float(h.area(.5, sec=sec))
        clamps = []
        for start, duration, density in t30c._pulses(d['injection_ma_cm2'][:, j], cfg['dt_ms']):
            clamp = h.IClamp(seg)
            clamp.delay, clamp.dur, clamp.amp = start, duration, density * area * .01
            clamps.append(clamp)
        initial_v = float(row['initial_voltage_mv'])
        initial_cai = float(cfg['initial_cai_mM'])
        initial_gate = t30.initial_states(np.array([initial_v]), np.array([initial_cai]))[0]
        h.finitialize(initial_v)
        seg.eca, seg.ena, seg.ek = 120., 55., -85.
        seg.cai = initial_cai
        try:
            seg.cai_CaDynamics_E2 = initial_cai
        except (AttributeError, LookupError):
            pass
        cad = seg.CaDynamics_E2
        cad.decay, cad.gamma, cad.depth = cfg['decay_ms'], cfg['gamma'], cfg['depth_um']
        cad.minCai = 1e-4
        for k, name in enumerate(ionic.CHANNELS):
            mech = getattr(seg, name)
            setattr(mech, 'g'+name+'bar', float(ionic.GBAR[k] * d['multipliers'][j, k]))
            a = ionic.OFFSETS[k]
            for q, state_name in enumerate(ionic.STATE_NAMES[k]):
                setattr(mech, state_name, float(initial_gate[a+q]))
        h.fcurrent()
        if abs(float(seg.cai)-initial_cai) > 1e-12:
            raise RuntimeError('Native initial calcium state was not installed')
        for n in range(steps):
            voltage[n, j] = float(seg.v)
            calcium[n, j] = float(seg.cai)
            ica[n, j] = float(seg.ica)
            for k, name in enumerate(ionic.CHANNELS):
                a = ionic.OFFSETS[k]
                mech = getattr(seg, name)
                for q, state_name in enumerate(ionic.STATE_NAMES[k]):
                    gates[n, j, a+q] = float(getattr(mech, state_name))
            if n < steps-1:
                h.fadvance()
        h.delete_section(sec=sec)
        if (j+1) % 4 == 0:
            print(f'[GIADA Task32] native {j+1}/{count}', flush=True)
    return {'voltage': voltage, 'calcium': calcium, 'ica': ica, 'gates': gates}


def calcium_step(cai: np.ndarray, ica: np.ndarray, cfg: dict) -> np.ndarray:
    persistence = np.exp(-cfg['dt_ms']/cfg['decay_ms'])
    source = -10000 * ica * cfg['gamma'] / (2 * FARADAY * cfg['depth_um'])
    return (1e-4 + (cai-1e-4)*persistence
            + source*cfg['decay_ms']*(1-persistence))


def _ca_current(v: np.ndarray, state: np.ndarray, mult: np.ndarray) -> np.ndarray:
    hva = ionic.GBAR[0] * mult[..., 0] * state[..., 0]**2 * state[..., 1] * (v-120.)
    lva = ionic.GBAR[1] * mult[..., 1] * state[..., 2]**2 * state[..., 3] * (v-120.)
    return hva + lva


def _sk_update(state_next: np.ndarray, state_old: np.ndarray,
               cai_next: np.ndarray, dt: float) -> None:
    index = int(ionic.OFFSETS[9])
    inf = np.asarray(sk_inf(cai_next)) if np.ndim(cai_next) == 0 else (
        1/(1+(0.00043/np.maximum(cai_next, 1e-7))**4.8))
    state_next[..., index] = inf + (state_old[..., index]-inf)*np.exp(-dt)


def rollout(d: dict, cfg: dict, base: dict, scheme: str,
            model=None, torch=None, frozen_cai: bool = False) -> dict:
    steps, count = len(d['time_ms']), len(d['rows'])
    seeds = len(cfg['model_seeds']) if model is not None else 1
    v = np.empty((seeds, steps, count))
    cai = np.empty_like(v)
    states = np.empty((seeds, steps, count, 18))
    v[:, 0] = [row['initial_voltage_mv'] for row in d['rows']]
    cai[:, 0] = cfg['initial_cai_mM']
    states[:, 0] = t30.initial_states(v[0, 0], cai[0, 0])
    multipliers = np.broadcast_to(d['multipliers'], (seeds, count, 11))
    for k in range(steps-1):
        if model is None:
            next_state = ionic.exact_step(v[:, k], cai[:, k], states[:, k], cfg['dt_ms'])
        else:
            next_state = t30._candidate_gate_step(model, torch, v[:, k], cai[:, k],
                                                   states[:, k], cfg['dt_ms'])
        if frozen_cai:
            cai[:, k+1] = cai[:, k]
        else:
            source_state = states[:, k] if scheme == 'old_state_ica' else next_state
            if scheme not in cfg['source_hypotheses']:
                raise ValueError(scheme)
            source = _ca_current(v[:, k], source_state, multipliers)
            cai[:, k+1] = calcium_step(cai[:, k], source, cfg)
        _sk_update(next_state, states[:, k], cai[:, k+1], cfg['dt_ms'])
        states[:, k+1] = next_state
        v[:, k+1] = t30.voltage_step(v[:, k], next_state, multipliers,
                                     d['injection_ma_cm2'][k], base)
        if not (np.isfinite(v[:, k+1]).all() and np.isfinite(cai[:, k+1]).all()
                and np.isfinite(next_state).all()):
            raise RuntimeError(f'Nonfinite coupled rollout at step {k+1}')
    return {'voltage': v, 'calcium': cai, 'gates': states}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(a)-np.asarray(b)))))


def episode_metrics(prediction: dict, native: dict, d: dict, cfg: dict,
                    indices: list[int], horizon_ms: float, seed_index: int = 0) -> list[dict]:
    length = int(round(horizon_ms/cfg['dt_ms']))+1
    rows = []
    for j in indices:
        v = prediction['voltage'][seed_index, :length, j]
        ca = prediction['calcium'][seed_index, :length, j]
        gates = prediction['gates'][seed_index, :length, j]
        truth_v = native['voltage'][:length, j]
        truth_ca = native['calcium'][:length, j]
        truth_gates = native['gates'][:length, j]
        rows.append({'episode_index': j, **d['rows'][j], 'horizon_ms': horizon_ms,
                     'voltage_rmse_mv': rmse(v, truth_v),
                     'cai_rmse_mM': rmse(ca, truth_ca),
                     'sk_gate_rmse': rmse(gates[:, int(ionic.OFFSETS[9])],
                                          truth_gates[:, int(ionic.OFFSETS[9])]),
                     'finite': bool(np.isfinite(v).all() and np.isfinite(ca).all()
                                    and np.isfinite(gates).all()),
                     'occupancy_violations': int(np.count_nonzero((gates < 0) | (gates > 1))),
                     'physical_voltage_violations': int(np.count_nonzero(
                         (v < cfg['physical_voltage_range_mv'][0]) |
                         (v > cfg['physical_voltage_range_mv'][1])))})
    return rows


def aggregate(rows: list[dict]) -> dict:
    values = [row['voltage_rmse_mv'] for row in rows]
    return {'pooled_voltage_rmse_mv': rmse(values, np.zeros(len(values))),
            'worst_episode_voltage_rmse_mv': max(values),
            'worst_episode_cai_rmse_mM': max(row['cai_rmse_mM'] for row in rows),
            'worst_episode_sk_gate_rmse': max(row['sk_gate_rmse'] for row in rows),
            'finite': all(row['finite'] for row in rows),
            'occupancy_violations': sum(row['occupancy_violations'] for row in rows),
            'physical_voltage_violations': sum(row['physical_voltage_violations'] for row in rows)}


def run(root: Path, teacher: Path, output: Path, cfg: dict, revision: str) -> dict:
    base = verify(root, teacher, cfg)
    d = design(cfg, base)
    build, hashes = compile_native(teacher, output, cfg)
    from neuron import load_mechanisms
    load_mechanisms(str(build.resolve()))
    native = native_reference(d, cfg, base)
    if not all(np.isfinite(array).all() for array in native.values()):
        raise RuntimeError('Nonfinite native voltage/calcium/gates/current')
    np.savez_compressed(output / 'native_traces.npz', **native)
    calibration = [j for j, row in enumerate(d['rows'])
                   if row['protocol'] in cfg['calibration_protocols']]
    confirmation = [j for j, row in enumerate(d['rows'])
                    if row['protocol'] in cfg['confirmation_protocols']]
    if len(calibration) != 8 or len(confirmation) != 8:
        raise RuntimeError('Task32 calibration/confirmation support changed')
    formulas = {scheme: rollout(d, cfg, base, scheme)
                for scheme in cfg['source_hypotheses']}
    calibration_scores = {scheme: aggregate(episode_metrics(
        result, native, d, cfg, calibration, cfg['primary_horizon_ms']))
        for scheme, result in formulas.items()}
    selected = min(cfg['source_hypotheses'], key=lambda scheme:
                   calibration_scores[scheme]['worst_episode_voltage_rmse_mv'])
    formula_rows = episode_metrics(formulas[selected], native, d, cfg,
                                   confirmation, cfg['primary_horizon_ms'])
    floor = aggregate(formula_rows)
    floor_pass = bool(floor['finite'] and floor['occupancy_violations'] == 0
        and floor['physical_voltage_violations'] == 0
        and floor['pooled_voltage_rmse_mv'] <= cfg['formula_floor_pooled_limit_mv']
        and floor['worst_episode_voltage_rmse_mv'] <= cfg['formula_floor_worst_episode_limit_mv']
        and floor['worst_episode_cai_rmse_mM'] <= cfg['formula_floor_cai_rmse_limit_mM'])
    frozen = rollout(d, cfg, base, selected, frozen_cai=True)
    negative_control = []
    for j, row in enumerate(d['rows']):
        if row['protocol'] == 'rest':
            continue
        negative_control.append({'episode_index': j, **row,
            'peak_cai_difference_mM': float(np.max(np.abs(
                formulas[selected]['calcium'][0, :, j]-frozen['calcium'][0, :, j]))),
            'peak_sk_difference': float(np.max(np.abs(
                formulas[selected]['gates'][0, :, j, int(ionic.OFFSETS[9])]
                -frozen['gates'][0, :, j, int(ionic.OFFSETS[9])]))),
            'voltage_rmse_mv': rmse(formulas[selected]['voltage'][0, :, j],
                                    frozen['voltage'][0, :, j])})
    models = {}
    if floor_pass:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Task32 frozen model evaluation requires CUDA')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        loaded, seeds = ionic.load_frozen(root, ionic.config(root), torch, 'cuda')
        if list(seeds) != cfg['model_seeds']:
            raise RuntimeError('Frozen model seed contract changed')
        for family in cfg['families']:
            predicted = rollout(d, cfg, base, selected,
                                loaded[family, cfg['frozen_arm']], torch)
            models[family] = {}
            for seed_index, seed in enumerate(seeds):
                rows = episode_metrics(predicted, native, d, cfg, confirmation,
                                       cfg['primary_horizon_ms'], seed_index)
                primary = aggregate(rows)
                diagnostic = aggregate(episode_metrics(predicted, native, d, cfg,
                    confirmation, cfg['diagnostic_horizon_ms'], seed_index))
                primary['passed'] = bool(primary['finite']
                    and primary['occupancy_violations'] == 0
                    and primary['physical_voltage_violations'] == 0
                    and primary['pooled_voltage_rmse_mv'] <= cfg['candidate_pooled_voltage_limit_mv']
                    and primary['worst_episode_voltage_rmse_mv'] <= cfg['candidate_worst_episode_voltage_limit_mv']
                    and primary['worst_episode_cai_rmse_mM'] <= cfg['candidate_cai_rmse_limit_mM']
                    and primary['worst_episode_sk_gate_rmse'] <= cfg['candidate_sk_gate_rmse_limit'])
                models[family][str(seed)] = {'primary': primary,
                                            'diagnostic_80ms': diagnostic, 'rows': rows}
            print(f'[GIADA Task32] frozen family {family} evaluated', flush=True)
    report = {'schema_version': cfg['schema_version'], 'valid': True,
              'native_floor_admissible': floor_pass,
              'scientific_primary_passed': bool(floor_pass and models and all(
                  result['primary']['passed'] for family in models.values()
                  for result in family.values())),
              'source_hypothesis_selected_on_calibration_only': selected,
              'calibration_scores': calibration_scores,
              'confirmation_formula_floor': floor,
              'confirmation_formula_rows': formula_rows,
              'formula_80ms_diagnostic': aggregate(episode_metrics(
                  formulas[selected], native, d, cfg, confirmation,
                  cfg['diagnostic_horizon_ms'])),
              'frozen_cai_negative_control': negative_control,
              'models': models, 'episode_design': d['rows'],
              'native_calcium_range_mM': [float(native['calcium'].min()),
                                           float(native['calcium'].max())],
              'native_voltage_range_mv': [float(native['voltage'].min()),
                                           float(native['voltage'].max())],
              'canonical_channel_sha256': hashes, 'code_revision': revision,
              'training_performed': False, 'fresh_used_for_selection': False,
              'gate_d_performance_status': 'NO_GO_UNCHANGED',
              'scope': cfg['scope']}
    write(output / 'final_report.json', report)
    return report
