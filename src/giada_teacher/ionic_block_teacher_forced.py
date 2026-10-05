"""Original Task28: explicit eleven-channel teacher-forced ionic composition.

Five channels reuse *frozen* Task27 rates. The other six retain their exact
NMODL rate equations; there is no claim that those six have been learned.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from . import heterogeneous_mechanism_composition as five
from . import single_gate_transfer as single
from . import calcium_gate_transfer as sk
from .hh_family_transfer import rates as hh_rates, sha, update, write
from .hh_family_transfer import TEACHER_REVISION

CHANNELS = ('Ca_HVA', 'Ca_LVAst', 'NaTa_t', 'NaTs2_t', 'Nap_Et2',
            'Ih', 'Im', 'K_Pst', 'K_Tst', 'SK_E2', 'SKv3_1')
STATE_NAMES = (('m', 'h'),) * 5 + (('m',), ('m',), ('m', 'h'), ('m', 'h'), ('z',), ('m',))
POWERS = np.array([2, 2, 3, 3, 3, 1, 1, 2, 4, 1, 1], dtype=np.int64)
REVERSALS = np.array([120, 120, 55, 55, 55, -45, -85, -85, -85, -85, -85.], dtype=np.float64)
GBAR = np.array([1e-5] * 9 + [1e-6, 1e-5], dtype=np.float64)
OFFSETS = np.cumsum([0] + [len(names) for names in STATE_NAMES])
assert OFFSETS[-1] == 18


def config(root: Path) -> dict:
    return json.loads((root / 'experiments/task28_ionic_block_teacher_forced.json').read_text(encoding='utf-8'))


def verify_parent(root: Path, cfg: dict) -> tuple[dict, dict, Path]:
    source = root / cfg['parent_result_dir']
    if {name: sha(source / name) for name in cfg['parent_sha256']} != cfg['parent_sha256']:
        raise RuntimeError('Task27 parent bytes changed')
    report = json.loads((source / 'final_report.json').read_text(encoding='utf-8'))
    freeze = json.loads((source / 'selection_freeze.json').read_text(encoding='utf-8'))
    audit = json.loads((source / 'result_audit.json').read_text(encoding='utf-8'))
    claimed = freeze.pop('freeze_sha256')
    digest = hashlib.sha256(json.dumps(freeze, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != claimed or not audit['valid'] or not all(audit['checks'].values()):
        raise RuntimeError('Task27 freeze or audit invalid')
    if not report['valid'] or not report['task28_preparation_authorized'] or report['fresh_used_for_selection']:
        raise RuntimeError('Task27 scientific prerequisite failed')
    if tuple(cfg['learned_channels']) != five.CHANNELS or tuple(cfg['channels']) != CHANNELS:
        raise RuntimeError('Task28 channel coverage changed')
    return report, freeze, source / 'artifact_bundle.zip'


def rates(channel: str, voltage: np.ndarray, calcium_mM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonical imposed-input rates; a length-one second gate is never invented."""
    v = np.asarray(voltage, dtype=np.float64)
    ca = np.asarray(calcium_mM, dtype=np.float64)
    if channel in five.CHANNELS:
        from . import calcium_pair_composition as calcium
        from . import sodium_family_composition as sodium
        return (calcium if channel in calcium.CHANNELS else sodium).rates(channel, v)
    if channel in ('Ih', 'Im'):
        return single.rates(channel, v)
    if channel == 'K_Pst':
        return hh_rates(channel, v)
    if channel == 'K_Tst':
        shifted = v + 10
        qt = 2.3 ** 1.3
        inf = np.stack((1 / (1 + np.exp(-shifted / 19)),
                        1 / (1 + np.exp((shifted + 66) / 10))), -1)
        tau = np.stack(((.34 + .92 * np.exp(-((shifted + 71) / 59) ** 2)) / qt,
                        (8 + 49 * np.exp(-((shifted + 73) / 23) ** 2)) / qt), -1)
        return inf, tau
    if channel == 'SK_E2':
        return sk.canonical_inf(ca), np.ones_like(ca)
    if channel == 'SKv3_1':
        return 1 / (1 + np.exp((v - 18.7) / -9.7)), 4 / (1 + np.exp((v + 46.56) / -44.14))
    raise ValueError(channel)


def exact_step(voltage: np.ndarray, calcium_mM: np.ndarray, state: np.ndarray,
               dt_ms: np.ndarray) -> np.ndarray:
    v = np.asarray(voltage, dtype=np.float64)
    ca = np.asarray(calcium_mM, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] != 18 or np.any(ca < 0):
        raise ValueError('18 explicit gate states and nonnegative imposed cai required')
    result = np.empty_like(state)
    for k, channel in enumerate(CHANNELS):
        inf, tau = rates(channel, v, ca)
        a, b = OFFSETS[k:k + 2]
        result[..., a:b] = update(state[..., a:b], inf, tau, dt_ms) if b-a == 2 else (
            single.update(state[..., a], inf, tau, dt_ms)[..., None])
    return result


def currents(voltage: np.ndarray, state: np.ndarray, multipliers: np.ndarray | None = None,
             reversals: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(voltage, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    gbar = GBAR if multipliers is None else GBAR * np.asarray(multipliers, dtype=np.float64)
    reversal = REVERSALS if reversals is None else np.asarray(reversals, dtype=np.float64)
    output = []
    for k, names in enumerate(STATE_NAMES):
        a = OFFSETS[k]
        opening = state[..., a] ** POWERS[k]
        if len(names) == 2:
            opening *= state[..., a + 1]
        output.append(gbar[k] * opening * (v - reversal[k]))
    return np.stack(output, -1)


def panel_multipliers() -> dict[str, np.ndarray]:
    ones = np.ones(len(CHANNELS))
    panels = {'canonical': ones}
    for label, indices in [('calcium_x4', (0, 1)), ('sodium_x4', (2, 3, 4)),
                           ('potassium_x4', (6, 7, 8, 9, 10))]:
        values = ones.copy(); values[list(indices)] = 4
        panels[label] = values
    values = ones.copy(); values[[0, 1, 5]] = 4
    panels['opposite_sign_cancellation'] = values
    return panels


def measure(voltage: np.ndarray, candidate: np.ndarray, truth: np.ndarray,
            cfg: dict) -> dict:
    gate = {}
    for k, name in enumerate(CHANNELS):
        a, b = OFFSETS[k:k + 2]
        error = candidate[..., a:b] - truth[..., a:b]
        gate[name] = {'rmse': float(np.sqrt(np.mean(error ** 2))),
                      'max_abs': float(np.max(np.abs(error))),
                      'occupancy_violations': int(np.count_nonzero((candidate[..., a:b] < 0) | (candidate[..., a:b] > 1)))}
    panels = {}
    for label, mult in panel_multipliers().items():
        predicted = currents(voltage, candidate, mult)
        target = currents(voltage, truth, mult)
        drive = GBAR[None, :] * mult[None, :] * (voltage[..., None] - REVERSALS[None, :])
        individual = np.zeros(len(CHANNELS))
        for k in range(len(CHANNELS)):
            mask = abs(drive[..., k]) > 1e-12
            individual[k] = float(np.sqrt(np.mean(((predicted[..., k] - target[..., k])[mask] / drive[..., k][mask]) ** 2))) if mask.any() else 0.
        total = (predicted - target).sum(-1) / np.maximum(np.abs(drive).sum(-1), 1e-12)
        cancellation = np.abs(target.sum(-1)) <= .1 * np.abs(target).sum(-1)
        panels[label] = {'worst_individual_normalized_rmse': float(max(individual)),
                         'per_channel_normalized_rmse': dict(zip(CHANNELS, map(float, individual))),
                         'total_normalized_rmse': float(np.sqrt(np.mean(total ** 2))),
                         'total_current_rmse_ma_cm2': float(np.sqrt(np.mean((predicted.sum(-1) - target.sum(-1)) ** 2))),
                         'cancellation_count': int(cancellation.sum()),
                         'cancellation_worst_individual_normalized_rmse': float(max(individual)) if cancellation.any() else None}
    return {'per_channel_gates': gate, 'current_panels': panels,
            'finite': bool(np.isfinite(candidate).all() and np.isfinite(truth).all())}


def load_frozen(root: Path, cfg: dict, torch, device: str):
    from .joint_heterogeneous_comparison import model_factory
    report, freeze, archive = verify_parent(root, cfg)
    output = {}
    seeds = freeze['config']['seeds']
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError('Task27 archive CRC failure')
        for family in cfg['frozen_families']:
            for arm in cfg['frozen_arms']:
                selected = report['selection'][family][arm]
                name = selected['checkpoint']
                raw = zipped.read(name)
                if hashlib.sha256(raw).hexdigest() != freeze['checkpoint_hashes'][name]:
                    raise RuntimeError('Task27 frozen checkpoint SHA-256 mismatch: ' + name)
                model = model_factory(torch, selected['width'], family, seeds).to(device)
                model.load_state_dict(torch.load(io.BytesIO(raw), map_location=device, weights_only=True))
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                output[family, arm] = model
    return output, seeds


def frozen_step(model, torch, device: str, voltage: np.ndarray, calcium_mM: np.ndarray,
                state: np.ndarray, dt_ms: np.ndarray) -> np.ndarray:
    """Only the first ten states are predicted; six formula mechanisms follow."""
    voltage = np.asarray(voltage, dtype=np.float64)
    calcium_mM = np.asarray(calcium_mM, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    dt_ms = np.broadcast_to(np.asarray(dt_ms, dtype=np.float64), voltage.shape)
    full = exact_step(voltage, calcium_mM, state, dt_ms)
    packed = np.c_[voltage, state[:, :10], dt_ms]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(packed), 2048):
            values = torch.tensor(packed[start:start + 2048], device=device, dtype=torch.float32)
            expanded = values[None].expand(model.w1.shape[0], -1, -1)
            prediction = model(expanded)[0].detach().cpu().double().numpy()
            predictions.append(prediction.transpose(0, 2, 1, 3).reshape(model.w1.shape[0], -1, 10))
    learned = np.concatenate(predictions, axis=1)
    return np.concatenate((learned, np.broadcast_to(full[None, :, 10:], (len(learned), len(full), 8))), -1)


def support(seed: int, count: int, cfg: dict, domain: str) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    voltage = rng.uniform(*cfg['voltage_range_mv'], count)
    log_ca = rng.uniform(*cfg['calcium_log10_range_mM'], count)
    state = rng.uniform(0, 1, (count, 18))
    dt = rng.choice(cfg['dt_ms'], count)
    if domain == 'activation_boundary':
        voltage = rng.uniform(-75, 5, count)
    elif domain == 'state_extrema':
        state = rng.choice([0., 1.], (count, 18))
    elif domain == 'calcium_tail':
        log_ca = rng.choice([-7., -2.], count) + rng.uniform(-.02, .02, count)
    elif domain != 'in_support':
        raise ValueError(domain)
    return voltage, 10 ** log_ca, state, dt


def native_audit(teacher: Path, output: Path, cfg: dict) -> dict:
    """Independent NEURON one-step and current checks for the six formula arms.

    The five learned mechanisms were native-audited in Task27 and their exact
    parent result bytes are verified again before any frozen evaluation.
    """
    import shutil
    import subprocess
    import sys
    import neuron
    from neuron import h, load_mechanisms

    if neuron.__version__.split('+')[0] != '8.2.7':
        raise RuntimeError('NEURON 8.2.7 required')
    revision = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != TEACHER_REVISION:
        raise RuntimeError('Canonical teacher revision mismatch')
    mod_dir = teacher / 'L5PC_NEURON_simulation/mods'
    build = output / 'native_formula_six'
    build.mkdir(parents=True, exist_ok=False)
    names = cfg['canonical_formula_channels']
    manifest = json.loads((Path(__file__).resolve().parents[2] / 'experiments/teacher_mechanism_inventory_v1.json').read_text(encoding='utf-8'))
    inventory = {row['mechanism']['name']: row for row in manifest['mechanisms']}
    hashes = {}
    for name in names:
        source = mod_dir / (name + '.mod')
        if name not in inventory or not source.is_file():
            raise RuntimeError('Canonical source absent: ' + name)
        raw = source.read_text(encoding='utf-8')
        needle = 'SUFFIX ' + name
        if raw.count(needle) != 1:
            raise RuntimeError('Ambiguous NMODL suffix: ' + name)
        exposure = 'zInf, zTau' if name == 'SK_E2' else 'mInf, mTau, hInf, hTau' if name in ('K_Pst', 'K_Tst') else 'mInf, mTau'
        modified = raw.replace(needle, needle + '\n RANGE ' + exposure, 1)
        (build / source.name).write_text(modified, encoding='utf-8')
        hashes[name] = {'canonical_mod_sha256': sha(source),
                        'instrumented_mod_sha256': sha(build / source.name)}
    nrnivmodl = shutil.which('nrnivmodl') or str(Path(sys.executable).parent / 'nrnivmodl')
    if not Path(nrnivmodl).is_file():
        raise RuntimeError('nrnivmodl unavailable')
    with (build / 'compile.log').open('w', encoding='utf-8') as log:
        subprocess.run([nrnivmodl, str(build.resolve())], cwd=build,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    load_mechanisms(str(build.resolve()))
    h.CVode().active(0); h.secondorder = 0
    rows = []
    for name in names:
        sec = h.Section(name='task28_' + name)
        sec.L = sec.diam = 10
        sec.insert(name)
        segment = sec(.5)
        mech = getattr(segment, name)
        gbar_name = 'g' + name + 'bar'
        setattr(mech, gbar_name, 0.)
        worst_step = worst_rate = worst_current = worst_voltage = 0.
        for voltage in (-120., -90., -65., -40., -15., 20., 60.):
            for calcium in (1e-7, 1e-4, 1e-2):
                h.dt = .025
                h.finitialize(voltage)
                if name not in ('Ih',):
                    segment.ek = -85.
                if name == 'SK_E2':
                    segment.cai = calcium
                initial = []
                for index, state_name in enumerate(STATE_NAMES[CHANNELS.index(name)]):
                    value = (.31, .79)[index]
                    setattr(mech, state_name, value)
                    initial.append(value)
                expected_inf, expected_tau = rates(name, np.array([voltage]), np.array([calcium]))
                h.fadvance()
                got = np.array([getattr(mech, state_name) for state_name in STATE_NAMES[CHANNELS.index(name)]])
                target = (single.update(np.array(initial), np.asarray(expected_inf).reshape(-1),
                                        np.asarray(expected_tau).reshape(-1), .025))
                worst_step = max(worst_step, float(np.max(np.abs(got - target))))
                if name == 'SK_E2':
                    native_inf, native_tau = mech.zInf, mech.zTau
                elif name in ('K_Pst', 'K_Tst'):
                    native_inf = np.array([mech.mInf, mech.hInf])
                    native_tau = np.array([mech.mTau, mech.hTau])
                else:
                    native_inf, native_tau = mech.mInf, mech.mTau
                worst_rate = max(worst_rate, float(np.max(np.abs(np.asarray(native_inf) - expected_inf))),
                                 float(np.max(np.abs(np.asarray(native_tau) - expected_tau))))
                worst_voltage = max(worst_voltage, abs(float(segment.v) - voltage))
                gbar = 1e-6 if name == 'SK_E2' else 1e-5
                setattr(mech, gbar_name, gbar)
                h.fcurrent()
                channel_index = CHANNELS.index(name)
                opening = got[0] ** POWERS[channel_index] * (got[1] if len(got) == 2 else 1.)
                expected_current = gbar * opening * (voltage - REVERSALS[channel_index])
                observed_current = float(mech.ihcn if name == 'Ih' else mech.ik)
                worst_current = max(worst_current, abs(observed_current - expected_current))
                setattr(mech, gbar_name, 0.)
        rows.append({'channel': name, 'maximum_step_error': worst_step,
                     'maximum_rate_error': worst_rate,
                     'maximum_current_error_ma_cm2': worst_current,
                     'maximum_voltage_drift_mv': worst_voltage,
                     'passed': bool(worst_step <= cfg['oracle_max_gate_error']
                                    and worst_rate <= cfg['oracle_max_gate_error']
                                    and worst_current <= cfg['oracle_max_current_error_ma_cm2']
                                    and worst_voltage <= 1e-8)})
        print(f'[GIADA Task28] native {name}: step={worst_step:.3g} current={worst_current:.3g}', flush=True)
    parent, _, _ = verify_parent(Path(__file__).resolve().parents[2], cfg)
    report = {'valid': all(row['passed'] for row in rows), 'rows': rows,
              'canonical_teacher_revision': revision, 'formula_source_hashes': hashes,
              'task27_native_audit_reused_after_hash_verification': bool(parent['valid']),
              'concentration_source': 'imposed_cai_mM'}
    write(output / 'native_audit.json', report)
    if not report['valid']:
        raise RuntimeError('Task28 formula-channel native oracle failed')
    return report


def run(root: Path, output: Path, cfg: dict, revision: str) -> dict:
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda' and cfg.get('require_cuda', True):
        raise RuntimeError('Task28 frozen GPU evaluation requires CUDA')
    models, seeds = load_frozen(root, cfg, torch, device)
    native = json.loads((output / 'native_audit.json').read_text(encoding='utf-8'))
    if not native['valid']:
        raise RuntimeError('Task28 native audit failed')
    rows = []
    start = time.monotonic()
    domains = ('in_support', 'activation_boundary', 'state_extrema', 'calcium_tail')
    for domain_index, domain in enumerate(domains):
        for support_seed in cfg['fresh_seeds']:
            voltage, calcium, state, dt = support(support_seed + 100 * domain_index, cfg['sample_count'], cfg, domain)
            target = exact_step(voltage, calcium, state, dt)
            for (family, arm), model in models.items():
                prediction = frozen_step(model, torch, device, voltage, calcium, state, dt)
                for index, seed in enumerate(seeds):
                    metrics = measure(voltage, prediction[index], target, cfg)
                    max_gate = max(metrics['per_channel_gates'][name]['rmse'] for name in CHANNELS)
                    worst_individual = max(panel['worst_individual_normalized_rmse'] for panel in metrics['current_panels'].values())
                    worst_total = max(panel['total_normalized_rmse'] for panel in metrics['current_panels'].values())
                    passed = (metrics['finite'] and max_gate <= cfg['learned_gate_rmse_limit']
                              and worst_individual <= cfg['learned_current_normalized_rmse_limit']
                              and worst_total <= cfg['learned_total_current_normalized_rmse_limit']
                              and all(row['occupancy_violations'] == 0 for row in metrics['per_channel_gates'].values()))
                    rows.append({'domain': domain, 'support_seed': support_seed, 'family': family,
                                 'arm': arm, 'model_seed': seed, 'passed': bool(passed),
                                 'worst_gate_rmse': max_gate, 'worst_individual_current_normalized_rmse': worst_individual,
                                 'worst_total_current_normalized_rmse': worst_total, 'metrics': metrics})
        print(f'[GIADA Task28] frozen domain {domain} complete', flush=True)
    # Long paths keep voltage and calcium externally prescribed. There is no
    # substitution of predicted voltage or calcium into any mechanism.
    path_rows = []
    steps = cfg['path_steps']; t = np.arange(steps) * cfg['path_dt_ms']
    for path in range(cfg['path_count']):
        voltage = -65 + 45 * np.sin((t + path * 17) / (9 + 13 * path))
        voltage += 15 * ((t.astype(int) // (23 + 7 * path)) % 2)
        calcium = 10 ** (-5 + 1.5 * np.sin((t + path * 13) / (35 + 13 * path)))
        rng = np.random.default_rng(280505 + path)
        state0 = rng.uniform(0, 1, 18)
        exact_inf = np.empty((steps, 18)); exact_tau = np.empty((steps, 18))
        for channel_index, channel in enumerate(CHANNELS):
            a, b = OFFSETS[channel_index:channel_index + 2]
            inf, tau = rates(channel, voltage, calcium)
            exact_inf[:, a:b] = np.asarray(inf).reshape(steps, b - a)
            exact_tau[:, a:b] = np.asarray(tau).reshape(steps, b - a)
        exact = state0.copy(); teacher = []
        for k in range(steps):
            z = -np.expm1(-cfg['path_dt_ms'] / exact_tau[k])
            exact = (1-z) * exact + z * exact_inf[k]
            teacher.append(exact)
        teacher = np.stack(teacher)
        for (family, arm), model in models.items():
            # Freeze rate predictions along the imposed path, then integrate
            # each model seed's own state without teacher-state injection.
            zero = np.zeros((steps, 18)); zero[:, :10] = .5
            packed = np.c_[voltage, zero[:, :10], np.full(steps, cfg['path_dt_ms'])]
            with torch.no_grad():
                x = torch.tensor(packed, device=device, dtype=torch.float32)[None].expand(len(seeds), -1, -1)
                _, inf, tau = model(x)
                inf = inf.detach().cpu().double().numpy().transpose(0, 2, 1, 3)
                tau = tau.detach().cpu().double().numpy().transpose(0, 2, 1, 3)
            for index, seed in enumerate(seeds):
                state = state0.copy(); trajectory = []
                for k in range(steps):
                    z = -np.expm1(-cfg['path_dt_ms'] / tau[index, k])
                    state[:10] = ((1-z) * state[:10].reshape(5, 2) + z * inf[index, k]).reshape(10)
                    state[10:] = teacher[k, 10:]
                    trajectory.append(state.copy())
                candidate = np.stack(trajectory)
                metrics = measure(voltage, candidate, teacher, cfg)
                worst_gate = max(v['rmse'] for v in metrics['per_channel_gates'].values())
                worst_current = max(v['worst_individual_normalized_rmse'] for v in metrics['current_panels'].values())
                passed = worst_gate <= cfg['learned_path_gate_rmse_limit'] and worst_current <= cfg['learned_path_current_normalized_rmse_limit']
                path_rows.append({'path': path, 'family': family, 'arm': arm, 'model_seed': seed,
                                  'passed': bool(passed), 'worst_gate_rmse': worst_gate,
                                  'worst_individual_current_normalized_rmse': worst_current,
                                  'metrics': metrics})
        print(f'[GIADA Task28] imposed path {path+1}/{cfg["path_count"]} complete', flush=True)
    per_arm = {family: {arm: all(r['passed'] for r in rows + path_rows if r['family'] == family and r['arm'] == arm)
                        for arm in cfg['frozen_arms']} for family in cfg['frozen_families']}
    passed = native['valid'] and all(per_arm[f]['both'] for f in cfg['frozen_families'])
    report = {'schema_version': 'giada-roadmap-task28-v1', 'valid': True,
              'teacher_forced_ionic_block_passed': bool(passed), 'per_arm_passed': per_arm,
              'gate_d_completed': False, 'task29_authorized': False,
              'compute_gate_measured': False, 'fresh_used_for_selection': False,
              'training_performed': False, 'channels': list(CHANNELS),
              'learned_channels': cfg['learned_channels'], 'formula_channels': cfg['canonical_formula_channels'],
              'input_contract': cfg['input_contract'], 'current_contract': cfg['current_contract'],
              'fresh_rows': rows, 'path_rows': path_rows,
              'elapsed_seconds': time.monotonic() - start, 'code_revision': revision,
              'limits': cfg['limits']}
    write(output / 'final_report.json', report)
    return report
