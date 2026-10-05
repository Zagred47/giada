"""Task30c: native active NEURON reference for the frozen Task30b design."""
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
from .hh_family_transfer import write


def verify(root: Path, teacher: Path, cfg: dict) -> dict:
    parent = root / cfg['parent_result']
    if hashlib.sha256((parent / 'final_report.json').read_bytes()).hexdigest() != cfg['parent_report_sha256']:
        raise RuntimeError('Task30b report hash mismatch')
    if hashlib.sha256((parent / 'artifact_bundle.zip').read_bytes()).hexdigest() != cfg['parent_artifact_sha256']:
        raise RuntimeError('Task30b artifact hash mismatch')
    report = json.loads((parent / 'final_report.json').read_text(encoding='utf-8'))
    audit = json.loads((parent / 'result_audit.json').read_text(encoding='utf-8'))
    if not (report['valid'] and report['scientific_primary_passed'] and audit['valid']
            and audit['decision_grade_within_formula_map_scope'] and not audit['task31_authorized']):
        raise RuntimeError('Task30b decision contract mismatch')
    old = t30.config(root, Path(cfg['parent_config']).name)
    t30.verify_parent(root, old)
    if tuple(cfg['channels']) != ionic.CHANNELS:
        raise RuntimeError('Channel coverage changed')
    revision = subprocess.check_output(['git', '-c', f'safe.directory={teacher.resolve()}',
                                        '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != cfg['teacher_revision']:
        raise RuntimeError('Teacher revision mismatch')
    if cfg['primary_horizon_ms'] != old['primary_horizon_ms']:
        raise RuntimeError('Primary horizon changed')
    return old


def compile_native(teacher: Path, output: Path, cfg: dict) -> tuple[Path, dict]:
    source = teacher / 'L5PC_NEURON_simulation/mods'
    inventory = json.loads((Path(__file__).resolve().parents[2] / 'experiments/teacher_mechanism_inventory_v1.json').read_text(encoding='utf-8'))
    canonical = {row['mechanism']['name']: row['sha256'] for row in inventory['mechanisms']}
    build = output / 'compiled_native_eleven'
    build.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in cfg['channels']:
        mod = source / (name + '.mod')
        digest = hashlib.sha256(mod.read_bytes()).hexdigest()
        if digest != canonical[name]:
            raise RuntimeError(f'Canonical NMODL hash mismatch: {name}')
        shutil.copy2(mod, build / mod.name)
        hashes[name] = digest
    command = shutil.which('nrnivmodl') or str(Path(sys.executable).parent / 'nrnivmodl')
    if not Path(command).is_file():
        raise RuntimeError('nrnivmodl unavailable')
    with (build / 'compile.log').open('w', encoding='utf-8') as log:
        subprocess.run([command, str(build.resolve())], cwd=build, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return build, hashes


def _pulses(density: np.ndarray, dt: float) -> list[tuple[float, float, float]]:
    change = np.r_[0, np.flatnonzero(np.diff(density) != 0) + 1, len(density)]
    return [(float(a*dt), float((b-a)*dt), float(density[a]))
            for a, b in zip(change[:-1], change[1:]) if density[a] != 0]


def native_reference(design: dict, old: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    import neuron
    from neuron import h
    if neuron.__version__.split('+')[0] != '8.2.7':
        raise RuntimeError(f'NEURON 8.2.7 required, got {neuron.__version__}')
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = old['dt_ms']
    steps, count = old['steps'], len(design['rows'])
    voltage = np.empty((steps, count), dtype=np.float64)
    gates = np.empty((steps, count, 18), dtype=np.float64)
    initial_v = np.array([row['initial_voltage_mv'] for row in design['rows']], dtype=np.float64)
    initial_gate = t30.initial_states(initial_v, design['calcium_mM'])
    section_area = []
    for j, row in enumerate(design['rows']):
        sec = h.Section(name=f'giada_task30c_{j}')
        sec.L = sec.diam = 10.
        sec.nseg = 1
        sec.cm = old['cm_uf_cm2']
        sec.insert('pas')
        sec.g_pas = old['g_pas_s_cm2']
        sec.e_pas = old['e_pas_mv']
        for k, name in enumerate(ionic.CHANNELS):
            sec.insert(name)
        seg = sec(.5)
        area = float(h.area(.5, sec=sec))
        section_area.append(area)
        clamps = []
        for start, duration, density in _pulses(design['injection_ma_cm2'][:, j], old['dt_ms']):
            clamp = h.IClamp(seg)
            clamp.delay, clamp.dur, clamp.amp = start, duration, density * area * .01
            clamps.append(clamp)
        h.finitialize(float(initial_v[j]))
        seg.eca, seg.ena, seg.ek = 120., 55., -85.
        seg.cai = float(design['calcium_mM'][j])
        for k, name in enumerate(ionic.CHANNELS):
            mech = getattr(seg, name)
            setattr(mech, 'g'+name+'bar', float(ionic.GBAR[k] * design['multipliers'][j, k]))
            a, b = ionic.OFFSETS[k:k+2]
            for q, state_name in enumerate(ionic.STATE_NAMES[k]):
                setattr(mech, state_name, float(initial_gate[j, a+q]))
        h.fcurrent()
        for n in range(steps):
            voltage[n, j] = float(seg.v)
            for k, name in enumerate(ionic.CHANNELS):
                a = ionic.OFFSETS[k]
                mech = getattr(seg, name)
                for q, state_name in enumerate(ionic.STATE_NAMES[k]):
                    gates[n, j, a+q] = float(getattr(mech, state_name))
            if n < steps-1:
                h.fadvance()
                # No CaDynamics mechanism is inserted: cai must remain imposed.
                if abs(float(seg.cai) - design['calcium_mM'][j]) > 1e-10:
                    raise RuntimeError(f'Imposed cai drifted in episode {j}')
        h.delete_section(sec=sec)
        if (j+1) % 8 == 0:
            print(f'[GIADA Task30c] native episodes {j+1}/{count}', flush=True)
    return voltage, gates, {'neuron_version': neuron.__version__,
                            'section_area_um2_range': [min(section_area), max(section_area)],
                            'calcium_imposed': True, 'reversals_imposed': True}


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def evaluate(reference: np.ndarray, native: np.ndarray, candidates: dict,
             design: dict, old: dict, cfg: dict, seeds: list[int]) -> dict:
    primary = int(round(cfg['primary_horizon_ms']/old['dt_ms']))
    exposure = design['injection_ma_cm2'][:primary]
    active = np.any(exposure != 0, axis=0)
    if int(active.sum()) != 24:
        raise RuntimeError('Native primary exposure preflight failed')
    rows = []
    for j, episode in enumerate(design['rows']):
        for horizon in [cfg['primary_horizon_ms'], *cfg['diagnostic_horizons_ms']]:
            n = int(round(horizon/old['dt_ms'])) + 1
            base = {'episode_index': j, **episode, 'horizon_ms': horizon,
                    'active_before_primary': bool(active[j])}
            rows.append({**base, 'arm': 'formula_vs_native',
                         'voltage_rmse_mv': _rmse(reference[:n,j]-native[:n,j]),
                         'native_crossings': int(np.count_nonzero((native[:n-1,j] < old['event_threshold_mv']) & (native[1:n,j] >= old['event_threshold_mv']))),
                         'formula_crossings': int(np.count_nonzero((reference[:n-1,j] < old['event_threshold_mv']) & (reference[1:n,j] >= old['event_threshold_mv'])))})
            for (family, arm), rollout in candidates.items():
                for seed_index, seed in enumerate(seeds):
                    predicted = rollout['autonomous_voltage'][seed_index,:n,j]
                    rows.append({**base, 'arm': arm, 'family': family, 'seed': seed,
                                 'voltage_rmse_mv': _rmse(predicted-native[:n,j]),
                                 'candidate_crossings': int(np.count_nonzero((predicted[:-1] < old['event_threshold_mv']) & (predicted[1:] >= old['event_threshold_mv']))),
                                 'finite': bool(np.isfinite(predicted).all())})
    primary_rows = [r for r in rows if r['horizon_ms'] == cfg['primary_horizon_ms']]
    floor_values = [r['voltage_rmse_mv'] for r in primary_rows if r['arm'] == 'formula_vs_native']
    floor = {'pooled_rmse_mv': _rmse(np.array(floor_values)),
             'worst_episode_rmse_mv': max(floor_values)}
    floor['passed'] = bool(floor['pooled_rmse_mv'] <= cfg['native_floor_primary_pooled_limit_mv']
                           and floor['worst_episode_rmse_mv'] <= cfg['native_floor_primary_worst_episode_limit_mv'])
    models = {}
    for family in old['families']:
        models[family] = {}
        for seed in seeds:
            values = [r for r in primary_rows if r.get('family') == family and r['arm'] == 'both' and r.get('seed') == seed]
            errors = [r['voltage_rmse_mv'] for r in values]
            model = {'pooled_rmse_mv': _rmse(np.array(errors)), 'worst_episode_rmse_mv': max(errors),
                     'finite': all(r['finite'] for r in values)}
            model['passed_if_floor_valid'] = bool(floor['passed'] and model['finite']
                and model['pooled_rmse_mv'] <= cfg['candidate_primary_pooled_limit_mv']
                and model['worst_episode_rmse_mv'] <= cfg['candidate_primary_worst_episode_limit_mv'])
            models[family][str(seed)] = model
    return {'rows': rows, 'native_floor': floor, 'models': models,
            'primary_exposed_nonrest_episode_count': int(active.sum()),
            'primary_nonzero_injection_sample_count': int(np.count_nonzero(exposure))}


def run(root: Path, teacher: Path, output: Path, cfg: dict, revision: str) -> dict:
    old = verify(root, teacher, cfg)
    design = t30.episodes(old)
    reference, reference_gates = t30.formula_reference(design, old)
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('Task30c frozen model evaluation requires CUDA')
    models, seeds = ionic.load_frozen(root, ionic.config(root), torch, 'cuda')
    if list(seeds) != old['model_seeds']:
        raise RuntimeError('Frozen seed contract mismatch')
    candidates = {}
    for family in old['families']:
        for arm in old['arms']:
            candidates[family, arm] = t30.candidate_rollout(
                models[family, arm], torch, design, old, reference, reference_gates, len(seeds))
    build, hashes = compile_native(teacher, output, cfg)
    from neuron import load_mechanisms
    load_mechanisms(str(build.resolve()))
    native_v, native_gates, native_meta = native_reference(design, old)
    finite = bool(np.isfinite(native_v).all() and np.isfinite(native_gates).all())
    if not finite:
        raise RuntimeError('Nonfinite native active trajectories')
    metrics = evaluate(reference, native_v, candidates, design, old, cfg, list(seeds))
    report = {'schema_version': cfg['schema_version'], 'valid': True,
              'native_floor_admissible': metrics['native_floor']['passed'],
              'scientific_primary_passed': bool(metrics['native_floor']['passed'] and all(
                  m['passed_if_floor_valid'] for family in metrics['models'].values() for m in family.values())),
              'native_floor': metrics['native_floor'], 'models': metrics['models'],
              'primary_exposed_nonrest_episode_count': metrics['primary_exposed_nonrest_episode_count'],
              'primary_nonzero_injection_sample_count': metrics['primary_nonzero_injection_sample_count'],
              'native_metadata': native_meta, 'canonical_channel_sha256': hashes,
              'native_voltage_range_mv': [float(native_v.min()), float(native_v.max())],
              'rows': metrics['rows'], 'code_revision': revision,
              'task31_authorized': False, 'gate_d_performance_status': 'NO_GO_UNCHANGED',
              'training_performed': False, 'fresh_used_for_selection': False,
              'scope': cfg['scope']}
    write(output / 'final_report.json', report)
    return report
