"""Task35: independent local-compartment conductance-panel confirmation.

The NEURON teacher and frozen CUDA model execute in different processes. The
Task33 implementation supplies the already-audited causal synaptic interface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from . import ionic_block_teacher_forced as ionic
from . import task32_dynamic_calcium_feedback as t32
from . import task33_observable_synaptic_feedback as t33
from .hh_family_transfer import write


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(root: Path) -> tuple[dict, dict, dict, dict]:
    spec = json.loads((root / 'experiments/task35_full_local_compartment_preregistration.json').read_text())
    for key in ('parent_task33', 'parent_task34'):
        path = root / spec[f'{key}_report']
        if _sha(path) != spec[f'{key}_sha256']:
            raise RuntimeError(f'Task35 immutable {key} report changed')
    parent33 = json.loads((root / spec['parent_task33_report']).read_text())
    parent34 = json.loads((root / spec['parent_task34_report']).read_text())
    if not parent33['task33_passed'] or not parent34['valid']:
        raise RuntimeError('Task35 predecessor gate not passed')
    cfg, t32cfg, c3 = t33.contract(root)
    if spec['roadmap_id'] != 'Task35' or spec['dt_ms'] != cfg['dt_ms'] or spec['duration_ms'] != cfg['duration_ms']:
        raise RuntimeError('Task35 registered time/domain changed')
    if len(spec['panels']) != 4 or any(panel not in ionic.panel_multipliers() for panel in spec['panels']):
        raise RuntimeError('Task35 conductance panels changed')
    if len(spec['confirmation']['seeds']) != 2 or len(spec['confirmation']['initial_voltages_mv']) != 2 or len(spec['confirmation']['protocols']) != 2:
        raise RuntimeError('Task35 32-case factorial shape changed')
    forbidden = set(spec['confirmation']['seeds']) & set(cfg['confirmation']['seeds'])
    if forbidden or spec['calibration']['seed'] in spec['confirmation']['seeds']:
        raise RuntimeError('Task35 seed split overlap')
    cal = set(sum(spec['schedules_ms'][spec['calibration']['schedule']].values(), []))
    opened = set(sum(spec['schedules_ms'][spec['confirmation']['schedule']].values(), []))
    if cal & opened:
        raise RuntimeError('Task35 calibration event times reused')
    cfg = {**cfg, 'schedules_ms': spec['schedules_ms']}
    return spec, cfg, t32cfg, c3


def cases(spec: dict) -> list[dict]:
    confirmation = spec['confirmation']
    return [{'panel': panel, 'schedule': confirmation['schedule'], 'seed': seed,
             'initial_voltage_mv': voltage, 'protocol': protocol}
            for panel in spec['panels'] for seed in confirmation['seeds']
            for voltage in confirmation['initial_voltages_mv']
            for protocol in confirmation['protocols']]


def _panel_metrics(prediction: dict, natives: list[dict], spec: dict, seed_index: int = 0) -> dict:
    rows = cases(spec)
    result = {}
    for panel in spec['panels']:
        indices = [i for i, row in enumerate(rows) if row['panel'] == panel]
        subset = {key: value[:, :, indices, ...] for key, value in prediction.items()}
        result[panel] = t33.metrics(subset, [natives[i] for i in indices], seed_index)
    return result


def run_native(root: Path, teacher: Path, output: Path, revision: str) -> dict:
    import neuron
    from neuron import h, load_mechanisms

    spec, cfg, t32cfg, c3 = contract(root)
    if neuron.__version__.split('+')[0] != spec['neuron_version']:
        raise RuntimeError('Task35 NEURON version changed')
    teacher_revision = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if teacher_revision != spec['teacher_revision']:
        raise RuntimeError('Task35 canonical teacher revision changed')
    base = t32.verify(root, teacher, t32cfg)
    build = t33.compile_combined(root, teacher, output)
    load_mechanisms(str(build.resolve()))

    cal = spec['calibration']
    native = t33._native_case(h, cfg, base, t32cfg, c3, cal)
    shadow = t33.shadow_case(h, cfg, c3, cal, native)
    floor = t33.metrics(t33.coupled_rollout([native], [shadow], cfg, t32cfg, base), [native])
    calibration = {'passed': bool(t33._shadow_pass(shadow, cfg) and t33._floor_pass(floor, cfg)),
                   'formula_floor': floor,
                   'shadow': {key: shadow[key] for key in ('release_count', 'release_mismatch_count',
                        'max_shadow_current_error_na', 'max_shadow_receptor_state_error',
                        'max_rng_sequence_difference')}}
    write(output / 'calibration_report.json', calibration)
    if not calibration['passed']:
        report = {'schema_version': spec['schema_version'], 'valid': False,
                  'diagnosis': 'CALIBRATION_IMPLEMENTATION_FAILURE', 'confirmation_opened': False,
                  'model_judged': False, 'task35_passed': False, 'calibration': calibration,
                  'code_revision': revision}
        write(output / 'final_report.json', report)
        return report

    natives, shadows, summaries = [], [], []
    control = {}
    for i, row in enumerate(cases(spec), 1):
        native = t33._native_case(h, cfg, base, t32cfg, c3, row)
        shadow = t33.shadow_case(h, cfg, c3, row, native)
        if row['panel'] not in control:
            zero = t33._native_case(h, cfg, base, t32cfg, c3, row, zero_weight=True)
            zero_shadow = t33.shadow_case(h, cfg, c3, row, zero, zero_weight=True)
            control[row['panel']] = bool(t33._shadow_pass(zero_shadow, cfg)
                and zero_shadow['release_count'] == 0
                and np.max(np.abs(zero['native_i'])) == 0
                and np.array_equal(zero['rng_seq'][0], zero['rng_seq'][-1]))
        summaries.append({'case': row, 'shadow_passed': t33._shadow_pass(shadow, cfg),
                          'scheduled_count': shadow['scheduled_count'],
                          'release_count': shadow['release_count'],
                          'max_shadow_current_error_na': shadow['max_shadow_current_error_na'],
                          'max_shadow_receptor_state_error': shadow['max_shadow_receptor_state_error'],
                          'max_rng_sequence_difference': shadow['max_rng_sequence_difference'],
                          'release_mismatch_count': shadow['release_mismatch_count']})
        natives.append(native)
        shadows.append(shadow)
        if i % 4 == 0:
            print(f'[GIADA Task35] native {i}/{len(cases(spec))}', flush=True)
    prediction = t33.coupled_rollout(natives, shadows, cfg, t32cfg, base)
    panels = _panel_metrics(prediction, natives, spec)
    controls = {'zero_weight_zero_current_no_rng_draw_per_panel': all(control.values()),
                'scheduled_events_not_equated_to_releases': any(row['release_count'] < row['scheduled_count'] for row in summaries),
                'synaptic_voltage_effect_present': False,
                'teacher_current_not_model_input': True}
    # A matched zero-weight native counterfactual is measured on one case per panel.
    effects = {}
    for panel in spec['panels']:
        index = next(i for i, row in enumerate(cases(spec)) if row['panel'] == panel)
        zero = t33._native_case(h, cfg, base, t32cfg, c3, cases(spec)[index], zero_weight=True)
        effects[panel] = float(np.max(np.abs(natives[index]['voltage'] - zero['voltage'])))
    controls['synaptic_voltage_effect_present'] = max(effects.values()) >= cfg['gates']['min_synaptic_voltage_effect_mv']
    floor_pass = all(t33._floor_pass(value, cfg) for value in panels.values())
    shadow_pass = all(row['shadow_passed'] for row in summaries)
    np.savez_compressed(output / 'native_shadow_traces.npz',
        voltage=np.stack([row['voltage'] for row in natives]),
        calcium=np.stack([row['calcium'] for row in natives]),
        gates=np.stack([row['gates'] for row in natives]),
        base_g_us=np.stack([row['base_g_us'] for row in shadows]),
        injection=np.stack([row['injection'] for row in natives]),
        multipliers=np.stack([row['multipliers'] for row in natives]),
        area_um2=np.array(natives[0]['area_um2']))
    report = {'schema_version': spec['schema_version'], 'valid': True,
              'code_revision': revision, 'teacher_revision': teacher_revision,
              'calibration': calibration, 'confirmation_opened': True,
              'case_count': len(natives), 'case_summaries': summaries,
              'formula_floor_by_panel': panels, 'floor_passed': floor_pass,
              'shadow_passed': shadow_pass, 'negative_controls': controls,
              'synaptic_voltage_effect_mv_by_panel': effects,
              'model_judged': False, 'task35_passed': False,
              'diagnosis': 'READY_FOR_FROZEN_MODEL_EVALUATION' if
                  floor_pass and shadow_pass and all(controls.values()) else
                  'NATIVE_OR_INTERFACE_GATE_FAILED',
              'no_axial_or_whole_cell_claim': True, 'speedup_claim': False}
    write(output / 'premodel_report.json', report)
    if report['diagnosis'] != 'READY_FOR_FROZEN_MODEL_EVALUATION':
        write(output / 'final_report.json', report)
    return report


def evaluate_frozen(root: Path, teacher: Path, output: Path) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Task35 frozen model evaluation requires CUDA')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec, cfg, t32cfg, _ = contract(root)
    base = t32.verify(root, teacher, t32cfg)
    premodel = json.loads((output / 'premodel_report.json').read_text())
    if premodel['diagnosis'] != 'READY_FOR_FROZEN_MODEL_EVALUATION':
        raise RuntimeError('Task35 native floor did not authorize model judgment')
    with np.load(output / 'native_shadow_traces.npz') as bundle:
        natives = [{'voltage': bundle['voltage'][j], 'calcium': bundle['calcium'][j],
                    'gates': bundle['gates'][j], 'injection': bundle['injection'][j],
                    'multipliers': bundle['multipliers'][j],
                    'area_um2': float(bundle['area_um2'])}
                   for j in range(len(bundle['voltage']))]
        shadows = [{'base_g_us': bundle['base_g_us'][j]} for j in range(len(natives))]
    if len(natives) != len(cases(spec)) or any(not np.array_equal(
        natives[j]['multipliers'], ionic.panel_multipliers()[row['panel']])
        for j, row in enumerate(cases(spec))):
        raise RuntimeError('Task35 saved panel matrix changed')
    loaded, seeds = ionic.load_frozen(root, ionic.config(root), torch, 'cuda')
    if list(seeds) != t32cfg['model_seeds']:
        raise RuntimeError('Task35 frozen seed contract changed')
    models = {}
    for family in t32cfg['families']:
        prediction = t33.coupled_rollout(natives, shadows, cfg, t32cfg, base,
                                        loaded[family, t32cfg['frozen_arm']], torch)
        models[family] = {}
        for index, seed in enumerate(seeds):
            panels = _panel_metrics(prediction, natives, spec, index)
            models[family][str(seed)] = {'panels': panels,
                'passed': all(t33._candidate_pass(row, cfg) for row in panels.values())}
        write(output / 'model_results_partial.json', models)
        print(f'[GIADA Task35] frozen {family}: {len(seeds)} seeds', flush=True)
    passed = len(models) == len(t32cfg['families']) and all(
        row['passed'] for family in models.values() for row in family.values())
    report = {**premodel, 'model_judged': True, 'models': models,
              'task35_passed': passed,
              'diagnosis': 'CONFIRMED_FULL_DECLARED_LOCAL_COMPARTMENT' if passed else
                           'FROZEN_CANDIDATE_NO_GO_WITH_ADMISSIBLE_FLOOR',
              'training_performed': False, 'teacher_future_used_as_input': False,
              'full_cell_claim': False, 'speedup_claim': False}
    write(output / 'final_report.json', report)
    return report
