"""Materialize NEW Task37 development data and evaluator preflights on CPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.giada_teacher import task37_ionic_playground as p


def prefix_preflight(spec):
    import torch
    from src.giada_teacher import ie_residual_cost as old
    from src.giada_teacher import cc_n1_contract_memory as n1
    from src.giada_teacher import task32_dynamic_calcium_feedback as t32
    _, cfg, calcium, _, _ = n1.contract(ROOT)
    torch.set_num_threads(1)
    rng = np.random.default_rng(spec['reference_preflight']['seed'])
    n = spec['reference_preflight']['random_and_boundary_points']
    v = rng.uniform(-100, 60, n)
    v[:9] = [-100, 60, -66, -64.4, -60, -38, -32, -27, -17]
    ca = 10**rng.uniform(-7, -2, n)
    gates = rng.uniform(0, 1, (n, 18)); mult = rng.uniform(.25, 4, (n, 11))
    rev = np.tile(p.ionic.REVERSALS, (n, 1))
    # Compute same-tick Ca with the independent existing NumPy implementation.
    pre, _, _ = p.recurrent_reference(v[None], ca[None], gates, mult, rev, .1)
    next_ca = t32.calcium_step(ca, t32._ca_current(v, pre[0], mult), calcium)
    ng, currents, source = p.recurrent_reference(v[None], next_ca[None], gates, mult, rev, .1)
    def run(dtype):
        t = lambda x: torch.tensor(x, dtype=dtype)
        nc, gg, gs, ge = old.ionic_prefix(torch, t(v), t(ca), t(gates), t(mult), cfg, calcium)
        gc, ic = p.readout(v, ng[0], mult, rev)
        errors = dict(gate=float(np.max(np.abs(gg.numpy()-ng[0]))),
            calcium_mM=float(np.max(np.abs(nc.numpy()-next_ca))),
            sum_g=float(np.max(np.abs(gs.numpy()-gc.sum(-1)))),
            sum_gE=float(np.max(np.abs(ge.numpy()-(gc*rev).sum(-1)))))
        # Individual currents independently reconstructed from original gate output.
        _, original_currents = p.readout(v, gg.numpy(), mult, rev)
        errors['current_ma_cm2'] = float(np.max(np.abs(original_currents-currents[0])))
        limits = spec['reference_preflight']
        assert errors['gate'] <= limits['gate_atol'], errors
        assert errors['calcium_mM'] <= limits['calcium_atol_mM'], errors
        assert errors['current_ma_cm2'] <= limits['current_atol_ma_cm2'], errors
        assert errors['sum_g'] <= limits['sum_g_atol_s_cm2'], errors
        return errors
    result = dict(valid=True, points=n, float64=run(torch.float64), float32=run(torch.float32),
        native_simulation_performed=False, reference='Previously audited NumPy calcium source and original independent PyTorch ionic prefix, fixed Eca=120mV',
        source_current_matches_channel_sum=bool(np.array_equal(source, currents[..., :2].sum(-1))))
    # Real recurrent reset, rather than merely perturbing output currents.
    vv = np.full((800, 1), -40.); cc = np.full_like(vv, 1e-4)
    initial = np.full((1, 18), .8)
    args = (vv, cc, initial, np.ones((1, 11)), p.ionic.REVERSALS[None], .1)
    g, i, _ = p.recurrent_reference(*args)
    bg, bi, _ = p.recurrent_reference(*args, reset_tick=400)
    reset = p.evaluate_currents(i[:, 0], bi[:, 0], .1, spec['quality'], g[:, 0], bg[:, 0])
    assert not reset['passed']
    result['actual_recurrent_reset'] = reset
    return result


def qualitative_figure(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(1000)*.1
    signal = np.exp(-.5*((t-30)/2)**2)*1e-3
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    for axis, k in zip(axes.flat, (0, 2, 5, 6, 7, 9)):
        actual = signal*(1+k*.1)*(-1 if k < 5 else 1)
        wrong = actual + np.linspace(0, 3e-4, len(t))
        axis.plot(t, actual, label='Reference', lw=1.5)
        axis.plot(t, wrong, label='Injected drift control', lw=1.1, ls='--')
        axis.set_title(p.ionic.CHANNELS[k]); axis.set_ylabel('Current (mA/cm²)')
        axis.axhline(0, color='0.6', lw=.5); axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]: axis.set_xlabel('Time (ms)')
    fig.suptitle('Evaluator fixture: inward currents negative; no learned model')
    fig.tight_layout()
    fig.savefig(output / 'qualitative_control_fixture.png', dpi=140)
    plt.close(fig)


def run(output):
    spec = json.loads((ROOT / p.CONFIG).read_text())
    if (output / 'final_report.json').exists():
        raise FileExistsError('Complete Task37 output exists; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    p.save_json(output / 'preregistration.json', spec)
    sources = [p.CONFIG, 'src/giada_teacher/task37_ionic_playground.py',
               'scripts/run_task37_playground.py', 'tests/test_task37_ionic_playground.py',
               'src/giada_teacher/ionic_block_teacher_forced.py',
               'src/giada_teacher/ie_residual_cost.py',
               'experiments/task36_ionic_abstraction_contract_v1.json']
    provenance = dict(code_revision_at_execution=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources={x: p.digest(ROOT / x) for x in sources}, config_sha256=p.digest(ROOT / p.CONFIG),
        runtime=dict(python=sys.version, numpy=np.__version__), training_performed=False)
    p.save_json(output / 'code_provenance.json', provenance)
    print('[GIADA Task37] independent phase/dtype and evaluator controls', flush=True)
    preflight = prefix_preflight(spec)
    controls = p.sensitivity_controls(spec)
    p.save_json(output / 'reference_preflight.json', preflight)
    p.save_json(output / 'evaluator_controls.json', controls)
    manifest, cases, rows, signatures, history_pairs = [], [], [], set(), []
    start = time.perf_counter()
    for family in spec['families']:
        for role in spec['roles']:
            inputs, meta = p.make_paths(spec, role, family)
            g, currents, source = p.recurrent_reference(inputs['voltage'], inputs['bridge_ca'],
                inputs['initial_gates'], inputs['multipliers'], inputs['reversals'], spec['dt_ms'])
            if family == 'same_v_history':
                split = int(len(g)*.4)
                for j in range(0, len(meta), 2):
                    assert np.array_equal(inputs['voltage'][split:, j], inputs['voltage'][split:, j+1])
                    assert np.array_equal(inputs['bridge_ca'][:, j], inputs['bridge_ca'][:, j+1])
                    delta = float(np.max(np.abs(currents[split:, j]-currents[split:, j+1])))
                    assert delta > 1e-12
                    history_pairs.append(dict(role=role, pair=j//2, current_future_delta_ma_cm2=delta,
                        same_initial_state=True, same_current_voltage_and_ca=True, same_parameters=True))
            path = output / f'{role}__{family}.npz'
            arrays = {**inputs, 'gate_targets': g, 'current_targets': currents, 'calcium_source_current': source}
            np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in arrays.items()})
            for j, metadata in enumerate(meta):
                assert metadata['input_sha256'] not in signatures, 'Duplicate history across partitions'
                signatures.add(metadata['input_sha256'])
                report = p.evaluate_currents(currents[:, j], currents[:, j].astype(np.float32), spec['dt_ms'],
                    spec['quality'], g[:, j], g[:, j].astype(np.float32), inputs['multipliers'][j] == 0)
                assert report['passed'], (metadata['id'], [x for x in report['channels'] if not x['passed']], report['gate'])
                rows.append(dict(case_id=metadata['id'], role=role, family=family, storage_conversion=report))
            cases.extend(meta)
            manifest.append(dict(file=path.name, sha256=p.digest(path), bytes=path.stat().st_size,
                role=role, family=family, trajectories=len(meta), ticks=len(currents),
                scalar_current_targets=int(currents.size), imposed_voltage_pulses=sum(m['pulse_count'] for m in meta),
                emerging_spike_count=None, support_scope='imposed V/Ca ionic transitions, not autonomous membrane'))
        print(f'[GIADA Task37] {family}: 5 partitions saved; elapsed {(time.perf_counter()-start)/60:.1f} min', flush=True)
    p.save_json(output / 'dataset_manifest.json', manifest)
    p.save_json(output / 'cases.json', cases)
    p.save_json(output / 'storage_conversion_evaluation.json', rows)
    p.save_json(output / 'matched_history_probe.json', history_pairs)
    qualitative_figure(output)
    report = dict(valid=True, task=37, decision='GO_TASK38_IMPLEMENTATION_WITH_GPU_PREFLIGHT',
        scope=spec['scope'], reference_preflight_passed=preflight['valid'],
        evaluator_controls_passed=controls['valid'], defective_controls_detected=len(controls['controls_detected']),
        actual_recurrent_reset_detected=True, partitions=list(spec['roles']),
        matched_history_pairs=len(history_pairs),
        trajectory_count=len(cases), family_count=len(spec['families']), file_count=len(manifest),
        total_biological_ms=sum(m['duration_ms'] for m in cases),
        rare_trajectory_count=sum(m['rare'] for m in cases),
        no_cross_partition_history_overlap=len(signatures) == len(cases),
        storage_conversion_all_passed=all(x['storage_conversion']['passed'] for x in rows),
        event_support_by_family={f: sum(x['event']['status'] == 'evaluable' for r in rows if r['family'] == f for x in r['storage_conversion']['channels']) for f in spec['families']},
        confirmation_generated=False, confirmation_accessed=False, training_performed=False,
        emerging_spikes_validated=False, dynamical_bifurcations_validated=False,
        coupled_behavior_gate_deferred_to_tasks=[40, 41], speedup_measured=False,
        quality=spec['quality'], cost_contract=spec['cost_contract'],
        training_draft=spec['task38_39_draft'], limitations=spec['limits'],
        elapsed_seconds=time.perf_counter()-start,
        source_hashes=provenance['sources'])
    p.save_json(output / 'final_report.json', report)
    print(json.dumps({k: report[k] for k in ('valid', 'task', 'trajectory_count', 'family_count', 'defective_controls_detected', 'decision')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/results/task37_playground_v1')
    run(parser.parse_args().output)
