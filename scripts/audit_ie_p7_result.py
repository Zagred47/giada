"""Audit archived IE-P7 v4 without training or accessing a new test set."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit():
    import numpy as np
    import torch
    from src.giada_teacher import ie_native_operational_confirmation as p7
    torch.set_num_threads(1)
    folder = ROOT / 'experiments/results/ie_p7_kaggle_v4'
    read = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    r, freeze, spec = read('final_report.json'), read('freeze.json'), read('preregistration.json')
    assert r['valid'] and r['model_judged'] and not r['training_performed']
    assert r['code_revision'] == freeze['code_revision'] == (folder / 'code_revision.txt').read_text().strip()
    assert (folder / 'preregistration.json').read_bytes() == (ROOT / p7.CONFIG).read_bytes()
    assert hashlib.sha256((folder / 'preregistration.json').read_bytes()).hexdigest() == freeze['preregistration_sha256']
    assert hashlib.sha256((ROOT / spec['checkpoint']).read_bytes()).hexdigest() == freeze['checkpoint_sha256']
    assert not freeze['fresh_accessed'] and not freeze['training_performed'] and not r['candidate_promoted']
    for name, key in [('native_report.json', 'native_report'), ('native_quality.json', 'native_quality'),
                      ('fresh_evaluation.json', 'fresh'), ('operational_timing.json', 'operational')]:
        assert read(name) == r[key]
    assert len(r['core_rows']) == len(r['core_preflights']) == 56
    assert len(r['operational']['rows']) == len(r['operational']['preflights']) == 56
    assert all(x['valid'] for x in r['core_preflights'] + r['operational']['preflights'])
    assert not r['graph_failures']
    for rows, operational in [(r['core_rows'], False), (r['operational']['rows'], True)]:
        for row in rows:
            control = next(x for x in rows if x['batch'] == row['batch'] and x['mode'] == row['mode'] and x['name'] == 'formula_vectorized')
            times = lambda x: [s['wall_ms'] for s in x['samples']] if operational else x['timings']['wall_ms']
            assert len(times(row)) == (10 if operational else 20)
            assert np.isfinite(times(row)).all() and min(times(row)) > 0
            assert p7.p4.paired_summary(times(control), times(row)) == row['saving_vs_formula']
    quality_flags = {}
    floor = r['native_quality']['formula_vectorized']['episodes']
    for name, row in r['native_quality'].items():
        ms = row['episodes']
        absolute = all(p7.acceptable(m, spec['quality']) for m in ms)
        ni = all(m['voltage_rmse_mv'] - f['voltage_rmse_mv'] <= .02 and m['calcium_max_mm'] - f['calcium_max_mm'] <= 1e-6 for m, f in zip(ms, floor))
        spike = all(m['spike_count_equal'] and (m['spike_timing_max_ms'] is None or m['spike_timing_max_ms'] <= .2) for m in ms)
        assert (absolute, ni, spike, absolute and ni and spike) == (row['absolute_passed'], row['noninferiority_passed'], row['spike_passed'], row['passed'])
        f = r['fresh'][name]
        quality_flags[name] = row['passed'] and f['fresh_gate']['rmse'] <= .001 and f['fresh_gate']['max_abs'] <= .01 and not f['occupancy_violations'] and all(m['rmse'] <= .005 and m['max_abs'] <= .03 for m in f['families'].values())
    assert quality_flags == r['quality_passed']
    _, cfg, ca, _, base = p7.contract(ROOT)
    p6 = p7.p6
    parent_spec = json.loads((ROOT / p6.CONFIG).read_text())
    parent = torch.load(ROOT / parent_spec['parent_checkpoint'], map_location='cpu', weights_only=True)
    ckpt = torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=True)
    ops = {'formula_vectorized': None, 'lut025': p7.p4.lut_operator(torch, device='cpu')}
    for knots in (81, 161):
        model = p6.local(torch, parent, knots, 'cpu')
        model.load_state_dict(ckpt[f'local{knots}'])
        for i, seed in enumerate((17, 29, 43)):
            ops[f'local{knots}_seed{seed}'] = p6.local_operator(torch, model, i)
    with np.load(folder / 'fresh_native_traces.npz') as z:
        data = {k: z[k] for k in z.files}
    assert data['voltage'].shape == (32, 401) and data['gates'].shape == (32, 401, 18)
    discrepancies = {}
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    mult, inj, syn = (t(data[k]) for k in ('multipliers', 'injection', 'base_g_us'))
    with torch.inference_mode():
        for name, op in ops.items():
            v, c, g = (t(data[k][:, 0]) for k in ('voltage', 'calcium', 'gates'))
            vs, cs, gs = [v], [c], [g]
            for tick in range(400):
                v, c, g, *_ = p7.p4.core(torch, v, c, g, mult, inj[:, tick], syn[:, tick], float(data['area_um2']), cfg, ca, base, op)
                vs.append(v); cs.append(c); gs.append(g)
            vv, cc, gg = (torch.stack(a, 1).numpy() for a in (vs, cs, gs))
            measured = [p7.array_metrics(vv[j], cc[j], gg[j], {k: data[k][j] for k in ('voltage', 'calcium', 'gates')}) for j in range(32)]
            delta = max(abs(m['voltage_rmse_mv'] - q['voltage_rmse_mv']) for m, q in zip(measured, r['native_quality'][name]['episodes']))
            assert delta < 2e-4, (name, delta)
            assert all(p7.acceptable(m, spec['quality']) for m in measured)
            discrepancies[name] = delta
    def fast(summary):
        return summary['median_saved_fraction'] >= .1 and summary['bootstrap_95pct'][0] > 0
    primary = {}
    for family in ('local81', 'local161'):
        for seed in (17, 29, 43):
            name = f'{family}_seed{seed}'
            core = next(x for x in r['core_rows'] if x['batch'] == 642 and x['mode'] == 'cuda_graph' and x['name'] == name)
            total = next(x for x in r['operational']['rows'] if x['batch'] == 642 and x['mode'] == 'cuda_graph' and x['name'] == name)
            computed = dict(quality=quality_flags[name], core_cost=fast(core['saving_vs_formula']), operational_cost=fast(total['saving_vs_formula']),
                core_vs_hybrids=all(fast(p7.p4.paired_summary(x['timings']['wall_ms'], core['timings']['wall_ms'])) for x in r['core_rows'] if x['batch'] == 642 and x['mode'] == 'cuda_graph' and x['name'].startswith('hybrid_')),
                operational_vs_hybrids=all(fast(s) for s in total['saving_vs_hybrids'].values()))
            assert computed == r['family_decisions'][family]['seeds'][str(seed)]
            primary[name] = dict(core_saving=core['saving_vs_formula']['median_saved_fraction'], operational_saving=total['saving_vs_formula']['median_saved_fraction'], **computed)
    return dict(valid=True, code_revision=r['code_revision'], core_cells=56, operational_cells=56,
        quality_flags_recomputed=14, native_cpu_replays=discrepancies, primary=primary,
        native_behavior_support=all(x['behavior_support'] for x in r['native_quality'].values()),
        candidate_promoted=False, training_performed=False, new_test_accessed=False,
        files={p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in sorted(folder.iterdir()) if p.name != 'result_audit.json'})


if __name__ == '__main__':
    result = audit()
    path = ROOT / 'experiments/results/ie_p7_kaggle_v4/result_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'primary')}, indent=2))
