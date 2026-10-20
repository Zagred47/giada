"""Frozen Task40: causal ionic substitution in the unchanged Task35 compartment.

Historical native traces are reused as opened diagnostic fixtures, never as a
new independent confirmation. Presynaptic shadow conductances are exogenous;
NMDA's voltage block is recomputed from the candidate's own voltage.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from . import task39j_atomic_relaxation as j
from . import ionic_block_teacher_forced as ionic
from . import task35_full_local_compartment as t35
from . import task33_observable_synaptic_feedback as t33
from . import task30_autonomous_voltage as t30
from . import cc_n1_contract_memory as n1
from . import ie_residual_cost as cost
from .gate_d_compute_audit import _rates

p = j.p
CONFIG = 'experiments/task40_preregistration_v1.json'
FAMILIES = dict(calcium=(0, 1), sodium=(2, 3, 4), potassium=(6, 7, 8, 9, 10),
                ih=(5,), repaired_pair=(4, 7), all=tuple(range(11)))
_CONSTANTS = {}


def constants(torch, v):
    key = (str(v.device), v.dtype)
    if key not in _CONSTANTS:
        _CONSTANTS[key] = tuple(torch.tensor(x, device=v.device, dtype=v.dtype)
                               for x in (ionic.GBAR, ionic.REVERSALS, (0., 0., -80., -97.)))
    return _CONSTANTS[key]


def membrane(torch, v, conductance, injection, syn, area, cfg, base):
    # Same twelve-step Task35 Newton solve, constants resident before capture.
    _, rev, synrev = constants(torch, v)
    density = syn*(100./area)
    fixed = torch.cat((density[...,:1],torch.zeros_like(density[...,1:2]),density[...,2:]),-1)
    cap = 1e-3*base['cm_uf_cm2']/cfg['dt_ms']
    denom = cap+base['g_pas_s_cm2']+conductance.sum(-1)+fixed.sum(-1)
    numer = cap*v+base['g_pas_s_cm2']*base['e_pas_mv']+(conductance*rev).sum(-1)+(fixed*synrev).sum(-1)+injection
    nv = numer/denom
    for _ in range(12):
        exponent = torch.exp((-.08*nv).clamp(-80,80))/3.57
        nmda = density[...,1]/(1+exponent)
        derivative = .08*density[...,1]*exponent/(1+exponent).square()
        nv = nv-(denom*nv-numer+nmda*nv)/(denom+nmda+derivative*nv)
    return nv


def contract(root):
    spec = json.loads((root / CONFIG).read_text())
    for name, expected in spec['sources'].items():
        if p.digest(root / name) != expected:
            raise RuntimeError('Task40 immutable source changed: ' + name)
    _, cfg, calcium, _ = t35.contract(root)
    base = t30.config(root, Path(calcium['base_config']).name)
    quality = json.loads((root / p.CONFIG).read_text())['quality']
    assert cfg['dt_ms'] == spec['dt_ms'] == .1
    assert spec['current_quality'] == quality and spec['compartment_gates'] == cfg['gates']
    report = json.loads((root / 'experiments/results/task35_final_report.json').read_text())
    assert report['task35_passed'] and report['floor_passed'] and report['shadow_passed']
    audit = json.loads((root / 'experiments/results/task39j_audited_v1/attribution_audit.json').read_text())
    assert audit['valid'] and audit['task40_preparation_authorized']
    return spec, cfg, calcium, base


def load_models(torch, root, device):
    spec = json.loads((root / j.CONFIG).read_text())
    weights = j.parent_weights(torch, root, spec, device)
    parent = j.i.model_class(torch)('plain', spec['seeds']).to(device)
    parent.load_state_dict(weights)
    candidate = j.make_model(torch, weights, j.arms()[0], spec['seeds'], device)
    source = root / 'experiments/results/task39j_audited_v1'
    freeze = json.loads((source / 'selection_freeze.json').read_text())
    payload = {k: v for k, v in freeze.items() if k != 'freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == freeze['freeze_sha256']
    assert p.digest(source / 'frozen_checkpoint.pt') == freeze['checkpoint_sha256']
    candidate.load_state_dict(torch.load(source / 'frozen_checkpoint.pt', map_location=device, weights_only=True)['J1'])
    for model in (parent, candidate):
        model.eval()
        for par in model.parameters():
            par.requires_grad_(False)
            assert torch.isfinite(par).all()
    return dict(I3=parent, J1=candidate)


def design(root, spec):
    with np.load(root / 'experiments/results/task35_native_shadow_traces.npz') as z:
        data = {k: z[k].copy() for k in z.files}
    rows = t35.cases(json.loads((root / 'experiments/task35_full_local_compartment_preregistration.json').read_text()))
    meta = [dict(id=i, source_index=i, intervention='none', historical_native=True, **r) for i, r in enumerate(rows)]
    selected = [i for i, r in enumerate(rows) if r['panel'] == 'canonical']
    arrays = {k: [data[k]] for k in data if k != 'area_um2'}
    for intervention in spec['interventions']:
        extra = {k: data[k][selected].copy() for k in arrays}
        if intervention.startswith('block_'):
            extra['multipliers'][:, ionic.CHANNELS.index(intervention[6:])] = 0
        elif intervention == 'zero_synapses':
            extra['base_g_us'][:] = 0
        elif intervention.startswith('injection_'):
            extra['injection'] *= float(intervention.split('_')[1])
        else:
            raise ValueError(intervention)
        for k in arrays:
            arrays[k].append(extra[k])
        for i in selected:
            meta.append(dict(id=len(meta), source_index=i, intervention=intervention, historical_native=False, **rows[i]))
    result = {k: np.concatenate(v) for k, v in arrays.items()}
    result['area_um2'] = float(data['area_um2'])
    return result, meta


def readout(torch, v, g, mult):
    openings = []
    for k, names in enumerate(ionic.STATE_NAMES):
        x = g[..., int(ionic.OFFSETS[k])].pow(int(ionic.POWERS[k]))
        if len(names) == 2:
            x = x * g[..., int(ionic.OFFSETS[k]) + 1]
        openings.append(x)
    gbar, rev, _ = constants(torch, v)
    conductance = torch.stack(openings, -1) * gbar * mult
    return conductance, conductance * (v[..., None] - rev)


def learned_sk(torch, model, ca, previous):
    # Exactly the frozen calcium-only head; no second voltage-trunk evaluation.
    cc = (ca.clamp_min(1e-9).log10() + 4.5) / 2.5
    raw = torch.bmm(torch.nn.functional.silu(torch.bmm(cc[..., None], model.sa) + model.sb), model.sc) + model.sd[:, None]
    return previous + torch.sigmoid(raw[..., 1]) * (torch.sigmoid(raw[..., 0]) - previous)


def make_core(torch, cfg, calcium, base, area, model=None, channels=(), ionic_only=False):
    learned = set(channels)
    def core(v, ca, state, mult, injection, syn):
        neural = model(v, ca, state) if model is not None else None
        parts = []
        for k, name in enumerate(ionic.CHANNELS):
            a, b = map(int, ionic.OFFSETS[k:k+2])
            if k == 9:
                value = state[..., a:b]  # SK consumes the new calcium below.
            elif k in learned:
                value = neural[..., a:b]
            else:
                u, tau = _rates(name, v, ca, torch)
                if b-a == 1:
                    u, tau = u[..., None], tau[..., None]
                z = -torch.expm1(-cfg['dt_ms'] / tau)
                value = (1-z) * state[..., a:b] + z*u
            parts.append(value)
        g = torch.cat(parts, -1)
        _, current = readout(torch, v, g, mult)
        persistence = math.exp(-cfg['dt_ms']/calcium['decay_ms'])
        source = -10000 * current[..., :2].sum(-1) * calcium['gamma'] / (2 * 96485.33212 * calcium['depth_um'])
        nc = 1e-4 + (ca-1e-4)*persistence + source*calcium['decay_ms']*(1-persistence)
        if 9 in learned:
            sk = learned_sk(torch, model, nc, state[..., 16])
        else:
            inf = 1 / (1 + (.00043/nc.clamp_min(1e-7)).pow(4.8))
            sk = inf + (state[..., 16]-inf)*math.exp(-cfg['dt_ms'])
        g = torch.cat((g[..., :16], sk[..., None], g[..., 17:]), -1)
        conductance, current = readout(torch, v, g, mult)
        nv = v if ionic_only else membrane(torch, v, conductance, injection, syn, area, cfg, base)
        return nv, nc, g, current
    return core


class Stepper:
    """Persistent resident state; identical outputs in eager and CUDA Graph mode."""
    def __init__(self, torch, core, args, graph=False):
        self.torch, self.core = torch, core
        self.args = [x.clone() for x in args]
        self.graph = None
        if graph:
            for _ in range(3):
                self._tick()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._tick()
        self.reset(args)

    def _tick(self):
        self.out = self.core(*self.args)
        for x, y in zip(self.args[:3], self.out[:3]):
            x.copy_(y)

    def reset(self, args):
        for x, y in zip(self.args, args):
            x.copy_(y)

    def tick(self, injection=None, syn=None, imposed=None):
        if injection is not None:
            self.args[4].copy_(injection)
            self.args[5].copy_(syn)
        if imposed is not None:
            self.args[0].copy_(imposed[0]); self.args[1].copy_(imposed[1])
        if self.graph is None:
            self._tick()
        else:
            self.graph.replay()
        return self.out


def numpy_reference(data, cfg, calcium, base):
    v, ca, g = (data[k][:, 0].copy() for k in ('voltage', 'calcium', 'gates'))
    vv, cc, gg, ii = [v.copy()], [ca.copy()], [g.copy()], []
    for tick in range(data['injection'].shape[1]):
        old = v.copy()
        v, ca, g = n1.numpy_step(v, ca, g, data['multipliers'], data['injection'][:, tick],
                               data['base_g_us'][:, tick], data['area_um2'], cfg, calcium, base)
        _, current = p.readout(old, g, data['multipliers'], ionic.REVERSALS)
        vv.append(v.copy()); cc.append(ca.copy()); gg.append(g.copy()); ii.append(current)
    return dict(voltage=np.stack(vv), calcium=np.stack(cc), gates=np.stack(gg), current=np.stack(ii))


def rollout(torch, core, data, device, imposed=None, graph=False):
    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    count = len(data['voltage'])
    def expanded(x):
        return t(x)[None].expand(3, *x.shape).contiguous()
    args = [expanded(data[k][:, 0]) for k in ('voltage', 'calcium', 'gates')]
    args += [expanded(data['multipliers']), expanded(data['injection'][:, 0]), expanded(data['base_g_us'][:, 0])]
    runner = Stepper(torch, core, args, graph)
    injection, syn = t(data['injection']).T, t(data['base_g_us']).transpose(0, 1)
    clamp = None if imposed is None else (t(imposed['voltage']), t(imposed['calcium']))
    values = [[x.clone()] for x in args[:3]] + [[]]
    for tick in range(injection.shape[0]):
        forced = None if clamp is None else (clamp[0][tick][None].expand(3, count), clamp[1][tick][None].expand(3, count))
        result = runner.tick(injection[tick][None].expand(3, count), syn[tick][None].expand(3, count, 4), forced)
        for target, x in zip(values, result):
            target.append(x.clone())
    return {k: torch.stack(v).cpu().numpy() for k, v in zip(('voltage', 'calcium', 'gates', 'current'), values)}


def spikes(v, dt):
    return (np.flatnonzero((v[:-1] < 0) & (v[1:] >= 0))+1)*dt


def metrics(pred, ref, data, meta, quality, gates, mode):
    rows = []
    for s, seed in enumerate((17, 29, 43)):
        for case, label in enumerate(meta):
            v, ca, g = (pred[k][:, s, case] for k in ('voltage', 'calcium', 'gates'))
            rv, rc, rg = (ref[k][:, case] for k in ('voltage', 'calcium', 'gates'))
            current = p.evaluate_currents(ref['current'][:, case], pred['current'][:, s, case], .1,
                                         quality, rg[1:], g[1:], data['multipliers'][case] == 0)
            finite = bool(all(np.isfinite(x).all() for x in (v, ca, g)))
            rms_v = float(np.sqrt(np.mean((v-rv)**2))) if finite else None
            rms_ca = float(np.sqrt(np.mean((ca-rc)**2))) if finite else None
            sr, sp = spikes(rv, .1), spikes(v, .1)
            matched = len(sr) == len(sp)
            timing = float(np.max(np.abs(sr-sp))) if matched and len(sr) else (0. if matched else None)
            voltage_ok = finite and rms_v <= gates['max_candidate_worst_voltage_rmse_mv'] and rms_ca <= gates['max_candidate_worst_cai_rmse_mM']
            physical = finite and not np.any((v < -140) | (v > 100) | (ca < 0)) and not np.any((g < -quality['occupancy_atol']) | (g > 1+quality['occupancy_atol']))
            spike_ok = matched and timing <= quality['onset_offset_max_ms']
            # For imposed inputs the output V is a one-step diagnostic, not an autonomous spike train.
            passed = bool(current['passed'] and physical and (mode != 'autonomous' or (voltage_ok and spike_ok)))
            rows.append(dict(case=label, seed=seed, passed=passed, current=current,
                voltage_rmse_mv=rms_v, calcium_rmse_mM=rms_ca, physical=bool(physical),
                reference_spikes_ms=sr.tolist(), predicted_spikes_ms=sp.tolist(), spike_count_equal=matched,
                spike_timing_error_ms=timing, voltage_domain_exits=int(np.sum((v < -100) | (v > 60))),
                calcium_domain_exits=int(np.sum((ca < 1e-7) | (ca > .01)))))
    pooled = [float(np.sqrt(np.mean([r['voltage_rmse_mv']**2 for r in rows if r['seed'] == s])))
              if all(r['voltage_rmse_mv'] is not None for r in rows if r['seed'] == s) else None for s in (17,29,43)]
    passed = all(r['passed'] for r in rows) and (mode != 'autonomous' or all(x is not None and x <= gates['max_candidate_pooled_voltage_rmse_mv'] for x in pooled))
    return dict(passed=passed, cases=rows, pooled_voltage_rmse_mv=pooled, passed_cases=sum(r['passed'] for r in rows), total=len(rows))


def preflight(torch, root, device):
    spec, cfg, calcium, base = contract(root)
    models = load_models(torch, root, device)
    data, _ = design(root, spec)
    rng = np.random.default_rng(40001)
    count = 64
    v = rng.uniform(-95, 40, count); ca = 10**rng.uniform(-7, -2, count); g = rng.uniform(0, 1, (count, 18))
    mult = rng.uniform(.25, 4, (count, 11)); inj = np.zeros(count); syn = np.zeros((count, 4))
    arrays = [v, ca, g, mult, inj, syn]
    args64 = [torch.tensor(x[None].repeat(3, axis=0), device=device, dtype=torch.float64) for x in arrays]
    core = make_core(torch, cfg, calcium, base, data['area_um2'])
    expected = n1.numpy_step(v, ca, g, mult, inj, syn, data['area_um2'], cfg, calcium, base)
    actual = core(*args64)
    errors = [float(np.max(np.abs(actual[k][0].cpu().numpy()-expected[k]))) for k in range(3)]
    assert max(errors) < 1e-8, errors
    args = [x.float() for x in args64]
    m = models['J1']
    a = learned_sk(torch, m, args[1], args[2][...,16])
    sk_error = float((a-m(args[0], args[1], args[2])[...,16]).abs().max())
    assert sk_error <= spec['current_quality']['occupancy_atol'], sk_error
    neural_core = make_core(torch, cfg, calcium, base, data['area_um2'], m, FAMILIES['all'])
    # Guard against accidentally restoring any canonical gate in the all-learned arm.
    global _rates
    original = _rates
    try:
        def forbidden(*a, **k):
            raise AssertionError('Canonical gate invoked in all-learned inference')
        _rates = forbidden
        neural_core(*args)
    finally:
        _rates = original
    parity = 0.
    if device == 'cuda':
        for fn in (core, neural_core):
            eager, captured = Stepper(torch, fn, args), Stepper(torch, fn, args, True)
            for _ in range(10):
                aa, bb = eager.tick(), captured.tick()
                parity = max(parity, max(float((a-b).abs().max()) for a,b in zip(aa,bb)))
        assert parity <= 1e-5, parity
    return dict(valid=True, formula_numpy_max_errors=errors, cuda_graph_max_error=parity,
                learned_sk_head_max_error=sk_error, all_learned_has_no_canonical_gate_calls=True,
                source_sha256=p.digest(Path(__file__)), device=device,
                gpu=torch.cuda.get_device_name() if device == 'cuda' else None)


def stress_replay(torch, root, output, models, spec, quality, device):
    oldspec = json.loads((root / j.CONFIG).read_text())
    scanner = j.i.old.Scanner(torch, list(models.values()), 2, oldspec['recurrence_chunk'], device)
    rows = []
    for family in ('same_v_history', 'activation_boundary'):
        with np.load(root / oldspec['dataset'] / ('stress_diagnostic__'+family+'.npz')) as z:
            a = {k:z[k] for k in z.files}
        result = scanner.run(a['voltage'], a['bridge_ca'], a['initial_gates'])
        for m, name in enumerate(models):
            for s, seed in enumerate(spec['seeds']):
                g = result[m][:,s]
                _, current = p.readout(a['voltage'],g,a['multipliers'],a['reversals'])
                for k in range(2):
                    measured = p.evaluate_currents(a['current_targets'][:,k],current[:,k],.1,quality,
                                                  a['gate_targets'][:,k],g[:,k],a['multipliers'][k] == 0)
                    rows.append(dict(model=name, family=family, case=k, seed=seed, metrics=measured))
        p.save_json(output/'stress_replay.json',rows)
        print('[GIADA Task40] checkpoint stress '+family,flush=True)
    previous = json.loads((root/'experiments/results/task39j_audited_v1/recurrence_evaluation.json').read_text())
    names = dict(J1='joint_wJ1', I3='joint_wfrozen39i_I3')
    checks = []
    for row in rows:
        expected = next(r for r in previous if r['model'] == names[row['model']] and r['role'] == 'stress_diagnostic'
                        and all(r[k] == row[k] for k in ('family','case','seed')))
        checks.append(row['metrics']['passed'] == expected['metrics']['passed'])
    p.save_json(output/'stress_reproduction_check.json',dict(valid=all(checks),compared=len(checks)))
    if not all(checks):
        raise RuntimeError('Frozen stress reproduction flags changed; model not reselected')
    return rows


def benchmark(torch, models, data, spec, cfg, calcium, base):
    results = []
    for batch in spec['cost']['batches']:
        ids = np.arange(batch) % len(data['voltage'])
        def expand(x):
            return torch.tensor(x,device='cuda',dtype=torch.float32)[None].expand(3,*x.shape).contiguous()
        args = [expand(data[k][ids,0]) for k in ('voltage','calcium','gates')]
        args += [expand(data['multipliers'][ids]),expand(data['injection'][ids,0]),expand(data['base_g_us'][ids,0])]
        for scope in ('ionic', 'integrated'):
            runners = {}
            for name, model in [('formula',None), *models.items()]:
                fn = make_core(torch,cfg,calcium,base,data['area_um2'],model,FAMILIES['all'] if model else (),scope=='ionic')
                runners[name] = Stepper(torch,fn,args,True)
            samples = {k:dict(cuda_ms=[],wall_ms=[]) for k in runners}
            for repetition in range(spec['cost']['repetitions']):
                names = list(runners) if repetition%2 == 0 else list(runners)[::-1]
                for name in names:
                    runner=runners[name];runner.reset(args)
                    torch.cuda.synchronize();start=time.perf_counter()
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                    for _ in range(spec['cost']['ticks']):runner.tick()
                    b.record();b.synchronize()
                    samples[name]['wall_ms'].append((time.perf_counter()-start)*1000)
                    samples[name]['cuda_ms'].append(a.elapsed_time(b))
            from .ie_ionic_cost import paired_summary
            comparison={k:paired_summary(samples['formula']['wall_ms'],samples[k]['wall_ms']) for k in models}
            finite={k:all(bool(torch.isfinite(x).all()) for x in runner.out) for k,runner in runners.items()}
            results.append(dict(batch_per_seed=batch,seeds=3,scope=scope,finite=finite,cost_qualification=False,
                execution='CUDA Graph, no host transfer inside timed block',samples=samples,comparison_vs_formula=comparison))
            print(f'[GIADA Task40] cost {scope} batch={batch}',flush=True)
    return results


def run(root, output, device='cuda'):
    import torch
    assert device == 'cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec,cfg,calcium,base=contract(root);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'final_report.json').exists()
    p.save_json(output/'preregistration.json',spec)
    with torch.inference_mode():
        pf=preflight(torch,root,device);p.save_json(output/'preflight.json',pf)
        models=load_models(torch,root,device)
        for name,model in models.items():torch.save(model.state_dict(),output/(name+'_frozen.pt'))
        print('[GIADA Task40] preflights passed; frozen checkpoints saved',flush=True)
        stress=stress_replay(torch,root,output,models,spec,spec['current_quality'],device)
        data,meta=design(root,spec);p.save_json(output/'cases.json',meta)
        golden=numpy_reference(data,cfg,calcium,base)
        # Native floor uses only the 32 actual historical NEURON episodes.
        natives=[dict(voltage=data['voltage'][k],calcium=data['calcium'][k]) for k in range(32)]
        pred={k:golden[k][:,:32][None] for k in ('voltage','calcium','gates')}
        floor=t33.metrics(pred,natives);floor_ok=t33._floor_pass(floor,cfg)
        p.save_json(output/'native_floor.json',dict(valid=floor_ok,metrics=floor,historical_replay=True,new_native_simulation=False))
        if not floor_ok:
            report=dict(valid=False,technical_failure=True,decision='FORMULA_NATIVE_FLOOR_FAILED_MODEL_NOT_JUDGED',model_judged=False)
            p.save_json(output/'final_report.json',report);return report
        # Imposed inputs: native on the original fixtures, formula controls for interventions.
        imposed={k:golden[k].copy() for k in ('voltage','calcium')}
        for k in imposed:imposed[k][:,:32]=data[k][:32].T
        reports={};references={};saved={}
        for mode in ('imposed','autonomous'):
            formula_core=make_core(torch,cfg,calcium,base,data['area_um2'])
            ref_all=rollout(torch,formula_core,data,device,imposed if mode=='imposed' else None,True)
            ref={k:v[:,0] for k,v in ref_all.items()};references[mode]=ref
            if mode=='autonomous':
                f32=t33.metrics({k:ref_all[k][:,:,:32].transpose(1,0,2,*range(3,ref_all[k].ndim)) for k in ('voltage','calcium','gates')},natives)
                if not t33._floor_pass(f32,cfg):
                    raise RuntimeError('FP32 GPU formula floor not admissible')
                p.save_json(output/'gpu_formula_floor.json',f32)
                current_floor=[p.evaluate_currents(golden['current'][:,case],ref['current'][:,case],.1,
                    spec['current_quality'],golden['gates'][1:,case],ref['gates'][1:,case],data['multipliers'][case]==0)
                    for case in range(len(meta))]
                p.save_json(output/'gpu_current_floor.json',dict(valid=all(r['passed'] for r in current_floor),cases=current_floor))
                if not all(r['passed'] for r in current_floor):
                    raise RuntimeError('FP32 formula current floor not admissible; model not judged')
            for name,model in models.items():
                for family,channels in FAMILIES.items():
                    key=name+'/'+family+'/'+mode
                    core=make_core(torch,cfg,calcium,base,data['area_um2'],model,channels)
                    candidate=rollout(torch,core,data,device,imposed if mode=='imposed' else None,True)
                    ref_metric={k:v.copy() for k,v in ref.items()}
                    # Native V/Ca endpoints for historical autonomous episodes, not just formula-relative error.
                    if mode=='autonomous':
                        for k in ('voltage','calcium'):ref_metric[k][:,:32]=data[k][:32].T
                    measured=metrics(candidate,ref_metric,data,meta,spec['current_quality'],cfg['gates'],mode)
                    reports[key]=measured
                    if family=='all':
                        saved[key.replace('/','__')]=candidate
                        intervention=[]
                        for s,seed in enumerate(spec['seeds']):
                            for case,row in enumerate(meta[32:],32):
                                origin=row['source_index']
                                mm=p.evaluate_intervention(ref['current'][:,origin],ref['current'][:,case],candidate['current'][:,s,origin],candidate['current'][:,s,case],.1,spec['current_quality'])
                                intervention.append(dict(case=case,seed=seed,metrics=mm))
                        measured['interventions']=intervention
                    p.save_json(output/'causal_results.json',reports)
                    print(f'[GIADA Task40] checkpoint {key} {measured["passed_cases"]}/{measured["total"]}',flush=True)
        np.savez_compressed(output/'primary_traces.npz',**{label+'__'+k:v for label,row in saved.items() for k,v in row.items()})
        np.savez_compressed(output/'reference_traces.npz',**{label+'__'+k:v for label,row in dict(golden=golden,**references).items() for k,v in row.items()})
        timing=benchmark(torch,models,data,spec,cfg,calcium,base);p.save_json(output/'cost.json',timing)
    primary=reports['J1/all/autonomous'];forced=reports['J1/all/imposed']
    passed=primary['passed'] and forced['passed'] and all(r['metrics']['passed'] for r in primary['interventions'])
    freeze=dict(candidate='J1',seeds=spec['seeds'],checkpoint_sha256=p.digest(output/'J1_frozen.pt'),
                protocol_sha256=p.digest(root/CONFIG),thresholds_changed=False,training_performed=False,
                task41_new_protocols_and_seeds_required=True,known_stress_failures=spec['known_stress'],
                inference_numerics='FP32 recurrent state, original J1 FP64 affine correction; no canonical gating in all-learned arm')
    p.save_json(output/'confirmation_contract.json',freeze)
    report=dict(valid=True,decision='CAUSAL_DOMAIN_PASSED_PREPARE_INDEPENDENT_CONFIRMATION' if passed else 'CAUSAL_REINSERTION_NO_GO',
                task40_causal_passed=passed,task41_preparation_authorized=passed,training_performed=False,
                thresholds_changed=False,independent_confirmation=False,speedup_claim=False,
                summary={k:dict(passed=v['passed'],cases_passed=v['passed_cases'],total=v['total']) for k,v in reports.items()},
                stress_diagnostic=[dict(model=r['model'],family=r['family'],case=r['case'],seed=r['seed'],passed=r['metrics']['passed']) for r in stress],
                limitations=spec['limits'],confirmation_contract=freeze)
    p.save_json(output/'final_report.json',report);return report
