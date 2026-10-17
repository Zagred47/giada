"""Preregistered small factorial diagnosis; all members share paired windows."""
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from . import task38_joint_transition as old

p = old.p
CONFIG = 'experiments/task39_learnability_preregistration_v1.json'


def arms():
    result = [dict(name=f'{bias}_{coverage}_{mode}', bias=bias, coverage=coverage, mode=mode, loss='base')
        for bias, coverage, mode in itertools.product(('joint', 'factorized'), ('fit', 'expanded'), ('tf', 'fr'))]
    result += [dict(name='joint_fit_tf_gate_only', bias='joint', coverage='fit', mode='tf', loss='gate'),
               dict(name='joint_fit_fr_relative', bias='joint', coverage='fit', mode='fr', loss='relative')]
    return result


def model_class(torch):
    class Model(old.model_class(torch)):
        def __init__(self, definitions, seeds):
            super().__init__(32, seeds*len(definitions))
            self.register_buffer('context', torch.tensor([int(a['bias']=='joint') for a in definitions for _ in seeds], dtype=torch.float32)[:, None, None])

        def forward(self, v, ca, state):
            previous = torch.cat((state[..., :16], state[..., 17:]), -1)
            inp = torch.cat(((v[..., None]+40)/80, (previous*2-1)*self.context), -1)
            raw = torch.bmm(torch.nn.functional.silu(torch.bmm(inp, self.a)+self.b), self.c)+self.d[:, None]
            candidate, fraction = torch.sigmoid(raw).split(17, -1)
            new = previous+fraction*(candidate-previous)
            inp_sk = torch.stack(((state[..., 16]*2-1)*self.context[..., 0], (ca.clamp_min(1e-9).log10()+4.5)/2.5), -1)
            sk = torch.sigmoid(torch.bmm(torch.nn.functional.silu(torch.bmm(inp_sk, self.sa)+self.sb), self.sc)+self.sd[:, None])
            z = state[..., 16]+sk[..., 1]*(sk[..., 0]-state[..., 16])
            return torch.cat((new[..., :16], z[..., None], new[..., 16:]), -1)
    return Model


def windows(root, role, families, count, length, seed):
    rng = np.random.default_rng(seed); rows, support = [], []
    for family in families:
        with np.load(root/f'{role}__{family}.npz') as z:
            a = {k: z[k] for k in z.files}
        ticks, cases = a['voltage'].shape
        events = (np.abs(np.diff(a['voltage'], axis=0, prepend=a['voltage'][:1])) > 1) | (np.abs(np.diff(np.log10(a['bridge_ca'].clip(1e-9)), axis=0, prepend=np.log10(a['bridge_ca'][:1].clip(1e-9)))) > .05)
        events[:25] = True; events[-25:] = True
        candidates = np.argwhere(events)
        starts = rng.integers(0, ticks-length+1, count); case = rng.integers(0, cases, count)
        selected = candidates[rng.integers(0, len(candidates), count//2)]
        starts[:count//2] = (selected[:, 0]-rng.integers(0, length, count//2)).clip(0, ticks-length)
        case[:count//2] = selected[:, 1]
        time_index = starts[:, None]+np.arange(length)[None]
        ci = np.broadcast_to(case[:, None], time_index.shape)
        pre = a['gate_targets'][np.maximum(time_index-1, 0), ci].copy()
        for j in np.flatnonzero(starts == 0): pre[j, 0] = a['initial_gates'][case[j]]
        data = np.concatenate((a['voltage'][time_index, ci, None], a['bridge_ca'][time_index, ci, None], pre,
            a['gate_targets'][time_index, ci], np.broadcast_to(a['multipliers'][case, None], (count, length, 11)),
            np.broadcast_to(a['reversals'][case, None], (count, length, 11)), a['current_targets'][time_index, ci]), -1)
        rows.append(data)
        support.append(dict(role=role, family=family, windows=count, unique_windows=len(set(zip(starts.tolist(), case.tolist()))),
            independent_trajectories=cases, assignment_sha256=hashlib.sha256(np.stack((starts, case)).tobytes()).hexdigest()))
    return np.concatenate(rows).astype('float32'), support


def objective(torch, model, batch, gs, cs, teacher, current_weight, relative_weight):
    state = batch[:, :, 0, 2:20]
    total = torch.zeros(model.a.shape[0], device=state.device)
    relative_scale = batch[..., 60:71].abs().amax(2).clamp_min(1e-7)
    for tick in range(batch.shape[2]):
        b = batch[:, :, tick]
        state = torch.where(teacher, b[..., 2:20], state)
        state = model(b[..., 0], b[..., 1], state)
        current = old.readout(torch, b[..., 0], state, b[..., 38:49], b[..., 49:60])
        total = total+((state-b[..., 20:38])/gs).square().mean((1, 2))
        total = total+current_weight*((current-b[..., 60:71])/cs).square().mean((1, 2))
        total = total+relative_weight*((current-b[..., 60:71])/relative_scale).square().mean((1, 2))
    return total/batch.shape[2]


def flags(torch, definitions, seeds, device):
    expanded = [a for a in definitions for _ in seeds]
    tensor = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    coverage = tensor([a['coverage']=='expanded' for a in expanded]).bool()[:, None, None, None]
    teacher = tensor([a['mode']=='tf' for a in expanded]).bool()[:, None, None]
    weight = tensor([a['loss']!='gate' for a in expanded])
    relative = tensor([a['loss']=='relative' for a in expanded])
    return coverage, teacher, weight, relative


def preflight(torch, spec, batch, gs, cs, device):
    definitions = arms(); model = model_class(torch)(definitions, spec['seeds']).to(device)
    _, teacher, weight, relative = flags(torch, definitions, spec['seeds'], device)
    original = {n: x.detach().clone() for n, x in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    objective(torch, model, batch, gs, cs, teacher, weight, relative).sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in model.parameters())
    old.clip(torch, model); optimizer.step(); error = 0.
    for arm_index, definition in enumerate(definitions):
        k = arm_index*3
        iso = model_class(torch)([definition], [17]).to(device)
        with torch.no_grad():
            for name, param in iso.named_parameters(): param.copy_(original[name][k:k+1])
        opt = torch.optim.Adam(iso.parameters(), lr=.001)
        _, tf, cw, rw = flags(torch, [definition], [17], device)
        objective(torch, iso, batch[k:k+1], gs, cs, tf, cw, rw).sum().backward(); old.clip(torch, iso); opt.step()
        error = max(error, max(float((param[0]-dict(model.named_parameters())[name][k]).abs().max()) for name, param in iso.named_parameters()))
    assert error <= spec['numeric_atol'], error
    return dict(valid=True, independent_adam_max_error=error, arm_types_tested=len(definitions), targets_per_member_per_update=spec['sequence_length']*spec['batch_sequences'])


def split_models(torch, model, definitions, seeds, device):
    result = []
    for k, definition in enumerate(definitions):
        m = model_class(torch)([definition], seeds).to(device)
        m.load_state_dict({n: x[k*3:(k+1)*3].clone() for n, x in model.state_dict().items()})
        result.append(m)
    return result


def severity(metrics):
    values = [metrics['gate']['rmse']/.005, metrics['gate']['max_abs']/.03]
    for c in metrics['channels']:
        values += [c['nrmse']/.01, c['nmax']/.05, abs(c['signed_charge_normalized'])/.02, c['absolute_charge_normalized']/.02]
    return max(values)


def contrasts(rows, definitions):
    scores = {}
    for a in definitions:
        rs = [r for r in rows if r['model']=='joint_w'+a['name'] and r['role']=='development']
        scores[a['name']] = {str(seed): dict(median=float(np.median([severity(r['metrics']) for r in rs if r['seed']==seed])),
            worst=max(severity(r['metrics']) for r in rs if r['seed']==seed)) for seed in (17, 29, 43)}
    pairs = []
    for axis, low, high in [('coverage', 'fit', 'expanded'), ('mode', 'tf', 'fr'), ('bias', 'joint', 'factorized')]:
        for a in definitions[:8]:
            if a[axis] != low: continue
            b = next(b for b in definitions[:8] if b[axis]==high and all(a[k]==b[k] for k in ('coverage', 'mode', 'bias') if k!=axis))
            pairs.append((axis, a['name'], b['name']))
    pairs += [('current_supervision', 'joint_fit_tf_gate_only', 'joint_fit_tf'), ('small_currents', 'joint_fit_fr', 'joint_fit_fr_relative')]
    result = []
    for axis, a, b in pairs:
        ratios = [scores[b][str(s)]['median']/max(scores[a][str(s)]['median'], 1e-12) for s in (17,29,43)]
        worst = [scores[b][str(s)]['worst']/max(scores[a][str(s)]['worst'], 1e-12) for s in (17,29,43)]
        result.append(dict(factor=axis, control=a, treatment=b, paired_seed_ratios=ratios, worst_ratios=worst,
            supported_in_this_suite=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    return dict(scores=scores, planned_contrasts=result)


def run(root, output, device='cuda'):
    import torch
    spec = json.loads((root/CONFIG).read_text()); quality = json.loads((root/p.CONFIG).read_text())
    output.mkdir(parents=True, exist_ok=True)
    if (output/'final_report.json').exists(): raise FileExistsError(output)
    p.save_json(output/'preregistration.json', spec)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert device != 'cuda' or torch.cuda.is_available()
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()): assert p.digest(root/spec['dataset']/row['file']) == row['sha256']
    parent = root/spec['parent']; freeze = json.loads((parent/'selection_freeze.json').read_text()); claim = freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest() == claim
    assert p.digest(parent/'frozen_checkpoint.pt') == spec['parent_checkpoint_sha256'] == freeze['checkpoint_sha256']
    baseline_spec = json.loads((root/old.CONFIG).read_text())
    assert p.digest(root/baseline_spec['baseline_checkpoint']) == baseline_spec['baseline_sha256']
    assert p.digest(root/baseline_spec['baseline_parent']) == baseline_spec['baseline_parent_sha256']
    normal = json.loads((parent/'fit_normalizers.json').read_text())
    gs, cs = [torch.tensor(normal[k], device=device) for k in ('gate', 'current')]
    definitions = arms(); model = model_class(torch)(definitions, spec['seeds']).to(device)
    families = list(quality['families']); support = []; pools = {}
    for role, count in [('fit', spec['windows_per_family']), ('fit_extra', spec['windows_per_family']), ('development', spec['development_windows_per_family'])]:
        data, info = windows(root/spec['dataset'], role, families, count, spec['sequence_length'], spec['data_seed']+len(pools)*1009)
        pools[role] = torch.as_tensor(data, device=device); support += info
    p.save_json(output/'window_support.json', support)
    coverage, teacher, weight, relative = flags(torch, definitions, spec['seeds'], device)
    n = model.a.shape[0]
    smoke = pools['fit'][:7][None].expand(n, -1, -1, -1)
    p.save_json(output/'optimizer_preflight.json', preflight(torch, spec, smoke, gs, cs, device))
    # Test every dependency mask with the shared captured scanner before training.
    old.recurrence_preflight(output, spec, split_models(torch, model, definitions, spec['seeds'], device), torch, device)
    sample_v = smoke[:, :, 0, 0]; sample_ca = smoke[:, :, 0, 1]; sample_x = smoke[:, :, 0, 2:20]
    with torch.no_grad():
        a = model(sample_v, sample_ca, sample_x); b = model(sample_v, sample_ca*10, sample_x)
        assert torch.equal(a[..., :16], b[..., :16]) and torch.equal(a[..., 17:], b[..., 17:])
        assert a.min()>=0 and a.max()<=1
        changed = sample_x.clone(); changed[..., 1:] = 1-changed[..., 1:]
        prediction = model(sample_v, sample_ca, changed)
        factorized = [i for i, arm in enumerate(definitions) if arm['bias']=='factorized']
        for arm_index in factorized: assert torch.equal(a[arm_index*3:arm_index*3+3, :, 0], prediction[arm_index*3:arm_index*3+3, :, 0])
        blocked = old.readout(torch, sample_v, a, torch.zeros_like(smoke[:,:,0,38:49]), smoke[:,:,0,49:60])
        assert torch.equal(blocked, torch.zeros_like(blocked))
        from .gate_d_compute_audit import exact_gpu
        example = pools['fit'][:512,0]
        ref,_ = exact_gpu(example[:,0],example[:,1],example[:,2:20],torch.full_like(example[:,0],.1),torch)
        ge = float((ref-example[:,20:38]).abs().max())
        ce = float((old.readout(torch,example[:,0],ref,example[:,38:49],example[:,49:60])-example[:,60:71]).abs().max())
        assert ge <= 1e-5 and ce <= 1e-7, (ge,ce)
    p.save_json(output/'phase_preflight.json',dict(valid=True, gate_error=ge,current_error=ce,blocked_exact=True,phase_A_Ca_invariant=True))
    print('[GIADA Task39] hashes, phase, optimizer and CUDA recurrence preflights passed', flush=True)
    fn = lambda batch: objective(torch, model, batch, gs, cs, teacher, weight, relative)
    execution = dict(compiled=False, members=n, gpu=torch.cuda.get_device_name() if device=='cuda' else 'CPU', torch=torch.__version__)
    try:
        compiled = torch.compile(fn)
        sample = pools['fit'][:spec['batch_sequences']][None].expand(n, -1, -1, -1)
        e = fn(sample); ge = torch.autograd.grad(e.sum(), tuple(model.parameters()))
        c = compiled(sample); gc = torch.autograd.grad(c.sum(), tuple(model.parameters()))
        forward = float(((c-e).abs()/e.abs().clamp_min(1)).max())
        grad = max(float(((x-y).abs()/y.abs().clamp_min(1)).max()) for x, y in zip(gc, ge))
        assert max(forward, grad) <= spec['numeric_atol'], (forward, grad)
        fn = compiled; execution.update(compiled=True, forward_relative_error=forward, gradient_relative_error=grad)
    except Exception as exc: execution['fallback_reason'] = str(exc)[:700]
    p.save_json(output/'execution.json', execution)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['learning_rate'])
    generator = torch.Generator(device=device).manual_seed(spec['data_seed']+4)
    best = np.full(n, np.inf); chosen = [None]*n; saved = [None]*n; curves = []
    exposure = np.zeros((len(definitions), len(families), 2), dtype=np.int64)
    def dev_score():
        values = [[], []]
        with torch.no_grad():
            for batch in pools['development'].split(spec['batch_sequences']):
                data = batch[None].expand(n, -1, -1, -1)
                for k, tf in enumerate((True, False)):
                    values[k].append(objective(torch, model, data, gs, cs, torch.full_like(teacher, tf), torch.ones_like(weight), torch.zeros_like(relative))*len(batch))
        return torch.maximum(*[torch.stack(v).sum(0)/len(pools['development']) for v in values]).cpu().numpy()
    start = time.perf_counter()
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            scores = dev_score(); assert np.isfinite(scores).all()
            for j, score in enumerate(scores):
                if score < best[j]:
                    best[j], chosen[j] = float(score), step
                    saved[j] = {k: x[j].detach().cpu().clone() for k, x in model.state_dict().items()}
            curves.append(dict(step=step, elapsed_seconds=time.perf_counter()-start, common_development_score=scores.tolist()))
            p.save_json(output/'training_curves.json', curves)
            torch.save(model.state_dict(), output/f'checkpoint_{step}.pt')
            print(f'[GIADA Task39] checkpoint {step}/60000 median={np.median(scores):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min', flush=True)
        if step == spec['checkpoints'][-1]: break
        idx = torch.randint(len(pools['fit']), (spec['batch_sequences'],), device=device, generator=generator)
        extra = torch.rand(spec['batch_sequences'], device=device, generator=generator)<.5
        expanded = torch.where(extra[:, None, None], pools['fit_extra'][idx], pools['fit'][idx])
        batch = torch.where(coverage, expanded[None], pools['fit'][idx][None])
        optimizer.zero_grad(set_to_none=True); fn(batch).sum().backward(); old.clip(torch, model); optimizer.step()
        fi, ex = (idx//spec['windows_per_family']).cpu().numpy(), extra.cpu().numpy()
        for k, arm in enumerate(definitions):
            actual = ex if arm['coverage']=='expanded' else np.zeros_like(ex)
            np.add.at(exposure[k], (fi, actual.astype(int)), spec['sequence_length'])
        if (step+1)%1000 == 0 and step+1 not in spec['checkpoints']:
            print(f'[GIADA Task39] update {step+1}/60000 elapsed={(time.perf_counter()-start)/60:.1f}min', flush=True)
    with torch.no_grad():
        for name, value in model.state_dict().items():
            for j in range(n): value[j].copy_(saved[j][name])
    torch.save(model.state_dict(), output/'frozen_checkpoint.pt')
    freeze = dict(checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'), chosen_steps=chosen, scores=best.tolist(), arms=definitions,
        development_only=True, stress_used_for_selection=False, independent_confirmation_accessed=False)
    freeze['freeze_sha256'] = hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest()
    p.save_json(output/'selection_freeze.json', freeze)
    p.save_json(output/'exposure.json', dict(arms=definitions, families=families, target_ticks_by_arm_family_role=exposure.tolist(), members_per_arm=3))
    models = split_models(torch, model, definitions, spec['seeds'], device)
    historical = torch.load(parent/'frozen_checkpoint.pt', map_location=device, weights_only=True)
    for w in (32,64):
        m = old.model_class(torch)(w, spec['seeds']).to(device); m.load_state_dict(historical[str(w)]); models.append(m)
    eval_spec = dict(spec, architectures=[a['name'] for a in definitions]+['historical38_w32','historical38_w64'])
    baseline_spec = json.loads((root/old.CONFIG).read_text())
    for key in ('baseline_parent','baseline_checkpoint'): eval_spec[key] = baseline_spec[key]
    summary = old.evaluate(root, output, eval_spec, models, torch, device, quality)
    rows = json.loads((output/'recurrence_evaluation.json').read_text())
    result = contrasts(rows, definitions); p.save_json(output/'paired_contrasts.json', result)
    strata = []
    for name in summary:
        for role in ('development','stress_diagnostic'):
            channels = [c for r in rows if r['model']==name and r['role']==role for c in r['metrics']['channels']]
            for label, select in [('weak', lambda c:c['scale_ma_cm2']<=1e-6), ('active',lambda c:c['scale_ma_cm2']>1e-6)]:
                cs_ = [c for c in channels if select(c)]
                strata.append(dict(model=name, role=role, stratum=label, channels=len(cs_),
                    median_nrmse=float(np.median([c['nrmse'] for c in cs_])) if cs_ else None,
                    worst_absolute_rmse=max([c['rmse_ma_cm2'] for c in cs_], default=None)))
    p.save_json(output/'current_strata.json', strata)
    # Probe the frozen learned field off the training-history manifold, not a new test claim.
    from .gate_d_compute_audit import exact_gpu
    rng = np.random.default_rng(spec['data_seed']+10)
    v = torch.tensor(rng.uniform(-100,60,256), dtype=torch.float32, device=device)
    ca = torch.tensor(10**rng.uniform(-7,-2,256), dtype=torch.float32, device=device)
    x = torch.tensor(rng.uniform(0,1,(256,18)), dtype=torch.float32, device=device)
    with torch.no_grad():
        ref,_ = exact_gpu(v, ca, x, torch.full_like(v,.1), torch)
        pred = model(v[None].expand(n,-1),ca[None].expand(n,-1),x[None].expand(n,-1,-1))
        changed = x.clone(); changed[:,1] = 1-changed[:,1]
        perturbed = model(v[None].expand(n,-1),ca[None].expand(n,-1),changed[None].expand(n,-1,-1))
        probe = dict(random_state_gate_rmse=(pred-ref).square().mean((1,2)).sqrt().cpu().tolist(),
            gate0_cross_dependence_on_gate1=(perturbed[...,0]-pred[...,0]).abs().amax(1).cpu().tolist(),
            source='fit-domain synthetic post-freeze diagnostics, not independent confirmation')
    p.save_json(output/'internal_probes.json', probe)
    passing = [a['name'] for a in definitions if summary['joint_w'+a['name']]['all_development_passed']]
    budgets = []
    a = np.asarray(next(r['common_development_score'] for r in curves if r['step']==30000)); b = np.asarray(curves[-1]['common_development_score'])
    for k, arm in enumerate(definitions):
        ratio = (b[k*3:k*3+3]/np.maximum(a[k*3:k*3+3],1e-12)).tolist()
        budgets.append(dict(arm=arm['name'], ratios_60k_over_30k=ratio, short_budget_support=bool(np.median(ratio)<=.8 and max(ratio)<1)))
    p.save_json(output/'budget_contrasts.json', budgets)
    final = dict(valid=True, task=39, decision='DEVELOPMENT_CANDIDATE_REQUIRES_AUDIT_BEFORE_TASK40' if passing else 'NO_GO_TASK40_BLOCKED',
        passing_arms=passing, summaries=summary, planned_contrasts=result['planned_contrasts'], budget_contrasts=budgets,
        independent_confirmation_accessed=False, task40_preparation_eligible=bool(passing), speedup_claimed=False,
        thresholds_changed=False, limits=spec['limits'])
    p.save_json(output/'final_report.json', final)
    return final
