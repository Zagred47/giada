"""Task38: explicit-state joint ionic recurrence, member-vectorized CUDA training."""
import json
import time
from pathlib import Path
import numpy as np
from . import task37_ionic_playground as p

CONFIG = 'experiments/task38_joint_transition_preregistration_v1.json'
IDS = tuple(i for i in range(18) if i != 16)


def model_class(torch):
    class Joint(torch.nn.Module):
        def __init__(self, width, seeds):
            super().__init__()
            rows = []
            for seed in seeds:
                gen = torch.Generator().manual_seed(seed)
                # Generate width64 first, so the shared prefix is paired.
                a = torch.randn(18, 64, generator=gen) / np.sqrt(18)
                c = torch.randn(64, 34, generator=gen) * .01
                sa = torch.randn(2, 8, generator=gen) / np.sqrt(2)
                sc = torch.randn(8, 2, generator=gen) * .01
                rows.append((a[:, :width], torch.zeros(1, width), c[:width],
                             torch.cat((torch.zeros(17), torch.full((17,), -3.))),
                             sa, torch.zeros(1, 8), sc, torch.tensor([0., -3.])))
            for k, name in enumerate(('a', 'b', 'c', 'd', 'sa', 'sb', 'sc', 'sd')):
                self.register_parameter(name, torch.nn.Parameter(torch.stack([r[k] for r in rows])))

        def forward(self, v, ca, state):
            old = state[..., IDS]
            inp = torch.cat(((v[..., None]+40)/80, old*2-1), -1)
            raw = torch.bmm(torch.nn.functional.silu(torch.bmm(inp, self.a)+self.b), self.c)+self.d[:, None, :]
            candidate, fraction = torch.sigmoid(raw).split(17, -1)
            new = old + fraction*(candidate-old)
            sk_in = torch.stack((state[..., 16]*2-1, (ca.clamp_min(1e-9).log10()+4.5)/2.5), -1)
            sk = torch.sigmoid(torch.bmm(torch.nn.functional.silu(torch.bmm(sk_in, self.sa)+self.sb), self.sc)+self.sd[:, None, :])
            z = state[..., 16] + sk[..., 1]*(sk[..., 0]-state[..., 16])
            return torch.cat((new[..., :16], z[..., None], new[..., 16:]), -1)
    return Joint


def readout(torch, v, gates, mult, rev):
    opening = []
    for k, names in enumerate(p.ionic.STATE_NAMES):
        a = p.OFFSETS[k]
        x = gates[..., a] ** int(p.ionic.POWERS[k])
        if len(names) == 2:
            x = x*gates[..., a+1]
        opening.append(x)
    g = torch.stack(opening, -1)*mult*torch.as_tensor(p.ionic.GBAR, device=v.device, dtype=v.dtype)
    return g*(v[..., None]-rev)


def clip(torch, model):
    norm = torch.stack([x.grad.flatten(1).square().sum(1) for x in model.parameters()]).sum(0).sqrt().clamp_min(1)
    for x in model.parameters():
        x.grad.div_(norm.reshape(-1, *([1]*(x.ndim-1))))


def pool(root, role, families, count, seed):
    rng = np.random.default_rng(seed)
    rows, family_ids = [], []
    for fi, family in enumerate(families):
        with np.load(root/f'{role}__{family}.npz') as a:
            t, n = a['voltage'].shape
            idx = rng.choice(t*n, min(count, t*n), replace=False)
            # Equal count per family, including short protocols, without invented data.
            if len(idx) < count:
                idx = np.tile(idx, (count+len(idx)-1)//len(idx))[:count]
            tick, case = idx//n, idx % n
            pre = a['gate_targets'][np.maximum(tick-1, 0), case].copy()
            pre[tick == 0] = a['initial_gates'][case[tick == 0]]
            rows.append(np.concatenate((a['voltage'][tick, case, None], a['bridge_ca'][tick, case, None],
                pre, a['gate_targets'][tick, case], a['multipliers'][case], a['reversals'][case],
                a['current_targets'][tick, case]), -1))
            family_ids.extend([fi]*count)
    return np.concatenate(rows).astype(np.float32), np.asarray(family_ids)


def loss(torch, model, batch, gate_scale, current_scale):
    v, ca, x, y, m, e, target = batch[:, 0], batch[:, 1], batch[:, 2:20], batch[:, 20:38], batch[:, 38:49], batch[:, 49:60], batch[:, 60:71]
    n = model.a.shape[0]
    pred = model(v[None].expand(n, -1), ca[None].expand(n, -1), x[None].expand(n, -1, -1))
    current = readout(torch, v, pred, m, e)
    return ((pred-y)/gate_scale).square().mean((1, 2)) + ((current-target)/current_scale).square().mean((1, 2))


def preflight(torch, device):
    cls = model_class(torch)
    m = cls(32, [17, 29, 43]).to(device)
    gen = torch.Generator(device=device).manual_seed(98318)
    v = torch.rand(3, 29, generator=gen, device=device)*160-100
    ca = torch.full_like(v, .0002); x = torch.rand(3, 29, 18, generator=gen, device=device)
    initial = {k: a.detach().clone() for k, a in m.named_parameters()}
    pred = m(v, ca, x)
    changed = m(v, ca*10, x)
    assert torch.equal(pred[..., IDS], changed[..., IDS]), 'Ca leaked into phase A'
    assert pred.min() >= 0 and pred.max() <= 1
    optimizer = torch.optim.Adam(m.parameters(), lr=.001)
    pred.square().mean((1, 2)).sum().backward(); clip(torch, m); optimizer.step()
    maximum = 0.
    for j, seed in enumerate([17, 29, 43]):
        iso = cls(32, [seed]).to(device)
        with torch.no_grad():
            for k, a in iso.named_parameters(): a.copy_(initial[k][j:j+1])
        opt = torch.optim.Adam(iso.parameters(), lr=.001)
        iso(v[j:j+1], ca[j:j+1], x[j:j+1]).square().mean().backward(); clip(torch, iso); opt.step()
        maximum = max(maximum, max(float((a[0]-dict(m.named_parameters())[k][j]).abs().max()) for k, a in iso.named_parameters()))
    assert maximum < 1e-5, maximum
    return dict(valid=True, independent_adam_max_error=maximum, phase_A_ca_invariant=True, occupancy_bounded=True)


class Scanner:
    """Fixed-size CUDA graph; state survives every chunk and input-buffer copy."""
    def __init__(self, torch, models, n, chunk, device):
        self.torch, self.models, self.chunk, self.n = torch, models, chunk, n
        self.v = torch.zeros(chunk, n, device=device)
        self.ca = torch.full_like(self.v, 1e-4)
        self.state = [torch.full((m.a.shape[0], n, 18), .5, device=device) for m in models]
        self.out = [torch.empty((chunk, *s.shape), device=device) for s in self.state]
        def step():
            for k, m in enumerate(models):
                x = self.state[k]
                for t in range(chunk):
                    x = m(self.v[t][None].expand(x.shape[0], -1), self.ca[t][None].expand(x.shape[0], -1), x)
                    self.out[k][t].copy_(x)
                self.state[k].copy_(x)
        self.step = step
        self.graph = None
        if str(device).startswith('cuda'):
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3): step()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(self.graph): step()

    def run(self, v, ca, initial):
        torch = self.torch
        for s in self.state: s.copy_(torch.as_tensor(initial, device=s.device)[None].expand_as(s))
        result = [[] for _ in self.models]
        with torch.no_grad():
            for start in range(0, len(v), self.chunk):
                count = min(self.chunk, len(v)-start)
                vv, cc = v[start:start+count], ca[start:start+count]
                if count < self.chunk:
                    vv = np.pad(vv, ((0, self.chunk-count), (0, 0)), mode='edge')
                    cc = np.pad(cc, ((0, self.chunk-count), (0, 0)), mode='edge')
                self.v.copy_(torch.as_tensor(vv, device=self.v.device)); self.ca.copy_(torch.as_tensor(cc, device=self.v.device))
                if self.graph: self.graph.replay()
                else: self.step()
                for k, out in enumerate(self.out): result[k].append(out[:count].cpu().numpy().copy())
        return [np.concatenate(x) for x in result]


def recurrence_preflight(output, spec, models, torch, device):
    scanner = Scanner(torch, models, 2, spec['recurrence_chunk'], device)
    # Check real recurrent graph execution against eager, not just one-step forward.
    rng = np.random.default_rng(98319); v = rng.uniform(-90, 30, (65, 2)).astype('float32')
    ca = np.full_like(v, 1e-4); x = np.full((2, 18), .4, dtype='float32')
    graph = scanner.run(v, ca, x)
    maximum = 0.
    with torch.no_grad():
        for k, m in enumerate(models):
            state = torch.as_tensor(x, device=device)[None].expand(3, -1, -1).clone(); eager = []
            for t in range(len(v)):
                state = m(torch.as_tensor(v[t], device=device)[None].expand(3, -1), torch.as_tensor(ca[t], device=device)[None].expand(3, -1), state)
                eager.append(state.cpu().numpy())
            maximum = max(maximum, float(np.max(np.abs(np.asarray(eager)-graph[k]))))
    assert maximum <= spec['numeric_atol'], maximum
    p.save_json(output/'recurrence_preflight.json', dict(valid=True, max_abs_error=maximum, chunk_boundary_test_ticks=65, no_state_reset=True))
    return scanner


def evaluate(root, output, spec, models, torch, device, quality):
    from . import ie_residual_repair as old
    parent = torch.load(root/spec['baseline_parent'], map_location='cpu', weights_only=True)
    frozen = torch.load(root/spec['baseline_checkpoint'], map_location=device, weights_only=True)
    baseline = old.local(torch, parent, 161, device); baseline.load_state_dict(frozen['local161'])
    rows, memory_probes = [], []
    scanner = recurrence_preflight(output, spec, models, torch, device)
    for role in ('development', 'stress_diagnostic'):
        for family in quality['families']:
            with np.load(root/spec['dataset']/f'{role}__{family}.npz') as z:
                a = {k: z[k] for k in z.files}
            predictions = scanner.run(a['voltage'], a['bridge_ca'], a['initial_gates'])
            # Frozen coefficient baseline is affine in old gates: coefficients precomputed,
            # then CPU recurrence, avoiding 100000 tiny GPU dispatches.
            with torch.no_grad():
                vv = torch.as_tensor(a['voltage'].reshape(-1), device=device)
                u, fraction = baseline(vv[None].expand(3, -1))
                u = u.reshape(3, *a['voltage'].shape, 17).cpu().numpy()
                fraction = fraction.reshape(3, *a['voltage'].shape, 17).cpu().numpy()
            sk_u, sk_z = p.coefficients(a['voltage'], a['bridge_ca'], .1)
            state = np.broadcast_to(a['initial_gates'], (3, *a['initial_gates'].shape)).copy()
            base = np.empty((len(a['voltage']), 3, 2, 18), dtype='float32')
            for t in range(len(base)):
                state[..., IDS] += fraction[:, t]*(u[:, t]-state[..., IDS])
                state[..., 16] += sk_z[t, :, 16]*(sk_u[t, :, 16]-state[..., 16])
                base[t] = state
            for name, gates in [(f'joint_w{w}', g) for w, g in zip(spec['architectures'], predictions)] + [('frozen_local161_SKcanonical', base)]:
                for member, seed in enumerate(spec['seeds']):
                    _, currents = p.readout(a['voltage'], gates[:, member], a['multipliers'], a['reversals'])
                    if family == 'same_v_history':
                        start = int(len(currents)*.4)
                        expected = a['current_targets'][start:, 0]-a['current_targets'][start:, 1]
                        actual = currents[start:, 0]-currents[start:, 1]
                        memory_probes.append(dict(role=role, model=name, seed=seed,
                            reference_tail_contrast=float(np.max(np.abs(expected))),
                            predicted_tail_contrast=float(np.max(np.abs(actual))),
                            contrast_error=float(np.max(np.abs(actual-expected))),
                            matched_initial_state_inputs_parameters=True))
                    for case in range(2):
                        report = p.evaluate_currents(a['current_targets'][:, case], currents[:, case], .1,
                            quality['quality'], a['gate_targets'][:, case], gates[:, member, case], a['multipliers'][case] == 0)
                        rows.append(dict(role=role, family=family, case=case, model=name, seed=seed, metrics=report))
            p.save_json(output/'recurrence_evaluation.json', rows)
            p.save_json(output/'matched_history_probes.json', memory_probes)
            print(f'[GIADA Task38] recurrence {role}/{family} complete', flush=True)
    summary = {}
    for name in sorted({r['model'] for r in rows}):
        selected = [r for r in rows if r['model'] == name and r['role'] == 'development']
        summary[name] = dict(all_development_passed=all(r['metrics']['passed'] for r in selected),
            passed_cases=sum(r['metrics']['passed'] for r in selected), total_cases=len(selected),
            worst_gate_rmse=max(r['metrics']['gate']['rmse'] for r in selected))
    return summary


def run(root, output, device='cuda'):
    import torch
    spec = json.loads((root/CONFIG).read_text())
    quality = json.loads((root/p.CONFIG).read_text())
    if (output/'final_report.json').exists(): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    p.save_json(output/'preregistration.json', spec)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('Registered CUDA device unavailable')
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()):
        assert p.digest(root/spec['dataset']/row['file']) == row['sha256']
    for name in ('baseline', 'baseline_parent'):
        key = 'baseline_checkpoint' if name == 'baseline' else name
        assert p.digest(root/spec[key]) == spec[name+'_sha256']
    print('[GIADA Task38] data hashes verified; independent Adam/phase preflight', flush=True)
    p.save_json(output/'optimizer_preflight.json', preflight(torch, device))
    families = list(quality['families'])
    fit, family = pool(root/spec['dataset'], 'fit', families, spec['pool_per_family'], spec['data_seed'])
    dev, _ = pool(root/spec['dataset'], 'development', families, spec['development_per_family'], spec['data_seed']+1)
    gate_scale = np.maximum(np.sqrt(np.mean((fit[:, 20:38]-fit[:, 2:20])**2, axis=0)), .001)
    current_scale = np.maximum(np.sqrt(np.mean(fit[:, 60:71]**2, axis=0)), 1e-7)
    p.save_json(output/'fit_normalizers.json', dict(gate=gate_scale.tolist(), current=current_scale.tolist(), source='fit only'))
    fit, dev = [torch.as_tensor(a, device=device) for a in (fit, dev)]
    gs, cs = [torch.as_tensor(a, device=device) for a in (gate_scale, current_scale)]
    models = [model_class(torch)(w, spec['seeds']).to(device) for w in spec['architectures']]
    # Exercise native graph capture before committing the training budget.
    recurrence_preflight(output, spec, models, torch, device)
    from .gate_d_compute_audit import exact_gpu
    b = dev[:512]
    with torch.no_grad():
        reference, _ = exact_gpu(b[:, 0], b[:, 1], b[:, 2:20], torch.full_like(b[:, 0], .1), torch)
        current = readout(torch, b[:, 0], reference, b[:, 38:49], b[:, 49:60])
        ge = float((reference-b[:, 20:38]).abs().max())
        ce = float((current-b[:, 60:71]).abs().max())
        assert ge < 1e-5 and ce < 1e-7, (ge, ce)
        assert torch.equal(readout(torch, b[:, 0], reference, torch.zeros_like(b[:, 38:49]), b[:, 49:60]), torch.zeros_like(current))
    p.save_json(output/'formula_preflight.json', dict(valid=True, gate_error=ge, current_error=ce, blocked_exact=True))
    optimizers = [torch.optim.Adam(m.parameters(), lr=spec['learning_rate']) for m in models]
    # Compilation is optional execution optimization; numerical comparison is mandatory.
    functions = [lambda b, m=m: loss(torch, m, b, gs, cs) for m in models]
    compile_info = []
    for k, fn in enumerate(functions):
        try:
            compiled = torch.compile(fn)
            with torch.no_grad():
                error = float((compiled(dev[:spec['batch_size']])-fn(dev[:spec['batch_size']])).abs().max())
            assert error <= spec['numeric_atol'], error
            functions[k] = compiled; compile_info.append(dict(enabled=True, max_loss_error=error))
        except Exception as exc:
            compile_info.append(dict(enabled=False, reason=str(exc)[:500]))
    p.save_json(output/'execution.json', dict(device=device, torch=torch.__version__, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
        compile=compile_info, members=6, parameters_per_member={str(w): sum(t.numel() for t in m.parameters())//3 for w, m in zip(spec['architectures'], models)}))
    best = [np.full(3, np.inf) for _ in models]; saved = [[None]*3 for _ in models]; chosen = [[None]*3 for _ in models]
    gen = torch.Generator(device=device).manual_seed(spec['data_seed']+2)
    exposure = np.zeros(len(families), dtype=np.int64); curves = []; start = time.perf_counter()
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            row = dict(step=step, elapsed_seconds=time.perf_counter()-start, models={})
            with torch.no_grad():
                for k, (w, model) in enumerate(zip(spec['architectures'], models)):
                    scores = torch.stack([loss(torch, model, b, gs, cs)*len(b) for b in dev.split(spec['batch_size'])]).sum(0)/len(dev)
                    score = scores.cpu().numpy(); assert np.isfinite(score).all()
                    row['models'][str(w)] = score.tolist()
                    for j in range(3):
                        if score[j] < best[k][j]:
                            best[k][j], chosen[k][j] = float(score[j]), step
                            saved[k][j] = {n: a[j].detach().cpu().clone() for n, a in model.named_parameters()}
            curves.append(row); p.save_json(output/'training_curves.json', curves)
            torch.save({str(w): m.state_dict() for w, m in zip(spec['architectures'], models)}, output/f'checkpoint_{step}.pt')
            print(f'[GIADA Task38] training {step}/{spec["checkpoints"][-1]} dev losses={row["models"]} elapsed={row["elapsed_seconds"]/60:.1f}min', flush=True)
        if step == spec['checkpoints'][-1]: break
        idx = torch.randint(len(fit), (spec['batch_size'],), generator=gen, device=device)
        exposure += np.bincount(family[idx.cpu().numpy()], minlength=len(families))
        batch = fit[idx]
        for fn, model, optimizer in zip(functions, models, optimizers):
            optimizer.zero_grad(set_to_none=True); fn(batch).sum().backward(); clip(torch, model); optimizer.step()
        if (step+1) % 1000 == 0 and step+1 not in spec['checkpoints']:
            print(f'[GIADA Task38] training {step+1}/30000 elapsed={(time.perf_counter()-start)/60:.1f}min', flush=True)
    with torch.no_grad():
        for k, model in enumerate(models):
            for n, a in model.named_parameters():
                for j in range(3): a[j].copy_(saved[k][j][n])
    torch.save({str(w): m.state_dict() for w, m in zip(spec['architectures'], models)}, output/'frozen_checkpoint.pt')
    freeze = dict(checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'), chosen_steps=chosen, development_loss=[b.tolist() for b in best],
        selection_used_stress=False, independent_confirmation_accessed=False, selection_rule=spec['selection'])
    freeze['freeze_sha256'] = __import__('hashlib').sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest()
    p.save_json(output/'selection_freeze.json', freeze)
    p.save_json(output/'exposure.json', dict(zip(families, exposure.tolist())))
    summary = evaluate(root, output, spec, models, torch, device, quality)
    passed = any(summary[f'joint_w{w}']['all_development_passed'] for w in spec['architectures'])
    final = dict(valid=True, task=38, decision='DEVELOPMENT_GO_REQUIRES_TASK39_40' if passed else 'DEVELOPMENT_NO_GO_DIAGNOSE_TASK39',
        summaries=summary, quality_thresholds_changed=False, independent_confirmation_accessed=False,
        autonomous_voltage_calcium=False, speedup_demonstrated=False, limits=spec['limits'])
    p.save_json(output/'final_report.json', final)
    return final
