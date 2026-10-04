"""Original roadmap Task19: independent Ih/Im single-state learnability."""
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .hh_family_transfer import TEACHER_REVISION, sha, write

CHANNELS = ('Ih', 'Im')


def rates(channel, voltage):
    v = np.asarray(voltage, dtype=np.float64).copy()
    if channel == 'Ih':
        # Preserve the exact canonical MOD branch, including its voltage mutation.
        v = np.where(v == -154.9, v + .0001, v)
        a = .001 * 6.43 * (v + 154.9) / np.expm1((v + 154.9) / 11.9)
        b = .001 * 193 * np.exp(v / 33.1)
        inf, tau = a / (a + b), 1 / (a + b)
    elif channel == 'Im':
        a = .0033 * np.exp(.1 * (v + 35))
        b = .0033 * np.exp(-.1 * (v + 35))
        inf, tau = a / (a + b), 1 / (a + b) / 2.3 ** 1.3
    else:
        raise ValueError(channel)
    if not np.isfinite(inf).all() or not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError('Invalid canonical rates')
    return inf, tau


def update(state, inf, tau, dt):
    z = -np.expm1(-np.asarray(dt) / tau)
    return (1 - z) * state + z * inf


def data(seed, count, domain=(-135, 75)):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(*domain, count), rng.uniform(0, 1, count),
                 rng.choice([.025, .1, .5, 1, 5, 25, 100], count)]


def state_grid(fresh=False):
    size, offset = (401, .37) if fresh else (201, .5)
    v = -135 + 210 * (np.arange(size) + offset) / size
    if fresh:
        v = np.r_[[-135., 75.], v]
    vv, dt, initial = np.meshgrid(v, [.025, .1, .5, 1, 5, 25, 100], [0., 1.], indexing='ij')
    return np.c_[vv.ravel(), initial.ravel(), dt.ravel()]


def measurements(x, prediction, inf, tau, channel):
    ti, tt = rates(channel, x[:, 0])
    target = update(x[:, 1], ti, tt, x[:, 2])
    error = prediction - target
    driving = x[:, 0] - (-45 if channel == 'Ih' else -85)
    return dict(gate_rmse=float(np.sqrt(np.mean(error ** 2))), gate_max_error=float(abs(error).max()),
        inf_rmse=float(np.sqrt(np.mean((inf - ti) ** 2))), log_tau_rmse=float(np.sqrt(np.mean((np.log(tau) - np.log(tt)) ** 2))),
        open_rmse=float(np.sqrt(np.mean(error ** 2))),
        current_rmse_ma_cm2=float(np.sqrt(np.mean((1e-5 * error * driving) ** 2))),
        occupancy_violations=int(np.sum((prediction < 0) | (prediction > 1))),
        finite=bool(np.isfinite(prediction).all() and np.isfinite(inf).all() and np.isfinite(tau).all() and (tau > 0).all()))


def passes(metrics, gates):
    return metrics['finite'] and all(metrics[k] <= bound for k, bound in gates.items())


def score(domains, gates):
    return max(sum(m[k] / gates[k] for k in ('gate_rmse', 'gate_max_error', 'inf_rmse', 'log_tau_rmse')) for m in domains.values())


def model_factory(torch, width, descriptors):
    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n = len(descriptors)
            shapes = [('w1', (n, width, 1)), ('b1', (n, width)), ('w2', (n, width, width)),
                      ('b2', (n, width)), ('wo', (n, 2, width)), ('bo', (n, 2))]
            for name, shape in shapes:
                value = torch.zeros(shape)
                if name.startswith('w'):
                    for index, (_, _, seed) in enumerate(descriptors):
                        gen = torch.Generator().manual_seed(seed + {'w1': 0, 'w2': 100, 'wo': 200}[name])
                        value[index].normal_(0, .1, generator=gen)
                setattr(self, name, torch.nn.Parameter(value))

        def forward(self, x):
            f = ((x[..., 0] + 30) / 100).unsqueeze(-1)
            f = torch.nn.functional.silu(torch.bmm(f, self.w1.transpose(1, 2)) + self.b1[:, None])
            f = torch.nn.functional.silu(torch.bmm(f, self.w2.transpose(1, 2)) + self.b2[:, None])
            raw = torch.bmm(f, self.wo.transpose(1, 2)) + self.bo[:, None]
            inf, tau = torch.sigmoid(raw[..., 0]), torch.exp(raw[..., 1].clamp(-12, 12))
            z = -torch.expm1(-x[..., 2] / tau)
            return (1 - z) * x[..., 1] + z * inf, inf, tau
    return Ensemble()


def native_audit(teacher, output):
    from neuron import h, load_mechanisms
    import neuron
    assert neuron.__version__.split('+')[0] == '8.2.7'
    assert subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip() == TEACHER_REVISION
    root = Path(output) / 'native'
    root.mkdir()
    hashes = {}
    for channel in CHANNELS:
        source = Path(teacher) / 'L5PC_NEURON_simulation/mods' / (channel + '.mod')
        raw = source.read_text()
        marker = 'SUFFIX ' + channel
        assert raw.count(marker) == 1
        modified = raw.replace(marker, marker + '\n RANGE mInf, mTau', 1)
        (root / source.name).write_text(modified)
        hashes[channel] = dict(canonical_mod_sha256=sha(source), instrumented_mod_sha256=sha(root / source.name))
    command = shutil.which('nrnivmodl') or str(Path(sys.executable).parent / 'nrnivmodl')
    assert Path(command).is_file(), 'nrnivmodl missing'
    with (root / 'compile.log').open('w') as log:
        subprocess.run([command, str(root.resolve())], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    load_mechanisms(str(root.resolve()))
    h.CVode().active(0)
    h.secondorder = 0
    rows = []
    singular = []
    for channel in CHANNELS:
        sec = h.Section(name='audit_' + channel)
        sec.L = sec.diam = 10
        sec.insert(channel)
        mechanism = getattr(sec(.5), channel)
        setattr(mechanism, 'g' + channel + 'bar', 0)
        if channel == 'Ih':
            mechanism.vshift = 0
        maximum_rate = maximum_gate = unexpected_drift = 0.
        for voltage in np.r_[np.linspace(-155, 95, 101), [-154.90001, -154.9, -154.89999, -135, -35, 75]]:
            inf, tau = rates(channel, voltage)
            h.finitialize(float(voltage))
            rate_offset = .0001 if channel == 'Ih' and voltage == -154.9 else 0.
            # NMODL alters the local voltage used by rates(), not the section clamp.
            unexpected_drift = max(unexpected_drift, abs(sec(.5).v - voltage))
            observed = np.array([mechanism.mInf, mechanism.mTau])
            expected = np.array([inf, tau])
            maximum_rate = max(maximum_rate, float(np.max(abs(observed - expected) / (1 + abs(expected)))))
            if rate_offset:
                singular.append(dict(channel=channel, requested_voltage_mv=float(voltage), observed_section_voltage_mv=float(sec(.5).v), effective_rate_voltage_mv=float(voltage+rate_offset), canonical_rate_offset_mv=rate_offset))
            for dt in (.025, .1, 1, 25, 100):
                h.dt = dt
                h.finitialize(float(voltage))
                mechanism.m = .31
                h.fadvance()
                target = update(.31, inf, tau, dt)
                maximum_gate = max(maximum_gate, abs(float(mechanism.m) - float(target)))
                unexpected_drift = max(unexpected_drift, abs(sec(.5).v - voltage))
        rows.append(dict(channel=channel, rate_scaled_max_error=maximum_rate, gate_max_error=maximum_gate, unexpected_voltage_drift_mv=unexpected_drift))
        h.delete_section(sec=sec)
    report = dict(valid=all(r['rate_scaled_max_error'] < 1e-7 and r['gate_max_error'] < 1e-7 and r['unexpected_voltage_drift_mv'] < 1e-8 for r in rows),
                  rows=rows, canonical_singular_voltage_mutation=singular, source_hashes=hashes,
                  contract='Ih shift0; Im fixed canonical qt; zero conductance; original equations preserved.')
    write(Path(output) / 'native_audit.json', report)
    if not report['valid']:
        raise RuntimeError('Task19 double oracle failed')
    return report


def equivalent(torch, model, width, descriptors, probe):
    from .joint_gate_symmetric_confirmation import _clip_per_seed
    clone = model_factory(torch, width, descriptors).to(probe.device)
    clone.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(clone.parameters(), lr=.003)
    prediction_error = adam_error = 0.
    # Check both channels/objectives, so loss masks and independent clipping are exercised.
    singles = []
    for index, descriptor in enumerate(descriptors):
        single = model_factory(torch, width, [descriptor]).to(probe.device)
        single.load_state_dict({k: v[index:index + 1].clone() for k, v in model.state_dict().items()})
        singles.append((single, torch.optim.Adam(single.parameters(), lr=.003)))
    prediction_error = max(float((clone(probe)[0][n] - s(probe[n:n+1])[0][0]).detach().abs().max()) for n, (s, _) in enumerate(singles))
    # Per-model loss scales differ deliberately; sum keeps each gradient independent.
    weights = torch.arange(1, len(descriptors) + 1, device=probe.device)
    clone(probe)[0].square().mean(1).mul(weights).sum().backward()
    _clip_per_seed(clone, torch, 1.)
    optimizer.step()
    for n, (single, opt) in enumerate(singles):
        single(probe[n:n+1])[0].square().mean().mul(weights[n]).backward()
        _clip_per_seed(single, torch, 1.)
        opt.step()
        adam_error = max(adam_error, max(float((v[n:n+1] - single.state_dict()[k]).detach().abs().max()) for k, v in clone.named_parameters()))
    return dict(width=width, prediction_error=prediction_error, adam_error=adam_error, valid=max(prediction_error, adam_error) < 1e-5)


def evaluate(model, values, descriptors, torch, device):
    inp = torch.tensor(values, dtype=torch.float32, device=device)[None].expand(len(descriptors), -1, -1)
    with torch.no_grad():
        result = [v.detach().cpu().double().numpy() for v in model(inp)]
    return [measurements(values, *(r[index] for r in result), channel) for index, (channel, _, _) in enumerate(descriptors)]


def rollout(model, descriptors, values, horizons, torch, device):
    inp = torch.tensor(values, dtype=torch.float32, device=device)[None].expand(len(descriptors), -1, -1)
    results = [{} for _ in descriptors]
    with torch.no_grad():
        _, inf, tau = model(inp)
        z = -torch.expm1(-inp[..., 2] / tau)
        state = inp[..., 1].clone()
        for step in range(1, max(horizons) + 1):
            state = (1 - z) * state + z * inf
            if step in horizons:
                pred = state.cpu().double().numpy()
                for index, (channel, _, _) in enumerate(descriptors):
                    ti, tt = rates(channel, values[:, 0])
                    target = update(values[:, 1], ti, tt, values[:, 2] * step)
                    error = pred[index] - target
                    results[index][str(step)] = dict(gate_rmse=float(np.sqrt(np.mean(error ** 2))), gate_max_error=float(abs(error).max()),
                        occupancy_violations=int(np.sum((pred[index] < 0) | (pred[index] > 1))), finite=bool(np.isfinite(pred[index]).all()))
    return results


def gpu_benchmark(torch, selected_models, values, output):
    """Measure eager forward kernels with equal inputs and outputs on CUDA."""
    if not torch.cuda.is_available():
        return dict(measured=False, reason='CPU smoke only')
    records = []
    x = torch.tensor(values, dtype=torch.float32, device='cuda')
    def analytic(channel):
        v = x[:, 0]
        if channel == 'Ih':
            v = torch.where(v == -154.9, v + .0001, v)
            a, b = .00643 * (v + 154.9) / torch.expm1((v + 154.9) / 11.9), .193 * torch.exp(v / 33.1)
            inf, tau = a / (a + b), 1 / (a + b)
        else:
            a, b = .0033 * torch.exp(.1 * (v + 35)), .0033 * torch.exp(-.1 * (v + 35))
            inf, tau = a / (a + b), 1 / (a + b) / 2.3 ** 1.3
        z = -torch.expm1(-x[:, 2] / tau)
        return (1-z)*x[:, 1] + z*inf, inf, tau
    for channel, model in selected_models.items():
        grid = np.linspace(-135, 75, 2049)
        i, t = rates(channel, grid)
        table = torch.tensor(np.c_[i, t], dtype=torch.float32, device='cuda')
        def lut():
            position = (x[:, 0] + 135) / 210 * 2048
            lower = position.floor().long().clamp(0, 2047)
            frac = (position - lower)[:, None]
            row = table[lower] * (1-frac) + table[lower+1] * frac
            inf, tau = row[:, 0], row[:, 1]
            z = -torch.expm1(-x[:, 2] / tau)
            return (1-z)*x[:, 1] + z*inf, inf, tau
        for family, call in [('mlp', lambda: model(x[None])), ('formula_torch', lambda: analytic(channel)), ('lut2049_torch', lut)]:
            with torch.no_grad():
                for _ in range(10):
                    call()
                torch.cuda.synchronize()
                samples = []
                for _ in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):
                        call()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / 100)
                p = call()[0].detach().cpu().double().numpy().reshape(-1)
            ti, tt = rates(channel, values[:, 0])
            target = update(values[:, 1], ti, tt, values[:, 2])
            records.append(dict(channel=channel, family=family, median_forward_ms=float(np.median(samples)),
                batch_size=len(values), gate_rmse=float(np.sqrt(np.mean((p-target)**2)))))
    report = dict(measured=True, records=records, scope='Eager CUDA resident batch4096; same gate/inf/tau outputs; no full-neuron or NEURON speedup claim.')
    write(Path(output) / 'gpu_benchmark.json', report)
    return report


def run(output, config, revision, backend=None):
    # Explicit backend injection reuses only orchestration, not channel equations.
    rates_fn = rates if backend is None else backend.rates
    model_fn = model_factory if backend is None else backend.model_factory
    evaluate_fn = evaluate if backend is None else backend.evaluate
    rollout_fn = rollout if backend is None else backend.rollout
    benchmark_fn = gpu_benchmark if backend is None else backend.gpu_benchmark
    channels = CHANNELS if backend is None else backend.CHANNELS
    from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest
    from .joint_gate_symmetric_confirmation import _clip_per_seed
    torch = configure_torch_runtime(17)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output = Path(output)
    if config['require_cuda']:
        assert device == 'cuda' and __import__('json').loads((output / 'native_audit.json').read_text())['valid']
    fit = data(190101, config['pool_size']) if backend is None else backend.fit(config)
    count = len(fit) // 4
    rng = np.random.default_rng(190102)
    if backend is None:
        fit[:count, 0] = rng.uniform(-120, -95, count)
        fit[count:2*count, 0] = rng.uniform(-45, -25, count)
    development = dict(uniform=data(190201, 2048), negative_tail=data(190202, 2048, (-135, -115)), state_extrema=state_grid()) if backend is None else backend.development()
    np.savez_compressed(output / 'fit_development.npz', fit=fit, **development)
    descriptors = [(c, o, s) for c in channels for o in config['objectives'] for s in config['seeds']]
    labels = [rates_fn(c, fit[:, 0]) for c, _, _ in descriptors]
    ti, tt = np.stack([r[0] for r in labels]), np.stack([r[1] for r in labels])
    ty = update(fit[:, 1], ti, tt, fit[:, 2])
    x, ti, tt, ty = [torch.tensor(a, dtype=torch.float32, device=device) for a in (fit, ti, tt, ty)]
    active = torch.tensor([o == 'rate_supervised' for _, o, _ in descriptors], dtype=torch.float32, device=device)
    models = {w: model_fn(torch, w, descriptors).to(device) for w in config['widths']}
    optimizers = {w: torch.optim.Adam(m.parameters(), lr=config['learning_rate']) for w, m in models.items()}
    probe = x[:32][None].expand(len(descriptors), -1, -1)
    preflight = [(equivalent(torch, m, w, descriptors, probe) if backend is None else backend.equivalent(torch, m, w, descriptors, probe)) for w, m in models.items()]
    write(output / 'equivalence_preflight.json', dict(valid=all(r['valid'] for r in preflight), rows=preflight, independent_adam_and_clipping=True))
    assert all(r['valid'] for r in preflight), 'Task19 vectorization equivalence failed'
    ladder, best = [], {}
    start = time.monotonic()
    rng = np.random.default_rng(190301)
    for step in range(config['checkpoints'][-1] + 1):
        if step in config['checkpoints']:
            for width, model in models.items():
                metrics = {k: evaluate_fn(model, a, descriptors, torch, device) for k, a in development.items()}
                torch.save(model.state_dict(), output / f'checkpoint_w{width}_step{step}.pt')
                for index, (channel, objective, seed) in enumerate(descriptors):
                    mm = {k: v[index] for k, v in metrics.items()}
                    ladder.append(dict(channel=channel, objective=objective, seed=seed, width=width, step=step, metrics=mm, score=score(mm, config['gates'])))
                for channel in channels:
                    for objective in config['objectives']:
                        indices = [n for n, (c, o, _) in enumerate(descriptors) if (c, o) == (channel, objective)]
                        candidate = dict(channel=channel, objective=objective, width=width, step=step, indices=indices,
                            score=max(score({k: v[n] for k, v in metrics.items()}, config['gates']) for n in indices))
                        key = channel + '/' + objective
                        if key not in best or candidate['score'] < best[key]['score']:
                            best[key] = candidate
            write(output / 'development_ladder.json', ladder)
            label = 'Task19' if backend is None else 'Task20'
            print(f'[GIADA {label}] checkpoint{step}/{config["checkpoints"][-1]} elapsed={(time.monotonic()-start)/60:.1f}min', flush=True)
        if step == config['checkpoints'][-1]:
            break
        indices = torch.tensor(rng.integers(0, len(fit), config['batch_size']), device=device)
        batch = x[indices][None].expand(len(descriptors), -1, -1)
        for width, model in models.items():
            optimizers[width].zero_grad(set_to_none=True)
            p, i, t = model(batch)
            loss = (p-ty[:, indices]).square().mean(1) + active*(config['inf_weight']*(i-ti[:, indices]).square().mean(1) + config['log_tau_weight']*(t.log()-tt[:, indices].log()).square().mean(1))
            loss.sum().backward()
            _clip_per_seed(model, torch, 1.)
            optimizers[width].step()
    by = {(r['channel'], r['objective'], r['seed'], r['width'], r['step']): r for r in ladder}
    contrasts = []
    for row in ladder:
        c, o, s, w, step = row['channel'], row['objective'], row['seed'], row['width'], row['step']
        alternatives = {}
        if o == 'transition_only':
            alternatives['rate_supervision'] = (c, 'rate_supervised', s, w, step)
        if w == 16 and 32 in models:
            alternatives['capacity'] = (c, o, s, 32, step)
        later = [t for t in config['checkpoints'] if t > step]
        if later:
            alternatives['budget'] = (c, o, s, w, later[0])
        for factor, key in alternatives.items():
            other = by[key]
            contrasts.append(dict(factor=factor, channel=c, seed=s, base_objective=o, treated_objective=other['objective'],
                base_width=w, treated_width=other['width'], base_step=step, treated_step=other['step'], score_difference=other['score']-row['score'],
                metric_differences={d: {k: other['metrics'][d][k]-row['metrics'][d][k] for k in config['gates']} for d in development}))
    write(output / 'paired_development_contrasts.json', contrasts)
    freeze = dict(selection=best, fresh_accessed=False, config=config, code_revision=revision,
                  checkpoint_hashes={p.name: sha(p) for p in output.glob('checkpoint_*.pt')})
    freeze['freeze_sha256'] = hashlib.sha256(__import__('json').dumps(freeze, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    write(output / 'selection_freeze.json', freeze)
    if backend is None:
        fresh = dict(in_support=data(190401, 4096), negative_tail=data(190404, 4096, (-135, -115)), state_extrema=state_grid(True),
                     ood_negative=data(190402, 2048, (-155, -135)), ood_positive=data(190403, 2048, (75, 95)))
        voltage = np.random.default_rng(190405).uniform(-135, 75, 128)
        roll_inputs = np.c_[np.repeat(voltage, 2), np.tile([0., 1.], len(voltage)), np.ones(2*len(voltage))]
    else:
        fresh = backend.fresh()
        roll_inputs = backend.roll_inputs()
    np.savez_compressed(output / 'fresh.npz', **fresh, rollout=roll_inputs)
    results, cache, benchmark_models = [], {}, {}
    for key, selected in best.items():
        width, step = selected['width'], selected['step']
        cp = output / f'checkpoint_w{width}_step{step}.pt'
        assert sha(cp) == freeze['checkpoint_hashes'][cp.name]
        model = models[width]
        model.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
        if (width, step) not in cache:
            cache[width, step] = ({d: evaluate_fn(model, a, descriptors, torch, device) for d, a in fresh.items()},
                                  rollout_fn(model, descriptors, roll_inputs, config['rollout_horizons_steps'], torch, device))
        mm, roll = cache[width, step]
        for index in selected['indices']:
            channel, objective, seed = descriptors[index]
            metrics = {d: rows[index] for d, rows in mm.items()}
            one_pass = all(passes(metrics[d], config['gates']) for d in ('in_support', 'negative_tail', 'state_extrema'))
            roll_pass = all(passes(v, config['rollout_gates']) for v in roll[index].values())
            results.append(dict(channel=channel, objective=objective, seed=seed, width=width, step=step,
                parameter_count=(width*width+5*width+2 if backend is None else backend.parameter_count(width)), dense_macs=(width*width+3*width if backend is None else width*width+2*width), metrics=metrics,
                rollout=roll[index], one_step_passed=one_pass, rollout_passed=roll_pass, passed=one_pass and roll_pass))
            if objective == 'rate_supervised' and seed == config['seeds'][0]:
                single = model_fn(torch, width, [(channel, objective, seed)]).to(device)
                single.load_state_dict({k: v[index:index+1].clone() for k, v in model.state_dict().items()})
                benchmark_models[channel] = single
    numerical = []
    for channel in channels:
        for size in (513, 2049):
            grid = np.linspace(-135, 75, size) if backend is None else np.linspace(-7, -2, size)
            ii, tt = rates_fn(channel, grid)
            for domain in ('in_support', 'negative_tail', 'state_extrema'):
                values = fresh[domain]
                inf, tau = np.interp(values[:, 0], grid, ii), np.interp(values[:, 0], grid, tt)
                m = (measurements if backend is None else backend.measurements)(values, update(values[:, 1], inf, tau, values[:, 2]), inf, tau, channel)
                numerical.append(dict(channel=channel, family=f'linear_lut_{size}_f64', domain=domain, metrics=m, passed=passes(m, config['gates'])))
    write(output / 'fresh_metrics.json', dict(learned=results, numerical=numerical))
    benchmark = benchmark_fn(torch, benchmark_models, fresh['in_support'], output)
    if not benchmark['measured']:
        write(output / 'gpu_benchmark.json', benchmark)
    supported = {c: all(r['passed'] for r in results if r['channel'] == c and r['objective'] == 'rate_supervised') for c in channels}
    report = dict(schema_version='giada-roadmap-task19-v1', valid=True, single_gate_transfer_passed=all(supported.values()),
        per_channel_transfer=supported, task20_authorized=all(supported.values()), fresh_used_for_selection=False,
        selection=best, learned=results, numerical=numerical, environment=environment_manifest(torch),
        training_and_evaluation_seconds=time.monotonic()-start, scope=config['limits'], gpu_benchmark=benchmark,
        shared_weights=False, full_neuron_speedup_claimed=False)
    if backend is not None:
        report = backend.finalize(report, benchmark_models, torch, device, output, config)
    write(output / 'final_report.json', report)
    return report
