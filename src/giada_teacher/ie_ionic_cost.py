"""IE-P1: paired resident-core cost with a gate-buffer diagnostic intervention."""
import json
import math
import time
from pathlib import Path
import numpy as np
from . import cc_n1_contract_memory as n1
from . import ionic_block_teacher_forced as ionic
from .gate_d_compute_audit import exact_gpu, hybrid_gpu


def core_with_gates(torch, v, ca, state, next_g, mult, injection, syn_g, area, cfg, calcium, base):
    """Task35 membrane map after gate update, preserving the calcium-before-SK phase."""
    from .gate_d_compute_audit import GBAR_VALUES, OFFSETS, POWERS, REVERSAL_VALUES
    gbar = torch.tensor(GBAR_VALUES, dtype=v.dtype, device=v.device) * mult
    ica = sum(gbar[..., k] * next_g[..., OFFSETS[k]].square() *
              next_g[..., OFFSETS[k]+1] * (v-120.) for k in (0, 1))
    persistence = math.exp(-cfg['dt_ms'] / calcium['decay_ms'])
    source = -10000 * ica * calcium['gamma'] / (2 * 96485.33212 * calcium['depth_um'])
    next_ca = 1e-4 + (ca-1e-4)*persistence + source*calcium['decay_ms']*(1-persistence)
    sk = OFFSETS[9]
    sk_inf = 1/(1+(.00043/next_ca.clamp_min(1e-7)).pow(4.8))
    sk_next = sk_inf + (state[..., sk]-sk_inf)*math.exp(-cfg['dt_ms'])
    next_g = torch.cat((next_g[..., :sk], sk_next[..., None], next_g[..., sk+1:]), -1)
    openings = []
    for k, names in enumerate(ionic.STATE_NAMES):
        opening = next_g[..., OFFSETS[k]].pow(POWERS[k])
        if len(names) == 2:
            opening = opening * next_g[..., OFFSETS[k]+1]
        openings.append(opening)
    ion_g = torch.stack(openings, -1) * gbar
    reversal = torch.tensor(REVERSAL_VALUES, dtype=v.dtype, device=v.device)
    syn_rev = torch.tensor((0., 0., -80., -97.), dtype=v.dtype, device=v.device)
    density = syn_g * (100./area)
    fixed = torch.cat((density[..., :1], torch.zeros_like(density[..., 1:2]), density[..., 2:]), -1)
    cap = 1e-3 * base['cm_uf_cm2'] / cfg['dt_ms']
    denom = cap + base['g_pas_s_cm2'] + ion_g.sum(-1) + fixed.sum(-1)
    numer = cap*v + base['g_pas_s_cm2']*base['e_pas_mv'] + (ion_g*reversal).sum(-1) + (fixed*syn_rev).sum(-1) + injection
    next_v = numer / denom
    for _ in range(12):
        exponent = torch.exp((-.08*next_v).clamp(-80, 80))/3.57
        nmda = density[..., 1]/(1+exponent)
        derivative = .08*density[..., 1]*exponent/(1+exponent).square()
        next_v = next_v - (denom*next_v-numer+nmda*next_v)/(denom+nmda+derivative*next_v)
    return next_v, next_ca, next_g


def paired_summary(full, cached):
    full, cached = np.asarray(full), np.asarray(cached)
    delta = 1-cached/full
    rng = np.random.default_rng(81502)
    medians = np.median(delta[rng.integers(0, len(delta), (2000, len(delta)))], axis=1)
    return {'median_saved_fraction': float(np.median(delta)),
            'bootstrap_95pct': np.quantile(medians, [.025, .975]).tolist(),
            'interpretation': 'Paired timing-block uncertainty on this device/run; not cross-device statistical replication.'}


def run(root, output):
    import torch
    from .gate_d_bottleneck_forensic import _profile
    root, output = Path(root), Path(output)
    spec = json.loads((root/'experiments/ie_p1_ionic_cost_preregistration_v1.json').read_text())
    n1spec, cfg, calcium, _, base = n1.contract(root)
    amendment = json.loads((root/'experiments/ie_p1_compile_amendment_v2.json').read_text())
    compile_limits = (n1spec['gpu']['compiled_vs_eager_voltage_atol_mv'],
                      n1spec['gpu']['compiled_vs_eager_cai_atol_mM'],
                      n1spec['gpu']['compiled_vs_eager_gate_atol'])
    assert compile_limits == (amendment['compiled_voltage_atol_mv'],
                              amendment['compiled_calcium_atol_mM'], amendment['compiled_gate_atol'])
    if not torch.cuda.is_available():
        raise RuntimeError('IE-P1 requires CUDA')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    models, seeds = ionic.load_frozen(root, ionic.config(root), torch, 'cuda')
    assert len(seeds) == spec['seed_count']
    data = np.load(root/'experiments/results/task35_native_shadow_traces.npz')
    rng = np.random.default_rng(spec['input_seed'])
    report = {'valid': False, 'schema_version': spec['schema_version'], 'training_performed': False,
              'candidate_promoted': False, 'whole_simulator_profile_complete': False,
              'scope': spec['scope'], 'exclusions': spec['exclusions'],
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'rows': [],
              'compile_amendment': amendment, 'numerical_errors': []}
    with torch.inference_mode():
        for batch in spec['batches']:
            episodes = rng.integers(0, 32, batch)
            ticks = rng.integers(0, 400, batch)
            def tensor(values):
                t = torch.tensor(values, dtype=torch.float32, device='cuda')
                return t[None].expand(3, *t.shape).contiguous()
            v, ca, gates = [tensor(data[k][episodes, ticks]) for k in ('voltage', 'calcium', 'gates')]
            mult = tensor(data['multipliers'][episodes])
            current, syn = [tensor(data[k][episodes, ticks]) for k in ('injection', 'base_g_us')]
            area = float(data['area_um2'])
            dt = torch.full_like(v, .1)
            funcs, compiled, errors = {}, {}, {}
            for arm in spec['arms']:
                model = None if arm == 'formula' else models[arm, 'both']
                def gate_fn(v, ca, gates, model=model):
                    return (exact_gpu(v, ca, gates, dt, torch) if model is None else
                            hybrid_gpu(model, v, ca, gates, dt, torch))[0]
                cached_g = gate_fn(v, ca, gates).clone()
                def full_fn(v, ca, gates, model=model):
                    return n1.torch_step(torch, v, ca, gates, mult, current, syn, area, cfg, calcium, base, model)
                def cached_fn(v, ca, gates, cached_g=cached_g):
                    return core_with_gates(torch, v, ca, gates, cached_g, mult, current, syn, area, cfg, calcium, base)
                funcs[arm, 'ionic_gates'] = gate_fn
                funcs[arm, 'membrane_core'] = full_fn
                funcs[arm, 'membrane_core_cached_gates'] = cached_fn
                expected = full_fn(v, ca, gates)
                actual = cached_fn(v, ca, gates)
                diff = [float((a-b).abs().max()) for a,b in zip(expected, actual)]
                assert all(e <= spec['preflight'][k] for e,k in zip(diff,
                    ('voltage_atol_mv','calcium_atol_mM','gate_atol'))), (arm, diff)
                errors[arm] = diff
            for key, fn in funcs.items():
                print(f'[GIADA IE-P1] compile {key[0]}/{key[1]} batch={batch}',flush=True)
                cfn = torch.compile(fn, backend='inductor', fullgraph=True)
                expected, actual = fn(v,ca,gates), cfn(v,ca,gates)
                aa = actual if isinstance(actual, tuple) else (actual,)
                ee = expected if isinstance(expected, tuple) else (expected,)
                limits = compile_limits if len(aa)==3 else (compile_limits[2],)
                differences = [float((a-b).abs().max()) for a,b in zip(aa,ee)]
                report['numerical_errors'].append({'batch':batch,'arm':key[0], 'component':key[1],
                                                   'maximum_errors':differences,'limits':limits})
                (output/'numeric_preflight.json').write_text(json.dumps(report['numerical_errors'],indent=2),encoding='utf-8')
                assert all(e <= lim for e,lim in zip(differences,limits)), (key,differences,limits)
                compiled[key] = cfn
                for _ in range(spec['warmup_calls']):
                    cfn(v,ca,gates)
            samples = {key: {'wall_ms': [], 'cuda_ms': []} for key in compiled}
            keys = list(compiled)
            for repetition in range(spec['repetitions']):
                for key in keys if repetition % 2 == 0 else keys[::-1]:
                    torch.cuda.synchronize()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall = time.perf_counter(); begin.record()
                    for _ in range(spec['calls_per_block']):
                        retained = compiled[key](v,ca,gates)
                    end.record(); end.synchronize()
                    samples[key]['wall_ms'].append((time.perf_counter()-wall)*1000/spec['calls_per_block'])
                    samples[key]['cuda_ms'].append(begin.elapsed_time(end)/spec['calls_per_block'])
            for arm in spec['arms']:
                row = {'arm': arm, 'batch_per_seed': batch, 'seed_count': 3, 'numeric_preflight_max_errors': errors[arm],
                       'timings': {name: samples[arm,name] for name in spec['components']},
                       'gate_buffer_diagnostic': paired_summary(samples[arm,'membrane_core']['wall_ms'],
                                                                samples[arm,'membrane_core_cached_gates']['wall_ms']),
                       'profiles': {name: _profile(lambda key=(arm,name): compiled[key](v,ca,gates),torch,2)
                                    for name in spec['components']}}
                report['rows'].append(row)
                (output/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(f'[GIADA IE-P1] {arm} batch={batch} saved-gate diagnostic={row["gate_buffer_diagnostic"]["median_saved_fraction"]:.3f}',flush=True)
    report['valid'] = True
    report['next'] = spec['next']
    (output/'final_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report
