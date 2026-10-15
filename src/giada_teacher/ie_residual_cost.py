"""IE-P2 residual-cost interventions; formula math and state phase unchanged."""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from . import cc_n1_contract_memory as n1
from . import ionic_block_teacher_forced as ionic
from .gate_d_compute_audit import exact_gpu, GBAR_VALUES, OFFSETS, POWERS, REVERSAL_VALUES
from .ie_ionic_cost import paired_summary

CONFIG = 'experiments/ie_p2_residual_cost_preregistration_v1.json'


def ionic_prefix(torch, v, ca, state, mult, cfg, calcium, gate_cache=None, conductance_cache=None):
    g = exact_gpu(v, ca, state, torch.full_like(v, cfg['dt_ms']), torch)[0] if gate_cache is None else gate_cache
    gbar = torch.tensor(GBAR_VALUES, dtype=v.dtype, device=v.device) * mult
    ica = sum(gbar[..., k] * g[..., OFFSETS[k]].square() * g[..., OFFSETS[k]+1] * (v-120.) for k in (0,1))
    persistence = math.exp(-cfg['dt_ms']/calcium['decay_ms'])
    source = -10000 * ica * calcium['gamma']/(2*96485.33212*calcium['depth_um'])
    next_ca = 1e-4 + (ca-1e-4)*persistence + source*calcium['decay_ms']*(1-persistence)
    sk = OFFSETS[9]
    inf = 1/(1+(.00043/next_ca.clamp_min(1e-7)).pow(4.8))
    next_sk = inf+(state[...,sk]-inf)*math.exp(-cfg['dt_ms'])
    g = torch.cat((g[...,:sk],next_sk[...,None],g[...,sk+1:]),-1)
    if conductance_cache is None:
        openings=[]
        for k,names in enumerate(ionic.STATE_NAMES):
            opening=g[...,OFFSETS[k]].pow(POWERS[k])
            if len(names)==2:
                opening=opening*g[...,OFFSETS[k]+1]
            openings.append(opening)
        ion_g=torch.stack(openings,-1)*gbar
        reverse=torch.tensor(REVERSAL_VALUES,dtype=v.dtype,device=v.device)
        gsum, gesum=ion_g.sum(-1),(ion_g*reverse).sum(-1)
    else:
        gsum,gesum=conductance_cache
    return next_ca,g,gsum,gesum


def membrane_solve(torch,v,gsum,gesum,injection,syn_g,area,cfg,base):
    reverse=torch.tensor((0.,0.,-80.,-97.),dtype=v.dtype,device=v.device)
    density=syn_g*(100./area)
    fixed=torch.cat((density[...,:1],torch.zeros_like(density[...,1:2]),density[...,2:]),-1)
    cap=1e-3*base['cm_uf_cm2']/cfg['dt_ms']
    denom=cap+base['g_pas_s_cm2']+gsum+fixed.sum(-1)
    numer=cap*v+base['g_pas_s_cm2']*base['e_pas_mv']+gesum+(fixed*reverse).sum(-1)+injection
    nv=numer/denom
    for _ in range(12):
        exponent=torch.exp((-.08*nv).clamp(-80,80))/3.57
        nmda=density[...,1]/(1+exponent)
        derivative=.08*density[...,1]*exponent/(1+exponent).square()
        nv=nv-(denom*nv-numer+nmda*nv)/(denom+nmda+derivative*nv)
    return nv


def functions(torch,v,ca,g,mult,injection,syn,area,cfg,calcium,base):
    """Bound fixed-input functions with an equal five-output diagnostic ABI."""
    raw_g=exact_gpu(v,ca,g,torch.full_like(v,cfg['dt_ms']),torch)[0].clone()
    cached=tuple(x.clone() for x in ionic_prefix(torch,v,ca,g,mult,cfg,calcium))
    voltage=membrane_solve(torch,v,cached[2],cached[3],injection,syn,area,cfg,base).clone()
    def build(mode):
        def step():
            if mode in ('cached_prefix','cached_both'):
                p=cached
            else:
                p=ionic_prefix(torch,v,ca,g,mult,cfg,calcium,
                    gate_cache=raw_g if mode=='cached_gates' else None,
                    conductance_cache=cached[2:] if mode=='cached_conductance' else None)
            nv=voltage if mode in ('cached_voltage','cached_both') else membrane_solve(
                torch,v,p[2],p[3],injection,syn,area,cfg,base)
            return nv,*p
        return step
    result={mode:build(mode) for mode in ('full','cached_gates','cached_conductance','cached_prefix','cached_voltage','cached_both')}
    result['ionic_prefix_only']=lambda:ionic_prefix(torch,v,ca,g,mult,cfg,calcium)
    result['membrane_solve_only']=lambda:membrane_solve(torch,v,cached[2],cached[3],injection,syn,area,cfg,base)
    result['legacy_abi']=lambda:n1.torch_step(torch,v,ca,g,mult,injection,syn,area,cfg,calcium,base)
    return result


def as_tuple(value):
    return value if isinstance(value,tuple) else (value,)


def differences(torch,actual,expected,limits):
    aa,ee=as_tuple(actual),as_tuple(expected)
    assert len(aa)==len(ee)==len(limits)
    errors=[float((a-b).abs().max()) for a,b in zip(aa,ee)]
    return {'maximum_errors':errors,'limits':list(limits),
            'valid':all(math.isfinite(e) and e<=limit for e,limit in zip(errors,limits))}


def benchmark(torch,funcs,spec):
    samples={key:{'wall_ms':[],'cuda_ms':[]} for key in funcs}
    keys=list(funcs)
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for repetition in range(spec['repetitions']):
        for key in keys if repetition%2==0 else keys[::-1]:
            torch.cuda.synchronize()
            wall=time.perf_counter();begin.record()
            for _ in range(spec['calls_per_block']):
                retained=funcs[key]()
            end.record();end.synchronize()
            samples[key]['wall_ms'].append((time.perf_counter()-wall)*1000/spec['calls_per_block'])
            samples[key]['cuda_ms'].append(begin.elapsed_time(end)/spec['calls_per_block'])
    return samples


def targeting(report,primary_batch):
    groups=[]
    for batch,mode in sorted({(r['batch_per_seed'],r['mode']) for r in report['rows']}):
        rows={r['variant']:r for r in report['rows'] if r['batch_per_seed']==batch and r['mode']==mode}
        if 'full' not in rows:continue
        perturbation=None
        if 'legacy_abi' in rows:
            perturbation=-paired_summary(rows['legacy_abi']['timings']['wall_ms'],rows['full']['timings']['wall_ms'])['median_saved_fraction']
        flags={}
        for label,name,threshold in [('H_ionic','cached_prefix',.30),('H_solver','cached_voltage',.30),
                                     ('H_readout','cached_conductance',.20)]:
            saving=rows.get(name,{}).get('saving_vs_full')
            flags[label]=None if saving is None else (saving['median_saved_fraction']>=threshold and saving['bootstrap_95pct'][0]>0)
        groups.append({'batch_per_seed':batch,'mode':mode,'is_primary':batch==primary_batch,
                       'instrumented_abi_cost_fraction':perturbation,
                       'abi_perturbed':None if perturbation is None else abs(perturbation)>=.10,
                       'hypothesis_flags':flags})
    dispatch=[{**row,'H_dispatch':row['median_saved_fraction']>=.50,
               'is_primary':row['batch']==primary_batch} for row in report.get('execution_contrasts',[])]
    return {'block_groups':groups,'execution':dispatch,
            'policy':'Diagnostic priorities only; qualify block conclusions where ABI is perturbed. No automatic candidate selection or training.'}


def run(root,output,revision):
    import torch
    root,output=Path(root),Path(output)
    spec=json.loads((root/CONFIG).read_text())
    if hashlib.sha256((root/spec['parent_report']).read_bytes()).hexdigest()!=spec['parent_sha256']:
        raise RuntimeError('IE-P1 parent changed')
    _,cfg,calcium,_,base=n1.contract(root)
    assert torch.cuda.is_available(),'CUDA required'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(128,torch._dynamo.config.cache_size_limit)
    data=np.load(root/'experiments/results/task35_native_shadow_traces.npz')
    rng=np.random.default_rng(spec['input_seed'])
    numeric=spec['numeric']
    limits=[numeric[k] for k in ('voltage_atol_mv','calcium_atol_mM','gate_atol','sum_g_atol_s_cm2','sum_gE_atol_ma_cm2')]
    report={'schema_version':spec['schema_version'],'valid':False,'code_revision':revision,
            'device':torch.cuda.get_device_name(),'torch':torch.__version__,'training_performed':False,
            'candidate_promoted':False,'whole_simulator_profile_complete':False,
            'scope':spec['scope'],'exclusions':spec['exclusions'],'rows':[],'preflights':[],'graph_failures':[]}
    def persist():
        (output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with torch.inference_mode():
        for batch in spec['batches']:
            ids=rng.integers(0,32,batch);ticks=rng.integers(0,400,batch)
            def tensor(a):
                x=torch.tensor(a,dtype=torch.float32,device='cuda')
                return x[None].expand(spec['seed_count'],*x.shape).contiguous()
            v,ca,g=[tensor(data[k][ids,ticks]) for k in ('voltage','calcium','gates')]
            mult=tensor(data['multipliers'][ids]);injection=tensor(data['injection'][ids,ticks]);syn=tensor(data['base_g_us'][ids,ticks])
            funcs=functions(torch,v,ca,g,mult,injection,syn,float(data['area_um2']),cfg,calcium,base)
            golden=funcs['full']();legacy=funcs['legacy_abi']()
            equivalence=differences(torch,golden[:3],legacy,limits[:3])
            report['preflights'].append({'batch':batch,'kind':'recomposition','variant':'full',**equivalence});persist()
            if not equivalence['valid']:raise RuntimeError('Recomposition mismatch: '+str(equivalence))
            compiled,graphs={},{}
            for name in spec['variants']:
                expected=funcs[name]()
                bounds=limits[1:] if name=='ionic_prefix_only' else limits[:1] if name=='membrane_solve_only' else limits[:3] if name=='legacy_abi' else limits
                if name.startswith('cached_'):
                    eq=differences(torch,expected,golden,limits)
                    report['preflights'].append({'batch':batch,'kind':'cache_equivalence','variant':name,**eq});persist()
                    if not eq['valid']:raise RuntimeError('Cache changed outputs: '+str(eq))
                print(f'[GIADA IE-P2] compile {name} batch={batch}',flush=True)
                fn=torch.compile(funcs[name],backend='inductor',fullgraph=True)
                eq=differences(torch,fn(),expected,bounds)
                report['preflights'].append({'batch':batch,'kind':'compile','variant':name,**eq});persist()
                if not eq['valid']:raise RuntimeError('Compiler mismatch: '+str(eq))
                for _ in range(spec['warmup_calls']):fn()
                compiled[name]=fn
                try:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):fn()
                    stream.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        held=fn()
                    graph.replay();torch.cuda.synchronize()
                    eq=differences(torch,held,expected,bounds)
                    if not eq['valid']:raise RuntimeError('Graph mismatch: '+str(eq))
                    def replay(graph=graph,held=held):
                        graph.replay()
                        return held
                    graphs[name]=replay
                    report['preflights'].append({'batch':batch,'kind':'cuda_graph','variant':name,**eq})
                except Exception as error:
                    report['graph_failures'].append({'batch':batch,'variant':name,'error':str(error)[:2000]})
            combined={(mode,name):call for mode,calls in [('compiled',compiled),('cuda_graph',graphs)] for name,call in calls.items()}
            for call in combined.values():
                for _ in range(3):call()
            timed=benchmark(torch,combined,spec)
            mode_samples={}
            for mode,calls in [('compiled',compiled),('cuda_graph',graphs)]:
                if not calls:continue
                mode_samples[mode]={name:timed[mode,name] for name in calls}
                full=mode_samples[mode].get('full')
                for name,timing in mode_samples[mode].items():
                    row={'batch_per_seed':batch,'seed_count':spec['seed_count'],'mode':mode,'variant':name,'timings':timing}
                    if full and name.startswith('cached_'):
                        row['saving_vs_full']=paired_summary(full['wall_ms'],timing['wall_ms'])
                    report['rows'].append(row)
                persist()
                print(f'[GIADA IE-P2] measured {len(calls)} variants mode={mode} batch={batch}',flush=True)
            if 'cuda_graph' in mode_samples and 'full' in graphs:
                report.setdefault('execution_contrasts',[]).append({'batch':batch,**paired_summary(
                    mode_samples['compiled']['full']['wall_ms'],mode_samples['cuda_graph']['full']['wall_ms'])})
            torch.cuda.synchronize()
    report['valid']=True
    report['targeting']=targeting(report,spec['primary_batch'])
    report['decision_policy']=spec['decision']
    (output/'final_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('[GIADA IE-P2] complete; no candidate trained or promoted',flush=True)
    return report
