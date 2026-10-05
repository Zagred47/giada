"""Compiled same-input bottleneck decomposition after the Gate D NO-GO."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from . import ionic_block_teacher_forced as ionic
from .gate_d_compute_audit import OFFSETS,_rates,_currents,exact_gpu,hybrid_gpu
from .hh_family_transfer import write


def config(root: Path) -> dict:
    return json.loads((root/'experiments/task28d_gate_d_bottleneck_forensic.json').read_text(encoding='utf-8'))


def verify_parent(root: Path, cfg: dict) -> None:
    folder=root/cfg['parent_result_dir']
    actual={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in cfg['parent_sha256']}
    if actual!=cfg['parent_sha256']:raise RuntimeError('Gate D compiled parent changed')
    report=json.loads((folder/'final_report.json').read_text(encoding='utf-8'))
    audit=json.loads((folder/'result_audit.json').read_text(encoding='utf-8'))
    if not(report['valid'] and not report['gate_d_robustly_supported'] and audit['valid']
           and not audit['task29_authorized']):
        raise RuntimeError('Gate D compiled NO-GO prerequisite not confirmed')


def _exact_core(v,ca,state,dt,torch):
    parts=[]
    for k,name in enumerate(ionic.CHANNELS[:5]):
        inf,tau=_rates(name,v,ca,torch)
        a,b=OFFSETS[k:k+2]
        z=-torch.expm1(-dt[...,None]/tau)
        parts.append((1-z)*state[...,a:b]+z*inf)
    return torch.cat(parts,-1)


def _neural_core(model,v,state,dt,torch):
    n,b=v.shape
    packed=torch.cat((v[...,None],state[...,:10],dt[...,None]),-1)
    return model(packed)[0].permute(0,2,1,3).reshape(n,b,10)


def _tail(v,ca,state,dt,torch):
    parts=[]
    for k,name in enumerate(ionic.CHANNELS[5:],start=5):
        inf,tau=_rates(name,v,ca,torch)
        a,b=OFFSETS[k:k+2]
        if b-a==1:inf,tau=inf[...,None],tau[...,None]
        z=-torch.expm1(-dt[...,None]/tau)
        parts.append((1-z)*state[...,a:b]+z*inf)
    return torch.cat(parts,-1)


def _timed(fn,torch):
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record();output=fn();end.record();end.synchronize()
    if isinstance(output,tuple):
        assert all(x.numel()>0 for x in output)
    else:assert output.numel()>0
    return float(start.elapsed_time(end))


def _profile(fn,torch,repetitions):
    try:
        from torch.profiler import profile,ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(repetitions):fn()
            torch.cuda.synchronize()
        kernels=[]
        for event in prof.events():
            if 'cuda' in str(event.device_type).lower():
                us=float(getattr(event,'device_time_total',0) or 0)
                kernels.append((str(event.name),us))
        top=sorted(kernels,key=lambda row:row[1],reverse=True)[:8]
        return {'valid':True,'kernel_event_count':len(kernels),
                'kernel_event_time_ms':sum(t for _,t in kernels)/1000,
                'top_kernel_events':[{'name':name[:160],'device_time_us':us} for name,us in top]}
    except Exception as error:
        return {'valid':False,'error':str(error)[:600]}


def run(root:Path,output:Path,cfg:dict,revision:str)->dict:
    import torch
    verify_parent(root,cfg)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('Registered CUDA device absent')
    task28=ionic.config(root)
    models,seeds=ionic.load_frozen(root,task28,torch,'cuda')
    if list(seeds)!=cfg['model_seeds']:raise RuntimeError('Frozen seeds changed')
    v,ca,state,dt=ionic.support(cfg['input_seed'],cfg['batch_size'],task28,'in_support')
    tensors=[torch.tensor(x,device='cuda',dtype=torch.float32) for x in (v,ca,state,dt)]
    tv,tc,ts,td=[x[None].expand(len(seeds),*x.shape).contiguous() for x in tensors]
    def compiled(fn):
        return torch.compile(fn,backend=cfg['compiler_backend'],mode=cfg['compiler_mode'],
                             fullgraph=cfg['fullgraph'])
    exact_core=compiled(lambda:_exact_core(tv,tc,ts,td,torch))
    tail=compiled(lambda:_tail(tv,tc,ts,td,torch))
    current_only=compiled(lambda:_currents(tv,ts,torch))
    exact_gates=compiled(lambda:torch.cat((_exact_core(tv,tc,ts,td,torch),
                                           _tail(tv,tc,ts,td,torch)),-1))
    exact_full=compiled(lambda:exact_gpu(tv,tc,ts,td,torch))
    common={'exact_core_five':exact_core,'tail_six':tail,
            'analytic_currents_only':current_only,'exact_gates':exact_gates,
            'exact_full':exact_full}
    rows=[]
    with torch.inference_mode():
        print('[GIADA Gate D forensic] compiling common exact components',flush=True)
        common_output={name:fn() for name,fn in common.items()}
        torch.cuda.synchronize()
        exact_core_value=common_output['exact_core_five']
        tail_value=common_output['tail_six']
        exact_gate_value=common_output['exact_gates']
        exact_full_value=common_output['exact_full']
        assert float((torch.cat((exact_core_value,tail_value),-1)-exact_gate_value).abs().max())<=cfg['numeric_tolerance_gate']
        assert float((exact_gate_value-exact_full_value[0]).abs().max())<=cfg['numeric_tolerance_gate']
        for family in cfg['families']:
            model=models[family,cfg['arm']]
            neural_core=compiled(lambda:_neural_core(model,tv,ts,td,torch))
            hybrid_gates=compiled(lambda:torch.cat((_neural_core(model,tv,ts,td,torch),
                                                    _tail(tv,tc,ts,td,torch)),-1))
            hybrid_full=compiled(lambda:hybrid_gpu(model,tv,tc,ts,td,torch))
            funcs={**common,'neural_core_five':neural_core,
                   'hybrid_gates':hybrid_gates,'hybrid_full':hybrid_full}
            print(f'[GIADA Gate D forensic] compiling {family} neural and composite components',flush=True)
            values={name:fn() for name,fn in funcs.items() if name not in common}
            torch.cuda.synchronize()
            gate_diff=float((torch.cat((values['neural_core_five'],tail_value),-1)-values['hybrid_gates']).abs().max())
            full_gate_diff=float((values['hybrid_gates']-values['hybrid_full'][0]).abs().max())
            full_current_diff=float((_currents(tv,values['hybrid_gates'],torch)-values['hybrid_full'][1]).abs().max())
            if gate_diff>cfg['numeric_tolerance_gate'] or full_gate_diff>cfg['numeric_tolerance_gate'] or full_current_diff>cfg['numeric_tolerance_current_ma_cm2']:
                raise RuntimeError(f'{family} decomposition changed output: {gate_diff},{full_gate_diff},{full_current_diff}')
            for _ in range(cfg['warmup_repetitions']):
                for fn in funcs.values():fn()
            torch.cuda.synchronize()
            traces={name:[] for name in funcs}
            names=tuple(funcs)
            for step in range(cfg['benchmark_repetitions']):
                order=names if step%2==0 else names[::-1]
                for name in order:traces[name].append(_timed(funcs[name],torch))
            timing={name:{'median_ms':statistics.median(samples),'min_ms':min(samples),
                          'max_ms':max(samples),'samples_ms':samples} for name,samples in traces.items()}
            comparisons={label:statistics.median((a-b)/a for a,b in zip(traces[left],traces[right]))
                         for label,left,right in (
                             ('neural_core_vs_exact_core_fraction','exact_core_five','neural_core_five'),
                             ('hybrid_gates_vs_exact_gates_fraction','exact_gates','hybrid_gates'),
                             ('hybrid_full_vs_exact_full_fraction','exact_full','hybrid_full'))}
            profiles={name:_profile(funcs[name],torch,cfg['profiler_repetitions'])
                      for name in ('exact_core_five','neural_core_five','exact_full','hybrid_full')}
            row={'family':family,'batch_size':cfg['batch_size'],'timing':timing,
                 'paired_reduction_fractions':comparisons,'profiles':profiles,
                 'decomposition_max_gate_difference':max(gate_diff,full_gate_diff),
                 'decomposition_max_current_difference_ma_cm2':full_current_diff}
            rows.append(row)
            print('[GIADA Gate D forensic] '+family+' '+json.dumps({
                name:round(timing[name]['median_ms'],4) for name in (
                    'exact_core_five','neural_core_five','tail_six','analytic_currents_only',
                    'exact_full','hybrid_full')}),flush=True)
    report={'schema_version':'giada-gate-d-bottleneck-forensic-v1','valid':True,
            'diagnostic_only':True,'gate_d_completed':False,'task29_authorized':False,
            'training_performed':False,'fresh_used_for_selection':False,
            'device_name':torch.cuda.get_device_name(),'torch_version':torch.__version__,
            'cuda_version':torch.version.cuda,'code_revision':revision,
            'rows':rows,'interpretation_contract':cfg['interpretation']}
    write(output/'final_report.json',report)
    return report
