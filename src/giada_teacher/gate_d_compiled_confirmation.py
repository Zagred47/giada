"""Symmetric Inductor confirmation of the eager Gate D result."""
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from . import ionic_block_teacher_forced as ionic
from .gate_d_compute_audit import exact_gpu,hybrid_gpu,_elapsed_ms
from .hh_family_transfer import write


def config(root):
    return json.loads((root/'experiments/task28c_gate_d_compiled_confirmation.json').read_text(encoding='utf-8'))


def verify_parent(root,cfg):
    directory=root/cfg['parent_result_dir']
    actual={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in cfg['parent_sha256']}
    if actual!=cfg['parent_sha256']:raise RuntimeError('Eager Gate D evidence changed')
    eager=json.loads((directory/'final_report.json').read_text(encoding='utf-8'))
    audit=json.loads((directory/'result_audit.json').read_text(encoding='utf-8'))
    if not eager['valid'] or not audit['valid'] or not audit['eager_registered_passed']:
        raise RuntimeError('Eager result prerequisite invalid')


def run(root:Path,output:Path,cfg:dict,revision:str):
    import torch
    verify_parent(root,cfg)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('Registered CUDA device unavailable')
    assert cfg['backend']=='torch.compile/inductor' and cfg['device']=='cuda'
    models,seeds=ionic.load_frozen(root,ionic.config(root),torch,'cuda')
    if list(seeds)!=cfg['model_seeds']:raise RuntimeError('Frozen model seeds changed')
    task28=ionic.config(root)
    batch=cfg['batch_size']
    v,ca,state,dt=ionic.support(cfg['input_seed'],batch,task28,'in_support')
    tensors=[torch.tensor(x,device='cuda',dtype=torch.float32) for x in (v,ca,state,dt)]
    tv,tc,ts,td=[x[None].expand(len(seeds),*x.shape).contiguous() for x in tensors]
    exact_eager=lambda:exact_gpu(tv,tc,ts,td,torch)
    exact_compiled=torch.compile(exact_eager,backend='inductor',mode=cfg['mode'],fullgraph=cfg['fullgraph'])
    rows=[]
    with torch.inference_mode():
        exact_reference=exact_eager()
        print('[GIADA Gate D compiled] compiling exact formula block',flush=True)
        exact_value=exact_compiled()
        torch.cuda.synchronize()
        exact_gate_difference=float((exact_value[0]-exact_reference[0]).abs().max().item())
        exact_current_difference=float((exact_value[1]-exact_reference[1]).abs().max().item())
        for family in cfg['families']:
            model=models[family,cfg['arm']]
            hybrid_eager=lambda:hybrid_gpu(model,tv,tc,ts,td,torch)
            hybrid_compiled=torch.compile(hybrid_eager,backend='inductor',mode=cfg['mode'],fullgraph=cfg['fullgraph'])
            hybrid_reference=hybrid_eager()
            print(f'[GIADA Gate D compiled] compiling {family} hybrid block',flush=True)
            hybrid_value=hybrid_compiled()
            torch.cuda.synchronize()
            gate_difference=float((hybrid_value[0]-hybrid_reference[0]).abs().max().item())
            current_difference=float((hybrid_value[1]-hybrid_reference[1]).abs().max().item())
            numerical_valid=all(math.isfinite(x) for x in (exact_gate_difference,exact_current_difference,gate_difference,current_difference))
            numerical_valid=numerical_valid and max(exact_gate_difference,gate_difference)<=cfg['maximum_compiled_vs_eager_gate_difference']
            numerical_valid=numerical_valid and max(exact_current_difference,current_difference)<=cfg['maximum_compiled_vs_eager_current_difference_ma_cm2']
            if not numerical_valid:
                raise RuntimeError(f'Compiled output mismatch {family}: '+str((exact_gate_difference,exact_current_difference,gate_difference,current_difference)))
            np_exact=ionic.exact_step(v,ca,state,dt)
            accuracy=[]
            candidate=hybrid_value[0].cpu().numpy()
            for index,seed in enumerate(seeds):
                measured=ionic.measure(v,candidate[index],np_exact,task28)
                worst_gate=max(x['rmse'] for x in measured['per_channel_gates'].values())
                worst_individual=max(x['worst_individual_normalized_rmse'] for x in measured['current_panels'].values())
                worst_total=max(x['total_normalized_rmse'] for x in measured['current_panels'].values())
                passed=(measured['finite'] and worst_gate<=task28['learned_gate_rmse_limit']
                        and worst_individual<=task28['learned_current_normalized_rmse_limit']
                        and worst_total<=task28['learned_total_current_normalized_rmse_limit']
                        and all(x['occupancy_violations']==0 for x in measured['per_channel_gates'].values()))
                accuracy.append({'seed':seed,'passed':bool(passed),'worst_gate_rmse':worst_gate,
                                 'worst_individual_current_normalized_rmse':worst_individual,
                                 'worst_total_current_normalized_rmse':worst_total})
            for _ in range(cfg['warmup_repetitions']):exact_compiled();hybrid_compiled()
            torch.cuda.synchronize()
            pairs=[]
            for index in range(cfg['benchmark_repetitions']):
                order=('exact','hybrid') if index%2==0 else ('hybrid','exact')
                timings={name:_elapsed_ms(exact_compiled if name=='exact' else hybrid_compiled,torch) for name in order}
                pairs.append(timings)
            gain=statistics.median((p['exact']-p['hybrid'])/p['exact'] for p in pairs)
            row={'family':family,'batch_size':batch,'exact_compiled_median_ms':statistics.median(p['exact'] for p in pairs),
                 'hybrid_compiled_median_ms':statistics.median(p['hybrid'] for p in pairs),
                 'paired_compute_reduction_fraction':gain,'material_reduction_passed':gain>=cfg['minimum_material_reduction'],
                 'accuracy_passed':all(x['passed'] for x in accuracy),'accuracy_rows':accuracy,
                 'exact_gate_difference':exact_gate_difference,'exact_current_difference_ma_cm2':exact_current_difference,
                 'hybrid_gate_difference':gate_difference,'hybrid_current_difference_ma_cm2':current_difference,
                 'paired_timings_ms':pairs}
            rows.append(row)
            print(f'[GIADA Gate D compiled] {family} exact={row["exact_compiled_median_ms"]:.4f}ms hybrid={row["hybrid_compiled_median_ms"]:.4f}ms gain={gain:.2%}',flush=True)
    passed=all(r['accuracy_passed'] and r['material_reduction_passed'] for r in rows)
    report={'schema_version':'giada-gate-d-compiled-confirmation-v1','valid':True,
            'eager_registered_passed':True,'compiled_confirmation_passed':passed,
            'gate_d_robustly_supported':passed,'task29_authorized':passed,
            'training_performed':False,'fresh_used_for_selection':False,
            'device_name':torch.cuda.get_device_name(),'torch_version':torch.__version__,
            'cuda_version':torch.version.cuda,'code_revision':revision,
            'compiler_backend':cfg['backend'],'compiler_mode':cfg['mode'],
            'rows':rows,'limits':cfg['limits']}
    write(output/'final_report.json',report)
    return report
