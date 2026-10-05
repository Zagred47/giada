"""Same-CUDA-device compute audit for the frozen Task28 hybrid ionic block.

Both branches compute all 18 gate endpoints and 11 analytic signed currents.
No host transfer, teacher NumPy execution, loading or training enters timing.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from . import ionic_block_teacher_forced as ionic
from .hh_family_transfer import write


def config(root: Path) -> dict:
    return json.loads((root / 'experiments/task28b_gate_d_compute.json').read_text(encoding='utf-8'))


def verify_parent(root: Path, cfg: dict) -> None:
    directory = root / cfg['task28_result_dir']
    actual = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in cfg['task28_result_sha256']}
    if actual != cfg['task28_result_sha256']:
        raise RuntimeError('Task28 result bytes changed')
    result = json.loads((directory / 'final_report.json').read_text(encoding='utf-8'))
    audit = json.loads((directory / 'result_audit.json').read_text(encoding='utf-8'))
    if not (result['valid'] and result['teacher_forced_ionic_block_passed'] and audit['valid']):
        raise RuntimeError('Task28 teacher-forced composition prerequisite failed')
    if result['gate_d_completed'] or result['task29_authorized']:
        raise RuntimeError('Gate D was already declared complete')


def _rates(name, v, ca, torch):
    exp, expm1, sig = torch.exp, torch.expm1, torch.sigmoid
    qt = 2.3 ** 1.3
    def seam(x, value):
        return torch.where(x == value, x + .0001, x)
    if name == 'Ca_HVA':
        x = seam(v, -27.)
        a = .055 * (-27-x) / expm1((-27-x)/3.8)
        b = .94*exp((-75-x)/17)
        ha = .000457*exp((-13-x)/50)
        hb = .0065/(exp((-x-15)/28)+1)
        return torch.stack((a/(a+b),ha/(ha+hb)),-1), torch.stack((1/(a+b),1/(ha+hb)),-1)
    if name == 'Ca_LVAst':
        x = v+10
        inf = torch.stack((sig((x+30)/6),sig(-(x+80)/6.4)),-1)
        tau = torch.stack(((5+20/(1+exp((x+25)/5)))/qt,
                           (20+50/(1+exp((x+40)/7)))/qt),-1)
        return inf,tau
    if name in ('NaTa_t','NaTs2_t','Nap_Et2'):
        if name == 'Nap_Et2':
            mi = sig((v+52.6)/4.6)
            x = seam(v,-38.)
            a = .182*(x+38)/(-expm1(-(x+38)/6))
            b = .124*(-x-38)/(-expm1(-(-x-38)/6))
            mt = 6/(a+b)/qt
            x = torch.where((x == -17.) | (x == -64.4),x+.0001,x)
            hi = sig(-(x+48.8)/10)
            a = -2.88e-6*(x+17)/(-expm1((x+17)/4.63))
            b = 6.94e-6*(x+64.4)/(-expm1(-(x+64.4)/2.63))
            ht = 1/(a+b)/qt
            return torch.stack((mi,hi),-1),torch.stack((mt,ht),-1)
        m_seam,h_seam = (-38.,-66.) if name == 'NaTa_t' else (-32.,-60.)
        x = seam(v,m_seam)
        a = .182*(x-m_seam)/(-expm1(-(x-m_seam)/6))
        b = .124*(-x+m_seam)/(-expm1(-(-x+m_seam)/6))
        mi,mt = a/(a+b), 1/(a+b)/qt
        x = seam(x,h_seam)
        if name == 'NaTa_t':
            a = -.015*(x+66)/(-expm1((x+66)/6))
            b = -.015*(-x-66)/(-expm1((-x-66)/6))
        else:
            a = -.015*(x+60)/(-expm1((x+60)/6))
            b = -.015*(-x-60)/(-expm1((-x-60)/6))
        return torch.stack((mi,a/(a+b)),-1),torch.stack((mt,1/(a+b)/qt),-1)
    if name == 'Ih':
        x = seam(v,-154.9)
        a = .001*6.43*(x+154.9)/expm1((x+154.9)/11.9)
        b = .001*193*exp(x/33.1)
        return a/(a+b),1/(a+b)
    if name == 'Im':
        a = .0033*exp(.1*(v+35));b = .0033*exp(-.1*(v+35))
        return a/(a+b),1/(a+b)/qt
    if name == 'K_Pst':
        x=v+10
        mi=sig((x+1)/12)
        mt=torch.where(x < -50,1.25+175.03*exp(x*.026),1.25+13*exp(-x*.026))/qt
        hi=sig(-(x+54)/11)
        ht=(360+(1010+24*(x+55))*exp(-((x+75)/48)**2))/qt
        return torch.stack((mi,hi),-1),torch.stack((mt,ht),-1)
    if name == 'K_Tst':
        x=v+10
        inf=torch.stack((sig(x/19),sig(-(x+66)/10)),-1)
        tau=torch.stack(((.34+.92*exp(-((x+71)/59)**2))/qt,
                         (8+49*exp(-((x+73)/23)**2))/qt),-1)
        return inf,tau
    if name == 'SK_E2':
        effective=torch.where(ca < 1e-7,ca+1e-7,ca)
        return 1/(1+(.00043/effective)**4.8),torch.ones_like(ca)
    if name == 'SKv3_1':
        return sig((v-18.7)/9.7),4/(1+exp((v+46.56)/-44.14))
    raise ValueError(name)


def _currents(v, state, torch):
    opening=[]
    for k,names in enumerate(ionic.STATE_NAMES):
        a=ionic.OFFSETS[k]
        x=state[...,a].pow(int(ionic.POWERS[k]))
        if len(names)==2:x=x*state[...,a+1]
        opening.append(x)
    opened=torch.stack(opening,-1)
    gb=torch.as_tensor(ionic.GBAR,device=v.device,dtype=v.dtype)
    reversal=torch.as_tensor(ionic.REVERSALS,device=v.device,dtype=v.dtype)
    return opened*gb*(v[...,None]-reversal)


def exact_gpu(v,ca,state,dt,torch):
    parts=[]
    for k,name in enumerate(ionic.CHANNELS):
        inf,tau=_rates(name,v,ca,torch)
        a,b=ionic.OFFSETS[k:k+2]
        if b-a==1:inf,tau=inf[...,None],tau[...,None]
        z=-torch.expm1(-dt[...,None]/tau)
        parts.append((1-z)*state[...,a:b]+z*inf)
    output=torch.cat(parts,-1)
    return output,_currents(v,output,torch)


def hybrid_gpu(model,v,ca,state,dt,torch):
    n,b=v.shape
    packed=torch.cat((v[...,None],state[...,:10],dt[...,None]),-1)
    learned=model(packed)[0].permute(0,2,1,3).reshape(n,b,10)
    parts=[learned]
    for k,name in enumerate(ionic.CHANNELS[5:],start=5):
        inf,tau=_rates(name,v,ca,torch)
        a,b=ionic.OFFSETS[k:k+2]
        if b-a==1:inf,tau=inf[...,None],tau[...,None]
        z=-torch.expm1(-dt[...,None]/tau)
        parts.append((1-z)*state[...,a:b]+z*inf)
    output=torch.cat(parts,-1)
    return output,_currents(v,output,torch)


def _elapsed_ms(fn,torch):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();states,currents=fn();end.record();end.synchronize()
    # Keep both outputs live. This is inference latency, not a partial-rate probe.
    assert states.shape[-1]==18 and currents.shape[-1]==11
    return start.elapsed_time(end)


def run(root:Path,output:Path,cfg:dict,revision:str)->dict:
    import torch
    verify_parent(root,cfg)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('Gate D registered CUDA device absent')
    models,seeds=ionic.load_frozen(root,ionic.config(root),torch,'cuda')
    assert list(seeds)==cfg['model_seeds']
    device='cuda';rows=[]
    task28=ionic.config(root)
    with torch.inference_mode():
        for batch in cfg['batch_sizes']:
            v,ca,state,dt=ionic.support(cfg['input_seed']+batch,batch,task28,'in_support')
            np_exact=ionic.exact_step(v,ca,state,dt)
            tensors=[torch.tensor(x,device=device,dtype=torch.float32) for x in (v,ca,state,dt)]
            tv,tc,ts,td=[x[None].expand(len(seeds),*x.shape).contiguous() for x in tensors]
            exact=lambda:exact_gpu(tv,tc,ts,td,torch)
            exact_out,exact_i=exact()
            exact_gate_error=float(np.max(np.abs(exact_out[0].double().cpu().numpy()-np_exact)))
            exact_current_error=float(np.max(np.abs(exact_i[0].double().cpu().numpy()-ionic.currents(v,np_exact))))
            if (not np.isfinite([exact_gate_error,exact_current_error]).all()
                or exact_gate_error>cfg['preflight_max_gate_error']
                or exact_current_error>cfg['preflight_max_current_error_ma_cm2']):
                raise RuntimeError(f'canonical GPU vs NumPy preflight mismatch: {exact_gate_error},{exact_current_error}')
            for family in cfg['families']:
                model=models[family,cfg['arm']]
                hybrid=lambda:hybrid_gpu(model,tv,tc,ts,td,torch)
                hybrid_out,hybrid_i=hybrid()
                assert torch.isfinite(hybrid_out).all() and torch.isfinite(hybrid_i).all()
                gate_rmse=float(torch.sqrt((hybrid_out-exact_out).square().mean()).item())
                current_rmse=float(torch.sqrt((hybrid_i-exact_i).square().mean()).item())
                accuracy_rows=[]
                hybrid_cpu=hybrid_out.cpu().numpy()
                for seed_index,seed in enumerate(seeds):
                    measured=ionic.measure(v,hybrid_cpu[seed_index],np_exact,task28)
                    max_gate=max(x['rmse'] for x in measured['per_channel_gates'].values())
                    max_individual=max(x['worst_individual_normalized_rmse'] for x in measured['current_panels'].values())
                    max_total=max(x['total_normalized_rmse'] for x in measured['current_panels'].values())
                    accurate=bool(measured['finite'] and max_gate<=task28['learned_gate_rmse_limit']
                                  and max_individual<=task28['learned_current_normalized_rmse_limit']
                                  and max_total<=task28['learned_total_current_normalized_rmse_limit']
                                  and all(x['occupancy_violations']==0 for x in measured['per_channel_gates'].values()))
                    accuracy_rows.append({'seed':seed,'passed':accurate,'worst_gate_rmse':max_gate,
                                          'worst_individual_current_normalized_rmse':max_individual,
                                          'worst_total_current_normalized_rmse':max_total})
                for _ in range(cfg['warmup_repetitions']):exact();hybrid()
                torch.cuda.synchronize()
                pairs=[]
                for index in range(cfg['benchmark_repetitions']):
                    order=('exact','hybrid') if index%2==0 else ('hybrid','exact')
                    times={name:_elapsed_ms(exact if name=='exact' else hybrid,torch) for name in order}
                    pairs.append(times)
                exact_median=statistics.median(p['exact'] for p in pairs)
                hybrid_median=statistics.median(p['hybrid'] for p in pairs)
                paired_gain=statistics.median((p['exact']-p['hybrid'])/p['exact'] for p in pairs)
                row={'family':family,'batch_size':batch,'model_seeds':list(seeds),
                     'exact_gpu_vs_numpy_max_gate_error':exact_gate_error,
                     'exact_gpu_vs_numpy_max_current_error_ma_cm2':exact_current_error,
                     'hybrid_vs_exact_gate_rmse':gate_rmse,'hybrid_vs_exact_current_rmse_ma_cm2':current_rmse,
                     'accuracy_rows':accuracy_rows,'accuracy_passed':all(x['passed'] for x in accuracy_rows),
                     'exact_median_ms':exact_median,'hybrid_median_ms':hybrid_median,
                     'median_paired_compute_reduction_fraction':paired_gain,
                     'material_reduction_passed':paired_gain>=cfg['material_compute_reduction_minimum'],
                     'paired_timings_ms':pairs}
                rows.append(row)
                print(f'[GIADA Gate D] {family} batch={batch} exact={exact_median:.4f}ms hybrid={hybrid_median:.4f}ms gain={paired_gain:.2%}',flush=True)
    primary=[r for r in rows if r['batch_size']==cfg['gate_d_batch_size']]
    passed=all(r['material_reduction_passed'] and r['accuracy_passed'] for r in primary)
    report={'schema_version':'giada-gate-d-compute-v1','valid':True,
            'gate_d_completed':passed,'task29_authorized':passed,
            'task28_teacher_forced_passed':True,'compute_reduction_passed':passed,
            'training_performed':False,'fresh_used_for_selection':False,
            'code_revision':revision,'device_name':torch.cuda.get_device_name(),
            'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
            'rows':rows,'limits':cfg['limits']}
    write(output/'final_report.json',report)
    return report
