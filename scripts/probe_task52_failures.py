import gzip,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task50_compact_memory as t
from src.giada_teacher import task52_closed_loop as q
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
with zipfile.ZipFile(ROOT/'artifacts/task51_kaggle_external/task51_final.zip') as z:
    cfg=json.loads(z.read('preregistration.json'));freeze=json.loads(z.read('selection_freeze.json'));state=torch.load(io.BytesIO(z.read('selected.pt')),map_location='cpu',weights_only=True)
indices=[min((i for i,m in enumerate(freeze['members']) if m['seed']==s),key=lambda i:freeze['development'][i]) for s in cfg['seeds']]
cfg['arms']=[dict(id='compact',widths=[2,3,3,1,1],trunk=16)];net=t.model_class(torch)(cfg,'cpu');net.load_state_dict({k:v[indices].clone() for k,v in state.items()});net.eval()
with zipfile.ZipFile(ROOT/'artifacts/task52_kaggle_external/task52_final.zip') as z:
    p=json.loads(z.read('preregistration.json'))
    evaluations=json.loads(gzip.decompress(z.read('evaluation.json.gz')))
    with np.load(io.BytesIO(z.read('traces.npz'))) as a:data={k:a[k] for k in a.files}
rows,_,mult,rev=q.design(p);v=data['reference_voltage'][:-1];ca=data['reference_calcium'][:-1];ref=data['reference_current'];T,S,C=v.shape
with torch.no_grad():
    weights=(net.out*net.om).numpy();bias=net.ob[:,0].numpy()
    low=1/(1+np.exp(-(bias+np.minimum(weights,0).sum(1))));high=1/(1+np.exp(-(bias+np.maximum(weights,0).sum(1))))
    factor=mult[None,None]*np.asarray(q.q.ionic.GBAR)[None,None,None]*(v[...,None]-rev[None,None])
    a=low[None,:,None]*factor;b=high[None,:,None]*factor
    nearest=np.clip(ref,np.minimum(a,b),np.maximum(a,b));err=nearest-ref
    scale=np.maximum(abs(ref).max(0),p['quality']['current_scale_floor_ma_cm2'])
    nr=np.sqrt((err**2).mean(0))/scale;nm=abs(err).max(0)/scale
    failed=(nr>p['quality']['current_nrmse_max'])|(nm>p['quality']['current_nmax_max'])
    actual=np.empty_like(nr)
    for row in evaluations:
        if row['arm']=='compact_imposed' and row['horizon_ms']==1000:
            actual[p['seeds'].index(row['seed']),row['case']]=[x['nrmse'] for x in row['current']['channels']]
    assert np.all(nr<=actual+1e-4*np.maximum(actual,1)), 'Decoder lower bound exceeds actual same-input error'
    initial=torch.tensor(q.q.steady(-65,1e-4),dtype=torch.float32)[None,None].expand(S,C,-1);h=net.initialize(initial)
    times=[];rates=[];jacobians=[]
    for tick in range(T):
        x=t.features(torch,torch.tensor(v[tick],dtype=torch.float32),torch.tensor(ca[tick],dtype=torch.float32))
        trunk=torch.nn.functional.silu(torch.bmm(x,net.w)+net.b)*net.tm
        raw=torch.bmm(trunk,net.u)+torch.bmm(h,net.r*net.rm)+net.c
        proposal,z=raw.chunk(2,-1);z=torch.sigmoid(z)
        if tick%100==0:
            rates.append(z[...,:10].numpy());times.append(tick*.1)
        if tick in (0,1000,5000,T-1):
            for seed in range(3):
                for case in (0,4,8):
                    def transition(h10):
                        full=torch.nn.functional.pad(h10,(0,8));rr=trunk[seed,case]@net.u[seed]+full@(net.r[seed]*net.rm[seed])+net.c[seed,0]
                        pp,zz=rr.chunk(2,-1)
                        return (full+torch.sigmoid(zz)*(torch.sigmoid(pp)-full))[:10]
                    with torch.enable_grad():
                        jac=torch.autograd.functional.jacobian(transition,h[seed,case,:10],vectorize=True)
                    jacobians.append(dict(time_ms=tick*.1,seed=seed,case=case,spectral_radius=float(torch.linalg.eigvals(jac).abs().max()),max_singular_value=float(torch.linalg.svdvals(jac).max())))
        h=(h+z*(torch.sigmoid(proposal)-h))*net.lm
    rate=np.stack(rates);tau=-.1/np.log1p(-np.clip(rate,1e-12,1-1e-7))
    families={};offset=0
    for name,width in zip(p['families'],[2,3,3,1,1]):
        val=tau[...,offset:offset+width];families[name]=dict(tau_min_ms=float(val.min()),tau_median_ms=float(np.median(val)),tau_max_ms=float(val.max()));offset+=width
result=dict(valid=True,source_indices=indices,scope='Read-only fixed-model probe; decoder box bounds are necessary not sufficient; no training',decoder_bounds=dict(lower=low.tolist(),upper=high.tolist(),channels=list(q.q.ionic.CHANNELS)),irreducible_current_error=dict(nrmse=nr.tolist(),nmax=nm.tolist(),failed_channel_cases=int(failed.sum()),total_channel_cases=int(failed.size),failed_trajectories=int(failed.any(-1).sum()),total_trajectories=int(S*C),failures_per_channel=failed.sum((0,1)).tolist(),lower_bound_checked_against_actual=True),direct_carry_timescales_only=families,local_full_state_jacobians=jacobians,observation='Direct carry timescale is not total memory because proposal/rate depend on state. Jacobians are local sampled probes, not a global memory theorem.')
dest=ROOT/'experiments/results/task52_audit_v1';dest.mkdir(exist_ok=True);(dest/'fixed_model_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(valid=True,decoder_failures=result['irreducible_current_error']['failed_trajectories'],direct_carry=families,jacobian_radius_max=max(x['spectral_radius'] for x in jacobians),jacobian_singular_max=max(x['max_singular_value'] for x in jacobians)),indent=2))
