"""Original Task25: frozen heterogeneous rate modules, ten explicit states."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from . import calcium_pair_composition as ca
from . import sodium_family_composition as na
from .hh_family_transfer import update,write,sha

CHANNELS=ca.CHANNELS+na.CHANNELS
POWERS=np.array([2,2,3,3,3])
REVERSALS=np.array([120,120,55,55,55])


def data(seed,count,dt,domain=(-135,75)):
    rng=np.random.default_rng(seed)
    return np.c_[rng.uniform(*domain,count),rng.uniform(0,1,(count,10)),rng.choice(dt,count)]


def extrema(cfg):
    states=np.r_[np.zeros((1,10)),np.ones((1,10)),np.eye(10),1-np.eye(10)]
    v=-135+210*(np.arange(cfg['extrema_nodes'])+cfg['extrema_node_offset'])/cfg['extrema_nodes']
    v=np.r_[v,-135,75,-27,-40,-90,na.SEAMS]
    vv,ss,dd=np.meshgrid(v,np.arange(len(states)),cfg['dt_values_ms'],indexing='ij')
    return np.c_[vv.ravel(),states[ss.ravel()],dd.ravel()]


def rates(v):
    rr=[ca.rates(c,v) for c in ca.CHANNELS]+[na.rates(c,v) for c in na.CHANNELS]
    return np.stack([a[0] for a in rr]),np.stack([a[1] for a in rr])


def targets(values):
    i,t=rates(values[:,0]);state=values[:,1:11].reshape(-1,5,2).transpose(1,0,2)
    return i,t,update(state,i,t,values[:,-1])


def verify_source(root):
    freeze=json.loads((root/'selection_freeze.json').read_text());claim=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==claim
    assert not freeze['fresh_accessed']
    return freeze


def load_frozen(root,cfg,torch,device):
    cr=root/cfg['calcium_source'];nr=root/cfg['sodium_source'];sr=root/cfg['sodium_shared_source']
    cf,nf,sf=[verify_source(p) for p in (cr,nr,sr)]
    calcium_report=json.loads((cr/'final_report.json').read_text())
    assert calcium_report['composition_passed'] and calcium_report['per_family_passed']['shared_heads']
    new=json.loads((nr/'final_report.json').read_text());assert new['primary_repair_passed'] and new['per_arm_passed']['frozen_conditioned']
    selected={'calcium_independent':(ca,cr,cf,cf['selection']['independent'],'independent'),
      'calcium_shared':(ca,cr,cf,cf['selection']['shared_heads'],'shared_heads'),
      'sodium_independent':(na,nr,nf,nf['selection']['decay_channel'],'independent'),
      'sodium_conditioned':(na,sr,sf,sf['selection']['conditioned'],'conditioned')}
    models={};sources={}
    for key,(module,folder,freeze,row,family) in selected.items():
        cp=folder/row['checkpoint'];assert sha(cp)==freeze['checkpoint_hashes'][cp.name]
        model=module.model_factory(torch,row['width'],family,cfg['seeds']).to(device)
        model.load_state_dict(torch.load(cp,map_location=device,weights_only=True));model.eval()
        for p in model.parameters():p.requires_grad_(False)
        models[key]=model;sources[key]=dict(path=cp.relative_to(root).as_posix(),sha256=sha(cp),selection=row,parameters=sum(p.numel() for p in model.parameters())//len(cfg['seeds']))
    return models,sources


def bundle(models,arm):
    return (models['calcium_shared' if arm in ('calcium_compressed','both_compressed') else 'calcium_independent'],
            models['sodium_conditioned' if arm in ('sodium_compressed','both_compressed') else 'sodium_independent'])


def forward(pair,x,torch):
    cx=torch.cat((x[...,:5],x[...,-1:]),-1)
    nx=torch.cat((x[...,:1],x[...,5:11],x[...,-1:]),-1)
    cc,nn=pair[0](cx),pair[1](nx)
    return tuple(torch.cat((a,b),1) for a,b in zip(cc,nn))


def current_metrics(v,pred,truth):
    po=pred[...,0]**POWERS[:,None]*pred[...,1];to=truth[...,0]**POWERS[:,None]*truth[...,1]
    panels=[]
    multipliers=[np.ones(5),np.array([1,1,0,0,0]),np.array([0,0,1,1,1]),np.array([.25,.25,4,4,4]),np.array([4,4,.25,.25,.25])]
    multipliers += [np.eye(5)[k] for k in range(5)]
    for g in multipliers:
        for e in (REVERSALS,np.array([40,160,25,55,85])):
            d=1e-5*g[:,None]*(v[None]-e[:,None]);err=(po-to)*d
            individual=err/np.maximum(abs(d),1e-12);total=err.sum(0)/np.maximum(abs(d).sum(0),1e-12)
            ti=to*d;ratio=abs(ti.sum(0))/np.maximum(abs(ti).sum(0),1e-12);mask=(abs(ti).sum(0)>1e-12)&(ratio<=.1)
            panels.append(dict(gbar_multipliers=g.tolist(),reversals_mv=e.tolist(),
                individual_current_normalized_rmse=float(np.sqrt(np.mean(individual**2))),
                per_channel_normalized_rmse={c:float(np.sqrt(np.mean(individual[k]**2))) for k,c in enumerate(CHANNELS)},
                total_current_normalized_rmse=float(np.sqrt(np.mean(total**2))),total_current_normalized_max=float(abs(total).max()),
                total_current_rmse_ma_cm2=float(np.sqrt(np.mean(err.sum(0)**2))),
                cancellation_count=int(mask.sum()),cancellation_total_normalized_rmse=float(np.sqrt(np.mean(total[mask]**2))) if mask.any() else None,
                cancellation_individual_normalized_rmse=float(np.sqrt(np.mean(individual[:,mask]**2))) if mask.any() else None))
    return dict(individual_current_normalized_rmse=max(v for p in panels for v in p['per_channel_normalized_rmse'].values()),
        total_current_normalized_rmse=max(p['total_current_normalized_rmse'] for p in panels),total_current_normalized_max=max(p['total_current_normalized_max'] for p in panels),panels=panels)


def measurements(values,pred,inf,tau,truth=None):
    ti,tt,target=targets(values)
    if truth is None:truth=target
    result={}
    for k,c in enumerate(CHANNELS):
        err=pred[k]-truth[k];op=pred[k,:,0]**POWERS[k]*pred[k,:,1]-truth[k,:,0]**POWERS[k]*truth[k,:,1]
        result[c]=dict(gate_rmse=float(np.sqrt(np.mean(err**2))),m_rmse=float(np.sqrt(np.mean(err[:,0]**2))),h_rmse=float(np.sqrt(np.mean(err[:,1]**2))),
            gate_max_error=float(abs(err).max()),inf_rmse=float(np.sqrt(np.mean((inf[k]-ti[k])**2))),log_tau_rmse=float(np.sqrt(np.mean((np.log(tau[k])-np.log(tt[k]))**2))),
            open_rmse=float(np.sqrt(np.mean(op**2))),occupancy_violations=int(((pred[k]<0)|(pred[k]>1)).sum()),
            finite=bool(np.isfinite(pred[k]).all() and np.isfinite(inf[k]).all() and np.isfinite(tau[k]).all() and (tau[k]>0).all()))
    result['currents']=current_metrics(values[:,0],pred,truth)
    return result


def passes(row,gates,current_gates):
    return all(row[c]['finite'] and all(row[c][k]<=limit for k,limit in gates.items()) for c in CHANNELS) and all(row['currents'][k]<=limit for k,limit in current_gates.items())


def predict(pair,values,seeds,torch,device):
    rows=[[] for _ in seeds]
    with torch.no_grad():
        for start in range(0,len(values),2048):
            x=torch.tensor(values[start:start+2048],device=device,dtype=torch.float32)[None].expand(len(seeds),-1,-1)
            out=[a.cpu().double().numpy() for a in forward(pair,x,torch)]
            for n in range(len(seeds)):rows[n].append([a[n] for a in out])
    return [tuple(np.concatenate([r[k] for r in row],axis=1) for k in range(3)) for row in rows]


def rollout(pair,seeds,torch,device,dt,held,horizons,voltage_seed,dtype='float32'):
    if held:
        v=np.repeat(np.random.default_rng(voltage_seed).uniform(-135,75,64),2)
        states=np.tile(np.array([[0.]*10,[1.]*10]),(64,1));paths=np.broadcast_to(v,(max(horizons),len(v))).copy()
    else:
        paths=ca.imposed_paths().T;states=np.tile(np.array([.1,.9,.8,.2,.3,.7,.6,.4,.2,.8]),(paths.shape[1],1))
    steps,b=paths.shape;rate_paths=paths[:1] if held else paths;rate_steps=len(rate_paths)
    values=np.c_[rate_paths.ravel(),np.zeros((rate_steps*b,10)),np.full(rate_steps*b,dt)]
    raw=predict(pair,values,seeds,torch,device)
    inf=np.stack([r[1].reshape(5,rate_steps,b,2) for r in raw]);tau=np.stack([r[2].reshape(5,rate_steps,b,2) for r in raw])
    it=torch.tensor(inf,device=device,dtype=getattr(torch,dtype));tt=torch.tensor(tau,device=device,dtype=getattr(torch,dtype))
    state=torch.tensor(states.reshape(b,5,2).transpose(1,0,2),device=device,dtype=getattr(torch,dtype))[None].expand(len(seeds),-1,-1,-1).clone()
    truth=states.reshape(b,5,2).transpose(1,0,2).copy();result=[{} for _ in seeds];predictions=[];truths=[]
    with torch.no_grad():
        for step in range(steps):
            ri=0 if held else step
            z=-torch.expm1(-dt/tt[:,:,ri]);state=(1-z)*state+z*it[:,:,ri]
            i,t=rates(paths[step]);truth=update(truth,i,t,dt)
            if held and step+1 in horizons:
                val=np.c_[paths[step],states,np.full(b,(step+1)*dt)]
                for n,p in enumerate(state.cpu().double().numpy()):result[n][str(step+1)]=measurements(val,p,inf[n,:,ri],tau[n,:,ri],truth)
            elif not held:
                predictions.append(state.cpu().double().numpy());truths.append(truth.copy())
    if not held:
        p=np.stack(predictions).transpose(1,2,0,3,4);truth=np.stack(truths).transpose(1,0,2,3)
        for n in range(len(seeds)):
            for path in range(b):
                val=np.c_[paths[:,path],np.zeros((steps,10)),np.full(steps,dt)]
                result[n][str(path)]=measurements(val,p[n,:,:,path],inf[n,:,:,path],tau[n,:,:,path],truth[:,:,path])
    return result


def native_audit(teacher,output):
    reports=[]
    for label,module in [('calcium','calcium_pair_composition'),('sodium','sodium_family_composition')]:
        sub=Path(output)/label;sub.mkdir(parents=True)
        code=f'from src.giada_teacher.{module} import native_audit;native_audit({str(teacher)!r},{str(sub)!r})'
        subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],check=True)
        reports.append(json.loads((sub/'native_audit.json').read_text()))
    report=dict(valid=all(r['valid'] for r in reports),reports=reports)
    write(Path(output)/'native_audit.json',report)
    if not report['valid']:raise RuntimeError('Native heterogeneous oracle failed')


def run(root,output,cfg,revision):
    import torch
    from .gpu_baseline_runtime import environment_manifest
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['require_cuda'] and device!='cuda':raise RuntimeError('CUDA required')
    assert json.loads((output/'native_audit.json').read_text())['valid']
    seeds=cfg['seeds'];models,sources=load_frozen(root,cfg,torch,device)
    freeze=dict(config=cfg,sources=sources,fresh_accessed=False,code_revision=revision,retraining_performed=False)
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();write(output/'composition_freeze.json',freeze)
    fresh=dict(in_support=data(cfg['fresh_seeds'][0],cfg['sample_count'],cfg['dt_values_ms']),activation_boundary=data(cfg['fresh_seeds'][1],cfg['sample_count'],cfg['dt_values_ms'],(-105,5)),state_extrema=extrema(cfg))
    np.savez_compressed(output/'fresh.npz',**fresh);results=[];contrasts=[];start=time.monotonic()
    for arm in cfg['arms']:
        pair=bundle(models,arm);domains={d:[measurements(v,*r) for r in predict(pair,v,seeds,torch,device)] for d,v in fresh.items()}
        held=rollout(pair,seeds,torch,device,1,True,cfg['held_horizons_ms'],cfg['fresh_seeds'][3]);held64=rollout(pair,seeds,torch,device,1,True,cfg['held_horizons_ms'],cfg['fresh_seeds'][3],'float64')
        paths={label:rollout(pair,seeds,torch,device,dt,False,[],0) for label,dt in cfg['path_dt_ms'].items()}
        identity=[]
        for p,i,t in predict(pair,fresh['in_support'],seeds,torch,device):
            state=fresh['in_support'][:,1:11].reshape(-1,5,2).transpose(1,0,2);order=[1,2,3,4,0]
            wrong=update(state,i[order],t[order],fresh['in_support'][:,-1]);identity.append(measurements(fresh['in_support'],wrong,i[order],t[order]))
        for n,seed in enumerate(seeds):
            dd={d:r[n] for d,r in domains.items()};pp={label:r[n] for label,r in paths.items()}
            passed=all(passes(r,cfg['gates'],cfg['current_gates']) for r in dd.values()) and all(passes(r,cfg['held_gates'],cfg['path_current_gates']) for r in held[n].values()) and all(passes(r,cfg['path_gates'],cfg['path_current_gates']) for path in pp.values() for r in path.values())
            count=sum(p.numel() for m in pair for p in m.parameters())//len(seeds)
            results.append(dict(arm=arm,seed=seed,parameter_count=count,metrics=dd,held_rollout=held[n],held_fp64_diagnostic=held64[n],path_rollout=pp,swapped_identity=identity[n],identity_rmse_ratio=float(np.median([identity[n][c]['gate_rmse']/max(dd['in_support'][c]['gate_rmse'],1e-12) for c in CHANNELS])),passed=passed))
        print('[GIADA Task25] frozen bundle '+arm+' evaluated',flush=True)
    for arm in cfg['arms'][1:]:
        for seed in seeds:
            a=next(r for r in results if r['arm']=='independent' and r['seed']==seed);b=next(r for r in results if r['arm']==arm and r['seed']==seed)
            contrasts.append(dict(arm=arm,seed=seed,parameter_difference=b['parameter_count']-a['parameter_count'],gate_rmse_difference={d:{c:b['metrics'][d][c]['gate_rmse']-a['metrics'][d][c]['gate_rmse'] for c in CHANNELS} for d in fresh}))
    numerical=[]
    grid=np.linspace(-135,75,2049);gi,gt=rates(grid)
    for domain,v in fresh.items():
        i,t,truth=targets(v);state=v[:,1:11].reshape(-1,5,2).transpose(1,0,2)
        for kind in ('formula_fp32','lut2049_fp64'):
            if kind=='formula_fp32':
                st,ii,tt=[torch.tensor(a,dtype=torch.float32,device=device) for a in (state,i,t)];z=-torch.expm1(-torch.tensor(v[:,-1],device=device,dtype=torch.float32)[None,:,None]/tt);p=((1-z)*st+z*ii).cpu().double().numpy();pi,pt=i,t
            else:
                pi=np.stack([np.stack([np.interp(v[:,0],grid,gi[k,:,g]) for g in range(2)],-1) for k in range(5)]);pt=np.stack([np.stack([np.interp(v[:,0],grid,gt[k,:,g]) for g in range(2)],-1) for k in range(5)]);p=update(state,pi,pt,v[:,-1])
            numerical.append(dict(kind=kind,domain=domain,metrics=measurements(v,p,pi,pt),selectable=False))
    passing={a:all(r['passed'] for r in results if r['arm']==a) for a in cfg['arms']};counts={a:next(r['parameter_count'] for r in results if r['arm']==a) for a in cfg['arms']}
    report=dict(schema_version='giada-task25-v1',valid=True,composition_passed=passing['independent'],task26_preparation_authorized=passing['independent'],per_arm_passed=passing,selected_parameter_counts=counts,
        compression_reuse={a:passing['independent'] and passing[a] and counts[a]<counts['independent'] for a in cfg['arms'][1:]},learned=results,numerical=numerical,paired_contrasts=contrasts,retraining_performed=False,fresh_used_for_selection=False,
        teacher_forced_only=True,full_neuron_speedup_claimed=False,scope=cfg['limits'],evaluation_seconds=time.monotonic()-start,environment=environment_manifest(torch))
    write(output/'paired_contrasts.json',contrasts);write(output/'final_report.json',report);return report
