"""Task22: three voltage-driven scalar gates, controlled weight sharing.

Each seed is an independent three-mechanism bundle. Loss weights, Adam,
clipping, data and duration support are identical across architecture families.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from . import single_gate_transfer as base
from . import slow_gate_transfer as slow
from .hh_family_transfer import write, sha

CHANNELS=('Ih','Im','Nap_h')
FAMILIES=('independent','shared_heads','conditioned')


def rates(channel, voltage):
    return slow.rates('multiscale_dt',voltage) if channel=='Nap_h' else base.rates(channel,voltage)


def data(seed,count,domain=(-135,75)):
    rng=np.random.default_rng(seed)
    return np.c_[rng.uniform(*domain,count),rng.uniform(0,1,count),rng.choice(slow.LONG_DT,count)]


def state_grid(fresh=False):
    n,offset=(401,.37) if fresh else (201,.5)
    voltage=-135+210*(np.arange(n)+offset)/n
    if fresh:voltage=np.r_[voltage,-135,75,-38,-17,-64.4]
    v,s,t=np.meshgrid(voltage,[0.,1.],slow.LONG_DT,indexing='ij')
    return np.c_[v.ravel(),s.ravel(),t.ravel()]


def model_factory(torch,width,family,seeds):
    if family not in FAMILIES:raise ValueError(family)
    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n=len(seeds);ind=family=='independent'
            shapes={'w1':(n,3,width,1) if ind else (n,width,4 if family=='conditioned' else 1),
                    'b1':(n,3,width) if ind else (n,width),
                    'w2':(n,3,width,width) if ind else (n,width,width),
                    'b2':(n,3,width) if ind else (n,width),
                    'wo':(n,3,2,width) if family!='conditioned' else (n,2,width),
                    'bo':(n,3,2) if family!='conditioned' else (n,2)}
            for name,shape in shapes.items():
                value=torch.zeros(shape)
                if name.startswith('w'):
                    for s,seed in enumerate(seeds):
                        gen=torch.Generator().manual_seed(seed+{'w1':0,'w2':100,'wo':200}[name])
                        # Identical trunk initialization across gates/families where shapes permit.
                        if ind:
                            sample=torch.randn(shape[2:],generator=gen)*.1
                            value[s]=sample[None].expand(3,*sample.shape)
                        elif name=='wo' and family=='shared_heads':
                            sample=torch.randn((2,width),generator=gen)*.1
                            value[s]=sample[None].expand(3,2,width)
                        elif name=='w1' and family=='conditioned':
                            value[s,:,0]=torch.randn((width,),generator=gen)*.1
                            value[s,:,1:]=torch.randn((width,3),generator=gen)*.1
                        else:value[s].normal_(0,.1,generator=gen)
                setattr(self,name,torch.nn.Parameter(value))

        def forward(self,x,identity_order=None):
            n,b,_=x.shape
            v=((x[...,0]+30)/100).unsqueeze(-1)
            if family=='independent':
                f=v[:,None].expand(-1,3,-1,-1)
                f=torch.nn.functional.silu(torch.einsum('scbi,scwi->scbw',f,self.w1)+self.b1[:,:,None])
                f=torch.nn.functional.silu(torch.einsum('scbi,scwi->scbw',f,self.w2)+self.b2[:,:,None])
                raw=torch.einsum('scbw,scow->scbo',f,self.wo)+self.bo[:,:,None]
            elif family=='shared_heads':
                f=torch.nn.functional.silu(torch.bmm(v,self.w1.transpose(1,2))+self.b1[:,None])
                f=torch.nn.functional.silu(torch.bmm(f,self.w2.transpose(1,2))+self.b2[:,None])
                raw=torch.einsum('sbw,scow->scbo',f,self.wo)+self.bo[:,:,None]
            else:
                ids=torch.eye(3,device=x.device,dtype=x.dtype)
                if identity_order is not None:ids=ids[identity_order]
                f=torch.cat([v[:,None].expand(-1,3,-1,-1),ids[None,:,None].expand(n,-1,b,-1)],-1).reshape(n,3*b,4)
                f=torch.nn.functional.silu(torch.bmm(f,self.w1.transpose(1,2))+self.b1[:,None])
                f=torch.nn.functional.silu(torch.bmm(f,self.w2.transpose(1,2))+self.b2[:,None])
                raw=(torch.bmm(f,self.wo.transpose(1,2))+self.bo[:,None]).reshape(n,3,b,2)
            if identity_order is not None and family!='conditioned':raw=raw[:,identity_order]
            inf=torch.sigmoid(raw[...,0]);tau=torch.exp(raw[...,1].clamp(-12,12))
            z=-torch.expm1(-x[:,None,:,2]/tau)
            return (1-z)*x[:,None,:,1]+z*inf,inf,tau
    return Ensemble()


def metrics(values,pred,inf,tau,channel):
    ti,tt=rates(channel,values[:,0]);error=pred-base.update(values[:,1],ti,tt,values[:,2])
    return dict(gate_rmse=float(np.sqrt(np.mean(error**2))),gate_max_error=float(abs(error).max()),
        inf_rmse=float(np.sqrt(np.mean((inf-ti)**2))),log_tau_rmse=float(np.sqrt(np.mean((np.log(tau)-np.log(tt))**2))),
        occupancy_violations=int(np.sum((pred<0)|(pred>1))),finite=bool(np.isfinite(pred).all() and np.isfinite(tau).all() and (tau>0).all()))


def evaluate(model,values,seeds,torch,device,swapped=False):
    x=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(len(seeds),-1,-1)
    with torch.no_grad():outputs=[v.cpu().double().numpy() for v in model(x,[1,2,0] if swapped else None)]
    return [{c:metrics(values,*(o[s,k] for o in outputs),c) for k,c in enumerate(CHANNELS)} for s in range(len(seeds))]


def equivalence(torch,width,family,seeds,device):
    from .joint_gate_symmetric_confirmation import _clip_per_seed
    m=model_factory(torch,width,family,seeds).to(device)
    x=torch.tensor(data(220001,32),dtype=torch.float32,device=device)[None].expand(len(seeds),-1,-1)
    opt=torch.optim.Adam(m.parameters(),lr=.003)
    singles=[];pred_error=0.;adam_error=0.
    for n,seed in enumerate(seeds):
        single=model_factory(torch,width,family,[seed]).to(device)
        single.load_state_dict({k:v[n:n+1].clone() for k,v in m.state_dict().items()})
        singles.append((single,torch.optim.Adam(single.parameters(),lr=.003)))
        pred_error=max(pred_error,float((m(x)[0][n]-single(x[n:n+1])[0][0]).detach().abs().max()))
    weights=torch.arange(1,len(seeds)+1,device=device)
    m(x)[0].square().mean((1,2)).mul(weights).sum().backward();_clip_per_seed(m,torch,1.);opt.step()
    for n,(single,optimizer) in enumerate(singles):
        single(x[n:n+1])[0].square().mean().mul(weights[n]).backward();_clip_per_seed(single,torch,1.);optimizer.step()
        adam_error=max(adam_error,max(float((v[n:n+1]-single.state_dict()[k]).detach().abs().max()) for k,v in m.named_parameters()))
    return dict(family=family,width=width,prediction_error=pred_error,adam_error=adam_error,valid=max(pred_error,adam_error)<1e-5)


def rollout(model,values,seeds,torch,device,horizons):
    x=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(len(seeds),-1,-1)
    result=[{c:{} for c in CHANNELS} for _ in seeds]
    with torch.no_grad():
        _,inf,tau=model(x);z=-torch.expm1(-x[:,None,:,2]/tau);state=x[:,None,:,1].expand(-1,3,-1).clone()
        for step in range(1,max(horizons)+1):
            state=(1-z)*state+z*inf
            if step in horizons:
                p=state.cpu().double().numpy()
                for k,c in enumerate(CHANNELS):
                    ti,tt=rates(c,values[:,0]);target=base.update(values[:,1],ti,tt,step*values[:,2])
                    for s in range(len(seeds)):
                        error=p[s,k]-target
                        result[s][c][str(step)]=dict(gate_rmse=float(np.sqrt(np.mean(error**2))),gate_max_error=float(abs(error).max()),occupancy_violations=int(np.sum((p[s,k]<0)|(p[s,k]>1))),finite=bool(np.isfinite(p[s,k]).all()))
    return result


def run(output,config,revision):
    from .gpu_baseline_runtime import configure_torch_runtime,environment_manifest
    from .joint_gate_symmetric_confirmation import _clip_per_seed
    torch=configure_torch_runtime(17);device='cuda' if torch.cuda.is_available() else 'cpu';output=Path(output)
    if config['require_cuda']:assert device=='cuda' and json.loads((output/'native_audit.json').read_text())['valid']
    seeds=config['seeds'];fit=data(220101,config['pool_size']);rng=np.random.default_rng(220102);n=len(fit)//4
    fit[:n,0]=rng.uniform(-120,-95,n);fit[n:2*n,0]=rng.uniform(-75,-25,n)
    dev=dict(uniform=data(220201,2048),negative_tail=data(220202,2048,(-135,-115)),state_extrema=state_grid())
    np.savez_compressed(output/'fit_development.npz',fit=fit,**dev)
    rr=[rates(c,fit[:,0]) for c in CHANNELS];ti,tt=np.stack([r[0] for r in rr]),np.stack([r[1] for r in rr]);ty=base.update(fit[:,1],ti,tt,fit[:,2])
    x,ti,tt,ty=[torch.tensor(v,dtype=torch.float32,device=device) for v in (fit,ti,tt,ty)]
    models={(f,w):model_factory(torch,w,f,seeds).to(device) for f in FAMILIES for w in config['widths']}
    pre=[equivalence(torch,w,f,seeds,device) for f,w in models]
    write(output/'equivalence_preflight.json',dict(valid=all(p['valid'] for p in pre),rows=pre,independent_seeds=True));assert all(p['valid'] for p in pre)
    optimizers={k:torch.optim.Adam(m.parameters(),lr=config['learning_rate']) for k,m in models.items()}
    ladder=[];best={};start=time.monotonic();rng=np.random.default_rng(220301)
    for step in range(config['checkpoints'][-1]+1):
        if step in config['checkpoints']:
            for (family,width),model in models.items():
                mm={d:evaluate(model,v,seeds,torch,device) for d,v in dev.items()}
                path=output/f'checkpoint_{family}_w{width}_step{step}.pt';torch.save(model.state_dict(),path)
                scores=[]
                for n,seed in enumerate(seeds):
                    domains={d:v[n] for d,v in mm.items()}
                    sc=max(base.score({d:domains[d][c] for d in domains},config['gates']) for c in CHANNELS);scores.append(sc)
                    ladder.append(dict(family=family,width=width,step=step,seed=seed,metrics=domains,score=sc))
                candidate=dict(family=family,width=width,step=step,score=max(scores),checkpoint=path.name)
                if family not in best or candidate['score']<best[family]['score']:best[family]=candidate
            write(output/'development_ladder.json',ladder)
            print(f'[GIADA Task22] checkpoint{step} elapsed={(time.monotonic()-start)/60:.1f}min',flush=True)
        if step==config['checkpoints'][-1]:break
        idx=torch.tensor(rng.integers(0,len(fit),config['batch_size']),device=device);batch=x[idx][None].expand(len(seeds),-1,-1)
        for k,model in models.items():
            optimizers[k].zero_grad(set_to_none=True);p,i,t=model(batch)
            loss=((p-ty[None,:,idx]).square().mean((1,2))+.1*(i-ti[None,:,idx]).square().mean((1,2))+.1*(t.log()-tt[None,:,idx].log()).square().mean((1,2)))
            loss.sum().backward();_clip_per_seed(model,torch,1.);optimizers[k].step()
    indexed={(r['family'],r['width'],r['step'],r['seed']):r for r in ladder}
    contrasts=[]
    for row in ladder:
        if row['family']=='independent':
            for family in ('shared_heads','conditioned'):
                other=indexed[family,row['width'],row['step'],row['seed']]
                contrasts.append(dict(family=family,width=row['width'],step=row['step'],seed=row['seed'],score_difference=other['score']-row['score'],
                    metric_differences={d:{c:{k:other['metrics'][d][c][k]-row['metrics'][d][c][k] for k in config['gates']} for c in CHANNELS} for d in dev}))
    write(output/'paired_architecture_contrasts.json',contrasts)
    freeze=dict(selection=best,fresh_accessed=False,config=config,code_revision=revision,checkpoint_hashes={p.name:sha(p) for p in output.glob('checkpoint_*.pt')})
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();write(output/'selection_freeze.json',freeze)
    fresh=dict(in_support=data(220401,4096),negative_tail=data(220404,4096,(-135,-115)),state_extrema=state_grid(True),ood_negative=data(220402,2048,(-155,-135)),ood_positive=data(220403,2048,(75,95)))
    v=np.random.default_rng(220405).uniform(-135,75,128);roll=np.c_[np.repeat(v,2),np.tile([0.,1.],128),np.ones(256)]
    np.savez_compressed(output/'fresh.npz',**fresh,rollout=roll)
    results=[]
    for family,selection in best.items():
        model=models[family,selection['width']];cp=output/selection['checkpoint'];assert sha(cp)==freeze['checkpoint_hashes'][cp.name]
        model.load_state_dict(torch.load(cp,map_location=device,weights_only=True));mm={d:evaluate(model,a,seeds,torch,device) for d,a in fresh.items()}
        swapped=evaluate(model,fresh['in_support'],seeds,torch,device,True);rr=rollout(model,roll,seeds,torch,device,config['rollout_horizons_steps'])
        for n,seed in enumerate(seeds):
            metrics_by_domain={d:rows[n] for d,rows in mm.items()}
            passed=all(base.passes(metrics_by_domain[d][c],config['gates']) for d in ('in_support','negative_tail','state_extrema') for c in CHANNELS) and all(base.passes(m,config['rollout_gates']) for c in CHANNELS for m in rr[n][c].values())
            results.append(dict(family=family,seed=seed,width=selection['width'],step=selection['step'],parameter_count=sum(p.numel() for p in model.parameters())//len(seeds),metrics=metrics_by_domain,rollout=rr[n],swapped_identity=swapped[n],passed=passed))
    passing={f:all(r['passed'] for r in results if r['family']==f) for f in FAMILIES}
    counts={f:next(r['parameter_count'] for r in results if r['family']==f) for f in FAMILIES}
    sharing={f:passing['independent'] and passing[f] and counts[f]<counts['independent'] for f in ('shared_heads','conditioned')}
    write(output/'fresh_metrics.json',results)
    report=dict(schema_version='giada-roadmap-task22-v1',valid=True,per_family_passed=passing,selected_parameter_counts=counts,sharing_with_fewer_parameters=sharing,
                sharing_confirmed=any(sharing.values()),task23_authorized=passing['independent'],selection=best,learned=results,fresh_used_for_selection=False,
                training_and_evaluation_seconds=time.monotonic()-start,environment=environment_manifest(torch),scope=config['limits'],full_neuron_speedup_claimed=False)
    write(output/'final_report.json',report);return report


def native_audit(teacher,output):
    # Native libraries are loaded in separate processes, never twice in one interpreter.
    root=Path(output)
    reports=[]
    for label,module in [('single','single_gate_transfer'),('slow','slow_gate_transfer')]:
        sub=root/('oracle_'+label);sub.mkdir()
        cmd=[sys.executable,'-c',f'from src.giada_teacher.{module} import native_audit; native_audit({str(teacher)!r},{str(sub)!r})']
        subprocess=__import__('subprocess');subprocess.run(cmd,check=True)
        reports.append(json.loads((sub/'native_audit.json').read_text()))
    report=dict(valid=all(r['valid'] for r in reports),reports=reports,scope='Canonical Ih/Im and Nap_Et2 h; independent native workers.')
    write(root/'native_audit.json',report);return report
