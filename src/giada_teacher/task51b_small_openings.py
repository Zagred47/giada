"""Static tiny-opening learnability, explicitly separated from temporal memory."""
import copy,gzip,hashlib,json,time,shutil
from pathlib import Path
import numpy as np
from . import task50_compact_memory as t
q=t.q;a=t.a

def plan(root):
    p=json.loads((root/'experiments/task51b_preregistration_v1.json').read_text());assert p['task']=='51b' and not p['thresholds_changed'];return p

def targets(torch,g):
    return torch.stack([g[...,int(lo)].pow(int(q.ionic.POWERS[k]))*(g[...,int(lo)+1] if hi-lo==2 else 1.) for k,(lo,hi) in enumerate(zip(q.ionic.OFFSETS[:-1],q.ionic.OFFSETS[1:]))],-1)

def inputs(torch,g):return g.clamp_min(1e-30).log10()/10

def model_class(torch):
    class Readout(torch.nn.Module):
        def __init__(self,p,fit_y,device):
            super().__init__();self.members=[dict(representation=r,loss=l,seed=s) for r in p['representations'] for l in p['losses'] for s in p['seeds']];E=len(self.members)
            def matrix(shape):return torch.nn.Parameter(torch.zeros(shape,device=device))
            self.w=matrix((E,18,32));self.b=matrix((E,1,32));self.o=matrix((E,32,11));self.skip=matrix((E,18,11));self.bias=matrix((E,1,11))
            mean=fit_y.clamp_min(1e-30).log10().mean(0);modes=[]
            with torch.no_grad():
                for i,m in enumerate(self.members):
                    gen=torch.Generator(device='cpu');gen.manual_seed(m['seed'])
                    for name in ('w','o','skip'):getattr(self,name)[i].copy_((torch.randn(getattr(self,name)[i].shape,generator=gen)*.02).to(device))
                    if m['representation']=='sigmoid_fit_bias':
                        opening=10**mean;self.bias[i,0].copy_(torch.logit(opening.clamp(1e-30,1-1e-7)))
                    elif m['representation']=='log_exp':self.bias[i,0].copy_(mean)
                    modes.append(m['representation']=='log_exp')
            self.register_buffer('log_mode',torch.tensor(modes,device=device)[:,None,None])
        def forward(self,x):
            raw=torch.bmm(torch.nn.functional.silu(torch.bmm(x,self.w)+self.b),self.o)+torch.bmm(x,self.skip)+self.bias
            return torch.where(self.log_mode,torch.pow(10.,raw.clamp_max(0)),torch.sigmoid(raw))
    return Readout

def dataset(torch,p,device):
    raw=t.dataset(torch,p,device);parts={}
    for role,info in p['roles'].items():
        ids=[i for i,m in enumerate(raw['meta']) if m['role']==role];g=raw['g'][::p['reachable_stride'],ids].reshape(-1,18);v=raw['v'][::p['reachable_stride'],ids].reshape(-1);clock=torch.arange(0,len(raw['g']),p['reachable_stride'],device=device).repeat_interleave(len(ids))*p['dt_ms']
        rng=np.random.default_rng(info['seed']+99);synthetic=torch.tensor(10**rng.uniform(-12,0,(p['synthetic_per_role'],18)),device=device,dtype=torch.float32)
        parts[role]=dict(g=torch.cat((g,synthetic)),v=torch.cat((v,torch.tensor(rng.uniform(-130,40,len(synthetic)),device=device,dtype=torch.float32))),reachable=len(g),clock=clock,histories=len(ids),ticks=len(raw['g'][::p['reachable_stride']]))
        parts[role]['y']=targets(torch,parts[role]['g'])
    return parts

def controls(torch,device):
    g=torch.tensor([[0.]*18,[1.]*18,[1e-12]*18,[.5]*18],device=device,dtype=torch.float64);y=targets(torch,g)
    exponents=torch.zeros((18,11),device=device,dtype=torch.float64)
    for k,(lo,hi) in enumerate(zip(q.ionic.OFFSETS[:-1],q.ionic.OFFSETS[1:])):
        exponents[int(lo),k]=q.ionic.POWERS[k]
        if hi-lo==2:exponents[int(lo)+1,k]=1
    roundtrip=10**(g[1:].log10()@exponents);assert torch.allclose(roundtrip,y[1:],rtol=1e-12,atol=0)
    assert torch.equal(y[0],torch.zeros(11,device=device,dtype=torch.float64)) and torch.equal(y[1],torch.ones(11,device=device,dtype=torch.float64))
    fp32=10**(g[1:].float().log10()@exponents.float());reference=y[1:].float()
    return dict(valid=True,zero_and_one_exact=True,log_product_roundtrip_max_relative=float(((roundtrip-y[1:]).abs()/y[1:]).max()),fp32_log_roundtrip_max_absolute=float((fp32-reference).abs().max()),fp32_underflow_count=int(((fp32==0)&(y[1:]>0)).sum()),scope='Analytic identity controls; neural exact-zero representability not asserted; FP32 underflow reported explicitly')

def train(torch,p,parts,out,device):
    fit=parts['fit'];dev=parts['development'];net=model_class(torch)(p,fit['y'],device);E=len(net.members);optim=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(951299);best=np.full(E,np.inf);selected=copy.deepcopy(net.state_dict());chosen=np.zeros(E,int);curve=[]
    # Balanced pools are fit-only: equal channel/decade categories, then sample a row.
    bins=torch.floor(fit['y'].clamp_min(1e-30).log10()).cpu().numpy();pools=[np.flatnonzero(bins[:,ch]==b) for ch in range(11) for b in np.unique(bins[:,ch])];started=time.monotonic()
    for update in range(1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.75:raise TimeoutError('51b training budget exceeded')
        uniform=rng.integers(0,len(fit['g']),p['batch_size']);balanced=np.array([rng.choice(pools[k]) for k in rng.integers(0,len(pools),p['batch_size'])]);index=torch.tensor(np.stack([balanced if m['loss']=='log_balanced' else uniform for m in net.members]),device=device)
        y=fit['y'][index];pred=net(inputs(torch,fit['g'][index]));absolute=(pred-y).square().mean((1,2));log=((pred+1e-30).log10()-(y+1e-30).log10()).square().mean((1,2))/900
        loss=torch.where(torch.tensor([m['loss']=='log_balanced' for m in net.members],device=device),log,absolute);optim.zero_grad(set_to_none=True);loss.sum().backward()
        norm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in net.parameters()).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in net.parameters():assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
        optim.step();assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
        if update%250==0:print(f'[GIADA51b] training {update}/{p["updates"]}',flush=True)
        if update in p['checkpoints']:
            with torch.no_grad():
                score=torch.zeros(E,device=device);abs_score=score.clone();n=0
                for start in range(0,len(dev['g']),512):
                    y=dev['y'][start:start+512];pr=net(inputs(torch,dev['g'][start:start+512])[None].expand(E,-1,-1));score+=((pr+1e-30).log10()-(y+1e-30).log10()).square().sum((1,2));abs_score+=(pr-y).square().sum((1,2));n+=y.numel()
                score/=n;abs_score/=n
            values=score.cpu().numpy();improved=values<best
            for key,v in net.state_dict().items():selected[key][improved]=v[improved].detach()
            best=np.minimum(best,values);chosen[improved]=update;curve.append(dict(update=update,development_log_mse=values.tolist(),development_absolute_mse=abs_score.cpu().tolist()));torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=optim.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve)
            public=Path('/kaggle/working')
            if public.is_dir():
                archive=Path(shutil.make_archive(str(public/'task51b_checkpoint_latest'),'zip',out))
                with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                a.n.p.save_json(public/'task51b_checkpoint_receipt.json',dict(valid=True,update=update,sha256=digest,bytes=archive.stat().st_size,partial=True))
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),development_log_mse=best.tolist(),fresh_accessed=False));return net

def evaluate(torch,p,net,parts,out,device):
    data=parts['fresh'];E=len(net.members);pred=[]
    for start in range(0,len(data['g']),512):pred.append(net(inputs(torch,data['g'][start:start+512])[None].expand(E,-1,-1)).cpu().numpy())
    pred=np.concatenate(pred,1);target=data['y'].cpu().numpy();voltage=data['v'].cpu().numpy();rows=[];floor=1e-7
    for i,member in enumerate(net.members):
        for group,selection in [('reachable',np.arange(data['reachable'])),('synthetic',np.arange(data['reachable'],len(target)))]:
            y=target[selection];pr=pred[i,selection];log=(np.log10(pr+1e-30)-np.log10(y+1e-30));rows.append(dict(**member,group=group,log_rmse=float(np.sqrt(np.mean(log**2))),opening_rmse=float(np.sqrt(np.mean((pr-y)**2))),max_opening_error=float(abs(pr-y).max())))
            for lo,hi in [(-100,-12),(-12,-8),(-8,-4),(-4,1)]:
                mask=(np.log10(y+1e-30)>=lo)&(np.log10(y+1e-30)<hi)
                if mask.any():rows.append(dict(**member,group=group,decade_bin=[lo,hi],count=int(mask.sum()),log_rmse=float(np.sqrt(np.mean(log[mask]**2))),opening_rmse=float(np.sqrt(np.mean((pr-y)[mask]**2)))))
        n=data['reachable'];shape=(data['ticks'],data['histories'],11);true=(target[:n]*np.array(q.ionic.GBAR)*(voltage[:n,None]-np.array(q.ionic.REVERSALS))).reshape(shape);got=(pred[i,:n]*np.array(q.ionic.GBAR)*(voltage[:n,None]-np.array(q.ionic.REVERSALS))).reshape(shape);scale=np.maximum(abs(true).max(0),floor);err=got-true;nr=np.sqrt(np.mean(err**2,axis=0))/scale;nm=abs(err).max(0)/scale
        rows.append(dict(**member,group='reachable_trajectory_current',nrmse=nr.tolist(),nmax=nm.tolist(),passed_amplitude_only=((nr<=.01)&(nm<=.05)).all(-1).tolist(),sample_interval_ms=p['dt_ms']*p['reachable_stride'],not_dense_temporal_or_charge_gate=True))
    with gzip.open(out/'evaluation.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
    np.savez_compressed(out/'fresh_predictions.npz',predicted=pred,target=target,voltage=voltage,reachable_count=data['reachable'])
    return rows

def benchmark(torch,p,net,parts,out,device):
    E=len(net.members);g=parts['fit']['g'][:p['cost']['batch']];x=inputs(torch,g)[None].expand(E,-1,-1);results={}
    for name,fn in [('packed_neural',lambda:net(x)),('analytic_product',lambda:targets(torch,g))]:
        fn();samples=[]
        for _ in range(p['cost']['repetitions']):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            for tick in range(100):fn()
            if device=='cuda':torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        results[name]=samples
    a.n.p.save_json(out/'cost.json',dict(samples_ms=results,ticks=100,packed_members=E,scope=p['cost']['scope'],not_matched_workload_speedup=True))

def run(torch,root,source,c,d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);started=time.monotonic();a.n.p.save_json(out/'preregistration.json',p);a.n.p.save_json(out/'controls.json',controls(torch,device));models,_,_=a.load_frozen(torch,root,source,c,d,out,device);before=a.n.weights_hash(models['P']);assert before==p['weights_sha256']
    with torch.no_grad():parts=dataset(torch,p,device)
    print('[GIADA51b] dataset ready; static readout training',flush=True);net=train(torch,p,parts,out,device)
    with torch.no_grad():
        rows=evaluate(torch,p,net,parts,out,device);benchmark(torch,p,net,parts,out,device)
    assert a.n.weights_hash(models['P'])==before;assert all(bool(torch.isfinite(v).all()) for v in net.parameters());assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='51b',P_frozen=True,finite_parameters=True,thresholds_changed=False,temporal_dynamics_tested=False,evaluation_rows=len(rows),elapsed_seconds=time.monotonic()-started,members=net.members,limitations=p['limitations']))
