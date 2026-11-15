"""Packed independent family-latent learners with shared paired minibatches."""
import copy,gzip,json,time
import numpy as np
from . import task49_causal_initialization as prev
q=prev.q;a=q.a

def plan(root):
    p=json.loads((root/'experiments/task50_preregistration_v1.json').read_text());assert p['task']=='50' and not p['thresholds_changed'];assert p['weights_sha256']==q.plan(root)['weights_sha256'];return p

def model_class(torch):
    class Packed(torch.nn.Module):
        def __init__(self,p,device):
            super().__init__();self.members=[dict(arm=arm['id'],seed=seed,widths=arm['widths'],trunk=arm['trunk']) for arm in p['arms'] for seed in p['seeds']];E=len(self.members);L=18;T=32
            generator=torch.Generator(device='cpu');generator.manual_seed(950001)
            def param(shape,scale=.1):return torch.nn.Parameter((torch.randn(shape,generator=generator)*scale).to(device))
            self.w=param((E,2,T));self.b=param((E,1,T),0);self.u=param((E,T,2*L));self.r=param((E,L,2*L));self.c=param((E,1,2*L),0);self.enc=param((E,18,L));self.eb=param((E,1,L),0);self.out=param((E,L,11));self.ob=param((E,1,11),0)
            lm=torch.zeros(E,1,L);tm=torch.zeros(E,1,T);rm=torch.zeros(E,L,2*L);em=torch.zeros(E,18,L);om=torch.zeros(E,L,11)
            gates=q.family_indices(p)
            for i,m in enumerate(self.members):
                lm[i,0,:sum(m['widths'])]=1;tm[i,0,:m['trunk']]=1;offset=0
                for (family,channels),width in zip(p['families'].items(),m['widths']):
                    sl=slice(offset,offset+width);rm[i,sl,sl]=1;rm[i,sl,slice(L+offset,L+offset+width)]=1;em[i,gates[family],sl]=1;om[i,sl,channels]=1;offset+=width
            for name,value in [('lm',lm),('tm',tm),('rm',rm),('em',em),('om',om)]:self.register_buffer(name,value.to(device))
        def initialize(self,g):return torch.sigmoid(torch.bmm(g,self.enc*self.em)+self.eb)*self.lm
        def opening(self,h):return torch.sigmoid(torch.bmm(h,self.out*self.om)+self.ob)
        def step(self,x,h):
            trunk=torch.nn.functional.silu(torch.bmm(x,self.w)+self.b)*self.tm
            raw=torch.bmm(trunk,self.u)+torch.bmm(h,self.r*self.rm)+self.c
            proposal,rate=raw.chunk(2,-1);return (h+torch.sigmoid(rate)*(torch.sigmoid(proposal)-h))*self.lm
    return Packed

def features(torch,v,ca):return torch.stack((v/100.,(ca.clamp_min(1e-9).log10()+4.5)/2.5),-1)
def current(torch,opening,v):
    bar=torch.tensor(q.ionic.GBAR,device=v.device,dtype=v.dtype);rev=torch.tensor(q.ionic.REVERSALS,device=v.device,dtype=v.dtype)
    return opening*bar*(v[...,None]-rev)

def dataset(torch,p,device):
    vh,ch,meta=prev.histories(p);g=torch.tensor(np.tile(q.steady(),(len(meta),1)),device=device,dtype=torch.float64);states=[];ops=[]
    for t in range(len(vh)):
        states.append(g.float());ops.append(torch.stack([g[:,int(lo)].pow(int(q.ionic.POWERS[k]))*(g[:,int(lo)+1] if hi-lo==2 else 1.) for k,(lo,hi) in enumerate(zip(q.ionic.OFFSETS[:-1],q.ionic.OFFSETS[1:]))],-1).float())
        v=torch.tensor(vh[t],device=device);ca=torch.tensor(ch[t],device=device);g=q.canonical_step(torch,g,v,ca,p['dt_ms'])
    return dict(v=torch.tensor(vh,device=device,dtype=torch.float32),ca=torch.tensor(ch,device=device,dtype=torch.float32),g=torch.stack(states),op=torch.stack(ops),meta=meta)

def rollout_loss(torch,net,data,ids,starts,length,scale):
    E=len(net.members);g=data['g'][starts,ids][None].expand(E,-1,-1);h=net.initialize(g);loss=torch.zeros(E,device=g.device)
    for k in range(length):
        v=data['v'][starts+k,ids];ca=data['ca'][starts+k,ids];target=data['op'][starts+k,ids];op=net.opening(h)
        delta=(current(torch,op,v[None])-current(torch,target,v))/scale
        loss+=delta.square().mean((1,2))+.1*(op-target).square().mean((1,2))
        h=net.step(features(torch,v,ca)[None].expand(E,-1,-1),h)
    return loss/length

def train(torch,p,data,out,device):
    net=model_class(torch)(p,device);E=len(net.members);fit=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='fit'],device=device);dev=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='development'],device=device)
    scale=current(torch,data['op'][:,fit],data['v'][:,fit]).abs().amax((0,1)).clamp_min(1e-7);optimizer=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(950019);best=np.full(E,np.inf);selected=copy.deepcopy(net.state_dict());chosen=np.zeros(E,dtype=int);curve=[]
    started=time.monotonic()
    for update in range(1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.75:raise TimeoutError('Training budget exhausted; retained checkpoints, no incomplete selection claimed')
        ids=fit[torch.tensor(rng.integers(0,len(fit),p['batch_size']),device=device)];starts=torch.tensor(rng.integers(0,len(data['v'])-p['chunk_ticks'],p['batch_size']),device=device)
        optimizer.zero_grad(set_to_none=True);loss=rollout_loss(torch,net,data,ids,starts,p['chunk_ticks'],scale);loss.sum().backward()
        norms=torch.zeros(E,device=device)
        for par in net.parameters():norms+=par.grad.reshape(E,-1).square().sum(-1)
        factors=(p['gradient_clip']/norms.sqrt().clamp_min(1e-12)).clamp_max(1)
        for par in net.parameters():par.grad.mul_(factors.reshape(E,*([1]*(par.ndim-1))))
        optimizer.step();assert all(bool(torch.isfinite(par).all()) for par in net.parameters())
        if update%100==0:print(f'[GIADA50] training {update}/{p["updates"]} loss={loss.detach().mean().item():.6g}',flush=True)
        if update in p['checkpoints']:
            with torch.no_grad():scores=sum(rollout_loss(torch,net,data,dev,torch.full_like(dev,start),min(256,len(data['v'])-start),scale) for start in (0,1500,2750) if start<len(data['v']))/sum(start<len(data['v']) for start in (0,1500,2750))
            values=scores.cpu().numpy();improved=values<best
            for key,value in net.state_dict().items():selected[key][improved]=value[improved].detach()
            best=np.minimum(best,values);chosen[improved]=update;curve.append(dict(update=update,development=values.tolist()))
            torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');a.n.p.save_json(out/'learning_curve.json',curve)
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),development=best.tolist(),fresh_accessed=False));return net

def evaluate(torch,p,net,P,data,out,device):
    ids=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='fresh'],device=device);E=len(net.members);rows=[];quality=a.spec(__import__('pathlib').Path(__file__).resolve().parents[2])[0]['quality']
    for cut in p['evaluation_cuts_ms']:
        initial=data['g'][round(cut/p['dt_ms'])-1,ids].double()
        # data g[t] is state BEFORE sample t; reconstruct exact state at cut.
        t=round(cut/p['dt_ms'])-1;initial=q.canonical_step(torch,initial,data['v'][t,ids].double(),data['ca'][t,ids].double(),p['dt_ms'])
        for future in p['future_protocols']:
            h=net.initialize(initial.float()[None].expand(E,-1,-1));reference=initial.clone();pg=initial.float()[None].expand(3,-1,-1).clone();scale=torch.zeros((len(ids),11),device=device,dtype=torch.float64)
            sq=torch.zeros((E+3,len(ids),11),device=device,dtype=torch.float64);mx=sq.clone();signed=sq.clone();absolute=sq.clone();reset_sq=torch.zeros((E,len(ids),11),device=device,dtype=torch.float64);hr=h.clone()
            for tick in range(round(max(p['horizons_ms'])/p['dt_ms'])+1):
                vv,cc=q.future_input(future,tick*p['dt_ms']);v=torch.full((len(ids),),vv,device=device);ca=torch.full_like(v,cc)
                ref=a.t.readout(torch,v.double(),reference,torch.ones((len(ids),11),device=device,dtype=torch.float64))[1]
                got=current(torch,net.opening(h),v[None]);pc=a.t.readout(torch,v[None].expand(3,-1),pg,torch.ones((3,len(ids),11),device=device))[1];allgot=torch.cat((got,pc)).double();err=allgot-ref
                scale=torch.maximum(scale,ref.abs());sq+=err.square();mx=torch.maximum(mx,err.abs());signed+=err;absolute+=allgot.abs()-ref.abs()
                reset_sq+=(current(torch,net.opening(hr),v[None]).double()-ref).square()
                if tick in [round(x/p['dt_ms']) for x in p['horizons_ms']]:
                    den=scale.clamp_min(1e-7);nrmse=(sq/(tick+1)).sqrt()/den;nmax=mx/den;charge=signed.abs()/((tick+1)*den);acharge=absolute.abs()/((tick+1)*den)
                    for j in range(E+3):
                        passed=(nrmse[j]<=quality['current_nrmse_max'])&(nmax[j]<=quality['current_nmax_max'])&(charge[j]<=quality['signed_charge_error_max'])&(acharge[j]<=quality['absolute_charge_error_max'])
                        member=net.members[j] if j<E else dict(arm='P',seed=p['seeds'][j-E])
                        rows.append(dict(**member,cut_ms=cut,future=future,horizon_ms=tick*p['dt_ms'],passed_by_history=passed.all(-1).cpu().tolist(),nrmse=nrmse[j].cpu().tolist(),nmax=nmax[j].cpu().tolist(),charge=charge[j].cpu().tolist(),absolute_charge=acharge[j].cpu().tolist(),reset_nrmse=(reset_sq[j]/(tick+1)).sqrt().div(den).cpu().tolist() if j<E else None))
                reference=q.canonical_step(torch,reference,v.double(),ca.double(),p['dt_ms']);pg=P(v[None].expand(3,-1),ca[None].expand(3,-1),pg)
                x=features(torch,v,ca)[None].expand(E,-1,-1);h=net.step(x,h);hr=net.step(x,hr)
                if tick==round(100/p['dt_ms']) and future=='pulse_recovery':hr=torch.zeros_like(hr)
            print(f'[GIADA50] evaluated cut={cut} future={future}',flush=True)
            with gzip.open(out/'evaluation.json.gz','wt',encoding='utf-8') as f:json.dump(rows,f,allow_nan=False)
    return rows

def benchmark(torch,p,net,P,out,device):
    n=p['cost']['batch'];E=len(net.members);v=torch.full((n,),-65.,device=device);ca=torch.full_like(v,5e-5);g=torch.tensor(np.tile(q.steady(),(n,1)),device=device,dtype=torch.float32);h=net.initialize(g[None].expand(E,-1,-1));x=features(torch,v,ca)[None].expand(E,-1,-1)
    # Packed latency is for all candidates at once; never label it per-model latency.
    functions={'packed_all_candidates':lambda:current(torch,net.opening(net.step(x,h)),v[None]),'P_three_seeds':lambda:a.t.readout(torch,v[None].expand(3,-1),P(v[None].expand(3,-1),ca[None].expand(3,-1),g[None].expand(3,-1,-1)),torch.ones((3,n,11),device=device))[1],'canonical_three_replicas':lambda:a.t.readout(torch,v[None].expand(3,-1).double(),q.canonical_step(torch,g[None].expand(3,-1,-1).double(),v[None].expand(3,-1).double(),ca[None].expand(3,-1).double()),torch.ones((3,n,11),device=device,dtype=torch.float64))[1]}
    samples={}
    for key,fn in functions.items():
        fn();values=[]
        for _ in range(p['cost']['repetitions']):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            for tick in range(p['cost']['ticks']):fn()
            if device=='cuda':torch.cuda.synchronize()
            values.append((time.perf_counter()-start)*1000)
        samples[key]=values
    a.n.p.save_json(out/'cost.json',dict(samples_ms=samples,packed_candidate_count=E,P_seed_count=3,not_per_candidate_speedup=True,scope=p['cost']['scope']))

def run(torch,root,source,c,d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);start=time.monotonic();a.n.p.save_json(out/'preregistration.json',p)
    models,_,_=a.load_frozen(torch,root,source,c,d,out,device);P=models['P'];before=a.n.weights_hash(P)
    with torch.no_grad():data=dataset(torch,p,device)
    net=train(torch,p,data,out,device)
    with torch.no_grad():
        rows=evaluate(torch,p,net,P,data,out,device);benchmark(torch,p,net,P,out,device)
    assert a.n.weights_hash(P)==before==p['weights_sha256'];a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='50',P_frozen=True,thresholds_changed=False,members=net.members,evaluation_rows=len(rows),elapsed_seconds=time.monotonic()-start,limitations=p['limitations'],speedup_claimed=False))
