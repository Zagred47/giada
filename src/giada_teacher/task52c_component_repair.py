"""Paired encoder x recurrence repair, conditional on one common decoder."""
import copy,gzip,hashlib,io,json,math,shutil,time,zipfile
from pathlib import Path
import numpy as np
from . import task52b_decoder_memory as b
t=b.t;q=b.q;a=b.a;parent=b.parent

def plan(root):
    p=json.loads((root/'experiments/task52c_preregistration_v1.json').read_text());assert not p['thresholds_changed'] and not p['abstraction_changed'];return p

def load_source(torch,package,device):
    with zipfile.ZipFile(package) as z:
        pp=json.loads(z.read('source52b/preregistration.json'));mean=torch.tensor(json.loads(z.read('source52b/fit_calibration.json'))['mean_opening'],device=device)
        state=torch.load(io.BytesIO(z.read('source52b/selected.pt')),map_location=device,weights_only=True)
    net=b.model_class(torch)(pp,mean,device);net.load_state_dict(state)
    return b.arm_view(torch,pp,net,mean,3,device),state

def calibration(torch,p,d,source):
    ids=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='fit'],device=d['v'].device)
    h=source.initialize(d['g'][0,ids][None].expand(3,-1,-1));hs=[];ys=[]
    with torch.no_grad():
        for k in range(len(d['v'])):
            if k%p['calibration_stride']==0:hs.append(h);ys.append(d['logop'][k,ids])
            h=source.step(t.features(torch,d['v'][k,ids],d['ca'][k,ids])[None].expand(3,-1,-1),h)
        hh=torch.stack(hs,1).flatten(1,2);yy=torch.cat(ys,0)
        logs=d['logop'][:,ids];low=logs.amin((0,1))-1;high=logs.amax((0,1)).clamp_max(0)
        return dict(h_mean=hh.mean(1,keepdim=True),h_std=hh.std(1,keepdim=True).clamp_min(p['feature_std_floor']),center=(low+high)/2,span=((high-low)/2).clamp_min(1),maxop=d['op'][:,ids].amax((0,1)).clamp_min(1e-8),h_fit=hh,y_fit=yy)

def model_class(torch):
    Base=t.model_class(torch)
    class Model(Base):
        def __init__(self,p,cal,source_state,device):
            super().__init__(p,device);E=len(self.members);ix=p['source_indices']*len(p['arms'])
            with torch.no_grad():
                for name,v in self.named_parameters():v.copy_(source_state[name][ix])
            for v in self.parameters():v.requires_grad_(False)
            for name in ('w','b','u','r','c','enc','eb'):getattr(self,name).requires_grad_(True)
            gen=torch.Generator();gen.manual_seed(9523001)
            # Identical decoder initialization per seed, across repair arms.
            rw=torch.randn((3,18,80),generator=gen)*.1;ro=torch.zeros(3,80,11)
            self.rw=torch.nn.Parameter(rw.repeat(len(p['arms']),1,1).to(device));self.rb=torch.nn.Parameter(torch.zeros(E,1,80,device=device))
            self.ro=torch.nn.Parameter(ro.repeat(len(p['arms']),1,1).to(device));self.skip=torch.nn.Parameter(torch.zeros(E,18,11,device=device));self.bias=torch.nn.Parameter(torch.zeros(E,1,11,device=device))
            wm=torch.zeros(E,18,80,device=device);om=torch.zeros(E,80,11,device=device);offset=0
            for f,channels in enumerate(p['families'].values()):
                width=p['arms'][0]['widths'][f];wm[:,offset:offset+width,f*16:(f+1)*16]=1;om[:,f*16:(f+1)*16,channels]=1;offset+=width
            for name,value in [('wm',wm),('rom',om),('hm',cal['h_mean'].repeat(len(p['arms']),1,1)),('hs',cal['h_std'].repeat(len(p['arms']),1,1)),('center',cal['center'][None,None].expand(E,1,-1)),('span',cal['span'][None,None].expand(E,1,-1))]:self.register_buffer(name,value.clone())
            self.register_buffer('encoder_mask',torch.tensor([arm['encoder_trainable'] for arm in p['arms'] for _ in p['seeds']],device=device))
            self.register_buffer('recurrence_mask',torch.tensor([arm['recurrence_trainable'] for arm in p['arms'] for _ in p['seeds']],device=device))
        def features(self,h):return (h-self.hm)/self.hs
        def normalized_raw(self,h):
            x=self.features(h);return torch.bmm(torch.nn.functional.silu(torch.bmm(x,self.rw*self.wm)+self.rb),self.ro*self.rom)+torch.bmm(x,self.skip*self.om)+self.bias
        def logopening(self,h):return (self.center+self.span*self.normalized_raw(h)).clamp_max(0)
        def opening(self,h):return 10**self.logopening(h)
        def mask_gradients(self):
            for name,v in self.named_parameters():
                if v.grad is None:continue
                mask=self.encoder_mask if name in ('enc','eb') else self.recurrence_mask if name in ('w','b','u','r','c') else None
                if mask is not None:v.grad.mul_(mask.reshape(-1,*([1]*(v.ndim-1))))
    return Model

def initialize_decoder(torch,p,net,cal):
    """Fit-only ridge head initialization, not a candidate selection."""
    E=len(net.members);h=cal['h_fit'].repeat(len(p['arms']),1,1)
    with torch.no_grad():
        x=net.features(h);hidden=torch.nn.functional.silu(torch.bmm(x,net.rw*net.wm)+net.rb)
        for e in range(min(3,E)):
            for k in range(11):
                hi=torch.nonzero(net.rom[e,:,k],as_tuple=True)[0];si=torch.nonzero(net.om[e,:,k],as_tuple=True)[0]
                X=torch.cat((hidden[e,:,hi],x[e,:,si],torch.ones((x.shape[1],1),device=x.device)),1).double()
                y=((cal['y_fit'][:,k]-net.center[e,0,k])/net.span[e,0,k]).double()
                reg=torch.eye(X.shape[1],device=x.device,dtype=torch.float64)*p['ridge']*len(X);reg[-1,-1]=1e-8
                coef=torch.linalg.solve(X.T@X+reg,X.T@y).float()
                net.ro[e,hi,k]=coef[:len(hi)];net.skip[e,si,k]=coef[len(hi):-1];net.bias[e,0,k]=coef[-1]
        for e in range(3,E):
            net.ro[e].copy_(net.ro[e%3]);net.skip[e].copy_(net.skip[e%3]);net.bias[e].copy_(net.bias[e%3])

def loss_step(torch,p,net,d,ids,starts,cal,backward=False):
    E=len(net.members);L=p['chunk_ticks'];h=net.initialize(d['g'][starts,ids][None].expand(E,-1,-1));indices=starts[None]+torch.arange(L,device=ids.device)[:,None]
    targets=d['op'][indices,ids[None]];current=t.current(torch,targets,d['v'][indices,ids[None]]);scale=current.abs().amax(0).clamp_min(1e-7)
    weight=torch.where(targets>=.2*cal['maxop'],5.,1.);weight=weight/weight.mean((0,1),keepdim=True)
    total=torch.zeros(E,device=ids.device);block=total.clone()
    for k in range(L):
        logs=net.logopening(h);op=10**logs;ll=((logs-d['logop'][starts+k,ids])/cal['span']).square()
        ci=((t.current(torch,op,d['v'][starts+k,ids][None])-current[k])/scale).square();am=((op-targets[k])/cal['maxop']).square()
        value=((ll+.25*ci+.25*am)*weight[k]).mean((1,2))/L
        if k==0:value=value+.25*ll.mean((1,2))
        total+=value.detach();block=block+value
        h=net.step(t.features(torch,d['v'][starts+k,ids],d['ca'][starts+k,ids])[None].expand(E,-1,-1),h)
        if backward and ((k+1)%p['bptt_ticks']==0 or k+1==L):block.sum().backward();h=h.detach();block=torch.zeros_like(total)
    return total

def train(torch,p,net,d,cal,out):
    device=d['v'].device;E=len(net.members);fit=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='fit'],device=device);dev=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='development'],device=device)
    pools=[torch.nonzero(d['op'][:,fit,k]>=.2*cal['maxop'][k],as_tuple=False).cpu().numpy() for k in range(11)]
    assert all(len(x) for x in pools);a.n.p.save_json(out/'support.json',dict(event_pool_counts=[len(x) for x in pools],fit_histories=len(fit),development_histories=len(dev),sampling=p['sampling']))
    original={k:v.detach().clone() for k,v in net.named_parameters() if k in ('w','b','u','r','c','enc','eb')}
    optimizer=torch.optim.Adam([v for v in net.parameters() if v.requires_grad],lr=p['learning_rate'],eps=p['adam_epsilon']);rng=np.random.default_rng(952399)
    best=np.full(E,np.inf);selected=copy.deepcopy(net.state_dict());chosen=np.zeros(E,int);curve=[];started=time.monotonic()
    for update in range(1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.58:raise TimeoutError('52c training budget; preserve checkpoints')
        episodes=rng.integers(0,len(fit),p['batch_size']);starts=rng.integers(0,len(d['v'])-p['chunk_ticks'],p['batch_size'])
        for j in range(p['batch_size']//2):
            pool=pools[rng.integers(11)];tick,episode=pool[rng.integers(len(pool))];episodes[j]=episode;starts[j]=np.clip(tick-p['chunk_ticks']//2,0,len(d['v'])-p['chunk_ticks']-1)
        ids=fit[torch.tensor(episodes,device=device)];starts=torch.tensor(starts,device=device)
        optimizer.zero_grad(set_to_none=True);loss=loss_step(torch,p,net,d,ids,starts,cal,True);net.mask_gradients()
        if update==1:a.n.p.save_json(out/'initial_gradients.json',{name:v.grad.reshape(E,-1).norm(dim=-1).cpu().tolist() for name,v in net.named_parameters() if v.grad is not None})
        pars=[v for v in net.parameters() if v.grad is not None];norm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in pars).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in pars:assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
        optimizer.step();assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
        if update%50==0:print(f'[GIADA52c] training {update}/{p["updates"]} loss={loss.mean().item():.6g} seconds={time.monotonic()-started:.1f}',flush=True)
        if update in p['checkpoints']:
            with torch.no_grad():scores=torch.stack([loss_step(torch,p,net,d,dev,torch.full_like(dev,min(round(ms/p['dt_ms']),len(d['v'])-p['chunk_ticks'])),cal) for ms in (0,150,275,650)]).mean(0)
            mask=scores.cpu().numpy()<best
            for k,v in net.state_dict().items():selected[k][mask]=v[mask].detach()
            best=np.minimum(best,scores.cpu().numpy());chosen[mask]=update;curve.append(dict(update=update,development=scores.cpu().tolist(),elapsed_seconds=time.monotonic()-started))
            for k,v in net.named_parameters():
                if k not in original:continue
                frozen=~(net.encoder_mask if k in ('enc','eb') else net.recurrence_mask);assert torch.equal(v[frozen],original[k][frozen]),'Frozen component changed'
            torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve);checkpoint(out,update)
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),fresh_accessed=False,frozen_components_verified=True))

def checkpoint(out,step):
    if Path('/kaggle/working').is_dir():
        path=Path(shutil.make_archive('/kaggle/working/task52c_checkpoint_latest','zip',out))
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        a.n.p.save_json(Path('/kaggle/working/task52c_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=path.stat().st_size,update=step,partial=True))

def arm_view(torch,p,net,cal,source,i,device):
    pp=copy.deepcopy(p);pp['arms']=[p['arms'][i]];view=model_class(torch)(pp,cal,source,device);view.load_state_dict({k:v[i*3:(i+1)*3].clone() for k,v in net.state_dict().items()})
    for v in view.parameters():v.requires_grad_(False)
    return view

def decoder_bounds(torch,net,out,label):
    """Conservative interval propagation, not proof that every output is reachable."""
    with torch.no_grad():
        lo=-net.hm/net.hs;hi=(1-net.hm)/net.hs;w=net.rw*net.wm
        zl=torch.bmm(lo,w.clamp_min(0))+torch.bmm(hi,w.clamp_max(0))+net.rb
        zh=torch.bmm(hi,w.clamp_min(0))+torch.bmm(lo,w.clamp_max(0))+net.rb
        silu=torch.nn.functional.silu
        low=torch.minimum(silu(zl),silu(zh));critical=(zl<=-1.2784645)&(zh>=-1.2784645);low=torch.where(critical,torch.full_like(low,-.278465),low)
        high=torch.maximum(silu(zl),silu(zh));o=net.ro*net.rom;s=net.skip*net.om
        rl=torch.bmm(low,o.clamp_min(0))+torch.bmm(high,o.clamp_max(0))+torch.bmm(lo,s.clamp_min(0))+torch.bmm(hi,s.clamp_max(0))+net.bias
        rh=torch.bmm(high,o.clamp_min(0))+torch.bmm(low,o.clamp_max(0))+torch.bmm(hi,s.clamp_min(0))+torch.bmm(lo,s.clamp_max(0))+net.bias
        a.n.p.save_json(out/(label+'_decoder_bounds.json'),dict(log_lower=(net.center+net.span*rl).clamp_max(0)[:,0].cpu().tolist(),log_upper=(net.center+net.span*rh).clamp_max(0)[:,0].cpu().tolist(),qualification='Conservative interval enclosure for latent box; inclusion is necessary, not sufficient, for representability.'))

def tiny_probe(torch,p,net,d,cal,out):
    """Independent throwaway optimization, never used to initialize/select main run."""
    probe=copy.deepcopy(net);fit=[i for i,m in enumerate(d['meta']) if m['role']=='fit'];ids=torch.tensor(fit[:2],device=d['v'].device);starts=torch.full_like(ids,min(1500,len(d['v'])-p['chunk_ticks']-1))
    opt=torch.optim.Adam([v for v in probe.parameters() if v.requires_grad],lr=p['learning_rate'],eps=p['adam_epsilon'])
    with torch.no_grad():first=loss_step(torch,p,probe,d,ids,starts,cal)
    for _ in range(p['probe_updates']):
        opt.zero_grad();loss_step(torch,p,probe,d,ids,starts,cal,True);probe.mask_gradients()
        pars=[v for v in probe.parameters() if v.grad is not None];E=len(probe.members);norm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in pars).sqrt();factor=(1/norm.clamp_min(1e-12)).clamp_max(1)
        for v in pars:v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
        opt.step()
    with torch.no_grad():last=loss_step(torch,p,probe,d,ids,starts,cal)
    a.n.p.save_json(out/'tiny_fit_probe.json',dict(initial=first.cpu().tolist(),final=last.cpu().tolist(),updates=p['probe_updates'],selection_eligible=False,main_initialization_unchanged=True,qualification='Small fixed-subset diagnostic; failure at this finite budget is not impossibility.'))

def collision_probe(torch,p,net,d,out):
    ids=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='fresh'][:4],device=d['v'].device);E=len(net.members);cut=min(1000,len(d['v'])-1)
    with torch.no_grad():
        h=net.initialize(d['g'][0,ids][None].expand(E,-1,-1))
        for k in range(cut):h=net.step(t.features(torch,d['v'][k,ids],d['ca'][k,ids])[None].expand(E,-1,-1),h)
        physical=d['g'][cut,ids].double();v=torch.full((len(ids),),-65.,device=h.device,dtype=torch.float64);ca=torch.full_like(v,1e-4);future=[]
        for _ in range(min(1000,len(d['v']))):
            future.append(t.current(torch,parent.opening(torch,physical),v));physical=q.canonical_step(torch,physical,v,ca,p['dt_ms'])
        future=torch.stack(future);scale=future.abs().amax((0,1)).clamp_min(1e-7);rows=[]
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                rows.append(dict(pair=[int(ids[i]),int(ids[j])],latent_distance=(h[:,i,:10]-h[:,j,:10]).norm(dim=-1).cpu().tolist(),initial_current_distance=float(((future[0,i]-future[0,j])/scale).square().mean().sqrt()),shared_future_current_distance=float(((future[:,i]-future[:,j])/scale).square().mean().sqrt())))
        a.n.p.save_json(out/'collision_probe.json',dict(rows=rows,cut_ms=cut*p['dt_ms'],same_future_voltage=-65,same_future_calcium=1e-4,qualification='Continuous distances only, not a new pass threshold; compare similar initial currents with divergent shared-input futures.'))

def diagnostics(torch,p,net,d,cal,out):
    from torch.utils.checkpoint import checkpoint as recompute
    E=len(net.members);device=d['v'].device;ids=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='fresh'][:p['oracle_initialization']['cases']],device=device)
    before=a.n.weights_hash(net)
    for v in net.parameters():v.requires_grad_(False)
    normal=net.initialize(d['g'][0,ids][None].expand(E,-1,-1)).detach();results=[]
    def objective(h,length):
        total=torch.zeros(E,device=device);block=p['oracle_initialization']['checkpoint_block']
        for start in range(0,length,block):
            end=min(start+block,length)
            def chunk(state,start=start,end=end):
                val=torch.zeros(E,device=device)
                for k in range(start,end):
                    val=val+((net.logopening(state)-d['logop'][k,ids])/cal['span']).square().mean((1,2))/length
                    state=net.step(t.features(torch,d['v'][k,ids],d['ca'][k,ids])[None].expand(E,-1,-1),state)
                return state,val
            h,value=recompute(chunk,h,use_reentrant=False) if h.requires_grad else chunk(h);total=total+value
        return total
    for ms in p['oracle_initialization']['fit_ms']:
        length=min(round(ms/p['dt_ms']),len(d['v']));best=objective(normal,length).detach();initial=best.clone()
        for seed in range(p['oracle_initialization']['starts']):
            gen=torch.Generator(device=device);gen.manual_seed(952370+seed)
            initial_logits=torch.logit(normal.clamp(1e-5,1-1e-5)) if seed==0 else torch.randn(normal.shape,generator=gen,device=device)
            logits=torch.nn.Parameter(initial_logits.clone());opt=torch.optim.Adam([logits],lr=p['oracle_initialization']['learning_rate'])
            for _ in range(p['oracle_initialization']['iterations']):
                opt.zero_grad();value=objective(torch.sigmoid(logits)*net.lm,length);value.sum().backward();opt.step();best=torch.minimum(best,value.detach())
        results.append(dict(horizon_ms=ms,normal_loss=initial.cpu().tolist(),oracle_best_loss=best.cpu().tolist(),selection_eligible=False))
        print('[GIADA52c] h0 diagnostic '+str(ms)+'ms',flush=True);a.n.p.save_json(out/'h0_diagnostics.json',results);checkpoint(out,p['updates'])
    with torch.no_grad():
        length=min(p['diagnostic_ticks'],len(d['v']));h=normal.clone();normal_sq=torch.zeros(E,device=device);encoded_sq=normal_sq.clone();latent=[];target=[]
        for k in range(length):
            expected=d['logop'][k,ids];normal_sq+=((net.logopening(h)-expected)/cal['span']).square().mean((1,2))/length
            enc=net.initialize(d['g'][k,ids][None].expand(E,-1,-1));encoded_sq+=((net.logopening(enc)-expected)/cal['span']).square().mean((1,2))/length
            if k%max(1,p['calibration_stride'])==0:latent.append(h.cpu().numpy());target.append(d['op'][k,ids].cpu().numpy())
            h=net.step(t.features(torch,d['v'][k,ids],d['ca'][k,ids])[None].expand(E,-1,-1),h)
        np.savez_compressed(out/'latent_probe.npz',latent=np.stack(latent),target=np.stack(target))
        # Full-state algebraic oracle verifies labels, not a candidate input.
        full=b.exact.log_targets(torch,d['g'][:length,ids].reshape(-1,18)).reshape(length,len(ids),11)
        oracle_error=float((full-d['logop'][:length,ids]).abs().max())
        result=dict(normal_log_loss=normal_sq.cpu().tolist(),teacher_reencode_log_loss=encoded_sq.cpu().tolist(),full_state_log_oracle_max_error=oracle_error,selection_eligible=False,weights_unchanged=a.n.weights_hash(net)==before)
    assert result['weights_unchanged'];a.n.p.save_json(out/'representation_diagnostics.json',result);collision_probe(torch,p,net,d,out)

def run(torch,root,source,c,d,package,out,device='cuda',override=None,coupled_override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);started=time.monotonic();a.n.p.save_json(out/'preregistration.json',p)
    models=parent.load(torch,root,package,source,c,d,out,device);old,state=load_source(torch,package,device);models['compact']=old
    before={k:a.n.weights_hash(v) for k,v in models.items()}
    with torch.no_grad():data=b.data(torch,p,device);cal=calibration(torch,p,data,old)
    torch.save({k:v for k,v in cal.items() if k not in ('h_fit','y_fit')},out/'calibration.pt')
    a.n.p.save_json(out/'calibration.json',{k:v.cpu().tolist() for k,v in cal.items() if k not in ('h_fit','y_fit')});a.n.p.save_json(out/'history_metadata.json',data['meta'])
    net=model_class(torch)(p,cal,state,device);initialize_decoder(torch,p,net,cal);torch.save(net.state_dict(),out/'initial.pt');decoder_bounds(torch,net,out,'initial');tiny_probe(torch,p,net,data,cal,out)
    train(torch,p,net,data,cal,out);frozen=a.n.weights_hash(net);decoder_bounds(torch,net,out,'selected');diagnostics(torch,p,net,data,cal,out)
    pp=parent.plan(root) if coupled_override is None else copy.deepcopy(coupled_override);pp['cost']=p['cost'];pp['runtime_budget_seconds']=p['runtime_budget_seconds']
    a.n.p.save_json(out/'unchanged_quality.json',dict(quality=pp['quality'],compartment_gates=pp['compartment_gates']));cases,inj,mult,rev=parent.design(pp);a.n.p.save_json(out/'cases.json',cases)
    inj=torch.tensor(inj,device=device,dtype=torch.float64);mult=torch.tensor(mult,device=device,dtype=torch.float64);rev=torch.tensor(rev,device=device,dtype=torch.float64);physics=a.n.contract(root)[1:];summaries={}
    with torch.no_grad():
        ref=parent.reference(torch,pp,inj,mult,rev,physics,device);np.savez_compressed(out/'reference.npz',**{k:v.cpu().numpy() for k,v in ref.items()})
        zero_v=torch.full_like(ref['voltage'][0],-65.);zero_ca=torch.full_like(zero_v,1e-4);zv=[zero_v];zc=[zero_ca]
        for k in range(len(inj)):zero_v,zero_ca,_=parent.physical(torch,zero_v,zero_ca,torch.zeros_like(ref['current'][0]),inj[k],rev,physics,p['dt_ms']);zv.append(zero_v);zc.append(zero_ca)
        passive=dict(voltage=torch.stack(zv)[:,:,None].cpu().numpy(),calcium=torch.stack(zc)[:,:,None].cpu().numpy(),current=np.zeros((len(inj),3,1,len(cases),11)))
        target=out/'passive';target.mkdir();pv=copy.deepcopy(pp);pv['arms']=['oracle_passive'];summaries['passive'],_=parent.evaluate(pv,passive,ref,cases,target)
        for label,compact in [('frozen52b',old)]+[(arm['id'],arm_view(torch,p,net,cal,state,i,device)) for i,arm in enumerate(p['arms'])]:
            target=out/label;target.mkdir();local=copy.deepcopy(pp);local['arms']=['canonical64','canonical32','P','P_J','compact','compact_imposed','compact_ca_imposed','compact_reset','oracle_all'] if label=='frozen52b' else ['compact','compact_imposed','compact_ca_imposed','compact_reset','oracle_all']
            current=dict(models,compact=compact);trace=parent.simulate(torch,local,current,inj,mult,rev,physics,ref,device,target);np.savez_compressed(target/'traces.npz',**trace)
            summary,rows=parent.evaluate(local,trace,ref,cases,target);summaries[label]=summary
            if label=='frozen52b':assert all(r['passed'] for r in rows if r['arm']=='canonical64')
            parent.benchmark(torch,local,current,device,target);checkpoint(out,p['updates'])
    assert before=={k:a.n.weights_hash(v) for k,v in models.items()};assert frozen==a.n.weights_hash(net)
    passed=[]
    for arm in p['arms']:
        with gzip.open(out/arm['id']/'interventions.json.gz','rt') as f:response=[r for r in json.load(f) if r['arm']=='compact']
        passed.append(all(row['all_quality_passed'] for row in summaries[arm['id']]['compact'].values()) and all(r['current']['conservative_passed'] and r['voltage_response_rmse_mv']<=pp['compartment_gates']['max_candidate_worst_voltage_rmse_mv'] and r['calcium_response_rmse_mM']<=pp['compartment_gates']['max_candidate_worst_cai_rmse_mM'] for r in response))
    a.n.p.save_json(out/'final_report.json',dict(valid=True,execution_valid=True,scientific_acceptance_passed=any(passed),promotion_authorized=False,complete=True,task='52c',training_updates=p['updates'],primary_training_updates=p['updates'],diagnostic_tiny_subset_updates=p['probe_updates'],decoder_ridge_initializations=3,selection_recomputed=True,thresholds_changed=False,abstraction_changed=False,logical_state=10,finite_parameters=all(bool(torch.isfinite(v).all()) for v in net.parameters()),postfreeze_weights_unchanged=True,frozen_controls_unchanged=True,summary=summaries,elapsed_seconds=time.monotonic()-started,limitations=p['limitations']))
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2;checkpoint(out,p['updates'])
