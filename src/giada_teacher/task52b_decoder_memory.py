import copy,gzip,hashlib,json,math,shutil,time
from pathlib import Path
import numpy as np
from . import task50_compact_memory as t
from . import task52_closed_loop as parent
from . import task51f_local_targets as exact
q=t.q;a=t.a
def plan(root):
    p=json.loads((root/'experiments/task52b_preregistration_v1.json').read_text());assert not p['thresholds_changed'] and not p['abstraction_changed'];return p
def model_class(torch):
    Base=t.model_class(torch)
    class Model(Base):
        def __init__(self,p,mean,device):
            super().__init__(p,device)
            for i,m in enumerate(self.members):
                arm=p['arms'][i//len(p['seeds'])];m.update(decoder=arm['decoder'],memory=arm['memory'])
                gen=torch.Generator();gen.manual_seed(m['seed'])
                with torch.no_grad():
                    for name,v in self.named_parameters():v[i].copy_((torch.randn(v[i].shape,generator=gen)*.05).to(device) if name not in ('b','c','eb','ob') else torch.zeros_like(v[i]))
                    self.u[i,:,18:]=0;self.r[i,:,18:]=0;self.out[i]=0
                    mu=mean.clamp(1e-12,1-1e-6)
                    self.ob[i,0]=torch.logit(mu) if m['decoder']=='sigmoid' else mu.log10()
                    if m['memory']=='multiscale':
                        tau=torch.tensor(p['initial_tau_ms'],device=device);z=-torch.expm1(-p['dt_ms']/tau)
                        self.c[i,0,18:28]=torch.logit(z)
            self.register_buffer('log_decoder',torch.tensor([m['decoder']=='log' for m in self.members],device=device)[:,None,None])
        def logopening(self,h):
            raw=torch.bmm(h,self.out*self.om)+self.ob
            return torch.where(self.log_decoder,raw.clamp_max(0),torch.nn.functional.logsigmoid(raw)/math.log(10))
        def opening(self,h):return 10**self.logopening(h)
    return Model
def data(torch,p,device):
    d=t.dataset(torch,p,device);shape=d['g'].shape
    d['logop']=exact.log_targets(torch,d['g'].reshape(-1,18)).reshape(*shape[:-1],11).float()
    return d
def loss_step(torch,net,d,ids,starts,length,p,backward=False):
    E=len(net.members);h=net.initialize(d['g'][starts,ids][None].expand(E,-1,-1))
    index=starts[None]+torch.arange(length,device=ids.device)[:,None]
    target_current=t.current(torch,d['op'][index,ids[None]],d['v'][index,ids[None]])
    scale=target_current.abs().amax(0).clamp_min(1e-7)
    total=torch.zeros(E,device=ids.device);block=torch.zeros_like(total)
    for k in range(length):
        target=d['op'][starts+k,ids];logs=net.logopening(h);op=10**logs
        diff=t.current(torch,op,d['v'][starts+k,ids][None])-target_current[k]
        ll=(logs-d['logop'][starts+k,ids]).square().mean((1,2))/900
        value=torch.asinh(diff/scale).square().mean((1,2))+.1*ll+.1*(op-target).square().mean((1,2))
        block=block+value/length+(4*ll if k==0 else 0);total+=value.detach()/length+(4*ll.detach() if k==0 else 0)
        h=net.step(t.features(torch,d['v'][starts+k,ids],d['ca'][starts+k,ids])[None].expand(E,-1,-1),h)
        if backward and ((k+1)%p['bptt_ticks']==0 or k+1==length):
            block.sum().backward();h=h.detach();block=torch.zeros_like(total)
    return total
def train(torch,p,d,out,device):
    fit=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='fit'],device=device);dev=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='development'],device=device)
    mean=d['op'][:,fit].mean((0,1));net=model_class(torch)(p,mean,device);E=len(net.members)
    opt=torch.optim.Adam(net.parameters(),lr=p['learning_rate'],eps=p['adam_epsilon']);rng=np.random.default_rng(952299)
    best=np.full(E,np.inf);chosen=np.zeros(E,int);selected=copy.deepcopy(net.state_dict());curve=[];start=time.monotonic()
    a.n.p.save_json(out/'fit_calibration.json',dict(mean_opening=mean.cpu().tolist(),fit_only=True))
    for update in range(1,p['updates']+1):
        if time.monotonic()-start>p['runtime_budget_seconds']*.65:raise TimeoutError('52b training budget; checkpoints retained')
        ids=fit[torch.tensor(rng.integers(0,len(fit),p['batch_size']),device=device)];starts=torch.tensor(rng.integers(0,len(d['v'])-p['chunk_ticks'],p['batch_size']),device=device)
        opt.zero_grad(set_to_none=True);loss=loss_step(torch,net,d,ids,starts,p['chunk_ticks'],p,True)
        norm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in net.parameters()).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in net.parameters():assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
        opt.step();assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
        if update%20==0:
            print(f'[GIADA52b] training {update}/{p["updates"]} loss={loss.mean().item():.6g}',flush=True)
            a.n.p.save_json(out/'training_progress.json',dict(recorded_completed_updates=update,finite_parameters=True,elapsed_seconds=time.monotonic()-start))
        if update in p['checkpoints']:
            with torch.no_grad():
                windows=[min(round(ms/p['dt_ms']),len(d['v'])-p['chunk_ticks']) for ms in (0,150,275,650)]
                score=torch.stack([loss_step(torch,net,d,dev,torch.full_like(dev,k),p['chunk_ticks'],p) for k in windows]).mean(0)
            values=score.cpu().numpy();mask=values<best
            for key,value in net.state_dict().items():selected[key][mask]=value[mask].detach()
            best=np.minimum(best,values);chosen[mask]=update;curve.append(dict(update=update,development=values.tolist(),elapsed_seconds=time.monotonic()-start))
            torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve)
            checkpoint(out,update)
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),fresh_accessed=False));return net,mean
def checkpoint(out,step):
    if Path('/kaggle/working').is_dir():
        path=Path(shutil.make_archive('/kaggle/working/task52b_checkpoint_latest','zip',out))
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        a.n.p.save_json(Path('/kaggle/working/task52b_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=path.stat().st_size,update=step,partial=True))

def initialization_diagnostics(torch,p,net,d,out):
    """Post-freeze oracles: change only h0, never use them for selection."""
    E=len(net.members);device=next(net.parameters()).device
    ids=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='fresh'][:p['oracle_initialization']['cases']],device=device)
    length=min(len(d['v']),p.get('diagnostic_ticks',1000));window=min(length,p['oracle_initialization']['window_ticks'])
    before=a.n.weights_hash(net)
    for par in net.parameters():par.requires_grad_(False)
    with torch.no_grad():normal=net.initialize(d['g'][0,ids][None].expand(E,-1,-1))
    latent=torch.nn.Parameter(torch.logit(normal.clamp(1e-6,1-1e-6)))
    optimizer=torch.optim.Adam([latent],lr=p['oracle_initialization']['learning_rate'])
    def score_initial():
        h=torch.sigmoid(latent)*net.lm;value=0
        for k in range(window):
            value=value+(net.logopening(h)-d['logop'][k,ids]).square().mean()/window
            h=net.step(t.features(torch,d['v'][k,ids],d['ca'][k,ids])[None].expand(E,-1,-1),h)
        return value
    # Keep the best h0 on the fitting window, including the normal initial state.
    best_latent=latent.detach().clone();best=float(score_initial().detach())
    for _ in range(p['oracle_initialization']['iterations']):
        optimizer.zero_grad();loss=score_initial();loss.backward();optimizer.step()
        value=float(score_initial().detach())
        if value<best:best=value;best_latent=latent.detach().clone()
    result={};traces={}
    with torch.no_grad():
        for mode in ('normal','future_h0_oracle','teacher_reencode'):
            h=normal.clone() if mode!='future_h0_oracle' else torch.sigmoid(best_latent)*net.lm
            got=[]
            for k in range(length):
                if mode=='teacher_reencode':h=net.initialize(d['g'][k,ids][None].expand(E,-1,-1))
                got.append(net.opening(h));h=net.step(t.features(torch,d['v'][k,ids],d['ca'][k,ids])[None].expand(E,-1,-1),h)
            pred=torch.stack(got);ref=d['op'][:length,ids];voltage=d['v'][:length,ids]
            err=t.current(torch,pred,voltage[:,None])-t.current(torch,ref,voltage)[:,None]
            den=t.current(torch,ref,voltage).abs().amax(0).clamp_min(1e-7)
            result[mode]=dict(nrmse=(err.square().mean(0).sqrt()/den).cpu().tolist(),nmax=(err.abs().amax(0)/den).cpu().tolist(),selection_eligible=mode=='normal')
            traces[mode]=pred.cpu().numpy()
        weights=net.out*net.om;lo=net.ob+weights.clamp_max(0).sum(1,keepdim=True);hi=net.ob+weights.clamp_min(0).sum(1,keepdim=True)
        lower=torch.where(net.log_decoder,10**lo.clamp_max(0),torch.sigmoid(lo));upper=torch.where(net.log_decoder,10**hi.clamp_max(0),torch.sigmoid(hi))
        result['decoder_box']=dict(lower=lower[:,0].cpu().tolist(),upper=upper[:,0].cpu().tolist(),qualification='Bounds for frozen weights and latent box only; not architecture impossibility.')
        raw=torch.bmm(t.features(torch,d['v'][0,ids],d['ca'][0,ids])[None].expand(E,-1,-1),net.w)+net.b
        raw=torch.bmm(torch.nn.functional.silu(raw)*net.tm,net.u)+torch.bmm(normal,net.r*net.rm)+net.c
        result['direct_carry']=dict(value=(1-torch.sigmoid(raw[...,18:28])).cpu().tolist(),qualification='Direct carry excludes proposal/rate state derivatives; not total memory.')
    # Full local Jacobians include both proposal and update-gate dependence.
    jacobians=[]
    for i in range(E):
        h=normal[:,0:1].detach().clone().requires_grad_(True)
        x=t.features(torch,d['v'][0,ids[:1]],d['ca'][0,ids[:1]])[None].expand(E,-1,-1)
        y=net.step(x,h);rows=[]
        for j in range(10):rows.append(torch.autograd.grad(y[i,0,j],h,retain_graph=True)[0][i,0,:10])
        matrix=torch.stack(rows);jacobians.append(dict(max_singular=float(torch.linalg.svdvals(matrix)[0]),spectral_radius=float(torch.linalg.eigvals(matrix).abs().max())))
    result.update(local_jacobians=jacobians,members=net.members,length_ticks=length,oracle_fit_ticks=window,oracle_best_fit_loss=best,fresh_used_for_selection=False,weights_unchanged=a.n.weights_hash(net)==before)
    assert result['weights_unchanged']
    np.savez_compressed(out/'initialization_diagnostics.npz',**traces,target=d['op'][:length,ids].cpu().numpy())
    a.n.p.save_json(out/'initialization_diagnostics.json',result)
    return result

def arm_view(torch,p,net,mean,index,device):
    pp=copy.deepcopy(p);pp['arms']=[p['arms'][index]];view=model_class(torch)(pp,mean,device)
    n=len(p['seeds']);view.load_state_dict({k:v[index*n:(index+1)*n].clone() for k,v in net.state_dict().items()})
    view.eval()
    for par in view.parameters():par.requires_grad_(False)
    return view

def run(torch,root,source,c,d,package,out,device='cuda',override=None,coupled_override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);started=time.monotonic()
    a.n.p.save_json(out/'preregistration.json',p)
    models=parent.load(torch,root,package,source,c,d,out,device);frozen_before={k:a.n.weights_hash(v) for k,v in models.items()}
    with torch.no_grad():dataset=data(torch,p,device)
    a.n.p.save_json(out/'history_metadata.json',dataset['meta'])
    net,mean=train(torch,p,dataset,out,device);freeze_hash=a.n.weights_hash(net)
    diagnostics=initialization_diagnostics(torch,p,net,dataset,out)
    pp=parent.plan(root) if coupled_override is None else copy.deepcopy(coupled_override)
    pp['cost']=p['cost'];pp['runtime_budget_seconds']=p['runtime_budget_seconds']
    a.n.p.save_json(out/'unchanged_quality.json',dict(quality=pp['quality'],compartment_gates=pp['compartment_gates']))
    cases,inj,mult,rev=parent.design(pp);a.n.p.save_json(out/'cases.json',cases)
    inj=torch.tensor(inj,device=device,dtype=torch.float64);mult=torch.tensor(mult,device=device,dtype=torch.float64);rev=torch.tensor(rev,device=device,dtype=torch.float64);physics=a.n.contract(root)[1:]
    summaries={}
    with torch.no_grad():
        ref=parent.reference(torch,pp,inj,mult,rev,physics,device)
        for label,compact in [('frozen52',models['compact'])]+[(arm['id'],arm_view(torch,p,net,mean,i,device)) for i,arm in enumerate(p['arms'])]:
            target=out/label;target.mkdir();local=copy.deepcopy(pp)
            local['arms']=['canonical64','canonical32','P','P_J','compact','compact_imposed','compact_ca_imposed','compact_reset','oracle_all'] if label=='frozen52' else ['compact','compact_imposed','compact_ca_imposed','compact_reset','oracle_all']
            current_models=dict(models,compact=compact)
            trace=parent.simulate(torch,local,current_models,inj,mult,rev,physics,ref,device,target)
            np.savez_compressed(target/'traces.npz',**trace)
            summary,rows=parent.evaluate(local,trace,ref,cases,target);summaries[label]=summary
            if label=='frozen52':
                assert all(r['passed'] for r in rows if r['arm']=='canonical64'),'Reference control failed'
                np.savez_compressed(out/'reference.npz',**{k:v.cpu().numpy() for k,v in ref.items()})
            parent.benchmark(torch,local,current_models,device,target);checkpoint(out,p['updates'])
    assert freeze_hash==a.n.weights_hash(net)
    assert frozen_before=={k:a.n.weights_hash(v) for k,v in models.items()}
    finite=all(bool(torch.isfinite(v).all()) for v in net.parameters());assert finite
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='52b',training_updates=p['updates'],selection_recomputed=True,thresholds_changed=False,abstraction_changed=False,logical_state=10,members=net.members,finite_parameters=finite,frozen_controls_unchanged=True,postfreeze_weights_unchanged=True,summary=summaries,initialization_diagnostics='initialization_diagnostics.json',elapsed_seconds=time.monotonic()-started,limitations=p['limitations']))
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2
    checkpoint(out,p['updates'])
