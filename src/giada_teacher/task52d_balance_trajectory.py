"""Decoder recalibration with a paired balance x autonomous-loss experiment."""
import copy,gzip,hashlib,io,json,shutil,time,zipfile
from pathlib import Path
import numpy as np
from . import task52c_component_repair as c
t=c.t;q=c.q;a=c.a;parent=c.parent
def plan(root):
    p=json.loads((root/'experiments/task52d_preregistration_v2.json').read_text());assert not p['thresholds_changed'] and not p['abstraction_changed'];return p
def load_model(torch,p,package,device):
    with zipfile.ZipFile(package) as z:
        state=torch.load(io.BytesIO(z.read('source52c/selected.pt')),map_location=device,weights_only=True);cal=torch.load(io.BytesIO(z.read('source52c/calibration.pt')),map_location=device,weights_only=True)
    net=c.model_class(torch)(p,cal,state,device);indices=p['source_indices']*len(p['arms']);net.load_state_dict({k:v[indices].clone() for k,v in state.items()})
    net.encoder_mask.fill_(False);net.recurrence_mask.fill_(True)
    for name,v in net.named_parameters():v.requires_grad_(name not in ('enc','eb','out','ob'))
    return net,cal,state
def view(torch,p,net,cal,state,index,device):
    pp=copy.deepcopy(p);pp['arms']=[p['arms'][index]];model=c.model_class(torch)(pp,cal,state,device);model.load_state_dict({k:v[3*index:3*(index+1)].clone() for k,v in net.state_dict().items()})
    for v in model.parameters():v.requires_grad_(False)
    return model
def coupled_data(torch,p,root,device):
    spec=parent.plan(root);spec['duration_ms']=p['coupled_training']['duration_ms'];cases,base,mult,rev=parent.design(spec)
    blocks=[];meta=[]
    for role in ('fit','development'):
        rng=np.random.default_rng(p['coupled_training'][role+'_seed']);blocks.append(base*rng.uniform(.8,1.2,(1,base.shape[1])))
        meta.extend(dict(case,role=role) for case in cases)
    inj=torch.tensor(np.concatenate(blocks,1),device=device,dtype=torch.float64);mult=torch.tensor(np.tile(mult,(2,1)),device=device,dtype=torch.float64);rev=torch.tensor(np.tile(rev,(2,1)),device=device,dtype=torch.float64)
    N=inj.shape[1];v=torch.full((N,),-65.,device=device,dtype=torch.float64);ca=torch.full_like(v,1e-4);g=torch.tensor(q.steady(-65.,1e-4),device=device)[None].expand(N,-1).clone();vs=[v];cs=[ca];gs=[g];curr=[];physics=a.n.contract(root)[1:]
    with torch.no_grad():
        for i in range(len(inj)):
            g=q.canonical_step(torch,g,v,ca,p['dt_ms']);cond=parent.opening(torch,g)*torch.tensor(q.ionic.GBAR,device=device)*mult
            v,ca,current=parent.physical(torch,v,ca,cond,inj[i],rev,physics,p['dt_ms']);vs.append(v);cs.append(ca);gs.append(g);curr.append(current)
    return dict(v=torch.stack(vs),ca=torch.stack(cs),g=torch.stack(gs).float(),current=torch.stack(curr),inj=inj,mult=mult,rev=rev,meta=meta,physics=physics)
def teacher_loss(torch,p,net,d,ids,starts,cal,balance_weight,backward):
    E=len(net.members);L=p['chunk_ticks'];h=net.initialize(d['g'][starts,ids][None].expand(E,-1,-1));idx=starts[None]+torch.arange(L,device=ids.device)[:,None]
    target=d['op'][idx,ids[None]];current=t.current(torch,target,d['v'][idx,ids[None]]);scale=current.abs().amax(0).clamp_min(1e-7);total_scale=current.abs().sum(-1).amax(0).clamp_min(1e-7)
    weights=torch.where(target>=.2*cal['maxop'],5.,1.);weights/=weights.mean((0,1),keepdim=True)
    common=torch.zeros(E,device=ids.device);balance=common.clone();block=common.clone();charge=torch.zeros(E,len(ids),device=ids.device)
    for k in range(L):
        logs=net.logopening(h);op=10**logs;pred=t.current(torch,op,d['v'][starts+k,ids][None]);error=pred-current[k]
        ll=((logs-d['logop'][starts+k,ids])/cal['span']).square();cl=(error/scale).square();ol=((op-target[k])/cal['maxop']).square()
        basic=((ll+.25*cl+.25*ol)*weights[k]).mean((1,2))/L
        if k==0:basic=basic+.25*ll.mean((1,2))
        net_error=error.sum(-1)/total_scale;charge=charge+net_error/L
        extra=(.25*net_error.square().mean(1)+.1*charge.square().mean(1))/L
        common+=basic.detach();balance+=extra.detach();block=block+basic+balance_weight*extra
        h=net.step(t.features(torch,d['v'][starts+k,ids],d['ca'][starts+k,ids])[None].expand(E,-1,-1),h)
        if backward and ((k+1)%p['bptt_ticks']==0 or k+1==L):block.sum().backward();h=h.detach();charge=charge.detach();block=torch.zeros_like(common)
    return common,balance
def trajectory_loss(torch,p,net,d,ids,starts,length,weight,backward):
    E=len(net.members);B=len(ids);burn=p['coupled_training']['burnin_ticks'];first=(starts-burn).clamp_min(0)
    h=net.initialize(d['g'][first,ids][None].expand(E,-1,-1));v=d['v'][first,ids][None].expand(E,-1).clone();ca=d['ca'][first,ids][None].expand(E,-1).clone()
    bar=torch.tensor(q.ionic.GBAR,device=v.device);mult=d['mult'][ids][None];rev=d['rev'][ids][None]
    # Align windows by a common fixed burn-in length; clamped start0 has none.
    with torch.no_grad():
        for j in range(burn):
            active=first+j<starts;tick=torch.minimum(first+j,starts)
            nh=net.step(t.features(torch,v.float(),ca.float()),h);cond=net.opening(nh).double()*bar*mult;nv,nca,_=parent.physical(torch,v,ca,cond,d['inj'][tick,ids][None],rev,d['physics'],p['dt_ms'])
            h=torch.where(active[None,:,None],nh,h);v=torch.where(active[None],nv,v);ca=torch.where(active[None],nca,ca)
    total=torch.zeros(E,device=v.device);block=total.clone()
    for k in range(length):
        tick=starts+k;h=net.step(t.features(torch,v.float(),ca.float()),h);cond=net.opening(h).double()*bar*mult;v,ca,_=parent.physical(torch,v,ca,cond,d['inj'][tick,ids][None],rev,d['physics'],p['dt_ms'])
        val=((v-d['v'][tick+1,ids][None])/p['coupled_training']['voltage_scale_mv']).square()+((ca-d['ca'][tick+1,ids][None])/p['coupled_training']['calcium_scale_mM']).square()
        value=.1*(val.mean(1)+.25*val.amax(1))/length
        total+=value.detach().float();block=block+weight*value
        if backward and ((k+1)%p['bptt_ticks']==0 or k+1==length):block.sum().backward();h=h.detach();v=v.detach();ca=ca.detach();block=torch.zeros_like(total)
    return total
def checkpoint(out,step):
    if Path('/kaggle/working').is_dir():
        path=Path(shutil.make_archive('/kaggle/working/task52d_checkpoint_latest','zip',out))
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        a.n.p.save_json(Path('/kaggle/working/task52d_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=path.stat().st_size,update=step,partial=True))
def train(torch,p,net,d,aut,cal,out):
    device=d['v'].device;E=len(net.members);fit=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='fit'],device=device);dev=torch.tensor([i for i,m in enumerate(d['meta']) if m['role']=='development'],device=device)
    afit=torch.tensor([i for i,m in enumerate(aut['meta']) if m['role']=='fit'],device=device);adev=torch.tensor([i for i,m in enumerate(aut['meta']) if m['role']=='development'],device=device)
    pools=[torch.nonzero(d['op'][:,fit,k]>=.2*cal['maxop'][k],as_tuple=False).cpu().numpy() for k in range(11)];assert all(len(x) for x in pools)
    a.n.p.save_json(out/'support.json',dict(teacher_pool_counts=[len(x) for x in pools],coupled_cases=aut['meta']))
    bw=torch.tensor([arm['balance'] for arm in p['arms'] for _ in p['seeds']],device=device);tw=torch.tensor([arm['trajectory'] for arm in p['arms'] for _ in p['seeds']],device=device)
    opt=torch.optim.Adam([v for v in net.parameters() if v.requires_grad],lr=p['learning_rate'],eps=p['adam_epsilon']);rng=np.random.default_rng(952599)
    encoder={k:v.clone() for k,v in net.state_dict().items() if k in ('enc','eb')};best=np.full(E,np.inf);chosen=np.zeros(E,int);selected=copy.deepcopy(net.state_dict());curve=[];started=time.monotonic()
    for update in range(1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.7:raise TimeoutError('52d training budget; checkpoints preserved')
        episodes=rng.integers(len(fit),size=p['batch_size']);starts=rng.integers(len(d['v'])-p['chunk_ticks'],size=p['batch_size'])
        for j in range(p['batch_size']//2):
            pool=pools[rng.integers(11)];tick,ep=pool[rng.integers(len(pool))];episodes[j]=ep;starts[j]=np.clip(tick-p['chunk_ticks']//2,0,len(d['v'])-p['chunk_ticks']-1)
        ai=afit[torch.tensor(rng.integers(len(afit),size=p['coupled_training']['batch_size']),device=device)];length=p['coupled_training']['horizon_before300' if update<=300 else 'horizon_after300']
        ast=rng.integers(len(aut['inj'])-length,size=len(ai))
        for j in range(len(ai)//2):ast[j]=np.clip(rng.choice([200,600,1500,2400,3370])-length//2,0,len(aut['inj'])-length-1)
        opt.zero_grad(set_to_none=True);common,bal=teacher_loss(torch,p,net,d,fit[torch.tensor(episodes,device=device)],torch.tensor(starts,device=device),cal,bw,True)
        tr=trajectory_loss(torch,p,net,aut,ai,torch.tensor(ast,device=device),length,tw,True)
        pars=[v for v in net.parameters() if v.grad is not None];norm=sum(v.grad.double().reshape(E,-1).square().sum(-1) for v in pars).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in pars:assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.to(v.dtype).reshape(E,*([1]*(v.ndim-1))))
        opt.step();assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
        if update%20==0:print(f'[GIADA52d] training {update}/{p["updates"]} elapsed={time.monotonic()-started:.1f}s',flush=True)
        if update in p['checkpoints']:
            with torch.no_grad():
                scores=[]
                for ms in (0,150,280):
                    st=torch.full_like(dev,min(round(ms/p['dt_ms']),len(d['v'])-p['chunk_ticks']));cc,bb=teacher_loss(torch,p,net,d,dev,st,cal,torch.ones(E,device=device),False)
                    ast=torch.full_like(adev,min(round(ms/p['dt_ms']),len(aut['inj'])-p['coupled_training']['horizon_after300']-1));tt=trajectory_loss(torch,p,net,aut,adev,ast,p['coupled_training']['horizon_after300'],torch.ones(E,device=device),False);scores.append(cc+bb+tt)
                score=torch.stack(scores).amax(0)
            values=score.cpu().numpy();mask=values<best
            for k,v in net.state_dict().items():selected[k][mask]=v[mask].detach()
            best=np.minimum(best,values);chosen[mask]=update;assert all(torch.equal(net.state_dict()[k],v) for k,v in encoder.items())
            curve.append(dict(update=update,development=values.tolist(),elapsed_seconds=time.monotonic()-started));torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve);checkpoint(out,update)
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),fresh_accessed=False,encoder_unchanged=True))
def run(torch,root,source,cost_source,recovery,package,out,device='cuda',override=None,coupled_override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);started=time.monotonic();a.n.p.save_json(out/'preregistration.json',p)
    models=parent.load(torch,root,package,source,cost_source,recovery,out,device);net,cal,state=load_model(torch,p,package,device);frozen=view(torch,p,net,cal,state,0,device);models['compact']=frozen;before={k:a.n.weights_hash(v) for k,v in models.items()}
    torch.save(net.state_dict(),out/'initial.pt')
    with torch.no_grad():data=c.b.data(torch,p,device);aut=coupled_data(torch,p,root,device)
    fit=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='fit'],device=data['v'].device)
    cal=dict(cal,maxop=data['op'][:,fit].amax((0,1)).clamp_min(1e-30));torch.save(cal,out/'calibration.pt')
    a.n.p.save_json(out/'history_metadata.json',data['meta']);train(torch,p,net,data,aut,cal,out);freeze=a.n.weights_hash(net)
    for v in net.parameters():v.requires_grad_(False)
    c.decoder_bounds(torch,net,out,'selected')
    pp=parent.plan(root) if coupled_override is None else copy.deepcopy(coupled_override);pp['cost']=p['cost'];pp['runtime_budget_seconds']=p['runtime_budget_seconds'];pp['families']['Ca_Ih']=[0,1,5]
    a.n.p.save_json(out/'unchanged_quality.json',dict(quality=pp['quality'],compartment_gates=pp['compartment_gates']));cases,inj,mult,rev=parent.design(pp);a.n.p.save_json(out/'cases.json',cases)
    inj=torch.tensor(inj,device=device,dtype=torch.float64);mult=torch.tensor(mult,device=device,dtype=torch.float64);rev=torch.tensor(rev,device=device,dtype=torch.float64);physics=a.n.contract(root)[1:];summaries={}
    with torch.no_grad():
        ref=parent.reference(torch,pp,inj,mult,rev,physics,device);np.savez_compressed(out/'reference.npz',**{k:v.cpu().numpy() for k,v in ref.items()})
        zv=torch.full_like(ref['voltage'][0],-65.);zc=torch.full_like(zv,1e-4);vs=[zv];cs=[zc]
        for k in range(len(inj)):zv,zc,_=parent.physical(torch,zv,zc,torch.zeros_like(ref['current'][0]),inj[k],rev,physics,p['dt_ms']);vs.append(zv);cs.append(zc)
        passive=dict(voltage=torch.stack(vs)[:,:,None].cpu().numpy(),calcium=torch.stack(cs)[:,:,None].cpu().numpy(),current=np.zeros((len(inj),3,1,len(cases),11)));dest=out/'passive';dest.mkdir();local=copy.deepcopy(pp);local['arms']=['oracle_passive'];summaries['passive'],_=parent.evaluate(local,passive,ref,cases,dest)
        for label,model in [('frozen52cDE',frozen)]+[(arm['id'],view(torch,p,net,cal,state,i,device)) for i,arm in enumerate(p['arms'])]:
            target=out/label;target.mkdir();local=copy.deepcopy(pp);local['arms']=['compact','compact_imposed','compact_ca_imposed','compact_reset','oracle_Ca','oracle_Ih','oracle_Ca_Ih','oracle_all']
            if label=='frozen52cDE':local['arms']=['canonical64','canonical32','P','P_J']+local['arms']
            current=dict(models,compact=model);trace=parent.simulate(torch,local,current,inj,mult,rev,physics,ref,device,target);np.savez_compressed(target/'traces.npz',**trace);summaries[label],rows=parent.evaluate(local,trace,ref,cases,target)
            if label=='frozen52cDE':assert all(r['passed'] for r in rows if r['arm']=='canonical64')
            parent.benchmark(torch,local,current,device,target);checkpoint(out,p['updates'])
    assert a.n.weights_hash(net)==freeze;assert before=={k:a.n.weights_hash(v) for k,v in models.items()}
    accepted=[]
    for arm in p['arms']:
        with gzip.open(out/arm['id']/'interventions.json.gz','rt') as f:rr=[r for r in json.load(f) if r['arm']=='compact']
        accepted.append(all(x['all_quality_passed'] for x in summaries[arm['id']]['compact'].values()) and all(r['current']['conservative_passed'] and r['voltage_response_rmse_mv']<=pp['compartment_gates']['max_candidate_worst_voltage_rmse_mv'] and r['calcium_response_rmse_mM']<=pp['compartment_gates']['max_candidate_worst_cai_rmse_mM'] for r in rr))
    a.n.p.save_json(out/'final_report.json',dict(valid=True,execution_valid=True,scientific_acceptance_passed=any(accepted),promotion_authorized=False,complete=True,task='52d',training_updates=p['updates'],selection_recomputed=True,thresholds_changed=False,abstraction_changed=False,decoder_recalibration_authorized=True,logical_state=10,finite_parameters=True,encoder_unchanged=True,postfreeze_weights_unchanged=True,frozen_controls_unchanged=True,summary=summaries,elapsed_seconds=time.monotonic()-started,limitations=p['limitations']))
    assert sum(x.stat().st_size for x in out.rglob('*') if x.is_file())<p['output_budget_mb']*1024**2;checkpoint(out,p['updates'])
