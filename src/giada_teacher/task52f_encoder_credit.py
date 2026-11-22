"""Encoder alignment x temporal credit, with unchanged quality guards."""
import copy,gzip,hashlib,io,json,time,zipfile,shutil
import numpy as np
from . import task52d_balance_trajectory as d
c=d.c;t=d.t;q=d.q;a=d.a;parent=d.parent

def checkpoint(out,step):
    from pathlib import Path
    if Path('/kaggle/working').is_dir():
        path=Path(shutil.make_archive('/kaggle/working/task52f_checkpoint_latest','zip',out))
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        a.n.p.save_json(Path('/kaggle/working/task52f_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=path.stat().st_size,update=step,partial=True))

def plan(root):
    p=json.loads((root/'experiments/task52f_preregistration_v1.json').read_text());assert not p['thresholds_changed'] and not p['abstraction_changed'];return p

def load_model(torch,p,package,device):
    with zipfile.ZipFile(package) as z:
        state=torch.load(io.BytesIO(z.read('source52d/selected.pt')),map_location=device,weights_only=True);cal=torch.load(io.BytesIO(z.read('source52d/calibration.pt')),map_location=device,weights_only=True)
    net=c.model_class(torch)(p,cal,state,device);ix=p['source_indices']*len(p['arms']);net.load_state_dict({k:v[ix].clone() for k,v in state.items()})
    net.encoder_mask.fill_(False);net.recurrence_mask.fill_(True)
    for name,v in net.named_parameters():v.requires_grad_(name not in ('enc','eb','out','ob'))
    return net,cal,state

def support_data(torch,p,device):
    T=round(p['history_ms']/p['dt_ms']);n=p['support_protocol']['count_per_role'];volts=[];calcium=[];meta=[]
    for role in ('fit','development'):
        rng=np.random.default_rng(p['support_protocol'][role+'_seed'])
        for j in range(n):
            v=np.full(T,-65.);ca=np.full(T,1e-4);time_ms=np.arange(T)*p['dt_ms']
            if j%2==0:
                start=rng.uniform(20,50);stop=rng.uniform(450,700);v[(time_ms>=start)&(time_ms<stop)]=rng.uniform(-115,-85)
                v[(time_ms>=stop)&(time_ms<stop+100)]=rng.uniform(-45,-20)
            else:
                for start in np.arange(40,800,rng.uniform(70,110)):
                    v[(time_ms>=start)&(time_ms<start+rng.uniform(2,8))]=rng.uniform(-15,30)
                ca[time_ms>100]=10**rng.uniform(-4,-2.5)
            volts.append(v);calcium.append(ca);meta.append(dict(role=role,protocol='slow_recovery' if j%2==0 else 'brief_bursts',index=j))
    vh=torch.tensor(np.array(volts).T,device=device,dtype=torch.float32);ch=torch.tensor(np.array(calcium).T,device=device,dtype=torch.float32)
    g=torch.tensor(q.steady(),device=device,dtype=torch.float64)[None].expand(len(meta),-1).clone();gs=[]
    for k in range(T):gs.append(g.float());g=q.canonical_step(torch,g,vh[k].double(),ch[k].double(),p['dt_ms'])
    gates=torch.stack(gs);logs=c.b.exact.log_targets(torch,gates.reshape(-1,18)).reshape(T,len(meta),11).float()
    return dict(v=vh,ca=ch,g=gates,logop=logs,op=10**logs,meta=meta)

def merge_data(torch,left,right):
    return {**{k:torch.cat((left[k],right[k]),1) for k in ('v','ca','g','logop','op')},'meta':left['meta']+right['meta']}

def teacher(torch,p,net,data,ids,starts,cal,extra,backward):
    E,B=ids.shape;L=p['chunk_ticks'];h=net.initialize(data['g'][starts,ids]);target=torch.stack([data['op'][starts+k,ids] for k in range(L)])
    ref=torch.stack([t.current(torch,target[k],data['v'][starts+k,ids]) for k in range(L)]);scale=ref.abs().amax(0).clamp_min(1e-7)
    weights=torch.where(target>=.2*cal['maxop'],5.,1.);weights/=weights.mean((0,2),keepdim=True)
    common=torch.zeros(E,device=ids.device);fam=common.clone();block=common.clone();errs=[];charge=torch.zeros_like(ref[0]);peak=torch.zeros_like(ref[0]);perchannel=torch.zeros(E,11,device=ids.device)
    for k in range(L):
        logs=net.logopening(h);op=10**logs;err=(t.current(torch,op,data['v'][starts+k,ids])-ref[k])/scale
        delta=logs-data['logop'][starts+k,ids];basic=(((delta/cal['span']).square()+.25*err.square()+.25*((op-target[k])/cal['maxop']).square())*weights[k]).mean((1,2))/L
        if k==0:basic=basic+.25*(delta/cal['span']).square().mean((1,2))
        charge=charge+err/L;peak=torch.maximum(peak,t.current(torch,op,data['v'][starts+k,ids]).abs()/scale)
        values=(torch.asinh(delta).square()+err.square()).mean(1)
        grouped=torch.stack([values[:,js].mean(1) for js in p['families'].values()],1);value=.1*(grouped.mean(1)+grouped.amax(1))/L
        perchannel+=err.detach().square().mean(1)/L;common+=basic.detach();fam+=value.detach();block=block+basic+extra*value
        h=net.step(t.features(torch,data['v'][starts+k,ids],data['ca'][starts+k,ids]),h)
        if backward and ((k+1)%p['bptt_ticks']==0 or k+1==L):
            if k+1==L:
                end=charge.square().mean((1,2))+(peak-ref.abs().amax(0)/scale).square().mean((1,2));block=block+.1*extra*end;fam+=.1*end.detach()
            block.sum().backward(retain_graph=k+1<L)
            detach=torch.tensor([(k+1)%arm['bptt']==0 or k+1==L for arm in p['arms'] for _ in p['seeds']],device=ids.device)[:,None,None]
            h=torch.where(detach,h.detach(),h);charge=torch.where(detach,charge.detach(),charge);peak=torch.where(detach,peak.detach(),peak);block=torch.zeros_like(common)
    return common,fam,perchannel

def trajectory(torch,p,net,d,ids,starts,length,weight,backward,components=False):
    E=len(net.members);B=len(ids);burn=p['coupled_training']['burnin_ticks'];first=(starts-burn).clamp_min(0)
    h=net.initialize(d['g'][first,ids][None].expand(E,-1,-1));v=d['v'][first,ids][None].expand(E,-1).clone();ca=d['ca'][first,ids][None].expand(E,-1).clone()
    bar=torch.tensor(q.ionic.GBAR,device=v.device);mult=d['mult'][ids][None];rev=d['rev'][ids][None]
    # Align windows by a common fixed burn-in length; clamped start0 has none.
    with torch.no_grad():
        for j in range(burn):
            active=first+j<starts;tick=torch.minimum(first+j,starts)
            nh=net.step(t.features(torch,v.float(),ca.float()),h);cond=net.opening(nh).double()*bar*mult;nv,nca,_=parent.physical(torch,v,ca,cond,d['inj'][tick,ids][None],rev,d['physics'],p['dt_ms'])
            h=torch.where(active[None,:,None],nh,h);v=torch.where(active[None],nv,v);ca=torch.where(active[None],nca,ca)
    parts=torch.zeros(E,2,device=v.device);total=torch.zeros(E,device=v.device);block=total.clone()
    for k in range(length):
        tick=starts+k;h=net.step(t.features(torch,v.float(),ca.float()),h);cond=net.opening(h).double()*bar*mult;v,ca,_=parent.physical(torch,v,ca,cond,d['inj'][tick,ids][None],rev,d['physics'],p['dt_ms'])
        vv=((v-d['v'][tick+1,ids][None])/p['coupled_training']['voltage_scale_mv']).square();cc=((ca-d['ca'][tick+1,ids][None])/p['coupled_training']['calcium_scale_mM']).square();val=vv+cc
        parts+=torch.stack((vv.mean(1),cc.mean(1)),1).detach()/length
        value=.1*(val.mean(1)+.25*val.amax(1))/length
        total+=value.detach().float();block=block+weight*value
        if backward and ((k+1)%p['bptt_ticks']==0 or k+1==length):
            block.sum().backward(retain_graph=k+1<length)
            detach=torch.tensor([(k+1)%arm['bptt']==0 or k+1==length for arm in p['arms'] for _ in p['seeds']],device=v.device)[:,None]
            h=torch.where(detach[:,:,None],h.detach(),h);v=torch.where(detach,v.detach(),v);ca=torch.where(detach,ca.detach(),ca);block=torch.zeros_like(total)
    return (total,parts) if components else total

def validate(torch,p,net,data,aut,cal):
    E=len(net.members);ids=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='development'],device=data['v'].device);ai=torch.tensor([i for i,m in enumerate(aut['meta']) if m['role']=='development'],device=ids.device)
    scores=[];channels=[];trajectories=[]
    with torch.no_grad():
        for ms in (0,150,650):
            starts=torch.full((E,len(ids)),min(round(ms/p['dt_ms']),len(data['v'])-p['chunk_ticks']-1),device=ids.device,dtype=torch.long)
            basic,fam,ch=teacher(torch,p,net,data,ids[None].expand(E,-1),starts,cal,torch.ones(E,device=ids.device),False)
            start=torch.full_like(ai,min(round(ms/p['dt_ms']),len(aut['inj'])-p['coupled_training']['horizon_after300']-1))
            tr,parts=trajectory(torch,p,net,aut,ai,start,p['coupled_training']['horizon_after300'],torch.ones(E,device=ids.device),False,components=True)
            scores.append(basic+fam+tr);channels.append(ch);trajectories.append(parts)
    return torch.stack(scores).amax(0),torch.stack(channels).amax(0),torch.stack(trajectories).amax(0)

def calibrate_encoder(torch,p,net,data,cal,out):
    E=len(net.members);device=data['v'].device;before={k:v.clone() for k,v in net.state_dict().items()};mask=torch.tensor([arm['recalibrate'] for arm in p['arms'] for _ in p['seeds']],device=device)
    for name,v in net.named_parameters():v.requires_grad_(name in ('enc','eb'))
    optimizer=torch.optim.Adam([net.enc,net.eb],lr=p['encoder_calibration']['learning_rate'],eps=p['adam_epsilon']);rng=np.random.default_rng(9527101);fit=np.array([i for i,m in enumerate(data['meta']) if m['role']=='fit']);curve=[]
    def loss(g,y):
        h=net.initialize(g[None].expand(E,-1,-1));delta=net.logopening(h)-y
        return torch.asinh(delta).square().mean((1,2))
    with torch.no_grad():initial=loss(data['g'][0,fit],data['logop'][0,fit])
    for step in range(p['encoder_calibration']['updates']):
        ids=torch.tensor(rng.choice(fit,p['encoder_calibration']['batch_size']),device=device);ticks=torch.tensor(rng.integers(len(data['v']),size=len(ids)),device=device);optimizer.zero_grad(set_to_none=True)
        value=loss(data['g'][ticks,ids],data['logop'][ticks,ids]);(value*mask).sum().backward()
        for v in (net.enc,net.eb):
            assert torch.isfinite(v.grad).all();v.grad.mul_(mask.reshape(E,*([1]*(v.ndim-1))))
        norm=sum(v.grad.double().reshape(E,-1).square().sum(1) for v in (net.enc,net.eb)).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in (net.enc,net.eb):v.grad.mul_(factor.to(v.dtype).reshape(E,*([1]*(v.ndim-1))))
        optimizer.step()
        assert torch.isfinite(net.enc).all() and torch.isfinite(net.eb).all()
        if (step+1)%20==0:curve.append(dict(step=step+1,fit_loss=value.detach().cpu().tolist()))
    for name,v in net.state_dict().items():
        if name not in ('enc','eb'):assert torch.equal(v,before[name])
        else:assert torch.equal(v[~mask],before[name][~mask])
    with torch.no_grad():final=loss(data['g'][0,fit],data['logop'][0,fit])
    for name,v in net.named_parameters():v.requires_grad_(name not in ('enc','eb','out','ob'))
    a.n.p.save_json(out/'encoder_calibration.json',dict(initial_fit_loss=initial.cpu().tolist(),final_fit_loss=final.cpu().tolist(),curve=curve,steps=p['encoder_calibration']['updates'],fit_only=True,other_weights_unchanged=True,control_encoders_unchanged=True))

def train(torch,p,net,data,aut,cal,out,ordinary_count):
    device=data['v'].device;E=len(net.members);fit=torch.tensor([i for i,m in enumerate(data['meta'][:ordinary_count]) if m['role']=='fit'],device=device);sup=torch.tensor([i for i,m in enumerate(data['meta']) if i>=ordinary_count and m['role']=='fit'],device=device);ai=torch.tensor([i for i,m in enumerate(aut['meta']) if m['role']=='fit'],device=device)
    resumed=(out/'resume_request.json').exists();start_update=0
    if resumed:
        baseline=json.loads((out/'validation_baseline.json').read_text());ch0=torch.tensor(baseline['channels'],device=device);tr0=torch.tensor(baseline['trajectory'],device=device);best=np.array(baseline['score']);chosen=np.zeros(E,int)
        selected=torch.load(out/'initial.pt',map_location=device,weights_only=True);curve=json.loads((out/'learning_curve.json').read_text())
        saved=torch.load(out/'resumable_training.pt',map_location=device,weights_only=True);start_update=int(saved['update']);assert start_update==json.loads((out/'resume_request.json').read_text())['checkpoint_update']
        for row in curve:
            assert row['update']<=start_update
            values=np.array(row['score']);mask=np.array(row['eligible'])&(values<best)
            if mask.any():
                checkpoint_state=torch.load(out/f'checkpoint_{row["update"]}.pt',map_location=device,weights_only=True)
                for key in selected:selected[key][mask]=checkpoint_state[key][mask]
            best=np.where(mask,values,best);chosen[mask]=row['update']
        net.load_state_dict(saved['model'])
    else:
        score0,ch0,tr0=validate(torch,p,net,data,aut,cal);best=score0.cpu().numpy();chosen=np.zeros(E,int);selected=copy.deepcopy(net.state_dict());curve=[]
        torch.save(selected,out/'initial.pt');calibrate_encoder(torch,p,net,data,cal,out);torch.save(net.state_dict(),out/'after_encoder_calibration.pt')
        a.n.p.save_json(out/'validation_baseline.json',dict(score=best.tolist(),channels=ch0.cpu().tolist(),trajectory=tr0.cpu().tolist()))
    encoder={k:v.clone() for k,v in net.state_dict().items() if k in ('enc','eb')}
    opt=torch.optim.Adam([v for v in net.parameters() if v.requires_grad],lr=p['learning_rate'],eps=p['adam_epsilon']);rng=np.random.default_rng(9526599);started=time.monotonic();extra=torch.tensor([arm['family_loss'] for arm in p['arms'] for _ in p['seeds']],device=device)
    if resumed:
        opt.load_state_dict(saved['optimizer'])
        # The original checkpoint did not store RNG state. Its sampler has only
        # these five NumPy draws per update; replay draws, not training.
        for update in range(1,start_update+1):
            rng.integers(len(fit),size=p['batch_size']);rng.integers(len(data['v'])-p['chunk_ticks'],size=p['batch_size']);rng.integers(len(sup),size=p['batch_size']//2)
            rng.integers(len(ai),size=p['coupled_training']['batch_size']);length=p['coupled_training']['horizon_before300' if update<=300 else 'horizon_after300'];rng.integers(len(aut['inj'])-length,size=p['coupled_training']['batch_size'])
        if 'rng' in saved:assert saved['rng']==rng.bit_generator.state
        a.n.p.save_json(out/'resume_state_verified.json',dict(valid=True,checkpoint_update=start_update,optimizer_restored=True,encoder_calibration_repeated=False,rng_reconstructed=True,selected_updates=chosen.tolist()))
    for update in range(start_update+1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.8:raise TimeoutError('52f training time limit; checkpoint retained')
        ii=fit[torch.tensor(rng.integers(len(fit),size=p['batch_size']),device=device)];ss=torch.tensor(rng.integers(len(data['v'])-p['chunk_ticks'],size=p['batch_size']),device=device)
        ids=ii[None].expand(E,-1).clone();starts=ss[None].expand(E,-1).clone();n=p['batch_size']//2;si=sup[torch.tensor(rng.integers(len(sup),size=n),device=device)]
        for arm_index,arm in enumerate(p['arms']):
            if arm['support']:ids[3*arm_index:3*(arm_index+1),:n]=si
        opt.zero_grad(set_to_none=True);basic,fam,ch=teacher(torch,p,net,data,ids,starts,cal,extra,True)
        aid=ai[torch.tensor(rng.integers(len(ai),size=p['coupled_training']['batch_size']),device=device)];length=p['coupled_training']['horizon_before300' if update<=300 else 'horizon_after300'];ast=torch.tensor(rng.integers(len(aut['inj'])-length,size=len(aid)),device=device)
        tr=trajectory(torch,p,net,aut,aid,ast,length,torch.ones(E,device=device),True)
        pars=[v for v in net.parameters() if v.grad is not None];norm=sum(v.grad.double().reshape(E,-1).square().sum(-1) for v in pars).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in pars:assert torch.isfinite(v.grad).all();v.grad.mul_(factor.to(v.dtype).reshape(E,*([1]*(v.ndim-1))))
        opt.step();assert all(torch.isfinite(v).all() for v in net.parameters())
        if update%20==0:print(f'[GIADA52f] training {update}/{p["updates"]}',flush=True)
        if update in p['checkpoints']:
            score,ch,tr=validate(torch,p,net,data,aut,cal);eligible=((ch<=ch0*(1+1e-6)+1e-12).all(1)&(tr<=tr0*(1+1e-6)+1e-12).all(1));values=score.cpu().numpy();mask=eligible.cpu().numpy()&(values<best)
            for k,v in net.state_dict().items():selected[k][mask]=v[mask].detach()
            best=np.where(mask,values,best);chosen[mask]=update;curve.append(dict(update=update,score=values.tolist(),eligible=eligible.cpu().tolist(),channel_errors=ch.cpu().tolist(),trajectory_error=tr.cpu().tolist(),gradient_norm=norm.cpu().tolist()))
            torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve);checkpoint(out,update)
    assert all(torch.equal(net.state_dict()[k],v) for k,v in encoder.items());net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),baseline_fallback=(chosen==0).tolist(),fresh_accessed=False,encoder_frozen_after_calibration=True))

def credit_probe(torch,p,net,data,cal,out):
    before=a.n.weights_hash(net);E=len(net.members);ids=torch.zeros(E,2,dtype=torch.long,device=data['v'].device);starts=torch.zeros_like(ids)
    gradients=[];length=min(512,len(data['v'])-1)
    for truncation in (min(128,length),length):
        pp=copy.deepcopy(p);pp['chunk_ticks']=length;pp['bptt_ticks']=truncation;net.zero_grad(set_to_none=True)
        for arm in pp['arms']:arm['bptt']=truncation
        teacher(torch,pp,net,data,ids,starts,cal,torch.ones(E,device=ids.device),True)
        vector=torch.cat([v.grad.detach().flatten() for v in net.parameters() if v.grad is not None]);assert torch.isfinite(vector).all();gradients.append(vector)
    cosine=torch.nn.functional.cosine_similarity(gradients[0][None],gradients[1][None]).item();net.zero_grad(set_to_none=True)
    assert before==a.n.weights_hash(net)
    a.n.p.save_json(out/'credit_assignment_probe.json',dict(valid=True,forward_ticks=length,truncations=[min(128,length),length],cosine=cosine,norms=[float(x.norm()) for x in gradients],weights_unchanged=True,qualification='Fit-only short-window gradient sensitivity, not proof about all slow timescales. No optimizer step.'))

def run(torch,root,source,cost_source,recovery,package,out,device='cuda',override=None,coupled_override=None,train_fn=None,task_name='52f'):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);started=time.monotonic();a.n.p.save_json(out/'preregistration.json',p)
    models=parent.load(torch,root,package,source,cost_source,recovery,out,device);net,cal,state=load_model(torch,p,package,device);frozen=d.view(torch,p,net,cal,state,0,device);models['compact']=frozen;before={k:a.n.weights_hash(v) for k,v in models.items()}
    with torch.no_grad():ordinary=c.b.data(torch,p,device);support=support_data(torch,p,device);data=merge_data(torch,ordinary,support);aut=d.coupled_data(torch,p,root,device)
    fit=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='fit'],device=device);cal=dict(cal,maxop=data['op'][:,fit].amax((0,1)).clamp_min(1e-30))
    if (out/'resume_request.json').exists():
        previous=torch.load(out/'calibration.pt',map_location=device,weights_only=True)
        assert all(torch.equal(v,previous[k]) for k,v in cal.items() if torch.is_tensor(v))
    else:
        torch.save(cal,out/'calibration.pt');a.n.p.save_json(out/'history_metadata.json',data['meta']);credit_probe(torch,p,net,support,cal,out)
    (train_fn or train)(torch,p,net,data,aut,cal,out,len(ordinary['meta']));freeze=a.n.weights_hash(net)
    for v in net.parameters():v.requires_grad_(False)
    pp=parent.plan(root) if coupled_override is None else copy.deepcopy(coupled_override);pp['cost']=p['cost'];pp['runtime_budget_seconds']=p['runtime_budget_seconds'];pp['families']['Ca_Ih']=[0,1,5];a.n.p.save_json(out/'unchanged_quality.json',dict(quality=pp['quality'],compartment_gates=pp['compartment_gates']));cases,inj,mult,rev=parent.design(pp);a.n.p.save_json(out/'cases.json',cases)
    inj=torch.tensor(inj,device=device,dtype=torch.float64);mult=torch.tensor(mult,device=device,dtype=torch.float64);rev=torch.tensor(rev,device=device,dtype=torch.float64);physics=a.n.contract(root)[1:];summaries={}
    with torch.no_grad():
        ref=parent.reference(torch,pp,inj,mult,rev,physics,device);np.savez_compressed(out/'reference.npz',**{k:v.cpu().numpy() for k,v in ref.items()})
        zv=torch.full_like(ref['voltage'][0],-65.);zc=torch.full_like(zv,1e-4);vs=[zv];cs=[zc]
        for k in range(len(inj)):zv,zc,_=parent.physical(torch,zv,zc,torch.zeros_like(ref['current'][0]),inj[k],rev,physics,p['dt_ms']);vs.append(zv);cs.append(zc)
        passive=dict(voltage=torch.stack(vs)[:,:,None].cpu().numpy(),calcium=torch.stack(cs)[:,:,None].cpu().numpy(),current=np.zeros((len(inj),3,1,len(cases),11)));dest=out/'passive';dest.mkdir();local=copy.deepcopy(pp);local['arms']=['oracle_passive'];summaries['passive'],_=parent.evaluate(local,passive,ref,cases,dest)
        for label,model in [('frozen52d_trajectory',frozen)]+[(arm['id'],d.view(torch,p,net,cal,state,i,device)) for i,arm in enumerate(p['arms'])]:
            target=out/label;target.mkdir();local=copy.deepcopy(pp);local['arms']=['compact','compact_imposed','compact_ca_imposed','compact_reset','oracle_Ca','oracle_Ih','oracle_Ca_Ih','oracle_all']
            if label=='frozen52d_trajectory':local['arms']=['canonical64','canonical32','P','P_J']+local['arms']
            current=dict(models,compact=model);trace=parent.simulate(torch,local,current,inj,mult,rev,physics,ref,device,target);np.savez_compressed(target/'traces.npz',**trace);summaries[label],rows=parent.evaluate(local,trace,ref,cases,target);parent.benchmark(torch,local,current,device,target);checkpoint(out,p['updates'])
            if label=='frozen52d_trajectory':assert all(r['passed'] for r in rows if r['arm']=='canonical64')
    assert a.n.weights_hash(net)==freeze and before=={k:a.n.weights_hash(v) for k,v in models.items()}
    accepted=[]
    for arm in p['arms']:
        with gzip.open(out/arm['id']/'interventions.json.gz','rt') as f:rr=[r for r in json.load(f) if r['arm']=='compact']
        accepted.append(all(x['all_quality_passed'] for x in summaries[arm['id']]['compact'].values()) and all(r['current']['conservative_passed'] and r['voltage_response_rmse_mv']<=pp['compartment_gates']['max_candidate_worst_voltage_rmse_mv'] and r['calcium_response_rmse_mM']<=pp['compartment_gates']['max_candidate_worst_cai_rmse_mM'] for r in rr))
    resumed_from=json.loads((out/'resume_request.json').read_text())['checkpoint_update'] if (out/'resume_request.json').exists() else 0
    a.n.p.save_json(out/'final_report.json',dict(valid=True,execution_valid=True,scientific_acceptance_passed=any(accepted),complete=True,task=task_name,finite_parameters=True,encoder_frozen_after_calibration=True,postfreeze_weights_unchanged=True,frozen_controls_unchanged=True,training_updates=p['updates'],resumed_from=resumed_from,new_training_updates=p['updates']-resumed_from,encoder_calibration_repeated=False if resumed_from else None,encoder_calibration_updates=p['encoder_calibration']['updates'],selection_recomputed=True,thresholds_changed=False,abstraction_changed=False,logical_state=10,promotion_authorized=False,summary=summaries,elapsed_seconds=time.monotonic()-started,limitations=p['limitations'],last_allowed_subtask='52j'))
    checkpoint(out,p['updates'])
