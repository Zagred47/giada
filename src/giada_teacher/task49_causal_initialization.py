"""State-available codecs and past-only warmup; P is never trained or altered."""
import gzip,json,time
from pathlib import Path
import numpy as np
from . import task48_state_minimality as q
a=q.a

def identity(p):
    value=q.identity(p);value['task']='49';return value

def plan(root):
    p=json.loads((root/'experiments/task49_preregistration_v1.json').read_text())
    assert p['task']=='49' and not p['thresholds_changed'] and not p['selection_performed'] and p['gradient_training_updates']==0
    assert p['weights_sha256']==q.plan(root)['weights_sha256']
    assert len({v['seed'] for v in p['roles'].values()})==3
    return p

def histories(p):
    vs=[];cs=[];meta=[]
    for role,info in p['roles'].items():
        config=dict(p,input_seed=info['seed'],history_count=info['count'])
        v,c,rows=q.history_inputs(config);vs.append(v);cs.append(c)
        offset=len(meta);meta.extend(dict(row,role=role,history=offset+i,input_seed=info['seed']) for i,row in enumerate(rows))
    return np.concatenate(vs,1),np.concatenate(cs,1),meta

def codec_fit(states,meta,p):
    fit=np.array([m['role']=='fit' for m in meta]);assert fit.any()
    data=states[:,fit].reshape(-1,18);result={}
    for family,gates in q.family_indices(p).items():
        value=data[:,gates];mean=value.mean(0);_,s,vt=np.linalg.svd(value-mean,full_matrices=False)
        result[family]=dict(gates=gates,mean=mean.tolist(),basis=vt.tolist(),singular_values=s.tolist(),fit_samples=len(value),fit_role_only=True)
    return result

def codec_decode(states,codec,widths):
    value=states.copy();latent=[]
    for family,row in codec.items():
        gates=row['gates'];mean=np.array(row['mean']);basis=np.array(row['basis'])[:widths[family]]
        code=(states[...,gates]-mean)@basis.T;value[...,gates]=np.clip(code@basis+mean,0,1);latent.append(code)
    return value,np.concatenate(latent,-1)

def coefficients_update(torch,u,z,state):
    return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)

def active_mask(p,tick):
    ends=np.rint(np.array(p['cuts_ms'])/p['dt_ms']).astype(int)
    lengths=np.rint(np.array(p['warmup_ms'])/p['dt_ms']).astype(int)
    return (tick>=np.maximum(0,ends[None]-lengths[:,None]))&(tick<ends[None])

def condition(torch,net,vh,ch,meta,p,device):
    n=len(meta);dt=p['dt_ms'];ends={round(v/dt):i for i,v in enumerate(p['cuts_ms'])};nc=len(ends);nw=len(p['warmup_ms'])
    base=np.broadcast_to(q.steady(),(3,n,18)).copy();g64=torch.tensor(base,device=device,dtype=torch.float64);g32=g64.float();gp=g32.clone()
    warm=g32[:,None,None].expand(3,nw,nc,n,18).clone();c64=[];c32=[];pp=[];ww=[]
    start=time.perf_counter()
    for tick in range(len(vh)):
        v=torch.tensor(vh[tick],device=device,dtype=torch.float32)[None].expand(3,-1);ca=torch.tensor(ch[tick],device=device,dtype=torch.float32)[None].expand(3,-1)
        u,z=net.raw(v,ca)
        if tick==0:assert torch.equal(coefficients_update(torch,u,z,gp),net(v,ca,gp)),'P forward changed by coefficient reuse'
        gp=coefficients_update(torch,u,z,gp);g64=q.canonical_step(torch,g64,v.double(),ca.double(),dt);g32=q.canonical_step(torch,g32,v,ca,dt)
        mask=torch.tensor(active_mask(p,tick),device=device)[None,:,:,None,None]
        warm=torch.where(mask,coefficients_update(torch,u[:,None,None],z[:,None,None],warm),warm)
        if tick+1 in ends:
            cut=ends[tick+1];c64.append(g64.cpu().numpy());c32.append(g32.cpu().numpy());pp.append(gp.cpu().numpy());ww.append(warm[:,:,cut].cpu().numpy())
        if tick%1000==0:print(f'[GIADA49] conditioning {tick}/{len(vh)}',flush=True)
    states=dict(canonical64=np.stack(c64),canonical32=np.stack(c32),P=np.stack(pp),warmup=np.stack(ww))
    # 400ms includes the complete available prefix at every registered cut.
    assert np.array_equal(states['warmup'][:,:,-1],states['P'])
    return states,time.perf_counter()-start

def initial_bank(states,meta,codec,p):
    nc,ns,n,_=states['P'].shape;arms=p['arms'];bank=np.empty((ns,len(arms),nc,n,18),dtype=np.float32)
    bank[:,0]=states['P'].transpose(1,0,2,3);bank[:,1]=bank[:,0].copy();bank[:,2]=states['canonical64'].transpose(1,0,2,3)
    latents={}
    for arm,widths in p['codec_widths'].items():
        decoded,latent=codec_decode(states['canonical64'][...,0,:,:],codec,widths)
        bank[:,arms.index(arm)]=decoded[None];latents[arm]=latent
    for j,budget in enumerate(p['warmup_ms']):bank[:,arms.index('warmup_'+str(budget))]=states['warmup'][:,:,j].transpose(1,0,2,3)
    shifted=np.arange(n)
    for role in p['roles']:
        ids=np.flatnonzero([m['role']==role for m in meta]);shifted[ids]=np.roll(ids,1)
    bank[:,-1]=states['canonical64'][:,0,shifted].astype(np.float32)[None]
    return bank.reshape(ns,len(arms),nc*n,18),latents

def metric_row(torch,pred,ref,scale,squares,maximum,signed,absolute,gsq,gmax,count):
    denom=count*scale
    return dict(current_nrmse=(squares/count).sqrt()/scale,current_nmax=maximum/scale,signed_charge=(signed.abs()/denom),absolute_charge=absolute.abs()/denom,gate_rmse=(gsq/count).sqrt(),gate_max=gmax)

def future(torch,net,bank,states,meta,p,name,out,device):
    dt=p['dt_ms'];n=len(meta)*len(p['cuts_ms']);steps=round(max(p['horizons_ms'])/dt);want={round(h/dt):h for h in p['horizons_ms']}
    # Reference histories used precisely recorded float32 input values; same future inputs for every arm.
    g=torch.tensor(states['canonical64'][:,0].reshape(n,18),device=device,dtype=torch.float64)
    references=[];gate_refs=[]
    for tick in range(steps+1):
        vv,cc=q.future_input(name,tick*dt);v=torch.full((n,),vv,device=device,dtype=torch.float64);ca=torch.full_like(v,cc)
        references.append(a.t.readout(torch,v,g,torch.ones((n,11),device=device,dtype=torch.float64))[1]);gate_refs.append(g)
        if tick<steps:g=q.canonical_step(torch,g,v,ca,dt)
    refs=torch.stack(references);grefs=torch.stack(gate_refs);del references,gate_refs
    current=torch.tensor(bank,device=device);g32=torch.tensor(states['canonical32'].transpose(1,0,2,3).reshape(3,n,18),device=device)
    shape=(*current.shape[:-1],11);squares=torch.zeros(shape,device=device,dtype=torch.float64);maximum=squares.clone();signed=squares.clone();absolute=squares.clone()
    gsq=torch.zeros(current.shape,device=device,dtype=torch.float64);gmax=gsq.clone();scale=torch.zeros((n,11),device=device,dtype=torch.float64)
    paired_sq=squares.clone();paired_max=squares.clone();numeric_sq=torch.zeros((3,n,11),device=device,dtype=torch.float64);numeric_max=numeric_sq.clone()
    selected=[ci*len(meta)+j for ci in range(len(p['cuts_ms'])) for role in p['roles'] for j in [k for k,m in enumerate(meta) if m['role']==role][:6]]
    rows=[];witness=[];quality=a.spec(Path(__file__).resolve().parents[2])[0]['quality']
    for tick in range(steps+1):
        vv,cc=q.future_input(name,tick*dt);v=torch.full((3,n),vv,device=device);ca=torch.full_like(v,cc)
        got=a.t.readout(torch,v[:,None].expand(current.shape[:-1]),current,torch.ones(shape,device=device))[1].double();ref=refs[tick];err=got-ref
        scale=torch.maximum(scale,ref.abs());squares+=err.square();maximum=torch.maximum(maximum,err.abs());signed+=err;absolute+=got.abs()-ref.abs()
        gateerr=current.double()-grefs[tick];gsq+=gateerr.square();gmax=torch.maximum(gmax,gateerr.abs())
        delta=got-got[:,:1];paired_sq+=delta.square();paired_max=torch.maximum(paired_max,delta.abs())
        numeric=a.t.readout(torch,v,g32,torch.ones((3,n,11),device=device))[1].double()-ref;numeric_sq+=numeric.square();numeric_max=torch.maximum(numeric_max,numeric.abs())
        assert torch.equal(current[:,0],current[:,1]),'Restart copy diverged'
        assert torch.equal(current[:,0],current[:,p['arms'].index('warmup_400')]),'Complete-prefix warmup diverged'
        if tick%10==0:witness.append(got[:,:,selected].float().cpu().numpy())
        if tick in want:
            floor=scale.clamp_min(quality['current_scale_floor_ma_cm2']);stats=metric_row(torch,got,ref,floor,squares,maximum,signed,absolute,gsq,gmax,tick+1)
            arrays={k:value.cpu().numpy() for k,value in stats.items()};paired=(paired_sq/(tick+1)).sqrt()/floor;paired_abs=paired_max/floor
            for ai,arm in enumerate(p['arms']):
                for ci,cut in enumerate(p['cuts_ms']):
                    for role in p['roles']:
                        ids=np.array([ci*len(meta)+j for j,m in enumerate(meta) if m['role']==role])
                        for si,seed in enumerate((17,29,43)):
                            values={k:value[si,ai,ids].tolist() for k,value in arrays.items()}
                            passed=(np.array(values['current_nrmse'])<=quality['current_nrmse_max'])&(np.array(values['current_nmax'])<=quality['current_nmax_max'])&(np.array(values['signed_charge'])<=quality['signed_charge_error_max'])&(np.array(values['absolute_charge'])<=quality['absolute_charge_error_max'])
                            rows.append(dict(future=name,horizon_ms=want[tick],arm=arm,cut_ms=cut,role=role,seed=seed,history_ids=[meta[int(j%len(meta))]['history'] for j in ids],metrics=values,all_channel_amplitude_charge_pass_by_history=passed.all(-1).tolist(),not_full_quality_release=True,paired_nrmse_to_P_history=paired[si,ai,ids].cpu().numpy().tolist(),paired_nmax_to_P_history=paired_abs[si,ai,ids].cpu().numpy().tolist()))
            rows.append(dict(kind='canonical32_numeric_control',future=name,horizon_ms=want[tick],current_nrmse_by_seed_channel=(numeric_sq/(tick+1)).sqrt().div(floor).amax(1).cpu().numpy().tolist(),current_nmax_by_seed_channel=numeric_max.div(floor).amax(1).cpu().numpy().tolist()))
        if tick<steps:
            u,z=net.raw(v,ca);current=coefficients_update(torch,u[:,None],z[:,None],current);g32=q.canonical_step(torch,g32,v,ca,dt)
        if tick%2000==0:print(f'[GIADA49] future {name} {tick}/{steps}',flush=True)
    assert bool(torch.isfinite(current).all()) and bool(((current>=0)&(current<=1)).all())
    np.savez_compressed(out/f'witness_{name}.npz',current=np.stack(witness),sample_dt_ms=dt*10,case_ids=selected)
    with gzip.open(out/f'metrics_{name}.json.gz','wt',encoding='utf-8') as stream:json.dump(rows,stream,allow_nan=False,separators=(',',':'))
    return len(rows)

def cost(torch,net,codec,p,device):
    n=p['cost']['batch_histories'];v=torch.full((3,n),-65.,device=device);ca=torch.full_like(v,5e-5);g=torch.tensor(np.tile(q.steady(),(3,n,1)),device=device,dtype=torch.float32)
    def measure(fn):
        fn();samples=[]
        for _ in range(p['cost']['repetitions']):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();fn()
            if device=='cuda':torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        return samples
    result={}
    for budget in p['warmup_ms']:
        def replay(budget=budget):
            value=g.clone()
            for _ in range(round(budget/p['dt_ms'])):value=net(v,ca,value)
            return value
        result['warmup_'+str(budget)]=measure(replay)
    for arm,widths in p['codec_widths'].items():
        packed=[(row['gates'],torch.tensor(row['mean'],device=device,dtype=torch.float32),torch.tensor(row['basis'][:widths[fam]],device=device,dtype=torch.float32)) for fam,row in codec.items()]
        def encode_decode():
            value=g.clone()
            for gates,mean,basis in packed:value[...,gates]=(((g[...,gates]-mean)@basis.T)@basis+mean).clamp(0,1)
            return value
        result[arm]=measure(encode_decode)
    result['physical_identity']=measure(lambda:g.clone())
    return dict(samples_ms=result,scope=p['cost']['scope'],includes_forward_replay=True,no_production_speedup_claim=True)

def run(torch,root,source,c,d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);save=lambda n,v:a.n.p.save_json(out/n,v);start=time.monotonic()
    save('preregistration.json',p);save('executed_protocol.json',identity(p));save('canonical_parity.json',q.parity(torch,device))
    models,_,_=a.load_frozen(torch,root,source,c,d,out,device);net=models['P'];before=a.n.weights_hash(net);assert before==p['weights_sha256']
    vh,ch,meta=histories(p);states,wall=condition(torch,net,vh,ch,meta,p,device);save('history_metadata.json',meta);np.savez_compressed(out/'restart_states.npz',**states)
    cut_ticks=np.rint(np.array(p['cuts_ms'])/p['dt_ms']).astype(int)-1
    np.savez_compressed(out/'restart_environment.npz',voltage_before_cut=vh[cut_ticks],calcium_before_cut=ch[cut_ticks],cuts_ms=p['cuts_ms'])
    codec=codec_fit(states['canonical64'][:,0],meta,p);save('fit_only_codecs.json',codec);bank,latent=initial_bank(states,meta,codec,p);np.savez_compressed(out/'initialization_bank.npz',states=bank,**latent)
    with np.load(out/'initialization_bank.npz') as restored:
        assert np.array_equal(bank,restored['states']);bank[:,1]=restored['states'][:,0]
    row_count=0
    for name in p['future_protocols']:
        row_count+=future(torch,net,bank,states,meta,p,name,out,device)
        if time.monotonic()-start>p['runtime_budget_seconds']:raise TimeoutError('49 runtime budget; partial panels retained')
    save('initialization_cost.json',cost(torch,net,codec,p,device));assert a.n.weights_hash(net)==before
    save('unchanged_quality.json',a.spec(root)[0]['quality'])
    report=dict(valid=True,complete=True,task='49',exploratory=True,gradient_training_updates=0,codec_fit_only=True,selection_performed=False,thresholds_changed=False,P_weights_hash_before=before,P_weights_hash_after=a.n.weights_hash(net),conditioning_wall_seconds=wall,elapsed_seconds=time.monotonic()-start,rows=row_count,finite_parameters_states=True,restart_bitwise_identity=True,complete_prefix_warmup_bitwise_identity=True,role_counts={r:sum(m['role']==r for m in meta) for r in p['roles']},limitations=p['limitations'],decision='Initialization diagnostics only; no compressed recurrent or coupled-system qualification, no speedup claim.')
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2
    save('final_report.json',report);return report
