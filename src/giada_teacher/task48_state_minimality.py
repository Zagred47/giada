"""Diagnostic state probes, not a compressed-model training or release gate."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from . import task46_spatial_cost as a

ionic=a.t.ionic

def plan(root):
    p=json.loads((root/'experiments/task48_preregistration_v1.json').read_text())
    assert p['task']=='48' and p['training_updates']==0 and not p['selection_performed'] and not p['thresholds_changed']
    assert sorted(sum(p['families'].values(),[]))==list(range(11))
    assert p['weights_sha256']==a.spec(root)[0]['freeze']['logical_weights_sha256']
    return p

def identity(p):
    return dict(task='48',plan_sha256=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest())

def steady(v=-65.,ca=5e-5):
    return np.concatenate([np.asarray(ionic.rates(ch,np.array([v]),np.array([ca]))[0]).reshape(-1) for ch in ionic.CHANNELS])

def openings(g):
    return np.stack([g[...,lo]**ionic.POWERS[k]*(g[...,lo+1] if hi-lo==2 else 1.) for k,(lo,hi) in enumerate(zip(ionic.OFFSETS[:-1],ionic.OFFSETS[1:]))],-1)

def currents(g,v):
    return openings(g)*np.asarray(ionic.GBAR)*(np.asarray(v)[...,None]-np.asarray(ionic.REVERSALS))

def collision_bank(p):
    states=[];pairs=[]
    for k,(lo,hi) in enumerate(zip(ionic.OFFSETS[:-1],ionic.OFFSETS[1:])):
        if hi-lo!=2:continue
        for opening in p['collision_openings']:
            ids=[]
            for fraction in p['collision_m_fractions']:
                g=steady();minimum=opening**(1./ionic.POWERS[k]);m=minimum+(1-minimum)*fraction
                g[lo]=m;g[lo+1]=opening/m**ionic.POWERS[k];ids.append(len(states));states.append(g)
            pairs.append(dict(channel=ionic.CHANNELS[k],channel_index=k,opening=opening,left=ids[0],right=ids[1],reachability='not_established'))
    states=np.array(states)
    for row in pairs:assert np.max(abs(openings(states[row['left']])-openings(states[row['right']])))<1e-14
    return states,pairs

def history_inputs(p):
    rng=np.random.default_rng(p['input_seed']);n=p['history_count'];ticks=round(p['history_ms']/p['dt_ms']);clock=np.arange(ticks)*p['dt_ms']
    v=np.empty((ticks,n));ca=np.empty_like(v);meta=[]
    for j in range(n):
        family=j%6;level=float(rng.uniform(-115,25));base=np.full(ticks,-65.)
        if family==0:base[clock<300]=level
        elif family==1:base[clock<180]=-110.;base[(clock>=180)&(clock<300)]=level
        elif family==2:base[clock<300]=np.where((clock[clock<300]%(10+j%7))<2,20.,-75.)
        elif family==3:base[clock<300]=-65+40*np.sin(clock[clock<300]*(.02+.003*(j%5)))
        elif family==4:base[clock<300]=np.linspace(-115,20,int((clock<300).sum()))
        else:base[clock<300]=level;base[(clock>=300)&(clock<350)]=-100.
        v[:,j]=base;ca[:,j]=5e-5
        ca[clock<350,j]=10**rng.uniform(-5,-3.5)
        meta.append(dict(id=j,kind=family,conditioning_level_mv=level,terminal_voltage_mv=-65.,terminal_calcium_mM=5e-5))
    return v,ca,meta

def future_input(name,t):
    if name=='rest':return -65.,5e-5
    if name=='pulse_recovery':return (-15. if 10<=t<30 or 80<=t<85 else -65.),5e-5
    if name=='hyper_calcium':return (-110. if t<150 else -65.),(2e-4 if 50<=t<250 else 5e-5)
    raise ValueError(name)

def coefficients(torch,v,ca):
    inf=[];tau=[]
    for name in ionic.CHANNELS:
        u,tau_k=a.t._rates(name,v,ca,torch)
        inf.append(u if u.ndim>v.ndim else u[...,None]);tau.append(tau_k if tau_k.ndim>v.ndim else tau_k[...,None])
    return torch.cat(inf,-1),torch.cat(tau,-1)

def canonical_step(torch,g,v,ca,dt=.1):
    inf,tau=coefficients(torch,v,ca);z=-torch.expm1(-dt/tau)
    return g+z*(inf-g)

def parity(torch,device):
    volts=np.array([-130.,-110.,-85.,-65.,-40.,-20.,0.,30.]);calcium=np.geomspace(1e-6,1e-3,len(volts))
    u,tau=coefficients(torch,torch.tensor(volts,device=device,dtype=torch.float64),torch.tensor(calcium,device=device,dtype=torch.float64))
    expected=[ionic.rates(ch,volts,calcium) for ch in ionic.CHANNELS]
    want_u=np.concatenate([np.asarray(row[0]).reshape(len(volts),-1) for row in expected],-1);want_tau=np.concatenate([np.asarray(row[1]).reshape(len(volts),-1) for row in expected],-1)
    errors=dict(inf_max_abs=float(abs(u.cpu().numpy()-want_u).max()),tau_max_relative=float((abs(tau.cpu().numpy()-want_tau)/np.maximum(abs(want_tau),1e-12)).max()))
    assert max(errors.values())<1e-10,errors
    return dict(valid=True,**errors,scope='NumPy versus Torch audited formulas; not a new native NEURON audit')

def neighbour_pairs(states,families):
    op=openings(states);cur=currents(states,-65.);views={'full_gates':states,'channel_openings':op,'family_currents':np.stack([cur[:,ks].sum(-1) for ks in families.values()],-1)}
    result=[]
    for name,view in views.items():
        scale=np.maximum(np.std(view,axis=0),1e-12);value=view/scale
        dist=np.sqrt(np.mean((value[:,None]-value[None,:])**2,-1));np.fill_diagonal(dist,np.inf)
        pairs=sorted({tuple(sorted((j,int(np.argmin(dist[j]))))) for j in range(len(states))})
        for left,right in pairs:result.append(dict(view=name,left=left,right=right,input_distance=float(dist[left,right]),exact_collision=bool(np.array_equal(view[left],view[right]))))
    return result

def family_indices(p):
    return {name:[g for ch in channels for g in range(int(ionic.OFFSETS[ch]),int(ionic.OFFSETS[ch+1]))] for name,channels in p['families'].items()}

def run(torch,root,source,ck45c,ck45d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;started=time.monotonic();out.mkdir(exist_ok=True)
    save=lambda name,value:a.n.p.save_json(out/name,value)
    save('executed_protocol.json',identity(p));save('preregistration.json',p)
    control=parity(torch,device);save('formula_parity.json',control)
    models,_,_=a.load_frozen(torch,root,source,ck45c,ck45d,out,device);net=models['P'];before=a.n.weights_hash(net);assert before==p['weights_sha256']
    seed_count=3;dt=p['dt_ms'];v_hist,ca_hist,meta=history_inputs(p);count=len(meta)
    base=np.tile(steady(),(seed_count,count,1));banks={};history_cost={}
    for label,dtype in [('canonical64',torch.float64),('canonical32',torch.float32),('P',torch.float32)]:
        g=torch.tensor(base,device=device,dtype=dtype);clock=time.monotonic()
        for tick,(vv,cc) in enumerate(zip(v_hist,ca_hist)):
            v=torch.tensor(vv,device=device,dtype=dtype)[None].expand(seed_count,-1);ca=torch.tensor(cc,device=device,dtype=dtype)[None].expand(seed_count,-1)
            g=net(v,ca,g) if label=='P' else canonical_step(torch,g,v,ca,dt)
            if tick%1000==0:print(f'[GIADA48] history {label} {tick}/{len(v_hist)}',flush=True)
        assert bool(torch.isfinite(g).all()) and bool(((g>=0)&(g<=1)).all())
        banks[label]=g.cpu().numpy();history_cost[label]=time.monotonic()-clock
    save('history_metadata.json',meta)
    np.savez_compressed(out/'reachable_states.npz',**banks)
    coll,pairs=collision_bank(p);refs=banks['canonical64'][0];neighbours=neighbour_pairs(refs,p['families']);indices=family_indices(p)
    probes=[];pert=[]
    for anchor in range(min(p['sensitivity_anchor_count'],count)):
        for gate in range(18):
            g=refs[anchor];eps=min(p['sensitivity_epsilon'],float(g[gate])*.25,float(1-g[gate])*.25)
            if eps<1e-10:continue
            ids=[]
            for sign in (-1,1):
                q=g.copy();q[gate]+=sign*eps;ids.append(len(pert));pert.append(q)
            probes.append(dict(anchor=anchor,gate=gate,epsilon=eps,left=ids[0],right=ids[1]))
    pert=np.array(pert).reshape(-1,18);initial=np.concatenate((refs,coll,pert,refs[:1],refs[:1]),0);offset=count;poffset=count+len(coll)
    states0=np.broadcast_to(initial,(3,*initial.shape)).copy()
    save('probe_design.json',dict(collision_pairs=pairs,neighbours=neighbours,sensitivity=probes,scope=p['scope'],history_states_are_canonical=True))
    results=[];snapshots={};witness={};thresholds=a.spec(root)[0]['quality'];save('unchanged_quality.json',thresholds)
    # Primary comparison shares canonical endpoint states, avoiding inherited history error confounding.
    for future in p['future_protocols']:
        for label,dtype in [('canonical64',torch.float64),('canonical32',torch.float32),('P',torch.float32)]:
            g=torch.tensor(states0,device=device,dtype=dtype);last=round(max(p['horizons_ms'])/dt);wanted={round(h/dt):h for h in p['horizons_ms']};samples={}
            trace=[];cpu_initial=currents(initial,-65.)
            initial_v=torch.full(g.shape[:-1],-65.,device=device,dtype=dtype)
            actual_initial=a.t.readout(torch,initial_v,g,torch.ones((*initial_v.shape,11),device=device,dtype=dtype))[1].cpu().numpy()
            channel_ids=torch.tensor([row['channel_index'] for row in pairs for _ in (0,1)],device=device)
            witness_ids=torch.arange(offset,poffset,device=device)
            for tick in range(last+1):
                vv,cc=future_input(future,tick*dt);v=torch.full(g.shape[:-1],vv,device=device,dtype=dtype);ca=torch.full_like(v,cc)
                current=a.t.readout(torch,v,g,torch.ones((*v.shape,11),device=device,dtype=dtype))[1]
                if tick in wanted:samples[str(wanted[tick])]=current.cpu().numpy();snapshots[f'{future}_{label}_{wanted[tick]}']=samples[str(wanted[tick])]
                # All 21 exact witnesses retained as short current traces, no full bank trajectories.
                trace.append(current[:,witness_ids,channel_ids].cpu().numpy())
                if tick%2000==0:print(f'[GIADA48] future {future} {label} {tick}/{last}',flush=True)
                if tick<last:g=net(v,ca,g) if label=='P' else canonical_step(torch,g,v,ca,dt)
                if time.monotonic()-started>p['runtime_budget_seconds']:raise TimeoutError('48 budget; saved partial state retained')
            assert bool(torch.isfinite(g).all()) and bool(((g>=0)&(g<=1)).all())
            trace=np.stack(trace);witness[future+'_'+label]=trace
            np.savez_compressed(out/f'panel_{future}_{label}.npz',witness_current=trace,**samples)
            for horizon,values in samples.items():
                assert np.array_equal(values[:,-1],values[:,-2]),'Identical-state control failed'
                for row in pairs:
                    l,r=offset+row['left'],offset+row['right'];ch=row['channel_index'];end=round(float(horizon)/dt)+1
                    tr=trace[:end,:,row['left']];pr=trace[:end,:,row['right']];scale=np.maximum(np.maximum(abs(tr).max(0),abs(pr).max(0)),1e-7)
                    gap=abs(tr-pr).max(0)/scale
                    results.append(dict(kind='exact_opening_collision',future=future,model=label,horizon_ms=float(horizon),channel=row['channel'],opening=row['opening'],initial_current_gap=float(abs(cpu_initial[l,ch]-cpu_initial[r,ch])),actual_dtype_initial_current_gap=abs(actual_initial[:,l,ch]-actual_initial[:,r,ch]).tolist(),max_future_gap_normalized=gap.tolist(),reachability='not_established',shared_scale_triangle_bound_only=True))
                for row in neighbours:
                    l,r=row['left'],row['right'];scale=np.maximum(np.maximum(abs(values[:,l]),abs(values[:,r])),1e-7)
                    results.append(dict(kind='reachable_neighbours',future=future,model=label,horizon_ms=float(horizon),**row,endpoint_future_gap_normalized=(abs(values[:,l]-values[:,r])/scale).max(-1).tolist(),impossibility_proof=False))
                for anchor in range(min(p['sensitivity_anchor_count'],count)):
                    for family,gates in indices.items():
                        columns=[]
                        for row in probes:
                            if row['anchor']==anchor and row['gate'] in gates:columns.append((values[:,poffset+row['right']]-values[:,poffset+row['left']])/(2*row['epsilon']))
                        if columns:
                            jac=np.stack(columns,-1);singular=np.linalg.svd(jac,compute_uv=False)
                            results.append(dict(kind='local_sensitivity',future=future,model=label,horizon_ms=float(horizon),anchor=anchor,family=family,singular_values=singular.tolist(),units='current_density_per_gate',not_global_minimal_rank=True))
            for anchor in range(min(p['sensitivity_anchor_count'],count)):
                for family,gates in indices.items():
                    columns=[]
                    for row in probes:
                        if row['anchor']==anchor and row['gate'] in gates:
                            columns.append(np.concatenate([(values[:,poffset+row['right']]-values[:,poffset+row['left']])/(2*row['epsilon']) for values in samples.values()],-1))
                    if columns:results.append(dict(kind='stacked_horizon_observability',future=future,model=label,anchor=anchor,family=family,singular_values=np.linalg.svd(np.stack(columns,-1),compute_uv=False).tolist(),horizons_ms=p['horizons_ms'],not_global_minimal_rank=True))
            save('diagnostic_rows.json',results)
    # Per-panel files already preserve all arrays; do not duplicate them in a second archive.
    pca={}
    for family,gates in indices.items():
        value=refs[:,gates];s=np.linalg.svd(value-value.mean(0),compute_uv=False);pca[family]=dict(gates=gates,singular_values=s.tolist(),centered=True,scaling='raw_occupancy',not_minimal_dynamical_dimension=True)
    assert a.n.weights_hash(net)==before
    controls=dict(valid=True,formula_parity=control,identical_state_future=True,finite_parameters_and_states=True,occupancy_preserved=True,weights_hash_before=before,weights_hash_after=a.n.weights_hash(net),single_gate_opening_invertibility=all(ionic.POWERS[k]>0 for k in (5,6,9,10)))
    comparisons=[]
    for future in p['future_protocols']:
        for horizon in p['horizons_ms']:
            ref=snapshots[f'{future}_canonical64_{horizon}'][:,:count]
            for label in ('canonical32','P'):
                pred=snapshots[f'{future}_{label}_{horizon}'][:,:count]
                comparisons.append(dict(future=future,horizon_ms=horizon,model=label,endpoint_current_rmse_by_seed_channel=np.sqrt(np.mean((pred-ref)**2,axis=1)).tolist(),not_full_trajectory_quality_gate=True))
    report=dict(valid=True,complete=True,task='48',exploratory=True,training_updates=0,selection_performed=False,thresholds_changed=False,diagnostic_rows=len(results),families=pca,controls=controls,history_wall_seconds=history_cost,history_gate_rmse_by_seed={label:np.sqrt(np.mean((value-banks['canonical64'])**2,axis=1)).tolist() for label,value in banks.items()},paired_current_comparisons=comparisons,elapsed_seconds=time.monotonic()-started,limitations47=p['limitations47'],scope=p['scope'],decision=p['decision'],speedup_claimed=False,compressed_model_qualified=False)
    save('final_report.json',report)
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2
    return report
