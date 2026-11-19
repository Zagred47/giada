"""Exploratory single-compartment composition of frozen compact memory."""
import copy,gzip,hashlib,io,json,math,time,zipfile
from pathlib import Path
import numpy as np
from . import task50_compact_memory as latent
from . import task51f_local_targets as readout
from . import task37_ionic_playground as metrics
from . import multi_event_evaluation as events
q=latent.q;a=latent.a
def plan(root):
    p=json.loads((root/'experiments/task52_preregistration_v1.json').read_text());old=json.loads((root/'experiments/task46_preregistration_v1.json').read_text())
    assert p['task']=='52' and p['training_updates']==0 and not p['thresholds_changed'];p['quality']=old['quality'];p['compartment_gates']=old['compartment_gates'];return p
def load(torch,root,package,source,c,d,out,device):
    p=plan(root);models,_,_=a.load_frozen(torch,root,source,c,d,out,device)
    with zipfile.ZipFile(package) as bundle:
        archives={k:bundle.read(k+'.zip') for k in p['inputs']}
    for k,data in archives.items():assert hashlib.sha256(data).hexdigest()==p['inputs'][k]['sha256']
    with zipfile.ZipFile(io.BytesIO(archives['task51'])) as z:
        cp=json.loads(z.read('preregistration.json'));freeze=json.loads(z.read('selection_freeze.json'));state=torch.load(io.BytesIO(z.read('selected.pt')),map_location=device,weights_only=True)
    indices=[min((i for i,m in enumerate(freeze['members']) if m['seed']==seed),key=lambda i:freeze['development'][i]) for seed in p['seeds']]
    specs=[freeze['members'][i] for i in indices];assert all(m['widths']==[2,3,3,1,1] and m['trunk']==16 for m in specs)
    cp=copy.deepcopy(cp);cp['arms']=[dict(id='compact',widths=[2,3,3,1,1],trunk=16)];compact=latent.model_class(torch)(cp,device);compact.load_state_dict({k:v[indices].clone() for k,v in state.items()})
    with zipfile.ZipFile(io.BytesIO(archives['task51j'])) as z:
        rp=json.loads(z.read('preregistration.json'));rf=json.loads(z.read('selection_freeze.json'));rs=torch.load(io.BytesIO(z.read('checkpoint_5000.pt')),map_location=device,weights_only=True)
    ri=[i for i,m in enumerate(rf['members']) if m['arm']=='sgd'];assert len(ri)==3
    rp=copy.deepcopy(rp);rp['arms']=[next(m for m in rp['arms'] if m['id']=='sgd')]
    ro=readout.model_class(torch)(rp,torch.ones((1,11),device=device)*.1,device);ro.load_state_dict({k:v[ri].clone() for k,v in rs.items()})
    models.update(compact=compact,J=ro)
    for m in models.values():
        m.eval()
        for v in m.parameters():v.requires_grad_(False);assert bool(torch.isfinite(v).all())
    a.n.p.save_json(out/'frozen_contract.json',dict(compact_source_indices=indices,compact_source_members=specs,development_only_choice=True,readout_source_indices=ri,readout_checkpoint='final SGD',logical_compact_state=10,padded_temporary=18,weights={k:a.n.weights_hash(v) for k,v in models.items()}))
    return models
def design(p):
    clock=np.arange(round(p['duration_ms']/p['dt_ms']))*p['dt_ms'];injections=[];mults=[];revs=[];rows=[]
    for protocol in p['protocols']:
        for intervention in p['interventions']:
            inj=np.zeros(len(clock))
            if protocol=='burst':
                for start in (20,40,60,80,100):inj[(clock>=start)&(clock<start+2)]=.018
            elif protocol=='hyper_rebound':inj[(clock>=35)&(clock<245)]=-.0034;inj[(clock>=337)&(clock<353)]=.0165
            elif protocol=='relaxation':inj[(clock>=20)&(clock<70)]=.012
            else:raise ValueError(protocol)
            mult=np.ones(11);rev=np.array(q.ionic.REVERSALS,dtype=float)
            if intervention=='block_Ca':mult[:2]=0
            elif intervention=='block_Na':mult[2:5]=0
            elif intervention=='reverse_Ca':rev[:2]+=20
            rows.append(dict(protocol=protocol,intervention=intervention));injections.append(inj);mults.append(mult);revs.append(rev)
    return rows,np.array(injections).T,np.array(mults),np.array(revs)
def opening(torch,g):
    return torch.stack([g[...,int(lo)].pow(int(q.ionic.POWERS[k]))*(g[...,int(lo)+1] if hi-lo==2 else 1.) for k,(lo,hi) in enumerate(zip(q.ionic.OFFSETS[:-1],q.ionic.OFFSETS[1:]))],-1)
def physical(torch,v,ca,conductance,injection,rev,physics,dt):
    _,calcium,base=physics;current=conductance*(v[...,None]-rev)
    cap=.001*base['cm_uf_cm2']/dt;leak=base['g_pas_s_cm2']
    nv=(cap*v+leak*base['e_pas_mv']+(conductance*rev).sum(-1)+injection)/(cap+leak+conductance.sum(-1))
    decay=math.exp(-dt/calcium['decay_ms']);source=-10000*current[...,:2].sum(-1)*calcium['gamma']/(2*96485.33212*calcium['depth_um'])
    nc=1e-4+(ca-1e-4)*decay+source*calcium['decay_ms']*(1-decay)
    return nv,nc,current
def compact_step(torch,net,v,ca,h):return net.step(latent.features(torch,v.float(),ca.float()),torch.nn.functional.pad(h,(0,8)))[...,:10]
def compact_opening(torch,net,h):return net.opening(torch.nn.functional.pad(h,(0,8)))
def reference(torch,p,inj,mult,rev,physics,device):
    C=mult.shape[0];T=len(inj);v=torch.full((3,C),-65.,dtype=torch.float64,device=device);ca=torch.full_like(v,1e-4);g=torch.tensor(q.steady(-65.,1e-4),device=device)[None,None].expand(3,C,-1).clone()
    vhist=[v.clone()];chist=[ca.clone()];curr=[];conds=[]
    for t in range(T):
        g=q.canonical_step(torch,g,v,ca,p['dt_ms']);conductance=opening(torch,g)*torch.tensor(q.ionic.GBAR,device=device,dtype=torch.float64)*mult
        v,ca,current=physical(torch,v,ca,conductance,inj[t],rev,physics,p['dt_ms']);vhist.append(v);chist.append(ca);curr.append(current);conds.append(conductance)
    assert bool(torch.isfinite(v).all()) and bool(torch.isfinite(ca).all()) and bool(torch.isfinite(g).all())
    return dict(voltage=torch.stack(vhist),calcium=torch.stack(chist),current=torch.stack(curr),conductance=torch.stack(conds))
def simulate(torch,p,models,inj,mult,rev,physics,ref,device,out):
    C=mult.shape[0];A=len(p['arms']);T=len(inj);B=A*C;arms=p['arms'];dt=p['dt_ms']
    v=torch.full((3,A,C),-65.,dtype=torch.float64,device=device);ca=torch.full_like(v,1e-4);state=torch.tensor(q.steady(-65.,1e-4),device=device)[None,None,None].expand(3,A,C,-1).clone()
    h=models['compact'].initialize(state.float().reshape(3,B,18))[...,:10];bar=torch.tensor(q.ionic.GBAR,device=device,dtype=torch.float64);mm=mult[None];rr=rev[None]
    typ=torch.tensor([0 if n.startswith('canonical') else 1 if n.startswith('P') else 2 for n in arms],device=device)[None,:,None,None]
    c64=torch.tensor([n in ('canonical64','canonical_J') for n in arms],device=device)[None,:,None,None]
    vh=torch.empty((T+1,3,A,C),device=device);ch=torch.empty_like(vh);ih=torch.empty((T,3,A,C,11),device=device);vh[0]=v;ch[0]=ca
    started=time.monotonic()
    for tick in range(T):
        for i,name in enumerate(arms):
            if name=='compact_imposed':v[:,i]=ref['voltage'][tick];ca[:,i]=ref['calcium'][tick]
            elif name=='compact_ca_imposed':ca[:,i]=ref['calcium'][tick]
            if name=='compact_reset' and tick==round(100/dt):h[:,i*C:(i+1)*C]=0
        vflat=v.reshape(3,B);caflat=ca.reshape(3,B)
        gc=q.canonical_step(torch,state,v,ca,dt)
        gf=q.canonical_step(torch,state.float(),v.float(),ca.float(),dt).double();gc=torch.where(c64,gc,gf)
        gp=models['P'](vflat.float(),caflat.float(),state.float().reshape(3,B,18)).reshape(3,A,C,18).double()
        state=torch.where(typ==0,gc,torch.where(typ==1,gp,state))
        h=compact_step(torch,models['compact'],vflat,caflat,h)
        op=opening(torch,state)
        learned=models['J'](readout.b.inputs(torch,state.float().reshape(3,B,18))).reshape(3,A,C,11).double()
        lop=compact_opening(torch,models['compact'],h).reshape(3,A,C,11).double()
        for i,name in enumerate(arms):
            if name in ('canonical_J','P_J'):op[:,i]=learned[:,i]
            elif not name.startswith('canonical') and not name.startswith('P'):op[:,i]=lop[:,i]
        cond=op*bar*mm
        for i,name in enumerate(arms):
            if name.startswith('oracle_'):
                channels=list(range(11)) if name=='oracle_all' else p['families'][name.removeprefix('oracle_')]
                view=cond[:,i];view[...,channels]=ref['conductance'][tick][...,channels]
        v,ca,current=physical(torch,v,ca,cond,inj[tick][None],rr,physics,dt)
        vh[tick+1]=v;ch[tick+1]=ca;ih[tick]=current
        if tick%1000==0:
            assert bool(torch.isfinite(v).all()) and bool(torch.isfinite(ca).all()) and bool(torch.isfinite(h).all())
            print(f'[GIADA52] coupled {tick}/{T}',flush=True)
            a.n.p.save_json(out/'progress.json',dict(step=tick,total=T,elapsed_seconds=time.monotonic()-started))
        if time.monotonic()-started>p['runtime_budget_seconds']*.8:raise TimeoutError('52 time budget')
    assert bool(torch.isfinite(v).all()) and bool(torch.isfinite(ca).all()) and bool(torch.isfinite(h).all())
    return dict(voltage=vh.cpu().numpy(),calcium=ch.cpu().numpy(),current=ih.cpu().numpy())

def current_metrics(ref,pred,dt,quality):
    r=metrics.evaluate_currents(ref,pred,dt,quality)
    multi=[events.evaluate_events(ref[:,k],pred[:,k],dt,quality,details=False) for k in range(11)]
    r['multi_event']=multi;r['original_passed']=r['passed'];r['conservative_passed']=r['passed'] and all(v['passed'] for v in multi)
    return r

def evaluate(p,trace,ref,rows,out):
    quality=p['quality'];gates=p['compartment_gates'];results=[];responses=[]
    reference={k:v.cpu().numpy() for k,v in ref.items() if k!='conductance'}
    for i,arm in enumerate(p['arms']):
        for horizon in p['horizons_ms']:
            end=round(horizon/p['dt_ms'])
            for s,seed in enumerate(p['seeds']):
                for c,case in enumerate(rows):
                    cm=current_metrics(reference['current'][:end,s,c],trace['current'][:end,s,i,c],p['dt_ms'],quality)
                    vr=float(np.sqrt(np.mean((trace['voltage'][:end+1,s,i,c]-reference['voltage'][:end+1,s,c])**2)))
                    cr=float(np.sqrt(np.mean((trace['calcium'][:end+1,s,i,c]-reference['calcium'][:end+1,s,c])**2)))
                    physical=bool(np.isfinite(trace['voltage'][:end+1,s,i,c]).all() and np.isfinite(trace['calcium'][:end+1,s,i,c]).all() and (trace['voltage'][:end+1,s,i,c]>=-140).all() and (trace['voltage'][:end+1,s,i,c]<=100).all() and (trace['calcium'][:end+1,s,i,c]>=0).all())
                    imposed=arm=='compact_imposed';oracle=arm.startswith('oracle_');passv=vr<=gates['max_candidate_worst_voltage_rmse_mv'];passca=cr<=gates['max_candidate_worst_cai_rmse_mM']
                    results.append(dict(arm=arm,seed=seed,case=c,**case,horizon_ms=horizon,voltage_rmse_mv=vr,calcium_rmse_mM=cr,physical=physical,imposed=imposed,oracle=oracle,selection_eligible=not imposed and not oracle and arm!='compact_ca_imposed',passed=bool(cm['conservative_passed'] and passv and passca and physical),current=cm))
                    if case['intervention']!='none':
                        base=next(j for j,x in enumerate(rows) if x['protocol']==case['protocol'] and x['intervention']=='none')
                        delta=current_metrics(reference['current'][:end,s,c]-reference['current'][:end,s,base],trace['current'][:end,s,i,c]-trace['current'][:end,s,i,base],p['dt_ms'],quality)
                        vd=(trace['voltage'][:end+1,s,i,c]-trace['voltage'][:end+1,s,i,base])-(reference['voltage'][:end+1,s,c]-reference['voltage'][:end+1,s,base])
                        cd=(trace['calcium'][:end+1,s,i,c]-trace['calcium'][:end+1,s,i,base])-(reference['calcium'][:end+1,s,c]-reference['calcium'][:end+1,s,base])
                        responses.append(dict(arm=arm,seed=seed,case=c,horizon_ms=horizon,current=delta,voltage_response_rmse_mv=float(np.sqrt(np.mean(vd**2))),calcium_response_rmse_mM=float(np.sqrt(np.mean(cd**2)))))
        print('[GIADA52] evaluated '+arm,flush=True)
    with gzip.open(out/'evaluation.json.gz','wt') as f:json.dump(results,f,allow_nan=False)
    with gzip.open(out/'interventions.json.gz','wt') as f:json.dump(responses,f,allow_nan=False)
    summary={}
    for arm in p['arms']:
        summary[arm]={}
        for horizon in p['horizons_ms']:
            subset=[r for r in results if r['arm']==arm and r['horizon_ms']==horizon]
            pooled=float(np.sqrt(np.mean([r['voltage_rmse_mv']**2 for r in subset])))
            passed=sum(r['passed'] for r in subset)
            summary[arm][str(horizon)]=dict(passed=passed,total=len(subset),worst_voltage_rmse=max(r['voltage_rmse_mv'] for r in subset),pooled_voltage_rmse=pooled,pooled_voltage_passed=pooled<=gates['max_candidate_pooled_voltage_rmse_mv'],all_quality_passed=passed==len(subset) and pooled<=gates['max_candidate_pooled_voltage_rmse_mv'])
    return summary,results

def benchmark(torch,p,models,device,out):
    B=p['cost']['batch'];v=torch.full((3,B),-65.,device=device);ca=torch.full_like(v,1e-4);state=torch.tensor(q.steady(-65,1e-4),device=device,dtype=torch.float32)[None,None].expand(3,B,-1).clone();h=models['compact'].initialize(state)[...,:10]
    functions={'canonical64':lambda:opening(torch,q.canonical_step(torch,state.double(),v.double(),ca.double(),p['dt_ms'])),'P_full18':lambda:opening(torch,models['P'](v,ca,state)),'J_readout_only':lambda:models['J'](readout.b.inputs(torch,state)),'compact10_step_readout':lambda:compact_opening(torch,models['compact'],compact_step(torch,models['compact'],v,ca,h))}
    samples={}
    for key,fn in functions.items():
        fn();vals=[]
        for _ in range(p['cost']['repetitions']):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(p['cost']['ticks']):fn()
            if device=='cuda':torch.cuda.synchronize()
            vals.append((time.perf_counter()-start)*1000)
        samples[key]=vals
    a.n.p.save_json(out/'cost.json',dict(samples_ms=samples,state_bytes_per_instance=dict(P=72,compact=40,compact_padded_temporary=72),batch=B,seeds=3,scope=p['cost']['scope'],whole_neuron_speedup_claimed=False))

def run(torch,root,source,c,d,package,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);start=time.monotonic()
    a.n.p.save_json(out/'preregistration.json',p);a.n.p.save_json(out/'unchanged_quality.json',dict(quality=p['quality'],compartment_gates=p['compartment_gates']))
    models=load(torch,root,package,source,c,d,out,device);before={k:a.n.weights_hash(v) for k,v in models.items()}
    rows,inj,mult,rev=design(p);a.n.p.save_json(out/'cases.json',rows)
    inj=torch.tensor(inj,device=device,dtype=torch.float64);mult=torch.tensor(mult,device=device,dtype=torch.float64);rev=torch.tensor(rev,device=device,dtype=torch.float64);physics=a.n.contract(root)[1:]
    with torch.no_grad():
        parity=q.parity(torch,device);ref=reference(torch,p,inj,mult,rev,physics,device);trace=simulate(torch,p,models,inj,mult,rev,physics,ref,device,out)
        np.savez_compressed(out/'traces.npz',**trace,reference_voltage=ref['voltage'].cpu().numpy(),reference_calcium=ref['calcium'].cpu().numpy(),reference_current=ref['current'].cpu().numpy())
        summary,results=evaluate(p,trace,ref,rows,out);benchmark(torch,p,models,device,out)
    after={k:a.n.weights_hash(v) for k,v in models.items()};assert before==after
    control=[r for r in results if r['arm']=='canonical64'];floor=all(r['voltage_rmse_mv']<=p['compartment_gates']['max_formula_floor_worst_voltage_rmse_mv'] and r['calcium_rmse_mM']<=p['compartment_gates']['max_formula_floor_worst_cai_rmse_mM'] and r['current']['conservative_passed'] for r in control)
    assert floor,'Discrete self-reference floor failed; do not attribute candidate error'
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='52',exploratory=True,training_updates=0,selection_recomputed=False,thresholds_changed=False,reference_kind='canonical discrete old-Ca gate splitting; no new NEURON audit',formula_parity=parity,reference_floor_passed=floor,finite_parameters=all(all(bool(torch.isfinite(x).all()) for x in v.parameters()) for v in models.values()),weights_unchanged=True,summary=summary,elapsed_seconds=time.monotonic()-start,limitations=p['limitations']))
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2
