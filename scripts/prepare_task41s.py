"""Deterministic41s same-head repair; fitting never opens evaluation panels."""
import copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import audit_task41q as q
from prepare_task41r import feature_tables
from research_memory.mirror import write_json
from src.giada_teacher import task41r_direct_timing as r
p=r.p

def crossing_rows(torch,base,prior,seeds,window=False,coverage=False):
    spec=dict(prior,stress_seeds=seeds);meta=r.n.stress_data(spec)
    if coverage:
        for index,row in enumerate(meta):
            clock=np.arange(len(row['v']))
            factor=(.5,1.,2.)[index%3]
            row['v']=np.interp(clock*factor,clock,row['v']).astype('float32')
            row['ca']=np.interp(clock*factor,clock,row['ca']).astype('float32')
            iv=(-70.,-35.)[row['history']]
            row['initial']=r.n.p.coefficients(np.array(iv),np.array(row['ca'][0]),.1)[0]
    v=np.stack([x['v'] for x in meta],1).astype('float32');ca=np.stack([x['ca'] for x in meta],1).astype('float32')
    initial=np.stack([x['initial'] for x in meta])[:,list(p.EQ)];mult=np.array([spec['panels'][x['panel']] for x in meta])
    uv,iv=np.unique(v,return_inverse=True);uc,ic=np.unique(ca,return_inverse=True);iv=iv.reshape(v.shape);ic=ic.reshape(ca.shape)
    vu,vz=q.actual(torch,base,uv,np.full_like(uv,1e-4));cu,cz=q.actual(torch,base,np.full_like(uc,-65),uc)
    vu=q.sigmoid(vu[...,list(p.EQ)]);vz=q.sigmoid(vz[...,list(p.EQ)]);cu=q.sigmoid(cu[...,16]);cz=q.sigmoid(cz[...,16])
    tv,tz=r.n.p.coefficients(uv.astype('float64'),np.full(len(uv),1e-4),.1);tc,tcz=r.n.p.coefficients(np.full(len(uc),-65.),uc.astype('float64'),.1)
    f=feature_tables(torch,base,uv);fc=feature_tables(torch,base,uc,True)
    state=initial.copy();truth=[]
    for tick in range(len(v)):
        u=tv[iv[tick]][:,list(p.EQ)].copy();z=tz[iv[tick]][:,list(p.EQ)].copy();u[:,5]=tc[ic[tick],16];z[:,5]=tcz[ic[tick],16]
        state+=z*(u-state);truth.append(state.copy())
    truth=np.stack(truth);currents={name:r.channel_current(truth,v,mult,name) for name in r.GROUPS};schedule={};events=[]
    for name,signal in currents.items():
        for k in range(v.shape[1]):
            rr=signal[:,k];event=r.n.p.event_metrics(rr,rr,.1,prior['quality']['current_scale_floor_ma_cm2'],prior['quality']['peak_plateau_relative_band'])
            if event['status']!='evaluable':continue
            excursion=abs(rr-rr[0]);active=np.flatnonzero(excursion>=.1*excursion.max())
            for kind,center in [('onset',int(active[0])),('offset',int(active[-1]))]:
                slope=abs(rr[min(center+1,len(rr)-1)]-rr[max(center-1,0)])/.2
                denominator=max(float(slope)*prior['quality']['onset_offset_max_ms'],1e-10)
                for tick in sorted({max(0,min(len(rr)-1,center+d)) for d in ((-100,-10,-1,0,1,10,100) if window else (-1,0,1))}):
                    schedule.setdefault(tick,[]).append((name,k,denominator))
                    events.append(dict(channel=name,case=k,kind=kind,tick=tick,denominator=denominator))
    state=np.broadcast_to(initial,(3,*initial.shape)).copy();su=np.zeros((*state.shape,65));sz=np.zeros_like(su)
    rows={name:[] for name in r.GROUPS};targets={name:[] for name in r.GROUPS}
    for tick in range(len(v)):
        u=vu[:,iv[tick]].copy();z=vz[:,iv[tick]].copy();u[:,:,5]=cu[:,ic[tick]];z[:,:,5]=cz[:,ic[tick]]
        features=np.broadcast_to(f[:,iv[tick],None,:],(*state.shape,65)).copy();features[:,:,5]=0;features[:,:,5,:fc.shape[-1]]=fc[:,ic[tick]]
        state,su,sz=r.affine_sensitivity(state,su,sz,u,z,features)
        for name,k,denominator in schedule.get(tick,[]):
            _,ids=r.GROUPS[name];derivatives=r.current_derivatives(state[:,k],v[tick,k],mult[k],name)
            parts=[]
            for kind,gate in r.keys_for(name,True):
                position=p.EQ.index(gate);derivative=derivatives[list(ids).index(position)]
                parts.append((su if kind==0 else sz)[:,k,position,:fc.shape[-1] if gate==16 else 65]*derivative[:,None])
            rows[name].append(np.concatenate(parts,-1)/denominator)
            pred=r.channel_current(state[:,k],v[tick,k],mult[k],name)
            targets[name].append((currents[name][tick,k]-pred)/denominator)

    result={}
    for name in ('Nap_Et2','K_Pst'):
        b=np.stack(rows[name],1);y=np.stack(targets[name],1)
        # Identical512 residual rows/channel/seed/linearization in every arm.
        ix=np.floor(np.linspace(0,len(y[0])-1,512)).astype(int)
        result[name]=(b[:,ix],y[:,ix])
    return result,events

def matrices(base):
    spec=json.loads((ROOT/'experiments/task41p_preregistration_v1.json').read_text())
    fit,_,_=p.pool(torch,base,spec,'cpu');v=fit['input_v'];ca=fit['input_ca']
    ru,rz=q.actual(torch,base,v,ca);white={}
    for kind,gate in r.HEADS:
        if gate not in (8,9,12,13):continue
        raw=torch.tensor((rz if kind else ru)[...,gate]);f=fit['fv']
        w=[]
        for s in range(3):
            value=torch.sigmoid(raw[s]);a=f[s]*(value*(1-value))[:,None]*1e4/np.sqrt(len(v))
            _,sv,vh=torch.linalg.svd(a,full_matrices=False)
            w.append(vh.T/torch.sqrt(sv.square()+(1e-6*sv[0]).square())[None])
        white[(kind,gate)]=w
    return white

def unpack(current,origin,keys,seed):
    return torch.cat([torch.tensor(current[f'h{k}_{g}'][seed]-origin[f'h{k}_{g}'][seed],dtype=torch.float64) for k,g in keys])

def step_weights(current,origin,keys,seed,delta,alpha):
    result=copy.deepcopy(current);offset=0
    for k,g in keys:
        key=f'h{k}_{g}';width=current[key].shape[1]
        result[key][seed]=(torch.tensor(origin[key][seed]).double()+alpha*delta[offset:offset+width]).float().numpy()
        offset+=width
    return result

def fit_arm(base,start,prior,spec,white,window,iterative,coverage):
    current=copy.deepcopy(start);records=[]
    for iteration in range(4 if iterative else 1):
        net=r.materialize(torch,base,current);jac,_=crossing_rows(torch,net,prior,spec['fit_path_seeds'],window,coverage)
        candidate=copy.deepcopy(current)
        for name in ('Nap_Et2','K_Pst'):
            keys=r.keys_for(name,True)
            for seed in range(3):
                b,y=jac[name];w=torch.block_diag(*[white[key][seed] for key in keys])
                # Center the displacement penalty on the frozen41r incumbent.
                displacement=unpack(current,start,keys,seed)
                # Solve for next displacement, relinearized at current.
                target=y[seed]+b[seed]@displacement.numpy()
                delta=r.timing_step(torch,torch.zeros_like(displacement),w,b[seed],target)
                trial=step_weights(current,start,keys,seed,delta,1.)
                for k,g in keys:candidate[f'h{k}_{g}'][seed]=trial[f'h{k}_{g}'][seed]
        if not iterative:
            records.append(dict(iteration=iteration,alpha=1.,acceptance='single unguarded linearized step'))
            current=candidate;continue
        old_score=sum(float(np.mean(y*y)) for b,y in jac.values())
        # Exact nonlinear training-path loss, never development/stress, decides backtracking.
        accepted=False
        for alpha in (1.,.5,.25,.125):
            trial={k:(current[k]+alpha*(candidate[k]-current[k])).astype('float32') for k in current}
            actual,_=crossing_rows(torch,r.materialize(torch,base,trial),prior,spec['fit_path_seeds'],window,coverage)
            score=sum(float(np.mean(y*y)) for b,y in actual.values())
            records.append(dict(iteration=iteration,alpha=alpha,old_score=old_score,actual_score=score,accepted=score<old_score))
            if score<old_score:current=trial;accepted=True;break
        print(f'[GIADA41s prepare] window={window} coverage={coverage} iter={iteration} accepted={accepted}',flush=True)
        if not accepted:break
    return current,records

def main():
    torch.set_num_threads(1);started=time.time()
    folder=ROOT/'experiments/results/task41s_protocol_v1';folder.mkdir(exist_ok=True)
    spec=json.loads((ROOT/'experiments/task41s_preregistration_v1.json').read_text())
    prior=r.n.contract(ROOT)[0]
    assert spec['quality']==prior['quality']
    excluded=prior['stress_seeds']+prior['native_seeds']
    for task in ('p','r'):
        previous=json.loads((ROOT/f'experiments/task41{task}_preregistration_v1.json').read_text())
        excluded+=previous.get('fit_path_seeds',[])+previous.get('development_seeds',[])
    assert not set(spec['fit_path_seeds']+spec['development_seeds'])&set(excluded)
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
    base=r.n.load_models(torch,ROOT,'cpu')[0]['dual'];original=r.head_weights(base)
    previous=ROOT/'experiments/results/task41r_protocol_v1'
    with np.load(previous/'heads_timing_frozen_k.npz') as a:start=dict(a)
    with np.load(previous/'heads_timing_joint_k.npz') as a:joint=dict(a)
    outputs={'incumbent':start,'joint_control':joint}
    for gate,label in ((12,'activation'),(13,'inactivation')):
        value=copy.deepcopy(start)
        for kind in (0,1):value[f'h{kind}_{gate}']=joint[f'h{kind}_{gate}'].copy()
        outputs['probe_'+label]=value
    white=matrices(r.materialize(torch,base,start));logs={}
    for window,iterative in ((False,False),(False,True),(True,False),(True,True)):
        name=('window' if window else 'crossing')+('_iterative' if iterative else '_single')
        outputs[name],logs[name]=fit_arm(base,start,prior,spec,white,window,iterative,False)
    outputs['window_iterative_coverage'],logs['window_iterative_coverage']=fit_arm(base,start,prior,spec,white,True,True,True)
    manifest=dict(valid=True,baseline_weights_sha256=r.n.weights_hash(base),incumbent_weights_sha256=r.n.weights_hash(r.materialize(torch,base,start)),models={},fit_log=logs,
        stochastic_updates=0,behavioral_selection_performed=False,elapsed_seconds=time.time()-started,rows_per_channel=512,fit_paths=24)
    for name,weights in outputs.items():
        for key in weights:
            if key not in ('h0_8','h0_9','h1_8','h1_9','h0_12','h0_13','h1_12','h1_13'):assert np.array_equal(weights[key],start[key])
        net=r.materialize(torch,base,weights);assert all(torch.isfinite(x).all() for x in net.parameters())
        assert sum(x.numel() for x in net.parameters())==sum(x.numel() for x in base.parameters())
        path=folder/f'heads_{name}.npz';np.savez_compressed(path,**weights)
        manifest['models'][name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=r.n.p.digest(path),weights_sha256=r.n.weights_hash(net))
    assert ['baseline',*outputs]==spec['arms']
    write_json(folder/'fit_manifest.json',manifest)
    print('[GIADA41s prepare] all weights frozen; no behavioral panel opened',flush=True)
if __name__=='__main__':main()
