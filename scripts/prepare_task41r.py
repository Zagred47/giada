"""Deterministic CPU preparation; no GPU, optimizer or behavioral selection."""
import copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import audit_task41q as q
from research_memory.mirror import write_json
from src.giada_teacher import task41r_direct_timing as r
p=r.p

def feature_tables(torch,base,values,calcium=False):
    rows=[]
    with torch.no_grad():
        for start in range(0,len(values),512):
            x=torch.tensor(values[start:start+512])[None].expand(3,-1)
            h=torch.nn.functional.silu(torch.bmm(((x.clamp_min(1e-9).log10()+4.5)/2.5 if calcium else (x+40)/80)[...,None],base.parent.sa if calcium else base.parent.a)+(base.parent.sb if calcium else base.parent.b))
            rows.append(torch.cat((h,torch.ones_like(h[...,:1])),-1).double().numpy())
    return np.concatenate(rows,1)

def crossing_rows(torch,base,prior,seeds):
    spec=dict(prior,stress_seeds=seeds);meta=r.n.stress_data(spec)
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
                for tick in sorted({max(0,center-1),center,min(len(rr)-1,center+1)}):
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
        if tick%3000==0:print(f'[GIADA41r prepare] causal Jacobian {tick}/{len(v)}',flush=True)
    return {name:(np.stack(rows[name],1),np.stack(targets[name],1)) for name in rows},events

def main():
    import torch
    torch.set_num_threads(1);start=time.time();folder=ROOT/'experiments/results/task41r_protocol_v1';folder.mkdir(exist_ok=True)
    spec=json.loads((ROOT/'experiments/task41r_preregistration_v1.json').read_text());prior=r.n.contract(ROOT)[0]
    spec['quality']=prior['quality'];spec['recovery_sha256']='6d2557b43a3026b35a6976a26b8ee33f1008ba6d61a57772a7d17e9f3f363a63'
    assert not set(spec['fit_path_seeds']+spec['development_seeds'])&set(prior['stress_seeds']+prior['native_seeds'])
    write_json(ROOT/'experiments/task41r_preregistration_v1.json',spec)
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin');base=r.n.load_models(torch,ROOT,'cpu')[0]['dual'];original=r.head_weights(base);base_hash=r.n.weights_hash(base)
    source=ROOT/'experiments/results/task41q_audit_v1/direct_weights_ridge_1e-06.npz';source_hash=r.n.p.digest(source)
    with np.load(source) as z:wv=z['wv'];wc=z['wc']
    frozen={k:v.copy() for k,v in original.items()};eq=copy.deepcopy(frozen)
    for i,col in enumerate(p.COLS):
        kind=1 if col>=17 else 0;gate=col-17 if kind else col;frozen[f'h{kind}_{gate}']=wv[:,:,i]
        if not kind:eq[f'h0_{gate}']=wv[:,:,i]
    eq['h0_16']=wc[:,:,0];frozen['h0_16']=wc[:,:,0]
    old=json.loads((ROOT/'experiments/task41p_preregistration_v1.json').read_text());fit,_,_=p.pool(torch,base,old,'cpu');inputs_v=fit.pop('input_v');inputs_ca=fit.pop('input_ca')
    raw_u,raw_z=q.actual(torch,base,inputs_v,inputs_ca)
    all_u,all_z=r.n.p.coefficients(inputs_v.astype('float64'),inputs_ca.astype('float64'),.1)
    joint=copy.deepcopy(frozen);whitens={};details=[]
    for kind,gate in r.HEADS:
        features=fit['fc'] if gate==16 else fit['fv'];raw=fit['sk_logits'][...,0] if gate==16 else (torch.tensor(raw_z if kind else raw_u)[...,gate])
        # Reuse the exact original41q raw anchor logits where stored.
        col=gate+17*kind
        if gate!=16 and col in p.COLS:raw=fit['raw'][...,p.COLS.index(col)]
        target=torch.tensor((all_z if kind else all_u)[:,gate]);matrices=[]
        for s in range(3):
            if kind and gate in (12,13):
                delta,_=q.ridge_delta(torch,features[s],raw[s],target,spec['ridge_relative']);joint[f'h1_{gate}'][s]=(torch.tensor(original[f'h1_{gate}'][s]).double()+delta).float().numpy()
            value=torch.sigmoid(raw[s]);a=features[s]*(value*(1-value))[:,None]*1e4/np.sqrt(len(target))
            _,singular,vh=torch.linalg.svd(a,full_matrices=False);regularizer=spec['ridge_relative']*singular[0]
            matrices.append(vh.T/torch.sqrt(singular.square()+regularizer.square())[None])
        whitens[(kind,gate)]=matrices
    jac,events=crossing_rows(torch,base,prior,spec['fit_path_seeds'])
    outputs={'q_equilibrium':eq,'coefficient_frozen_k':frozen,'coefficient_joint_k':joint}
    for with_k,coef in [(False,frozen),(True,joint)]:
        new=copy.deepcopy(coef)
        for name in r.GROUPS:
            keys=r.keys_for(name,with_k);allkeys=r.keys_for(name,True);b,y=jac[name]
            indices=[];offset=0
            for key in allkeys:
                width=original[f'h{key[0]}_{key[1]}'].shape[1]
                if key in keys:indices.extend(range(offset,offset+width))
                offset+=width
            for s in range(3):
                delta0=torch.cat([torch.tensor(coef[f'h{k}_{g}'][s]).double()-torch.tensor(original[f'h{k}_{g}'][s]).double() for k,g in keys])
                whitening=torch.block_diag(*[whitens[key][s] for key in keys])
                updated=r.timing_step(torch,delta0,whitening,b[s][:,indices],y[s]);offset=0
                for k,g in keys:
                    width=original[f'h{k}_{g}'].shape[1]
                    new[f'h{k}_{g}'][s]=(torch.tensor(original[f'h{k}_{g}'][s]).double()+updated[offset:offset+width]).float().numpy();offset+=width
        outputs['timing_joint_k' if with_k else 'timing_frozen_k']=new
    manifest=dict(valid=True,baseline_weights_sha256=base_hash,saved41q_weights_sha256=source_hash,models={},fit_events=events,stochastic_updates=0,behavioral_selection_performed=False,elapsed_seconds=time.time()-start)
    for name,weights in outputs.items():
        net=r.materialize(torch,base,weights);assert all(torch.isfinite(x).all() for x in net.parameters())
        assert sum(x.numel() for x in net.parameters())==sum(x.numel() for x in base.parameters())
        path=folder/f'heads_{name}.npz';np.savez_compressed(path,**weights)
        manifest['models'][name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=r.n.p.digest(path),weights_sha256=r.n.weights_hash(net))
    assert r.n.weights_hash(base)==base_hash;write_json(folder/'fit_manifest.json',manifest)
    print('[GIADA41r prepare] deterministic fits frozen; no behavioral selection',flush=True)
if __name__=='__main__':main()
