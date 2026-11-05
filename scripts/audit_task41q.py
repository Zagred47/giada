"""Audit existing41p checkpoints locally; deterministic diagnostic solves only."""
import copy,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research_memory.mirror import write_json
from src.giada_teacher import task41p_tail_repair as p

def sigmoid(x):return 1/(1+np.exp(-x))

def unpack_np(z,name):
    with np.load(io.BytesIO(z.read(name))) as a:return dict(a)

def actual(torch,net,v,ca):
    us=[];zs=[]
    with torch.inference_mode():
        for start in range(0,len(v),512):
            vv=torch.tensor(v[start:start+512])[None].expand(3,-1);cc=torch.tensor(ca[start:start+512])[None].expand(3,-1)
            u,z=net.raw(vv,cc);us.append(u.double().numpy());zs.append(z.double().numpy())
    return np.concatenate(us,1),np.concatenate(zs,1)

def coefficient_summary(ru,rz,data):
    uu=sigmoid(ru[...,list(p.EQ)]);zz=sigmoid(rz[...,[8,9]])
    target_u=data['u'].cpu().numpy();target_z=data['z'].cpu().numpy()[:,:2]
    eu=uu-target_u[None];ez=zz-target_z[None]
    return dict(equilibrium_rmse=np.sqrt((eu**2).mean(1)).tolist(),equilibrium_max_abs=abs(eu).max(1).tolist(),
        nap_update_rmse=np.sqrt((ez**2).mean(1)).tolist(),nap_log_update_rmse=np.sqrt(((np.log(zz.clip(1e-15))-np.log(target_z[None].clip(1e-15)))**2).mean(1)).tolist())

def objective(torch,wv,wc,data):
    total=torch.zeros((2,3),dtype=torch.float64);count=len(data['u'])
    with torch.no_grad():
        for start in range(0,count,512):
            ids=torch.arange(start,min(start+512,count))
            total+=p.loss(torch,wv,wc,data,ids,torch.tensor([[0.],[1.]]))*len(ids)
    return (total/count).tolist()

def fixed_rate_lower_bound(data):
    """Exact per-input optimum in u with z fixed, not a neural candidate."""
    u=data['u'].numpy()[None];z=data['base_z'].numpy();initial=data['initial'].numpy()[None]
    if 'raw' in data:z[:,:,:2]=sigmoid(data['raw'].numpy()[...,5:7])
    target_z=data['z'].numpy()[None];num=u.copy()+np.zeros_like(z);den=np.ones_like(z)
    terms=[]
    for ticks in (10,80,400,1000,5000,10000):
        a=np.exp(np.log1p(-z.clip(max=1-1e-15))*ticks)
        wanted=u+(initial-u)*np.exp(np.log1p(-target_z.clip(max=1-1e-15))*ticks)
        num+=(.25/6)*(1-a)*(wanted-a*initial);den+=(.25/6)*(1-a)**2;terms.append((a,wanted))
    best=(num/den).clip(0,1);loss=((best-u)/1e-4)**2
    for a,wanted in terms:loss+=(.25/6)*((a*initial+(1-a)*best-wanted)/1e-4)**2
    per_gate=loss.mean(1)
    # In joint arm Nap z is free: setting exact u,z yields zero there.
    joint=per_gate.copy();joint[:,:2]=0
    return dict(equilibrium_total_per_seed=per_gate.mean(1).tolist(),joint_total_per_seed=joint.mean(1).tolist(),
        per_gate_per_seed=per_gate.tolist(),mean_abs_optimal_u_shift=abs(best-u).mean(1).tolist(),
        scope='Analytic lower bound for41p objective with existing fixed z and arbitrary u per input, NOT an inference model or changed scientific gate. Joint bound permits exact Nap coefficients only; no capacity conclusion.')

def ridge_delta(torch,features,raw,target,relative):
    """Local probability-space least squares, not an asserted global optimum."""
    value=torch.sigmoid(raw);derivative=value*(1-value)
    a=features*derivative[:,None];rhs=target-value
    u,s,vh=torch.linalg.svd(a,full_matrices=False)
    if float(s[0])==0:
        return torch.zeros(features.shape[-1],dtype=features.dtype),dict(singular_values=s.tolist(),effective_rank=0,delta_norm=0.)
    reg=relative*s[0];delta=vh.T@((s/(s.square()+reg.square()))*(u.T@rhs))
    return delta,dict(singular_values=s.tolist(),effective_rank=int((s>s[0]*relative).sum()),delta_norm=float(delta.norm()))

def crossing_audit(torch,base,models,source_n,source_p):
    inputs=unpack_np(source_n,'fresh_stress_inputs.npz');truth=inputs['target_gates'];v=inputs['voltage'];ca=inputs['calcium']
    old=source_n.read('fresh_stress_results.json');old=json.loads(old)['dual']['rows']
    target=p.n.p.readout(v,truth,inputs['multipliers'],p.n.p.ionic.REVERSALS)[1]
    archived_gates=unpack_np(source_n,'stress_dual.npz')['gates'];meta=p.n.stress_data(p.n.contract(ROOT)[0]);rows=[]
    for arm,net in models.items():
        measured=json.loads(source_p.read(f'opened41n_{arm}/fresh_stress_results.json'))['dual']['rows']
        gates=unpack_np(source_p,f'opened41n_{arm}/stress_dual.npz')['gates']
        current=p.n.p.readout(v[:,None],gates,inputs['multipliers'],p.n.p.ionic.REVERSALS)[1]
        old_current=p.n.p.readout(v[:,None],archived_gates,inputs['multipliers'],p.n.p.ionic.REVERSALS)[1]
        for index,(before,after) in enumerate(zip(old,measured)):
            s,k=divmod(index,len(meta))
            for j,(a,b) in enumerate(zip(before['metrics']['channels'],after['metrics']['channels'])):
                if a['passed'] and b['passed']:continue
                rr=target[:,k,j];signal=abs(rr-rr[0]);active=np.flatnonzero(signal>=.1*signal.max())
                if not len(active):continue
                tick=int(active[-1]);vv=np.full(24,v[tick,k],dtype='float32');cc=np.full(24,ca[tick,k],dtype='float32')
                bu,bz=actual(torch,base,vv,cc);nu,nz=actual(torch,net,vv,cc)
                tu,tz=p.n.p.coefficients(np.array([v[tick,k]],dtype='float64'),np.array([ca[tick,k]],dtype='float64'),.1)
                lo,hi=p.n.p.OFFSETS[j:j+2]
                slope=abs(rr[min(tick+1,len(rr)-1)]-rr[max(tick-1,0)])/.2
                rows.append(dict(arm=arm,model_seed=before['model_seed'],case=k,family=before['family'],channel=a['channel'],
                    baseline_passed=a['passed'],checkpoint_passed=b['passed'],baseline_event=a['event'],checkpoint_event=b['event'],
                    reference_offset_ms=tick*.1,crossing_slope=float(slope),
                    baseline_current_error_at_crossing=float(old_current[tick,s,k,j]-rr[tick]),checkpoint_current_error_at_crossing=float(current[tick,s,k,j]-rr[tick]),
                    baseline_equilibrium_error=(sigmoid(bu[s,0,lo:hi])-tu[0,lo:hi]).tolist(),checkpoint_equilibrium_error=(sigmoid(nu[s,0,lo:hi])-tu[0,lo:hi]).tolist(),
                    baseline_update_error=(sigmoid(bz[s,0,lo:hi])-tz[0,lo:hi]).tolist(),checkpoint_update_error=(sigmoid(nz[s,0,lo:hi])-tz[0,lo:hi]).tolist()))
    return rows

def main():
    import torch
    torch.set_num_threads(1);started=time.time()
    folder=ROOT/'experiments/results/task41q_audit_v1';folder.mkdir(exist_ok=True)
    plan=json.loads((ROOT/'experiments/task41q_preregistration_v1.json').read_text());spec=json.loads((ROOT/'experiments/task41p_preregistration_v1.json').read_text())
    paths=[ROOT/'artifacts/task41p_kaggle_external/task41p_final.zip',ROOT/'artifacts/task41n_recovery_kaggle_external/task41n_recovery_final.zip']
    for path,expected in zip(paths,(plan['source41p_sha256'],plan['source41n_sha256'])):
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
    baseline=p.n.load_models(torch,ROOT,'cpu')[0]['dual'];before=p.n.weights_hash(baseline)
    fit,wv0,wc0=p.pool(torch,baseline,spec,'cpu');fv=fit.pop('input_v');fc=fit.pop('input_ca')
    validation=dict(spec,fit_seed=plan['validation_seed'],fit_points=plan['validation_points']);val,_,_=p.pool(torch,baseline,validation,'cpu');vv=val.pop('input_v');vc=val.pop('input_ca')
    baseline_fit=actual(torch,baseline,fv,fc);baseline_val=actual(torch,baseline,vv,vc)
    full_loss={'baseline':objective(torch,wv0[None].repeat(2,1,1,1),wc0[None].repeat(2,1,1,1),fit)}
    coefficients={'baseline':dict(fit=coefficient_summary(*baseline_fit,fit),validation=coefficient_summary(*baseline_val,val))};gap={};chosen={}
    with zipfile.ZipFile(paths[0]) as zp,zipfile.ZipFile(paths[1]) as zn:
        assert zp.testzip() is None and zn.testzip() is None
        saved_inputs=unpack_np(zp,'fit_inputs.npz');assert np.array_equal(fv,saved_inputs['voltage']) and np.array_equal(fc,saved_inputs['calcium'])
        freeze=json.loads(zp.read('selection_freeze.json'))
        for step in plan['checkpoints']:
            checkpoint=torch.load(io.BytesIO(zp.read(f'training_{step}.pt')),map_location='cpu',weights_only=True)
            full_loss[str(step)]=objective(torch,checkpoint['wv'],checkpoint['wc'],fit)
            for a,arm in enumerate(p.ARMS):
                net=p.materialize(torch,baseline,checkpoint['wv'][a],checkpoint['wc'][a],arm)
                states=torch.load(io.BytesIO(zp.read(f'{arm}_{step}.pt')),map_location='cpu',weights_only=True)
                assert all(torch.equal(value,states[key]) for key,value in net.state_dict().items())
                ru,rz=actual(torch,net,fv,fc);vu,vz=actual(torch,net,vv,vc);key=f'{arm}/{step}'
                coefficients[key]=dict(fit=coefficient_summary(ru,rz,fit),validation=coefficient_summary(vu,vz,val))
                sur=torch.bmm(fit['fv'],checkpoint['wv'][a].double())+fit['offset_v']
                real=np.stack([ru[...,k] for k in p.EQ[:-1]]+[rz[...,8],rz[...,9]],-1)
                gap[key]=dict(max_abs_logit_gap=float(abs(sur.numpy()-real).max()),rmse_logit_gap=float(np.sqrt(((sur.numpy()-real)**2).mean())))
                if step==freeze['selection'][arm]['step']:chosen[arm]=net
            print(f'[GIADA41q] checkpoint{step} full loss and coefficients audited',flush=True)
        crossings=crossing_audit(torch,baseline,chosen,zn,zp)
    direct=[]
    for relative in plan['direct_ridges_relative']:
        wv=wv0.double().clone();wc=wc0.double().clone();spectra=[]
        for s in range(3):
            for head in range(7):
                target=fit['u'][:,head] if head<5 else fit['z'][:,head-5]
                delta,info=ridge_delta(torch,fit['fv'][s],fit['raw'][s,:,head],target,relative)
                wv[s,:,head]+=delta;spectra.append(dict(seed=(17,29,43)[s],head=int(p.COLS[head]),**info))
            delta,info=ridge_delta(torch,fit['fc'][s],fit['sk_logits'][s,:,0],fit['u'][:,5],relative)
            wc[s,:,0]+=delta;spectra.append(dict(seed=(17,29,43)[s],head='SK_equilibrium',**info))
        for arm in p.ARMS:
            net=p.materialize(torch,baseline,wv.float(),wc.float(),arm)
            ru,rz=actual(torch,net,fv,fc);vu,vz=actual(torch,net,vv,vc)
            eq_wv=wv.float().clone();eq_wv[:,:,5:]=wv0[:,:,5:]
            original_loss=objective(torch,torch.stack((eq_wv,wv.float())),wc.float()[None].repeat(2,1,1,1),fit)[p.ARMS.index(arm)]
            direct.append(dict(relative_ridge=relative,arm=arm,fit=coefficient_summary(ru,rz,fit),validation=coefficient_summary(vu,vz,val),original_full_training_objective=original_loss,spectra=spectra,selection_eligible=False))
        np.savez_compressed(folder/f'direct_weights_ridge_{relative}.npz',wv=wv.float().numpy(),wc=wc.float().numpy(),equilibrium_update_columns_frozen=wv0[:,:,5:].numpy())
        print(f'[GIADA41q] same-feature direct diagnostic ridge={relative}',flush=True)
    assert p.n.weights_hash(baseline)==before
    report=dict(valid=True,complete=True,task='41q',source_hashes_verified=True,source_fit_inputs_bitwise_reproduced=True,full_training_objective=full_loss,
        coefficient_errors=coefficients,surrogate_materialization_gap=gap,crossing_audit=crossings,direct_fits=direct,fixed_rate_objective_lower_bound=fixed_rate_lower_bound(fit),
        baseline_unchanged=True,new_stochastic_training_updates=0,new_selection=False,thresholds_changed=False,
        original_fp32_failures_retained=[218,234],task42_authorized=False,independent_confirmation=False,limits=plan['limits'],elapsed_seconds=time.time()-started,
        numeric_scope='CPU coefficient audit and diagnostic least-squares, not GPU scientific requalification. Archived GPU timing/pass predicates are reused unchanged.')
    write_json(folder/'audit.json',report);print('[GIADA41q] audit complete',flush=True)
if __name__=='__main__':main()
