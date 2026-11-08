"""41t deterministic same-head fits, channel/seed-local safeguards and numerical audit."""
import copy,json,os,sys,time
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import prepare_task41s as s
from research_memory.mirror import write_json
r=s.r;p=r.p

@contextmanager
def memory_histories(enabled):
    original=r.n.stress_data
    def altered(spec):
        rows=original(spec)
        for row in rows:
            ca=float(row['ca'][0]);u0,_=r.n.p.coefficients(np.array(-85.),np.array(ca),.1)
            u,z=r.n.p.coefficients(np.array(-25.),np.array(ca),.1)
            state=u+(u0-u)*(1-z)**1000
            u,z=r.n.p.coefficients(np.array(-65.),np.array(ca),.1)
            row['initial']=u+(state-u)*(1-z)**(100 if row['history']==0 else 5000)
        return rows
    if enabled:r.n.stress_data=altered
    try:yield
    finally:r.n.stress_data=original

def rows(base,prior,seeds,memory=False):
    with memory_histories(memory):return s.crossing_rows(torch,base,prior,seeds,True,False)[0]

def raw64(net,v,ca):
    with torch.no_grad():
        u,z=net.raw(torch.tensor(v,dtype=torch.float64)[None].expand(3,-1),torch.tensor(ca,dtype=torch.float64)[None].expand(3,-1))
    return u.numpy(),z.numpy()

def head_reduction_audit(net,v,ca):
    """Isolate final GEMM cancellation, holding features and other corrections fixed."""
    with torch.no_grad():
        vv=torch.tensor(v)[None].expand(3,-1)
        f=torch.nn.functional.silu(torch.bmm(((vv+40)/80)[...,None],net.parent.a)+net.parent.b)
        a=torch.bmm(f,net.parent.c)+net.parent.d[:,None]
        b=torch.bmm(f.double(),net.parent.c.double())+net.parent.d[:,None].double()
    return dict(available=True,scope='Final voltage-head reduction FP32 vs FP64 with identical FP32 features; not full FP64 dynamics',
        per_head_max_abs_difference={str(i):float((a.double()-b)[...,i].abs().max()) for i in (8,9,12,13,25,26,29,30)})

def numerical_audit(base,start):
    net=r.materialize(torch,base,start);v=np.r_[np.full(20,-62.),np.full(20,-25.),np.full(80,-42.)].astype('float32');ca=np.full_like(v,1e-4)
    ru,rz=s.q.actual(torch,net,v,ca);u=s.q.sigmoid(ru);z=s.q.sigmoid(rz)
    features=s.feature_tables(torch,net,v);initial=r.n.p.coefficients(np.array(-85.),np.array(1e-4),.1)[0]
    def ending(w):
        a,b=s.q.actual(torch,r.materialize(torch,base,w),v,ca);state=np.broadcast_to(initial,(3,18)).copy()
        for tick in range(len(v)):state+=s.q.sigmoid(b[:,tick])*(s.q.sigmoid(a[:,tick])-state)
        return state
    rng=np.random.default_rng(9416100);audit=[]
    for gate in (8,9,12,13):
        for kind in (0,1):
            direction=rng.normal(size=65);direction/=np.linalg.norm(direction)
            state=np.broadcast_to(initial[gate],(3,)).copy();su=np.zeros((3,65));sz=su.copy()
            for tick in range(len(v)):
                state,su,sz=r.affine_sensitivity(state,su,sz,u[:,tick,gate],z[:,tick,gate],features[:,tick])
            predicted=(su if kind==0 else sz)@direction;trials=[]
            for eps in (1e-2,1e-3,1e-4):
                plus=copy.deepcopy(start);minus=copy.deepcopy(start);key=f'h{kind}_{gate}'
                plus[key]=(plus[key]+eps*direction).astype('float32');minus[key]=(minus[key]-eps*direction).astype('float32')
                fd=(ending(plus)[:,gate]-ending(minus)[:,gate])/(2*eps)
                trials.append(dict(epsilon=eps,finite_difference=fd.tolist(),relative_error=(abs(fd-predicted)/np.maximum(abs(predicted),1e-8)).tolist()))
            valid=all(min(t['relative_error'][seed] for t in trials)<.05 or abs(predicted[seed])<1e-8 for seed in range(3))
            audit.append(dict(gate=gate,kind=kind,predicted=predicted.tolist(),trials=trials,valid=valid))
    assert all(x['valid'] for x in audit),'Full materialized recurrence Jacobian mismatch: do not launch a repair'
    fp={}
    for name,weights in [('incumbent',start),('prior41s_window',dict(np.load(ROOT/'experiments/results/task41s_protocol_v1/heads_window_iterative.npz')))]:
        model=r.materialize(torch,base,weights);a,b=s.q.actual(torch,model,v,ca)
        try:
            aa,bb=raw64(copy.deepcopy(model).double(),v.astype('float64'),ca.astype('float64'))
            fp[name]=dict(available=True,max_logit_difference=float(max(abs(a-aa).max(),abs(b-bb).max())),head_max_abs={k:float(abs(w).max()) for k,w in weights.items()})
        except (RuntimeError,AssertionError) as error:
            fp[name]=dict(full_fp64_available=False,reason=repr(error),diagnostic_only=True,head_reduction=head_reduction_audit(model,v,ca))
    return dict(valid=True,full_recurrence_directional_checks=audit,fp64_diagnostic=fp,inference_precision_changed=False)

def geometries(net,conditioned):
    if not conditioned:return s.matrices(net),{}
    spec=json.loads((ROOT/'experiments/task41p_preregistration_v1.json').read_text());fit,_,_=p.pool(torch,net,spec,'cpu');f=fit['fv'];white={};diag={}
    for kind,gate in r.HEADS:
        if gate not in (8,9,12,13):continue
        values=[];details=[]
        for seed in range(3):
            # Truncated feature directions, not a reduced inference architecture.
            _,sing,vh=torch.linalg.svd(f[seed]/np.sqrt(f.shape[1]),full_matrices=False)
            keep=sing>=sing[0]*1e-3
            values.append(vh[keep].T/torch.sqrt(sing[keep].square()+(.001*sing[0]).square())[None])
            details.append(dict(rank=int(keep.sum()),singular_values=sing.tolist()))
        white[(kind,gate)]=values;diag[f'{kind}_{gate}']=details
    return white,diag

def projected_step(white,b,y):
    count=len(y);bb=torch.tensor(b,dtype=torch.float64)/np.sqrt(count);yy=torch.tensor(y,dtype=torch.float64)/np.sqrt(count)
    rank=white.shape[1]
    if not rank:return torch.zeros(white.shape[0],dtype=torch.float64)
    a=torch.cat((torch.eye(rank,dtype=torch.float64),bb@white),0)
    theta=torch.linalg.lstsq(a,torch.cat((torch.zeros(rank,dtype=torch.float64),yy)),rcond=1e-12,driver='gelsd').solution
    return white@theta

def rebase_control(start,white):
    """Same inference heads, minimum-norm projection onto reliable feature space."""
    value=copy.deepcopy(start)
    for kind,gate in r.keys_for('Nap_Et2',True):
        for seed in range(3):
            basis=torch.linalg.qr(white[(kind,gate)][seed],mode='reduced')[0]
            original=torch.tensor(start[f'h{kind}_{gate}'][seed]).double()
            value[f'h{kind}_{gate}'][seed]=(basis@(basis.T@original)).float().numpy()
    return value

def fit(base,start,prior,spec,white,channel,keys,protect,memory=False):
    current=copy.deepcopy(start);records=[];initial_net=r.materialize(torch,base,start)
    guard=rows(initial_net,prior,spec['guard_path_seeds'],True)[channel]
    guard_limit=np.mean(guard[1]**2,axis=1)
    allkeys=r.keys_for(channel,True);indices=[j for i,k in enumerate(allkeys) if k in keys for j in range(i*65,(i+1)*65)]
    for iteration in range(3):
        b,y=rows(r.materialize(torch,base,current),prior,spec['fit_path_seeds'],memory)[channel];old=np.mean(y*y,axis=1)
        candidate=copy.deepcopy(current)
        for seed in range(3):
            # Regularize displacement from incumbent in a fixed geometry.
            w=torch.block_diag(*[white[k][seed] for k in keys]);dis=s.unpack(current,start,keys,seed)
            delta=projected_step(w,b[seed][:,indices],y[seed]+b[seed][:,indices]@dis.numpy())
            updated=s.step_weights(current,start,keys,seed,delta,1.)
            for k,g in keys:candidate[f'h{k}_{g}'][seed]=updated[f'h{k}_{g}'][seed]
        accepted=np.zeros(3,dtype=bool);next_weights=copy.deepcopy(current)
        for alpha in (1.,.5,.25,.125):
            trial={k:(current[k]+alpha*(candidate[k]-current[k])).astype('float32') for k in current}
            net=r.materialize(torch,base,trial);actual=rows(net,prior,spec['fit_path_seeds'],memory)[channel][1];score=np.mean(actual**2,axis=1)
            preserved=np.mean(rows(net,prior,spec['guard_path_seeds'],True)[channel][1]**2,axis=1) if protect else None
            eligible=(score<old)&(~accepted)
            if protect:eligible&=preserved<=guard_limit*(1+1e-6)+1e-12
            for seed in np.flatnonzero(eligible):
                for k,g in keys:next_weights[f'h{k}_{g}'][seed]=trial[f'h{k}_{g}'][seed]
            records.append(dict(iteration=iteration,alpha=alpha,old_score=old.tolist(),actual_score=score.tolist(),guard_score=None if preserved is None else preserved.tolist(),guard_limit=guard_limit.tolist(),accepted_seeds=[(17,29,43)[i] for i in np.flatnonzero(eligible)]))
            accepted|=eligible
            if accepted.all():break
        print(f'[GIADA41t fit] {channel} protected={protect} memory={memory} iteration={iteration} accepted={accepted.tolist()}',flush=True)
        current=next_weights
        if not accepted.any():break
    return current,records

def main():
    torch.set_num_threads(1);started=time.time();folder=ROOT/'experiments/results/task41t_protocol_v1';folder.mkdir(exist_ok=True)
    spec=json.loads((ROOT/'experiments/task41t_preregistration_v1.json').read_text());prior=r.n.contract(ROOT)[0];assert spec['quality']==prior['quality']
    used=prior['native_seeds']+prior['stress_seeds']
    for name in ('r','s'):
        old=json.loads((ROOT/f'experiments/task41{name}_preregistration_v1.json').read_text());used+=old['fit_path_seeds']+old['development_seeds']
    fresh=spec['fit_path_seeds']+spec['guard_path_seeds']+spec['development_seeds'];assert len(set(fresh))==len(fresh) and not set(fresh)&set(used)
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin');base=r.n.load_models(torch,ROOT,'cpu')[0]['dual']
    with np.load(ROOT/'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz') as a:start=dict(a)
    audit=numerical_audit(base,start);write_json(folder/'numerical_preflight.json',audit)
    whites={};geometry={}
    for conditioned in (False,True):whites[conditioned],geometry[str(conditioned)]=geometries(r.materialize(torch,base,start),conditioned)
    outputs={'incumbent':start};logs={}
    for label,keys in [('equilibrium',[(0,13)]),('rate',[(1,13)]),('joint',[(0,13),(1,13)]),('activation',[(0,12),(1,12)])]:
        outputs['k_'+label],logs['k_'+label]=fit(base,start,prior,spec,whites[True],'K_Pst',keys,False)
    for conditioned in (False,True):
        for protected in (False,True):
            name='nap_'+('conditioned' if conditioned else 'original')+('_protected' if protected else '_free')
            outputs[name],logs[name]=fit(base,start,prior,spec,whites[conditioned],'Nap_Et2',r.keys_for('Nap_Et2',True),protected)
    outputs['nap_conditioned_memory'],logs['nap_conditioned_memory']=fit(base,start,prior,spec,whites[True],'Nap_Et2',r.keys_for('Nap_Et2',True),True,True)
    outputs['nap_rebased_only']=rebase_control(start,whites[True]);logs['nap_rebased_only']=[dict(operation='Reliable-feature minimum-norm head projection only; no trajectory fitting')]
    manifest=dict(valid=True,baseline_weights_sha256=r.n.weights_hash(base),incumbent_weights_sha256=r.n.weights_hash(r.materialize(torch,base,start)),models={},fit_log=logs,geometry=geometry,rows_per_channel=512,stochastic_updates=0,behavioral_selection_performed=False,elapsed_seconds=time.time()-started)
    for name,weights in outputs.items():
        allowed=('h0_12','h1_12','h0_13','h1_13') if name.startswith('k_') else ('h0_8','h1_8','h0_9','h1_9') if name.startswith('nap_') else ()
        for k in weights:
            if k not in allowed:assert np.array_equal(weights[k],start[k]),(name,k)
        model=r.materialize(torch,base,weights);assert all(torch.isfinite(v).all() for v in model.parameters())
        path=folder/f'heads_{name}.npz';np.savez_compressed(path,**weights)
        manifest['models'][name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=r.n.p.digest(path),weights_sha256=r.n.weights_hash(model))
    assert ['baseline',*outputs,'recomposed']==spec['arms']
    write_json(folder/'fit_manifest.json',manifest);print('[GIADA41t] candidates frozen; no development/stress opened',flush=True)
def add_rebase_only():
    # Complete this cheap diagnostic after an already frozen fit, without repeating it.
    torch.set_num_threads(1);folder=ROOT/'experiments/results/task41t_protocol_v1'
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin');base=r.n.load_models(torch,ROOT,'cpu')[0]['dual']
    with np.load(folder/'heads_incumbent.npz') as a:start=dict(a)
    white,_=geometries(r.materialize(torch,base,start),True);weights=rebase_control(start,white)
    model=r.materialize(torch,base,weights);path=folder/'heads_nap_rebased_only.npz';np.savez_compressed(path,**weights)
    manifest=json.loads((folder/'fit_manifest.json').read_text())
    manifest['models']['nap_rebased_only']=dict(path=path.relative_to(ROOT).as_posix(),sha256=r.n.p.digest(path),weights_sha256=r.n.weights_hash(model))
    manifest['fit_log']['nap_rebased_only']=[dict(operation='Reliable-feature minimum-norm head projection only; no trajectory fitting')]
    write_json(folder/'fit_manifest.json',manifest)

def refresh_k_only():
    torch.set_num_threads(1);folder=ROOT/'experiments/results/task41t_protocol_v1';spec=json.loads((ROOT/'experiments/task41t_preregistration_v1.json').read_text());prior=r.n.contract(ROOT)[0]
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin');base=r.n.load_models(torch,ROOT,'cpu')[0]['dual']
    with np.load(folder/'heads_incumbent.npz') as a:start=dict(a)
    white,_=geometries(r.materialize(torch,base,start),True);manifest=json.loads((folder/'fit_manifest.json').read_text())
    manifest['local_guarded_K_pilot']=manifest.get('local_guarded_K_pilot',{k:v for k,v in manifest['fit_log'].items() if k.startswith('k_')})
    for label,keys in [('equilibrium',[(0,13)]),('rate',[(1,13)]),('joint',[(0,13),(1,13)]),('activation',[(0,12),(1,12)])]:
        name='k_'+label;weights,log=fit(base,start,prior,spec,white,'K_Pst',keys,False)
        model=r.materialize(torch,base,weights);path=folder/f'heads_{name}.npz';np.savez_compressed(path,**weights)
        manifest['fit_log'][name]=log;manifest['models'][name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=r.n.p.digest(path),weights_sha256=r.n.weights_hash(model))
    write_json(folder/'fit_manifest.json',manifest)
if __name__=='__main__':
    if '--rebase-only' in sys.argv:add_rebase_only()
    elif '--refresh-k-only' in sys.argv:refresh_k_only()
    else:main()
