"""Atomic held-voltage precision: learned corrections, unchanged recurrent map.

Canonical functions label independent FIT anchors only. Candidate inference
never uses teacher u/z, time-to-event, or a development trajectory identifier.
"""
import hashlib
import json
import time
import numpy as np
from . import task39i_coupled_relaxation as i

p=i.p
CONFIG='experiments/task39j_preregistration_v1.json'
GATES=(8,9,12,13)


def arms():
    # Each contrast changes one registered package; full-batch comparisons
    # report exposure and wall time instead of pretending equal compute.
    return [dict(name=name,basis=basis,relaxation=loss,support=support,
                 full_batch=full,parent_trainable=unfreeze,channel_clip=clip)
            for name,basis,loss,support,full,unfreeze,clip in [
                ('J1','local',False,'uniform',False,False,False),
                ('J2','local',True,'uniform',False,False,False),
                ('J3','local',True,'uniform',True,False,False),
                ('J4','global',True,'uniform',False,False,False),
                ('J5','local',True,'sensitive',False,False,False),
                ('J6','local',True,'sensitive',True,False,False),
                ('J7','local',True,'uniform',False,False,True),
                ('J8','local',True,'uniform',False,True,False)]]


def contrasts():
    return [('objective','J1','J2'),('deterministic_refinement','J2','J3'),
            ('locality','J4','J2'),('coverage','J2','J5'),
            ('refinement_covered','J5','J6'),('coverage_fixed','J3','J6'),
            ('optimizer_interference','J2','J7'),('parent_interference','J2','J8')]


def model_class(torch):
    Parent=i.model_class(torch)
    class Model(Parent):
        def __init__(self,basis,seeds,parent_trainable=False):
            super().__init__('plain',seeds)
            for par in self.parameters():par.requires_grad_(parent_trainable)
            self.basis=basis
            self.correction=torch.nn.Parameter(torch.zeros(len(seeds),32,8))
            self.register_buffer('gate_map',torch.tensor(GATES),persistent=False)

        def features(self,v):
            if self.basis=='local':
                centers=torch.linspace(-100,60,32,device=v.device,dtype=v.dtype)
                return (1-(v[...,None]-centers).abs()/(160/31)).clamp_min(0)
            phase=(v+100)/160*torch.pi
            freq=torch.arange(1,17,device=v.device,dtype=v.dtype)
            return torch.cat((torch.cos(phase[...,None]*freq),torch.sin(phase[...,None]*freq)),-1)/4

        def raw(self,v,ca):
            u,z=super().raw(v,ca)
            extra=torch.bmm(self.features(v).double(),self.correction.double()).to(v.dtype)
            extra=.1*torch.tanh(extra/.1)
            # index_add is avoided for deterministic CUDA graph execution.
            zero=torch.zeros_like(u[...,:1]);du=[];dz=[]
            for gate in range(18):
                if gate in GATES:
                    k=GATES.index(gate);du.append(extra[...,k:k+1]);dz.append(extra[...,k+4:k+5])
                else:du.append(zero);dz.append(zero)
            return u+torch.cat(du,-1),z+torch.cat(dz,-1)
    return Model


def parent_weights(torch,root,spec,device):
    path=root/spec['parent']/'frozen_checkpoint.pt'
    assert p.digest(path)==spec['parent_sha256']
    frozen=json.loads((path.parent/'selection_freeze.json').read_text())
    aa=[a['name'] for a in frozen['arms'] if a['architecture']=='plain']
    index=aa.index('I3')*len(spec['seeds'])
    weights=torch.load(path,map_location=device,weights_only=True)['plain']
    return {k:v.clone() if k=='gbar' else v[index:index+len(spec['seeds'])].clone() for k,v in weights.items()}


def make_model(torch,weights,definition,seeds,device):
    model=model_class(torch)(definition['basis'],seeds,definition['parent_trainable']).to(device)
    state=model.state_dict();state.update(weights);model.load_state_dict(state)
    return model


def anchors(seed,count,sensitive=False):
    rng=np.random.default_rng(seed)
    v=rng.uniform(-100,60,count);ca=10**rng.uniform(-7,-2,count)
    initial=rng.uniform(0,1,(count,18))
    if sensitive:
        # Broad, independent voltage stratum; no dev voltages are copied.
        v[:count//2]=rng.uniform(-20,20,count//2)
        initial[:count//2]=10**rng.uniform(-4,0,(count//2,18))
    u,z=p.coefficients(v,ca,.1)
    # Complete held interval, shared nodes including the real first tick.
    ticks=np.unique(np.rint(np.geomspace(1,100000,65)))
    return dict(v=v.astype('float32'),ca=ca.astype('float32'),initial=initial,
                u=u,z=z,ticks=ticks.astype('float64'))


def tensors(torch,data,device):
    return {k:torch.tensor(v,device=device) for k,v in data.items()}


def loss_terms(torch,model,data,indices):
    v=data['v'][indices][None].expand(model.a.shape[0],-1)
    ca=data['ca'][indices][None].expand_as(v)
    ru,rz=model.raw(v,ca);ru=ru.double();rz=rz.double()
    target_u=data['u'][indices];target_z=data['z'][indices]
    # Censored logits avoid inf labels at exact saturation. No divisions by
    # predicted values and no detached rescaling of model gradients.
    expected_u=torch.logit(target_u.clamp(1e-7,1-1e-7))
    expected_z=torch.log(target_z.clamp_min(1e-12))
    coefficient=((ru[...,GATES]-expected_u[...,GATES]).square()+
                 (torch.nn.functional.logsigmoid(rz[...,GATES])-expected_z[...,GATES]).square()).mean((1,2))
    ticks=data['ticks'][None,None,:,None]
    x0=data['initial'][indices][None,:,None,list(GATES)]
    u=torch.sigmoid(ru[...,GATES])[:,:,None]
    decay=torch.nn.functional.logsigmoid(-rz[...,GATES])[:,:,None]
    state=x0-torch.expm1(ticks*decay)*(u-x0)
    tu=target_u[...,GATES][None,:,None]
    tz=target_z[...,GATES][None,:,None]
    reference=x0-torch.expm1(ticks*torch.log1p(-tz))*(tu-x0)
    # Current opening factors for Nap_Et2 and K_Pst (same drive cancels in the
    # normalization). Full relaxation origin is first tick, never a chunk reset.
    pred=torch.stack((state[...,0]**3*state[...,1],state[...,2]**2*state[...,3]),-1)
    ref=torch.stack((reference[...,0]**3*reference[...,1],reference[...,2]**2*reference[...,3]),-1)
    scale=ref.abs().amax(2,keepdim=True).clamp_min(1e-8)
    dt=torch.diff(data['ticks'])*.1
    slope=torch.diff(ref,dim=2).abs()/dt[None,None,:,None]
    slope=torch.cat((slope[:,:,:1],slope),2)
    # Bounded time-sensitive residual and a shape term. This is a FIT loss,
    # not a replacement for the unchanged discrete event evaluator.
    precision=(pred-ref)/(slope*.3).clamp_min(scale*1e-5)
    relaxation=(5*torch.tanh(precision/5)).square().mean((1,2,3))
    shape=((pred-ref)/scale).square().mean((1,2,3))
    preservation=((ru-expected_u).square()+
                  (torch.nn.functional.logsigmoid(rz)-expected_z).square()).mean((1,2))
    return torch.stack((coefficient,relaxation+shape,preservation),-1)


def clip(torch,model,channel=False):
    params=[x for x in model.parameters() if x.requires_grad and x.grad is not None]
    if channel:
        grad=model.correction.grad
        # Explicit independent clipping per two-gate channel, negative control
        # for optimizer coupling. Architecture and forward function unchanged.
        counts=torch.zeros(grad.shape[0],device=grad.device,dtype=torch.int64)
        for ids in ([0,1,4,5],[2,3,6,7]):
            norm=grad[...,ids].double().square().sum((1,2)).sqrt()
            counts+=(norm>1).to(torch.int64)
            grad[...,ids]*=(1/norm.clamp_min(1))[:,None,None]
        return counts
    norm=sum(x.grad.double().flatten(1).square().sum(1) for x in params).sqrt()
    for x in params:x.grad.mul_((1/norm.clamp_min(1)).reshape(-1,*([1]*(x.ndim-1))).to(x.dtype))
    return (norm>1).to(torch.int64)


def step(torch,model,opt,data,idx,weights,channel=False):
    opt.zero_grad(set_to_none=True)
    terms=loss_terms(torch,model,data,idx)
    (terms*weights).sum().backward();clipped=clip(torch,model,channel);opt.step()
    return terms.detach(),clipped


def run(root,output,device='cuda',smoke=False):
    import torch
    assert device=='cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());quality=json.loads((root/p.CONFIG).read_text())
    assert p.digest(root/p.CONFIG)==spec['quality_config_sha256']
    output.mkdir(parents=True,exist_ok=True);assert not (output/'checkpoint_0.pt').exists()
    p.save_json(output/'preregistration.json',spec)
    weights=parent_weights(torch,root,spec,device)
    cohorts=[];pools={}
    for support in ('uniform','sensitive'):
        data=anchors(spec['fit_seed'],spec['anchors'],support=='sensitive')
        np.savez_compressed(output/f'fit_{support}.npz',**data)
        pools[support]=tensors(torch,data,device)
    held=tensors(torch,anchors(spec['holdout_seed'],256),device)
    definitions=arms()
    fixed=torch.arange(spec['anchors'],device=device)
    for a in definitions:
        model=make_model(torch,weights,a,spec['seeds'],device)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=spec['lr'],eps=1e-8)
        # Calibrate auxiliary gradient on the same FIT subset, once. Final
        # coefficient and preservation weights do not change during training.
        terms=loss_terms(torch,model,pools['uniform'],fixed[:128])
        norms=[]
        params=[x for x in model.parameters() if x.requires_grad]
        for k in range(3):
            grads=torch.autograd.grad(terms[:,k].sum(),params,retain_graph=k<2)
            norms.append(sum(g.double().flatten(1).square().sum(1) for g in grads).sqrt())
        w=torch.zeros(3,3,device=device,dtype=torch.float64);w[:,0]=1
        if a['relaxation']:w[:,1]=(.1*norms[0]/norms[1].clamp_min(1e-20)).clamp(1e-8,1e4)
        # All arms receive the identical preservation objective, not just J8.
        w[:,2]=1
        cohorts.append(dict(key=a['name'],definitions=[a],model=model,opt=opt,weights=w,clips=torch.zeros(3,device=device,dtype=torch.int64)))
    shared_auxiliary=cohorts[1]['weights'][:,1].detach().clone()
    for a,g in zip(definitions,cohorts):
        if a['relaxation']:g['weights'][:,1]=shared_auxiliary
    p.save_json(output/'fit_loss_calibration.json',dict(weights={g['key']:g['weights'].tolist() for g in cohorts},source='initial J2 uniform FIT only, shared across treatments'))
    # Exact zero correction and frozen-channel preservation before training.
    baseline=i.model_class(torch)('plain',spec['seeds']).to(device);baseline.load_state_dict(weights)
    vv=pools['uniform']['v'][None].expand(3,-1);cc=pools['uniform']['ca'][None].expand(3,-1)
    with torch.no_grad():
        expected=baseline.raw(vv,cc)
        for g in cohorts:
            actual=g['model'].raw(vv,cc)
            assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    p.save_json(output/'initialization_preflight.json',dict(valid=True,exact_parent_function=True,learned_kinetics_only=True))
    rng=np.random.default_rng(spec['stream_seed']);curves=[];best=[None]*24;saved=[None]*24;chosen=[None]*24
    normal=json.loads((root/spec['normalizers']).read_text())
    dev,selection_held=(None,None) if smoke else i.h.development(root,spec,quality)
    checkpoints=[0,2] if smoke else spec['checkpoints'];start=time.perf_counter()
    for update in range(checkpoints[-1]+1):
        if update in checkpoints:
            torch.save({g['key']:g['model'].state_dict() for g in cohorts},output/f'checkpoint_{update}.pt')
            torch.save(dict(optimizers=[g['opt'].state_dict() for g in cohorts],rng=rng.bit_generator.state,step=update,clips={g['key']:g['clips'] for g in cohorts}),output/f'optimizer_{update}.pt')
            finite=all(bool(torch.isfinite(x).all()) for g in cohorts for x in g['model'].parameters())
            p.save_json(output/'checkpoint_phase.json',dict(step=update,parameters_finite=finite,phase='before_selection'))
            print(f'[GIADA Task39j] checkpoint_saved {update} before_selection',flush=True)
            assert finite,'Nonfinite weights; checkpoint preserved'
            with torch.no_grad():
                if smoke:
                    common=np.zeros(24);ranks=[(0,0,0)]*24;panel=[]
                else:common,_,ranks,panel=i.e.checkpoint_scores(torch,cohorts,dev,selection_held,normal,spec,device,quality)
                atomic={g['key']:loss_terms(torch,g['model'],held,torch.arange(256,device=device)).tolist() for g in cohorts}
                fit={g['key']:loss_terms(torch,g['model'],pools[a['support']],fixed).tolist() for a,g in zip(definitions,cohorts)}
            for number,g in enumerate(cohorts):
                for member in range(3):
                    n=number*3+member
                    if ranks[n] is not None and (best[n] is None or ranks[n]<best[n]):
                        best[n]=ranks[n];chosen[n]=update
                        saved[n]={k:v[member].detach().cpu().clone() for k,v in g['model'].named_parameters()}
            curves.append(dict(step=update,panel=panel,common_development_score=common.tolist(),atomic_nonselectable=atomic,fit=fit,
                clipping_counts={g['key']:g['clips'].tolist() for g in cohorts},elapsed_seconds=time.perf_counter()-start))
            p.save_json(output/'training_curves.json',i.e.diagnostic_json(curves))
        if update==checkpoints[-1]:break
        idx=torch.tensor(rng.integers(spec['anchors'],size=spec['batch']),device=device)
        for a,g in zip(definitions,cohorts):
            _,clipped=step(torch,g['model'],g['opt'],pools[a['support']],fixed if a['full_batch'] else idx,g['weights'],a['channel_clip'])
            g['clips'].add_(clipped)
        if (update+1)%100==0:
            print(f'[GIADA Task39j] update {update+1}/{checkpoints[-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    if smoke:
        p.save_json(output/'smoke.json',dict(valid=True,updates=2,all_parameters_finite=True,gpu=torch.cuda.get_device_name()))
        return dict(decision='PREFLIGHT_ONLY',valid=True)
    assert all(x is not None for x in saved)
    with torch.no_grad():
        for number,g in enumerate(cohorts):
            for member in range(3):
                for name,x in g['model'].named_parameters():x[member].copy_(saved[number*3+member][name])
    torch.save({g['key']:g['model'].state_dict() for g in cohorts},output/'frozen_checkpoint.pt')
    freeze=dict(arms=definitions,seeds=spec['seeds'],chosen_steps=chosen,ranks=best,development_only=True,stress_used_for_selection=False,
                independent_confirmation_accessed=False,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'))
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    models=[g['model'] for g in cohorts]+[baseline]
    old=json.loads((root/i.old.CONFIG).read_text());evaluation=dict(spec,architectures=[a['name'] for a in definitions]+['frozen39i_I3'])
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=old[key]
    with torch.no_grad():summary=i.old.evaluate(root,output,evaluation,models,torch,device,quality)
    rows=json.loads((output/'recurrence_evaluation.json').read_text());scores={}
    for a in definitions:
        scores[a['name']]={}
        for seed in spec['seeds']:
            rr=[r for r in rows if r['model']=='joint_w'+a['name'] and r['role']=='development' and r['seed']==seed]
            values=[i.q.severity(r['metrics']) for r in rr]
            scores[a['name']][str(seed)]=dict(median=float(np.median(values)),worst=max(values),passed=sum(r['metrics']['passed'] for r in rr))
    paired=[]
    for axis,control,treatment in contrasts():
        ratios=[scores[treatment][str(s)]['median']/max(scores[control][str(s)]['median'],1e-12) for s in spec['seeds']]
        worst=[scores[treatment][str(s)]['worst']/max(scores[control][str(s)]['worst'],1e-12) for s in spec['seeds']]
        paired.append(dict(axis=axis,control=control,treatment=treatment,ratios=ratios,worst_ratios=worst,
            supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05),severity_excludes_timing=True))
    p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=paired))
    p.save_json(output/'execution.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,wall_seconds=time.perf_counter()-start,
        anchor_examples_per_member={a['name']:checkpoints[-1]*(spec['anchors'] if a['full_batch'] else spec['batch']) for a in definitions},
        parallelism='Three independent seeds per arm vectorized; arm kernels enqueued in one worker; no claim of simultaneous streams',
        independent_confirmation_accessed=False))
    passing=[a['name'] for a in definitions if summary['joint_w'+a['name']]['all_development_passed']]
    final=dict(valid=True,task='39j',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',passing_arms=passing,summaries=summary,
               thresholds_changed=False,independent_confirmation_accessed=False,speedup_claimed=False)
    p.save_json(output/'final_report.json',final)
    return final
