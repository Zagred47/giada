"""Paired support x resolution x objective experiment on three learned functions.

B6 is frozen. Canonical kinetics supply FIT labels and diagnostic references
only. All candidate gates remain learned with the same 18 recurrent states.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from . import task41_independent_confirmation as prior

p,t,b,j,ionic=prior.p,prior.t,prior.b,prior.j,prior.ionic
CONFIG='experiments/task41b_preregistration_v1.json'
TARGETS=(('u',8),('z',7),('z',11))

def arms():
    return [dict(name=f'P{s}{c}{o}',support='sensitive' if s else 'uniform',basis=64 if c else 32,conditioned=bool(o))
            for s in (0,1) for c in (0,1) for o in (0,1)]

def contract(root):
    spec=json.loads((root/CONFIG).read_text());old,cfg,calcium,base=prior.contract(root)
    assert spec['quality']==old['current_quality'] and spec['compartment_gates']==old['compartment_gates']
    for name,sha in spec['sources'].items():assert p.digest(root/name)==sha,name
    assert spec['seeds']==[17,29,43] and spec['dt_ms']==.1 and spec['arms']==arms()
    return spec,cfg,calcium,base

def feature(torch,v,n):
    centers=torch.linspace(-100,60,n,device=v.device,dtype=v.dtype)
    return (1-(v[...,None]-centers).abs()/(160/(n-1))).clamp_min(0)

def model_class(torch):
    class Model(b.model_class(torch)):
        def __init__(self,definition,seeds):
            super().__init__(next(a for a in b.arms() if a['name']=='B6'),seeds)
            for par in self.parameters():par.requires_grad_(False)
            self.precision_definition=definition
            self.precision_correction=torch.nn.Parameter(torch.zeros(len(seeds),definition['basis'],3))
        def raw(self,v,ca):
            u,z=super().raw(v,ca)
            extra=torch.bmm(feature(torch,v,self.precision_definition['basis']).double(),self.precision_correction.double()).to(v.dtype)
            extra=.1*torch.tanh(extra/.1)
            du=torch.zeros_like(u[...,:1]);up=[];zp=[]
            for gate in range(18):
                up.append(extra[...,0:1] if gate==8 else du)
                zp.append(extra[...,1:2] if gate==7 else (extra[...,2:3] if gate==11 else du))
            return u+torch.cat(up,-1),z+torch.cat(zp,-1)
    return Model

def make_model(torch,parent,definition,seeds,device):
    model=model_class(torch)(definition,seeds).to(device);state=model.state_dict();state.update(parent.state_dict());model.load_state_dict(state)
    return model

def atomic(seed,count,sensitive=False):
    rng=np.random.default_rng(seed);pilot=rng.uniform(-100,60,count*4)
    ca=10**rng.uniform(-7,-2,count*4)
    if sensitive:
        a,az=p.coefficients(pilot-.2,ca,.1);c,cz=p.coefficients(pilot+.2,ca,.1)
        # Broad canonical FIT sensitivity, independent of opened Task41 voltages.
        score=np.abs(c[:,8]-a[:,8])/(.01+a[:,8]*(1-a[:,8]))
        score+=np.abs(np.log(cz[:,7])-np.log(az[:,7]))+np.abs(np.log(cz[:,11])-np.log(az[:,11]))
        score=np.minimum(score,np.quantile(score,.95));prob=.5/len(score)+.5*score/max(score.sum(),1e-30)
        chosen=rng.choice(len(pilot),count,replace=False,p=prob/prob.sum())
    else:chosen=np.arange(count)
    v=pilot[chosen];ca=ca[chosen]
    # Independent held, recovery, and second-pulse levels; all broad-domain.
    values=np.stack((v,rng.uniform(-100,60,count),rng.uniform(-100,60,count)),-1).astype('float32')
    calcium=np.broadcast_to(ca[:,None],values.shape).astype('float32').copy()
    u,z=p.coefficients(values.astype('float64'),calcium.astype('float64'),.1)
    return dict(v=values,ca=calcium,u=u,z=z,initial=rng.uniform(0,1,(count,18)),
        durations=rng.choice([1.,10.,80.,400.,4000.],(count,2)),
        ticks=np.array([1.,10.,80.,400.,4000.,12000.]))

def pool(torch,parent,data,device):
    with torch.no_grad():
        v=torch.tensor(data['v'],device=device);ca=torch.tensor(data['ca'],device=device)
        u,z=parent.raw(v.reshape(1,-1).expand(3,-1),ca.reshape(1,-1).expand(3,-1))
    result={k:torch.tensor(v,device=device) for k,v in data.items()}
    result.update(base_u=u.reshape(3,len(v),3,18).double(),base_z=z.reshape(3,len(v),3,18).double())
    return result

def batched_pool(torch,pools,definitions):
    values={};device=pools['uniform']['v'].device
    for key in pools['uniform']:
        if key=='ticks':values[key]=pools['uniform'][key];continue
        values[key]=torch.cat([pools[a['support']][key] if key.startswith('base_') else
            pools[a['support']][key][None].expand(3,*pools[a['support']][key].shape) for a in definitions])
    ff=[]
    for a in definitions:
        f=feature(torch,pools[a['support']]['v'],a['basis'])
        ff.extend([torch.nn.functional.pad(f,(0,64-a['basis']))]*3)
    values['features']=torch.stack(ff).double()
    values['member']=torch.arange(3*len(definitions),device=device)[:,None]
    return values

def terms(torch,w,data,idx,normalizers):
    member=data['member'];feat=data['features'][member,idx]
    extra=torch.bmm(feat.reshape(len(w),-1,64),w.double()).float().double().reshape(len(w),idx.shape[1],3,3)
    extra=(.1*torch.tanh(extra.float()/.1)).double()
    ru=data['base_u'][member,idx];rz=data['base_z'][member,idx]
    # Match deployed FP32 additions, leaving all other learned logits untouched.
    du=torch.nn.functional.one_hot(torch.tensor([8],device=w.device),18).double()
    dz=torch.nn.functional.one_hot(torch.tensor([7,11],device=w.device),18).double()
    ru=(ru+extra[...,:1]@du).float().double();rz=(rz+extra[...,1:]@dz).float().double()
    tu,tz=data['u'][member,idx],data['z'][member,idx]
    errors=torch.stack((ru[...,8]-torch.logit(tu[...,8].clamp(1e-7,1-1e-7)),
        torch.nn.functional.logsigmoid(rz[...,7])-tz[...,7].clamp_min(1e-12).log(),
        torch.nn.functional.logsigmoid(rz[...,11])-tz[...,11].clamp_min(1e-12).log()),-1)
    coefficient=(errors.square()/normalizers[:,None,None]).mean((1,2,3))
    u=torch.sigmoid(ru);decay=torch.nn.functional.logsigmoid(-rz)
    truth_decay=torch.log1p(-tz);state=data['initial'][member,idx];truth=state
    pred_open=[];true_open=[]
    for stage in range(3):
        ticks=data['ticks'][None,None,:,None]
        g=state[:,:,None]-torch.expm1(ticks*decay[:,:,stage,None])*(u[:,:,stage,None]-state[:,:,None])
        target=truth[:,:,None]-torch.expm1(ticks*truth_decay[:,:,stage,None])*(tu[:,:,stage,None]-truth[:,:,None])
        def current(a):
            opening=torch.stack((a[...,6]**3*a[...,7],a[...,8]**3*a[...,9],a[...,11]),-1)
            drive=data['v'][member,idx][:,:,stage,None,None]-torch.tensor(ionic.REVERSALS[[3,4,6]],device=w.device)
            gb=torch.tensor(ionic.GBAR[[3,4,6]],device=w.device,dtype=torch.float64)
            return opening*drive*gb
        pred_open.append(current(g));true_open.append(current(target))
        if stage<2:
            n=data['durations'][member,idx][:,:,stage,None]
            state=state-torch.expm1(n*decay[:,:,stage])*(u[:,:,stage]-state)
            truth=truth-torch.expm1(n*truth_decay[:,:,stage])*(tu[:,:,stage]-truth)
    pp,rr=torch.stack(pred_open,2),torch.stack(true_open,2)
    scale=rr.abs().amax((2,3),keepdim=True).clamp_min(1e-7)
    slope=torch.diff(rr,dim=3).abs()/(torch.diff(data['ticks'])*.1)[None,None,None,:,None]
    slope=torch.cat((slope[:,:,:,:1],slope),3)
    denom=(slope*.3).clamp_min(scale*1e-5)
    conditioned=(5*torch.tanh((pp-rr)/(5*denom))).square().mean((1,2,3,4))
    return torch.stack((coefficient,conditioned),-1)

def step(torch,w,opt,data,idx,weights,normalizers):
    opt.zero_grad(set_to_none=True);loss=terms(torch,w,data,idx,normalizers);(loss*weights).sum().backward()
    norm=w.grad.double().flatten(1).norm(dim=1)
    w.grad.mul_((1/norm.clamp_min(1))[:,None,None].float());opt.step()
    return loss.detach()

def development(seed,count=8):
    rng=np.random.default_rng(seed);steps=12000;v=np.empty((steps,count),dtype='float32');ca=np.empty_like(v);initial=[]
    meta=[]
    for k in range(count):
        levels=rng.uniform(-75,-25,3);pulse=round(rng.uniform(5,25)*10);recovery=round(rng.uniform(10,100)*10)
        v[:,k]=levels[1];v[:pulse,k]=levels[0]
        if k%2:v[pulse+recovery:pulse+recovery+20,k]=levels[2]
        ca[:,k]=10**rng.uniform(-4.7,-3.7)
        initial.append(p.coefficients(np.array(-95. if k%2==0 else -42.),np.array(ca[0,k]),.1)[0])
        meta.append(dict(id=k,family='held_recovery' if k%2==0 else 'paired_pulse',initial_voltage=-95. if k%2==0 else -42.))
    u,z=p.coefficients(v.astype('float64'),ca.astype('float64'),.1);g=np.array(initial);truth=[]
    for tick in range(steps):g=g+z[tick]*(u[tick]-g);truth.append(g.copy())
    return dict(voltage=v,calcium=ca,initial=np.array(initial),target_gates=np.stack(truth),multipliers=np.ones((count,11))),meta

def shadow_evaluate(torch,models,data,quality,device):
    """Preregistered CPU serial FP32 shadow using actual learned GPU coefficients."""
    v,ca=data['voltage'],data['calcium'];points,ix=np.unique(np.stack((v,ca),-1).reshape(-1,2),axis=0,return_inverse=True)
    uu=[];zz=[]
    with torch.no_grad():
        a=torch.tensor(points[:,0],device=device)[None].expand(3,-1);c=torch.tensor(points[:,1],device=device)[None].expand(3,-1)
        for model in models:
            u,z=model.raw(a,c);uu.append(torch.sigmoid(u).cpu().numpy());zz.append(torch.sigmoid(z).cpu().numpy())
    u,z=np.concatenate(uu),np.concatenate(zz);indices=ix.reshape(v.shape);members=len(u);n=v.shape[1]
    state=np.broadcast_to(data['initial'].astype('float32'),(members,n,18)).copy()
    currents=np.empty((len(v),members,n,11));squared=np.zeros((members,n));maximum=np.zeros((members,n));occupancy=np.zeros((members,n),dtype=int)
    for tick in range(len(v)):
        a=u[:,indices[tick]];b=z[:,indices[tick]];state=state+b*(a-state)
        delta=state.astype('float64')-data['target_gates'][tick]
        squared+=(delta*delta).mean(-1);maximum=np.maximum(maximum,np.abs(delta).max(-1))
        occupancy+=np.sum((state < -quality['occupancy_atol']) | (state>1+quality['occupancy_atol']),axis=-1)
        currents[tick]=p.readout(v[tick][None],state,data['multipliers'],ionic.REVERSALS)[1]
    target=p.readout(v,data['target_gates'],data['multipliers'],ionic.REVERSALS)[1];rows=[]
    for member in range(members):
        cases=[]
        for k in range(n):
            mm=p.evaluate_currents(target[:,k],currents[:,member,k],.1,quality)
            gate=dict(rmse=float(np.sqrt(squared[member,k]/len(v))),max_abs=float(maximum[member,k]),occupancy_violations=int(occupancy[member,k]))
            gate['passed']=bool(gate['rmse']<=quality['gate_rmse_max'] and gate['max_abs']<=quality['gate_max_abs_max'] and gate['occupancy_violations']==0)
            mm['gate']=gate;mm['passed']=mm['passed'] and gate['passed'];cases.append(mm)
        rows.append(cases)
    return rows

def severity(mm,q):
    values=[mm['gate']['rmse']/q['gate_rmse_max'],mm['gate']['max_abs']/q['gate_max_abs_max']]
    for c in mm['channels']:
        values.extend([c['nrmse']/q['current_nrmse_max'],c['nmax']/q['current_nmax_max'],c['signed_charge_normalized']/q['signed_charge_error_max'],c['absolute_charge_normalized']/q['absolute_charge_error_max']])
        e=c['event']
        if e['status']=='missing_event':values.append(1e12)
        elif e['status']=='evaluable':values.extend([e['peak_error_ms']/q['peak_timing_max_ms'],e['onset_error_ms']/q['onset_offset_max_ms'],e['offset_error_ms']/q['onset_offset_max_ms']])
    return max(values)

def update_models(torch,models,w):
    with torch.no_grad():
        for k,m in enumerate(models):m.precision_correction.copy_(w[k*3:k*3+3,:m.precision_definition['basis']])

def prepare(torch,root,device,spec):
    parent=prior.load_b6(torch,root,device);defs=arms();models=[make_model(torch,parent,a,spec['seeds'],device) for a in defs]
    datasets={s:atomic(spec['fit_seed'],spec['anchors'],s=='sensitive') for s in ('uniform','sensitive')}
    pools={s:pool(torch,parent,d,device) for s,d in datasets.items()};data=batched_pool(torch,pools,defs)
    select=atomic(spec['selection_seed'],spec['selection_anchors']);sp=pool(torch,parent,select,device)
    selection=batched_pool(torch,dict(uniform=sp,sensitive=sp),defs)
    w=torch.nn.Parameter(torch.zeros(24,64,3,device=device));idx=torch.arange(min(spec['batch'],spec['anchors']),device=device)[None].expand(24,-1)
    # Same uniform FIT-only output normalizers, shared across all arms per seed.
    ru,rz=pools['uniform']['base_u'],pools['uniform']['base_z'];u,z=pools['uniform']['u'],pools['uniform']['z']
    err=torch.stack((ru[...,8]-torch.logit(u[...,8].clamp(1e-7,1-1e-7)),torch.nn.functional.logsigmoid(rz[...,7])-z[...,7].log(),torch.nn.functional.logsigmoid(rz[...,11])-z[...,11].log()),-1)
    norm=err.square().mean((1,2)).clamp_min(1e-12).repeat(8,1).detach()
    loss=terms(torch,w,data,idx,norm);grads=[torch.autograd.grad(loss[:,i].sum(),w,retain_graph=i==0)[0].double().flatten(1).norm(dim=1) for i in (0,1)]
    aux=(.1*grads[0][:3]/grads[1][:3].clamp_min(1e-20)).clamp(1e-8,1e4).detach()
    weights=torch.zeros(24,2,device=device,dtype=torch.float64);weights[:,0]=1
    for k,a in enumerate(defs):
        if a['conditioned']:weights[k*3:k*3+3,1]=aux
    return parent,models,datasets,data,selection,w,norm,weights

def preflight(torch,root,output,device,spec):
    small={**spec,'anchors':64,'selection_anchors':16,'batch':8}
    parent,models,ds,data,selection,w,norm,weights=prepare(torch,root,device,small)
    v=torch.linspace(-95,50,19,device=device)[None].expand(3,-1);ca=torch.full_like(v,1e-4);g=torch.full((3,19,18),.5,device=device)
    with torch.no_grad():assert all(torch.equal(parent(v,ca,g),m(v,ca,g)) for m in models)
    idx=torch.arange(8,device=device)[None].expand(24,-1);opt=torch.optim.Adam([w],lr=spec['lr'],eps=1e-8)
    step(torch,w,opt,data,idx,weights,norm);isolated=0.
    for member in (0,5,12,23):
        dd={k:(vv if k=='ticks' else (torch.zeros(1,1,device=device,dtype=torch.long) if k=='member' else vv[member:member+1])) for k,vv in data.items()}
        wi=torch.nn.Parameter(torch.zeros(1,64,3,device=device));oi=torch.optim.Adam([wi],lr=spec['lr'],eps=1e-8)
        step(torch,wi,oi,dd,idx[:1],weights[member:member+1],norm[member:member+1]);isolated=max(isolated,float((wi[0]-w[member]).abs().max()))
    assert isolated<1e-6,isolated;update_models(torch,models,w)
    with torch.no_grad():
        for m in models:
            before=parent.raw(v,ca);after=m.raw(v,ca)
            masku=[k for k in range(18) if k!=8];maskz=[k for k in range(18) if k not in (7,11)]
            assert torch.equal(before[0][...,masku],after[0][...,masku]) and torch.equal(before[1][...,maskz],after[1][...,maskz])
    _,cfg,calcium,base=prior.contract(root);cfg=prior.calibration_config(json.loads((root/prior.CONFIG).read_text()),cfg)
    core=t.make_core(torch,cfg,calcium,base,100*np.pi,models[0],t.FAMILIES['all'])
    args=[v[:,:2],ca[:,:2],g[:,:2],torch.ones((3,2,11),device=device),torch.zeros((3,2),device=device),torch.zeros((3,2,4),device=device)]
    old=t._rates
    try:
        def forbidden(*a,**k):raise AssertionError('Canonical gate fallback')
        t._rates=forbidden
        with torch.no_grad():
            aa=t.Stepper(torch,core,args);bb=t.Stepper(torch,core,args,device=='cuda');error=0.
            for _ in range(4):error=max(error,max(float((a-b).abs().max()) for a,b in zip(aa.tick(),bb.tick())))
    finally:t._rates=old
    assert error<1e-5
    # Compare cached serial FP32 coefficients against the actual scanner.
    probe,meta=development(spec['development_seed'],2);probe={k:(a[:80] if k in ('voltage','calcium','target_gates') else a) for k,a in probe.items()}
    with torch.no_grad():actual=j.i.old.Scanner(torch,[models[0]],2,32,device).run(probe['voltage'],probe['calcium'],probe['initial'].astype('float32'))[0]
    state=torch.tensor(probe['initial'].astype('float32'),device=device)[None].expand(3,-1,-1);expected=[]
    with torch.no_grad():
        for tick in range(80):state=models[0](torch.tensor(probe['voltage'][tick],device=device)[None].expand(3,-1),torch.tensor(probe['calcium'][tick],device=device)[None].expand(3,-1),state);expected.append(state.clone())
    scanner_error=float(np.max(np.abs(actual-torch.stack(expected).cpu().numpy())));assert scanner_error<1e-5
    elapsed=time.perf_counter()
    for _ in range(10):step(torch,w,opt,data,idx,weights,norm)
    if device=='cuda':torch.cuda.synchronize()
    report=dict(valid=True,optimizer_isolation_max_error=isolated,graph_eager_error=error,scanner_error=scanner_error,
        finite=bool(torch.isfinite(w).all()),zero_initialization_exact=True,unaffected_functions_exact=True,
        no_canonical_fallback=True,vectorized_members=24,seconds_per_small_update=(time.perf_counter()-elapsed)/10,
        device=device,torch=torch.__version__,source_sha256=p.digest(Path(__file__)),protocol_sha256=p.digest(root/CONFIG))
    p.save_json(output/'preflight.json',report);return report

def run(root,output,device='cuda',smoke=False):
    import torch
    spec,cfg,calcium,base=contract(root);assert device=='cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output.mkdir(parents=True,exist_ok=True);assert not (output/'checkpoint_0.pt').exists()
    p.save_json(output/'preregistration.json',spec);pf=preflight(torch,root,output,device,spec)
    if smoke:
        result=dict(valid=True,decision='GPU_PREFLIGHT_ONLY',task='41b',thresholds_changed=False);p.save_json(output/'final_report.json',result);return result
    parent,models,datasets,data,selection,w,norm,weights=prepare(torch,root,device,spec)
    for name,d in datasets.items():np.savez_compressed(output/f'fit_{name}.npz',**d)
    dev,devmeta=development(spec['development_seed']);np.savez_compressed(output/'development_inputs.npz',**dev)
    p.save_json(output/'development_cases.json',devmeta)
    p.save_json(output/'fit_calibration.json',dict(normalizers=norm.tolist(),weights=weights.tolist(),source='initial uniform FIT only; shared per frozen parent seed'))
    opt=torch.optim.Adam([w],lr=spec['lr'],eps=1e-8);idx=torch.zeros((24,spec['batch']),device=device,dtype=torch.long)
    rng=np.random.default_rng(spec['stream_seed']);curves=[];best=[None]*24;saved=[None]*24;chosen=[None]*24;clock=time.perf_counter()
    baseline=shadow_evaluate(torch,[parent],dev,spec['quality'],device)
    p.save_json(output/'development_B6.json',baseline)
    all_selection=torch.arange(spec['selection_anchors'],device=device)[None].expand(24,-1)
    baseline_loss=terms(torch,w,selection,all_selection,norm).detach().clamp_min(1e-15)
    for update in range(spec['checkpoints'][-1]+1):
        if update in spec['checkpoints']:
            torch.save(dict(step=update,correction=w.detach().cpu(),optimizer=opt.state_dict(),rng=rng.bit_generator.state),output/f'checkpoint_{update}.pt')
            assert torch.isfinite(w).all(),'Nonfinite checkpoint saved'
            print(f'[GIADA Task41b] checkpoint_saved {update}',flush=True);update_models(torch,models,w)
            with torch.no_grad():common=terms(torch,w,selection,all_selection,norm);atomic_score=(common/baseline_loss).mean(-1).cpu().numpy()
            measured=shadow_evaluate(torch,models,dev,spec['quality'],device);ranks=[]
            for member,cases in enumerate(measured):
                rank=(sum(not r['passed'] for r in cases),max(severity(r,spec['quality']) for r in cases),float(atomic_score[member]))
                ranks.append(rank)
                if best[member] is None or rank<best[member]:best[member]=rank;chosen[member]=update;saved[member]=w[member].detach().clone()
            curves.append(dict(step=update,ranks=ranks,selection_terms=common.tolist(),elapsed_seconds=time.perf_counter()-clock))
            p.save_json(output/'training_curves.json',curves);p.save_json(output/f'development_{update}.json',measured)
            print(f'[GIADA Task41b] checkpoint selection {update} passed={sum(r[0]==0 for r in ranks)}/24 members',flush=True)
        if update==spec['checkpoints'][-1]:break
        common=rng.integers(0,spec['anchors'],spec['batch']);idx.copy_(torch.tensor(common,device=device)[None].expand(24,-1));step(torch,w,opt,data,idx,weights,norm)
        if (update+1)%1000==0:print(f'[GIADA Task41b] update {update+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-clock)/60:.1f}min',flush=True)
    with torch.no_grad():w.copy_(torch.stack(saved));update_models(torch,models,w)
    torch.save({a['name']:m.state_dict() for a,m in zip(arms(),models)},output/'frozen_checkpoint.pt')
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen,ranks=best,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'),
        selection='new development shadow all-case rank, then worst complete severity including timing, then common atomic score; earliest exact tie',
        old_task41_used_for_selection=False,old_task40b_used_for_selection=False,independent_confirmation=False)
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    print('[GIADA Task41b] checkpoint selection frozen',flush=True)
    from .task41b_evaluation import evaluate
    with torch.inference_mode():result=evaluate(torch,root,output,dict(B6=parent,**{a['name']:m for a,m in zip(arms(),models)}),spec,device)
    result['elapsed_seconds']=time.perf_counter()-clock;p.save_json(output/'final_report.json',result);return result
