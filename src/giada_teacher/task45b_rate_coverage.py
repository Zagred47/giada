"""45b coefficient attribution, fixed-feature repairs and paired coupled factorial.

41r is immutable; learned readout sidecars are frozen BEFORE coupled evaluation.
Native calibration never sees model outcomes; analytic oracles are offline only.
"""
import copy,hashlib,itertools,json,os,time
from contextlib import contextmanager
import numpy as np
from . import task45_controlled_coverage as a
y,x,n,t=a.y,a.x,a.n,a.t
BITS=list(itertools.product((0,1),repeat=3))
ARMS=('none',)+tuple('f'+''.join(map(str,b)) for b in BITS)

def spec(root):
    p=json.loads((root/'experiments/task45b_preregistration_v1.json').read_text())
    prior,cfg,ca,base=n.contract(root)
    assert p['quality']==prior['quality'] and p['compartment_gates']==cfg['gates']
    assert p['passive_gates']==a.spec(root)[0]['passive_gates']
    assert p['arms']==list(ARMS) and p['seeds']==[17,29,43] and p['dt_ms']==.1
    assert not p['thresholds_changed'] and not p['abstraction_changed']
    assert p['fit']['trainable_columns']==[10,27] and p['fit']['checkpoints']==[500,2000]
    return p,cfg,ca,base

def design(geo,p):
    if p.get('panel')=='regression44':return ORIGINAL44_DESIGN(geo,p)
    old=dict(p,protocols=p['original45_protocols'])
    rows,inj,mult=a.design(geo,old)
    if p.get('panel')=='safe_hyper':
        # Exactly same densities for each geometry/coupling. Reuse original
        # conditioning/rebound but replace the root hyperpolarizing pulse.
        base_rows,base_inj,base_mult=rows,inj,mult
        rows=[];injections=[];multipliers=[]
        clock=np.arange(len(inj))*.1
        for ci,coupling in enumerate(p['coupling']):
            for level,amplitude in zip(p['calibration']['target_minima_mv'],p['calibrated_amplitudes']):
                j=ci*3;value=base_inj[:,j].copy()
                value[(clock>=40)&(clock<200),0]=amplitude
                rows.append(dict(base_rows[j],protocol='45b_safe_hyper_'+str(int(-level))))
                injections.append(value);multipliers.append(base_mult[j])
        return rows,np.stack(injections,1),np.stack(multipliers)
    return rows,inj,mult

ORIGINAL44_DESIGN=y.previous.design

def panel_configs(root,p,out):
    calibration=json.loads((out/'calibration_freeze.json').read_text())
    assert calibration['candidate_outcomes_accessed'] is False
    safe=dict(p,panel='safe_hyper',calibrated_amplitudes=calibration['amplitudes_ma_cm2'])
    old44=json.loads((root/'experiments/task44_preregistration_v1.json').read_text())
    regression=dict(p,panel='regression44',duration_ms=old44['duration_ms'],protocols=old44['protocols'],geometries=old44['geometries'],prefix_horizons_ms=[old44['duration_ms']])
    return [dict(p,panel='original45'),safe,regression]

def masks(count,arms):
    result=np.zeros((len(arms),count,11),dtype=bool)
    for i,label in enumerate(arms):
        if label!='none':
            if label not in ARMS and label!='all':raise ValueError(label)
            result[i]=True
    return result

@contextmanager
def configured(p):
    before=y.spec,y.ARMS,y.previous.design,y.channel_masks
    y.spec=lambda root:(p,*n.contract(root)[1:]);y.ARMS=ARMS
    y.previous.design=design;y.channel_masks=masks
    try:yield
    finally:y.spec,y.ARMS,y.previous.design,y.channel_masks=before

def calibrate(h,p,cfg,ca,base):
    geo=p['geometries'][0];old=dict(p,panel='original45')
    rows,inj,mult=design(geo,old);clock=np.arange(len(inj))*.1
    def observe(amplitude):
        stimulus=inj[:,0].copy();stimulus[(clock>=40)&(clock<200),0]=amplitude
        v=x.native_episode(h,geo,rows[0],stimulus,mult[0],cfg,ca,base)['voltage']
        return float(v.min())
    records=[];amplitudes=[]
    for target in p['calibration']['target_minima_mv']:
        lo,hi=p['calibration']['amplitude_bracket_ma_cm2'];vl,vh=observe(lo),observe(hi)
        assert vl<=target<=vh,(target,vl,vh)
        for step in range(p['calibration']['bisection_steps']):
            mid=(lo+hi)/2;value=observe(mid)
            records.append(dict(target_mv=target,step=step,amplitude=mid,native_min_mv=value))
            if value<target:lo=mid
            else:hi=mid
        actual=observe(hi);assert actual>=target
        amplitudes.append(hi)
    return dict(valid=True,amplitudes_ma_cm2=amplitudes,teacher_only=True,candidate_outcomes_accessed=False,records=records)

def native(root,teacher,out):
    import neuron
    from neuron import h,load_mechanisms
    from . import task32_dynamic_calcium_feedback as cad
    p,cfg,ca,base=spec(root);assert neuron.__version__.split('+')[0]=='8.2.7'
    build,hashes=cad.compile_native(teacher,out,ca);load_mechanisms(str(build.resolve()))
    n.p.save_json(out/'calibration_freeze.json',calibrate(h,p,cfg,ca,base))
    # Native objects/load only once. x.native would reload a compiled library;
    # reuse its unchanged passive/active loop with the already loaded module.
    from unittest.mock import patch
    for panel in panel_configs(root,p,out):
        dest=out/panel['panel'];dest.mkdir()
        with configured(panel),patch.object(cad,'compile_native',return_value=(build,hashes)),patch.object(neuron,'load_mechanisms',return_value=True):
            y.native(root,teacher,dest)
        print('[GIADA45b] native '+panel['panel']+' saved',flush=True)

def features(torch,net,v):
    hidden=torch.nn.functional.silu(torch.bmm(((v+40)/80)[...,None],net.parent.a)+net.parent.b)
    return torch.cat((hidden,torch.ones_like(hidden[...,:1])),-1)

def head_weights(torch,net,columns=(10,27)):
    return torch.cat((net.parent.c[:,:,list(columns)],net.parent.d[:,None,list(columns)]),1).detach().clone()

def materialize(torch,frozen,ih,old,bit):
    net=copy.deepcopy(frozen)
    with torch.no_grad():
        if bit[0]:
            net.parent.c[:,:,26].copy_(old.parent.c[:,:,26]);net.parent.d[:,26].copy_(old.parent.d[:,26])
        for head,enabled in enumerate(bit[1:]):
            if enabled:
                col=(10,27)[head];net.parent.c[:,:,col].copy_(ih[:,:-1,head]);net.parent.d[:,col].copy_(ih[:,-1,head])
    for v in net.parameters():v.requires_grad_(False)
    return net.eval()

def batched_model(torch,frozen,models,arms=ARMS):
    """Exact materialized head evaluation, not a subtract/add logit shortcut.

    Ill-conditioned readouts make (new+old_raw-old_affine) unsafe in FP32.
    Fold arm×seed into the model-batch axis and call the ACTUAL frozen class;
    select only changed heads. Every other original logit remains bitwise.
    """
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.frozen=frozen;self.arms=tuple(arms)
            assert not bool(frozen.mask[[9,10]].any()),'Target readouts must belong to frozen parent, not calcium donor'
            self.ensemble=copy.deepcopy(frozen.parent)
            source=[frozen.parent if label=='none' else models[label].parent for label in arms]
            for key,param in self.ensemble.named_parameters():
                if param.ndim and param.shape[0]==3:
                    values=[dict(model.named_parameters())[key].detach() for model in source]
                    param.data=torch.stack(values,1).reshape(3*len(arms),*param.shape[1:]).clone()
                param.requires_grad_(False)
            for key,value in list(self.ensemble.named_buffers()):
                if value.ndim and value.shape[0]==3:
                    values=[dict(model.named_buffers())[key] for model in source]
                    owner,_,field=key.rpartition('.')
                    module=self.ensemble.get_submodule(owner) if owner else self.ensemble
                    setattr(module,field,torch.stack(values,1).reshape(3*len(arms),*value.shape[1:]).clone())
            enabled=[[False]*3 if label=='none' else list(BITS[int(label[1:],2)]) for label in arms]
            self.register_buffer('enabled',torch.tensor(enabled,device=frozen.a.device,dtype=torch.bool))
        def __getattr__(self,name):
            if name in ('a','sa','sb','sc','sd','gbar'):return getattr(super().__getattr__('frozen'),name)
            return super().__getattr__(name)
        def raw(self,v,ca):
            u,z=self.frozen.raw(v,ca)
            count=v.shape[1]//len(self.arms);assert count*len(self.arms)==v.shape[1]
            shape=(3,len(self.arms),count)
            eu,ez=self.ensemble.raw(v.reshape(3*len(self.arms),count),ca.reshape(3*len(self.arms),count))
            eu=eu.reshape(*shape,18);ez=ez.reshape(*shape,18)
            u=u.clone();z=z.clone()
            for head,gate in enumerate((9,10,10)):
                src=z if head in (0,2) else u
                val=src[...,gate].reshape(shape)
                replace=self.enabled[None,:,None,head]
                candidate=ez[...,gate] if head in (0,2) else eu[...,gate]
                src[...,gate]=torch.where(replace,candidate,val).reshape(3,-1)
            return u,z
        def forward(self,v,ca,state):
            u,z=self.raw(v,ca);return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)
    return Adapter().eval()

def fit_inputs(p):
    f=p['fit'];rng=np.random.default_rng(f['fit_seed']);q=(np.arange(f['points'])+rng.uniform(.1,.9,f['points']))/f['points']
    nominal=(-100+160*q).astype('float32');balanced=nominal.copy()
    balanced[::2]=(-135+35*q[::2]).astype('float32')
    devrng=np.random.default_rng(f['dev_seed']);dev=[]
    for lo,hi in ((-100,60),(-135,-100)):
        d=(lo+(hi-lo)*(np.arange(f['dev_points_per_bin'])+devrng.uniform(.1,.9,f['dev_points_per_bin']))/f['dev_points_per_bin']).astype('float32');dev.append(d)
    assert not set(np.concatenate((nominal,balanced)).tolist())&set(np.concatenate(dev).tolist())
    return np.stack((nominal,balanced)),np.concatenate(dev)

def targets(torch,v):
    inf,tau=t._rates('Ih',v.double(),torch.full_like(v.double(),1e-4),torch)
    z=-torch.expm1(-.1/tau)
    return torch.stack((torch.logit(inf),torch.logit(z)),-1)

def adapter_preflight(torch,adapter,models,count,p,device):
    """Check actual arm width against materialized checkpoints, not a surrogate.

    Unselected gates are evaluated with common original batch semantics. The
    comparison here is the updated gate map; retain original occupancy tolerance
    rather than inventing an epsilon that can hide a materially different head.
    """
    v=torch.linspace(-135.,60.,count,device=device)[None].expand(3,-1).repeat(1,len(ARMS))
    ca=torch.full_like(v,1e-4);state=torch.linspace(.02,.98,18,device=device)[None,None].expand(3,v.shape[1],-1)
    packed=adapter(v,ca,state);rows=[]
    for ai,arm in enumerate(ARMS[1:],1):
        wanted=models[arm](v[:,ai*count:(ai+1)*count],ca[:,ai*count:(ai+1)*count],state[:,ai*count:(ai+1)*count])
        diff=(packed[:,ai*count:(ai+1)*count]-wanted).abs()
        # No model-choice contrasts may include changes to unrelated heads from
        # an adapter's alternate arithmetic. Report full and targeted errors.
        limit=p['quality']['occupancy_atol'];error=float(diff.max())
        rows.append(dict(arm=arm,points_per_arm=count,max_gate_map_error=error,limit=limit,passed=error<=limit))
    assert all(r['passed'] for r in rows),'45b packed/materialized gate-map mismatch'
    return rows

def train(torch,frozen,p,out,device):
    fit,dev=fit_inputs(p);np.savez_compressed(out/'fit_dev_inputs.npz',fit=fit,development=dev)
    digest=hashlib.sha256(fit.tobytes()+dev.tobytes()).hexdigest()
    f=p['fit'];w0=head_weights(torch,frozen).double();records=[];candidates={};lsrows=[]
    # [support,seed,point,feature] and [support,seed,point,head]
    Fs=[];Rs=[];whitens=[];conditioning=[]
    with torch.no_grad():
        for vv in fit:
            v=torch.tensor(vv,device=device)[None].expand(3,-1);ff=features(torch,frozen,v).double()
            ru,rz=frozen.raw(v,torch.full_like(v,1e-4));raw=torch.stack((ru[...,10],rz[...,10]),-1).double()
            residual=targets(torch,v)-raw
            _,s,vh=torch.linalg.svd(ff,full_matrices=False)
            whitening=vh.transpose(-1,-2)/((s/(ff.shape[1]**.5)).clamp_min(1e-5))[:,None,:]
            conditioning.append(dict(singular_values=s.cpu().tolist(),whitening_floor=1e-5))
            Fs.append(ff);Rs.append(residual);whitens.append(whitening)
        F=torch.stack(Fs);R=torch.stack(Rs);W=torch.stack(whitens);G=F@W
        vd=torch.tensor(dev,device=device)[None].expand(3,-1);fd=features(torch,frozen,vd).double()
        ud,zd=frozen.raw(vd,torch.full_like(vd,1e-4));rawdev=torch.stack((ud[...,10],zd[...,10]),-1).double();truth=targets(torch,vd)
        def metrics(weights):
            # Score the actual quantized inference readout, not the idealized
            # float64 affine projection used for the optimizer diagnostic.
            probe=copy.deepcopy(frozen)
            for head,col in enumerate((10,27)):
                probe.parent.c[:,:,col].copy_(weights[:,:-1,head].float())
                probe.parent.d[:,col].copy_(weights[:,-1,head].float())
            pu,pz=probe.raw(vd,torch.full_like(vd,1e-4));pred=torch.stack((pu[...,10],pz[...,10]),-1).double()
            err=(pred-truth).square().reshape(3,2,f['dev_points_per_bin'],2).mean(2).sqrt()
            return dict(worst=float(err.max()),per_seed_bin_head_rmse=err.cpu().tolist())
        baseline_dev=metrics(w0)
        # Solve precisely the SAME whitened ridge objective, diagnostic only.
        eye=torch.eye(G.shape[-1],device=device,dtype=torch.float64)
        lhs=G.transpose(-1,-2)@G/G.shape[-2]+f['ridge']*eye
        rhs=G.transpose(-1,-2)@R/G.shape[-2]
        sol=torch.linalg.solve(lhs,rhs)
        for support,label in enumerate(f['supports']):
            lsrows.append(dict(support=label,selectable=False,**metrics((w0+W[support]@sol[support]).float())))
    theta=torch.nn.Parameter(torch.zeros((*G.shape[:2],G.shape[-1],2),device=device,dtype=torch.float64))
    optimizer=torch.optim.Adam([theta],lr=f['learning_rate'])
    for step in range(1,max(f['checkpoints'])+1):
        optimizer.zero_grad(set_to_none=True)
        residual=G@theta-R;scores=residual.square().mean(-2)+f['ridge']*theta.square().sum(-2)
        scores.sum().backward();norm=theta.grad.square().sum(-2,keepdim=True).sqrt().clamp_min(1e-30)
        theta.grad.mul_((f['gradient_clip']/norm).clamp_max(1));optimizer.step()
        assert bool(torch.isfinite(theta).all()),'Nonfinite45b readout repair'
        if step in f['checkpoints']:
            torch.save(dict(step=step,theta=theta.detach().cpu(),optimizer=optimizer.state_dict(),whitening=W.cpu()),out/f'training_{step}.pt')
            with torch.no_grad():
                for si,label in enumerate(f['supports']):
                    weights=(w0+W[si]@theta[si]).float();key=label+'_'+str(step);candidates[key]=weights
                    assert bool(torch.isfinite(weights).all()),'Nonfinite quantized45b weights'
                    row=dict(candidate=key,support=label,step=step,**metrics(weights));records.append(row)
                    np.savez_compressed(out/(key+'.npz'),weights=weights.cpu().numpy())
            n.p.save_json(out/'development_results.json',dict(baseline=baseline_dev,candidates=records,least_squares_diagnostic=lsrows))
            print('[GIADA45b] paired fit checkpoint '+str(step)+' finite',flush=True)
    eligible=[r for r in records if np.isfinite(r['worst'])]
    assert eligible,'No finite DEV candidate; no oracle or silent fallback allowed'
    chosen=min(eligible,key=lambda r:(r['worst'],r['step'],f['supports'].index(r['support'])))
    weights=candidates[chosen['candidate']].detach()
    freeze=dict(valid=True,chosen=chosen,inputs_sha256=digest,selection_uses_coupled_outcomes=False,selection_uses_stress=False,
        weights_sha256=hashlib.sha256(weights.cpu().numpy().tobytes()).hexdigest(),all_candidates=records,baseline_development=baseline_dev,least_squares_diagnostic=lsrows,
        conditioning=conditioning,objective_is_ideal_affine_fit=True,development_uses_actual_quantized_model=True)
    n.p.save_json(out/'repair_freeze.json',freeze)
    return weights,freeze

def timing_probes(pred,ref,geo,rows,mode):
    result=[]
    for case,row in enumerate(rows):
        for site in (0,len(geo['parents'])-1):
            signal=ref['current'][:,0,0,case,site,5].astype('float64')
            baseline=signal[0];distance=abs(signal-baseline);peak=int(np.argmax(distance));amplitude=distance[peak]
            if amplitude==0:continue
            level=.1*amplitude;above=distance>=level
            crossings=np.flatnonzero(above[1:]!=above[:-1])+1
            for tick in crossings:
                slope=(distance[min(tick+1,len(signal)-1)]-distance[max(0,tick-1)])/.2
                for si,seed in enumerate((17,29,43)):
                    for ai,arm in enumerate(ARMS[1:],1):
                        current=pred['current'][:,si,ai,case,site,5]
                        pdistance=abs(current-baseline)
                        cand=np.flatnonzero((pdistance[1:]>=level)!=(pdistance[:-1]>=level))+1
                        # Nearest crossing is descriptive only, does NOT replace
                        # registered first/last event matching or any quality gate.
                        nearest=None if not len(cand) else int(cand[np.argmin(abs(cand-tick))])
                        residual=float(pdistance[tick]-distance[tick])
                        result.append(dict(geometry=geo['id'],case=case,node=site,seed=seed,arm=arm,mode=mode,reference_tick=int(tick),
                            reference_slope_ma_cm2_per_ms=float(slope),current_residual=residual,
                            linear_predicted_shift_ms=None if slope==0 else -residual/float(slope),
                            diagnostic_nearest_crossing_shift_ms=None if nearest is None else (nearest-int(tick))*.1,
                            registered_event=n.p.event_metrics(signal,current,.1,1e-7,.001),
                            diagnostic_only=True,registered_event_matching_unchanged=True))
    return result

def atomic_paths(torch,frozen,old,ref,p,out,label,device):
    """Oracle coefficient2x2 on saved paths, not a selectable inference model."""
    # Root/distal, all physical cases. Teacher path shared by model seeds.
    v=np.concatenate((ref['voltage'][:,0,0,:,0],ref['voltage'][:,0,0,:,-1]),1).astype('float32')
    c=np.concatenate((ref['calcium'][:,0,0,:,0],ref['calcium'][:,0,0,:,-1]),1).astype('float32')
    vv=torch.tensor(v,device=device);cc=torch.tensor(c,device=device)
    logs=[];oldlogs=[]
    # Vectorize time×path as the batch, keeping independent model seed axis.
    with torch.no_grad():
        flatv=vv.reshape(-1)[None].expand(3,-1);flatc=cc.reshape(-1)[None].expand(3,-1)
        for net,dest in ((frozen,logs),(old,oldlogs)):
            u,z=net.raw(flatv,flatc)
            for value in (u[..., [9,10]],z[..., [9,10]]):dest.append(torch.sigmoid(value).reshape(3,*vv.shape,2).permute(1,0,2,3))
        truth=[];ztruth=[]
        for name,gate in (('Nap_Et2',1),('Ih',None)):
            inf,tau=t._rates(name,vv.double(),cc.double(),torch)
            if gate is not None:inf=inf[...,gate];tau=tau[...,gate]
            truth.append(inf);ztruth.append(-torch.expm1(-.1/tau))
        trueu=torch.stack(truth,-1)[:,None].expand(-1,3,-1,-1);truez=torch.stack(ztruth,-1)[:,None].expand(-1,3,-1,-1)
        initial=trueu[0];names=['learned','oracle_inf','oracle_rate','oracle_both','Nap_inf_rollback_negative_control','Nap_rate_rollback','Nap_both_rollback']
        coeff=[]
        for name in names:
            u=logs[0].double();z=logs[1].double()
            if name in ('oracle_inf','oracle_both'):u=trueu
            if name in ('oracle_rate','oracle_both'):z=truez
            if name.startswith('Nap_'):
                u=u.clone();z=z.clone()
                if name in ('Nap_inf_rollback_negative_control','Nap_both_rollback'):u[...,0]=oldlogs[0][...,0]
                if name in ('Nap_rate_rollback','Nap_both_rollback'):z[...,0]=oldlogs[1][...,0]
            coeff.append((u,z))
        states=torch.stack([initial.clone() for _ in names]);reference=initial.clone();fp32=initial.float();fp64=initial.clone()
        reset_sse=torch.zeros((len(names),3,v.shape[1],2),device=device,dtype=torch.float64)
        rows=[]
        for tick in range(len(v)-1):
            nextref=reference+truez[tick]*(trueu[tick]-reference)
            for j,(u,z) in enumerate(coeff):
                one=reference+z[tick]*(u[tick]-reference);reset_sse[j]+=(one-nextref).square()
                states[j]+=z[tick]*(u[tick]-states[j])
            fp32+=logs[1][tick]*(logs[0][tick]-fp32)
            fp64+=logs[1][tick].double()*(logs[0][tick].double()-fp64)
            reference=nextref
            if round((tick+1)*.1,8) in p['atomic']['horizons_ms']:
                for j,name in enumerate(names):
                    rows.append(dict(control=name,horizon_ms=(tick+1)*.1,free_gate_endpoint_abs=(states[j]-reference).abs().cpu().tolist(),
                        teacher_reset_one_step_rmse=(reset_sse[j]/(tick+1)).sqrt().cpu().tolist(),oracle_selectable=False))
        report=dict(label=label,controls=rows,fp32_vs_fp64_max_gate_error=float((fp32.double()-fp64).abs().max()),gate_order=['Nap_Et2.h','Ih.m'],paths='all cases, root then distal',selection_used=False)
        n.p.save_json(out/('atomic_'+label+'.json'),report)
        return report

def save_targeted(path,pred):
    # No precision reduction. Persist all targeted channels/gates for every arm,
    # plus complete baseline/full-repair tensors, avoiding9 redundant full copies.
    values={k:pred[k] for k in ('voltage','calcium')}
    values['gates_Nap_h_Ih_m']=pred['gates'][...,[9,10]]
    values['current_Nap_Ih']=pred['current'][...,[4,5]]
    for k in ('gates','current'):values[k+'_f000_f111']=pred[k][:,:,[1,8]]
    np.savez_compressed(path,**values)

def atomic_clamps(torch,frozen,old,models,p,out,device):
    voltages=np.asarray(p['atomic']['voltages_mv'],dtype='float32')
    v=torch.tensor(voltages,device=device)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
    labels={'41r':frozen,'pre41r':old,**models};result=[]
    with torch.inference_mode():
        for name,net in labels.items():
            rawu,rawz=net.raw(v,ca);u=torch.sigmoid(rawu).double();z=torch.sigmoid(rawz).double()
            for channel,gate,local in [('Nap_Et2',9,1),('Ih',10,None)]:
                tu,tau=t._rates(channel,v.double(),ca.double(),torch)
                if local is not None:tu=tu[...,local];tau=tau[...,local]
                tz=-torch.expm1(-.1/tau)
                for initial in p['atomic']['initial_gate_states']:
                    for horizon in p['atomic']['horizons_ms']:
                        truth=tu+(initial-tu)*torch.exp(-horizon/tau)
                        controls={}
                        for tag,cu,cz in [('learned',u[...,gate],z[...,gate]),('oracle_inf',tu,z[...,gate]),('oracle_rate',u[...,gate],tz),('oracle_both',tu,tz)]:
                            value=cu+(initial-cu)*torch.pow(1-cz,round(horizon/.1))
                            controls[tag]=dict(max_abs_error=float((value-truth).abs().max()),per_seed_voltage_abs=(value-truth).abs().cpu().tolist())
                        result.append(dict(model=name,channel=channel,gate=gate,initial=initial,horizon_ms=horizon,controls=controls))
    n.p.save_json(out/'constant_clamp_probes.json',dict(voltages_mv=voltages.tolist(),rows=result,oracles_selectable=False))

def summary(rows):
    result={}
    for arm in ARMS[1:]:
        result[arm]={}
        for mode in ('imposed','autonomous'):
            pool=[r for r in rows if r['arm']==arm and r['mode']==mode]
            result[arm][mode]=dict(total=len(pool),eligible=sum(r['reference_valid'] for r in pool),
                passed=sum(r['reference_valid'] and r['current_passed'] and r['compartment_passed'] for r in pool),
                regressions_vs_f000=sum(r.get('regression_vs_f000',False) for r in pool))
    return result

def reproduction_audit(root,rows):
    historical=json.loads((root/'experiments/results/task45_audited_v1/audit.json').read_text())
    checks=[]
    for old in historical['horizons']:
        pool=[r for r in rows if r['panel']=='original45' and r['arm']=='f000' and r['mode']==old['mode'] and r['horizon_ms']==old['horizon_ms']]
        # Unit fixtures and an intentionally partial recovery cannot claim a
        # complete historical reproduction; retain explicit scope instead.
        if len(pool)!=old['total']:continue
        valid=[r for r in pool if r['reference_valid']];passed=sum(r['current_passed'] and r['compartment_passed'] for r in valid)
        checks.append(dict(horizon_ms=old['horizon_ms'],mode=old['mode'],old_passed=old['passed'],new_passed=passed,
            old_interpretable=old['interpretable'],new_interpretable=len(valid),passed=passed==old['passed'] and len(valid)==old['interpretable']))
    wanted={(r['geometry'],r['case'],r['seed'],r['mode']) for r in historical['full_horizon_failed_cases']}
    got={(r['geometry'],r['case'],r['seed'],r['mode']) for r in rows if r['panel']=='original45' and r['arm']=='f000' and r['horizon_ms']==1000 and r['reference_valid'] and not(r['current_passed'] and r['compartment_passed'])}
    return dict(complete=len(checks)==6,checks=checks,full_horizon_identity_match=got==wanted,
        passed=bool(len(checks)==6 and all(r['passed'] for r in checks) and got==wanted),not_a_new_quality_threshold=True)

def evaluate(torch,root,source,out,device='cuda'):
    p,cfg,ca,base=spec(root);cache=out.parent/'verified_source45b';x.repair.p.r.verify_source(root,source,cache)
    calibration=json.loads((out/'calibration_freeze.json').read_text())
    assert calibration['valid'] and calibration['teacher_only'] and not calibration['candidate_outcomes_accessed']
    os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
    old=n.load_models(torch,root,device)[0]['dual'];assert n.weights_hash(old)==p['freeze']['baseline_weights_sha256']
    with np.load(root/p['freeze']['heads_path']) as z:frozen=x.repair.materialize(torch,old,dict(z))
    assert n.weights_hash(frozen)==p['freeze']['logical_weights_sha256']
    torch.save(frozen.state_dict(),out/'frozen41r.pt')
    weights,freeze=train(torch,frozen,p,out,device)
    models={arm:materialize(torch,frozen,weights,old,bit) for arm,bit in zip(ARMS[1:],BITS)}
    for arm,net in models.items():torch.save(net.state_dict(),out/(arm+'.pt'))
    atomic_clamps(torch,frozen,old,models,p,out,device)
    adapter=batched_model(torch,frozen,models)
    allrows=[];execution={};parity=[];atomic=[];adapter_checks=[]
    with torch.inference_mode():
        for panel in panel_configs(root,p,out):
            dest=out/panel['panel'];floor=json.loads((dest/'native_floor.json').read_text())
            rows=[]
            with configured(panel):
                for geo in panel['geometries']:
                    parity.extend(dict(panel=panel['panel'],geometry=geo['id'],**r) for r in y.endpoint_preflight(torch,root,frozen,geo,panel,cfg,ca,base,device))
                    # Same input batch, f000 must equal original frozen logits.
                    core,args,_,_=y.setup(torch,adapter,geo,panel,cfg,ca,base,device,ARMS)
                    adapter_checks.extend(dict(panel=panel['panel'],geometry=geo['id'],**r) for r in adapter_preflight(torch,adapter,models,args[0].shape[1]//len(ARMS),p,device))
                    n.p.save_json(out/'adapter_preflight.json',adapter_checks)
                    original=frozen.raw(args[0].float(),args[1]);actual=adapter.raw(args[0].float(),args[1]);width=args[0].shape[1]//len(ARMS)
                    assert all(torch.equal(u[:,width:2*width],v[:,width:2*width]) for u,v in zip(original,actual))
                    with np.load(dest/('native_'+geo['id']+'.npz')) as z:teacher=dict(z)
                    repeated,info=y.rollout(torch,None,geo,panel,cfg,ca,base,device,ARMS)
                    ref={k:v[:,:,:1].copy() for k,v in repeated.items()};del repeated
                    execution[panel['panel']+'/'+geo['id']+'/formula']=info
                    if panel['panel']=='original45':atomic.append(atomic_paths(torch,frozen,old,ref,p,dest,geo['id'],device))
                    physical=design(geo,panel)[0]
                    for mode,forcing in [('imposed',ref),('autonomous',None)]:
                        pred,info=y.rollout(torch,adapter,geo,panel,cfg,ca,base,device,ARMS,forcing)
                        controls=[]
                        for key in ref:
                            error=float(abs(pred[key][:,:,0]-ref[key][:,:,0]).max());limit=y.previous.permutation_limit(panel,key)
                            controls.append(dict(field=key,max_abs_error=error,limit=limit,passed=error<=limit))
                        assert all(r['passed'] for r in controls),'45b formula execution control mismatch'
                        info['null_control']=controls;execution[panel['panel']+'/'+geo['id']+'/'+mode]=info
                        for horizon in panel['prefix_horizons_ms']:
                            if horizon>panel['duration_ms']:continue
                            ticks=round(horizon/.1);tt={k:v[:ticks+1] for k,v in teacher.items()}
                            measured=y.score(a.crop(pred,ticks),a.crop(ref,ticks),tt,geo,physical,ARMS,mode,panel,cfg,floor['valid'])
                            controls={(r['case'],r['seed']):r for r in measured if r['arm']=='none'}
                            incumbent={(r['case'],r['seed']):r for r in measured if r['arm']=='f000'}
                            for r in measured:
                                control=controls[(r['case'],r['seed'])];prior=incumbent[(r['case'],r['seed'])]
                                r.update(panel=panel['panel'],horizon_ms=horizon,
                                    reference_valid=bool(r['judgment_eligible'] and control['compartment_passed'] and control['current_passed']),
                                    regression_vs_f000=bool(prior['current_passed'] and prior['compartment_passed'] and not(r['current_passed'] and r['compartment_passed'])))
                            rows.extend(measured)
                        n.p.save_json(dest/('timing_'+geo['id']+'_'+mode+'.json'),timing_probes(pred,ref,geo,physical,mode))
                        save_targeted(dest/('traces_'+geo['id']+'_'+mode+'.npz'),pred)
                        n.p.save_json(dest/'results.json',rows);n.p.save_json(out/'execution.json',execution)
                        print('[GIADA45b] '+panel['panel']+'/'+geo['id']+'/'+mode+'9 paired arms saved',flush=True)
                        del pred
                    np.savez_compressed(dest/('reference_'+geo['id']+'.npz'),**ref);del ref
            allrows.extend(rows)
        # Opened stress is a regression audit AFTER the FIT/DEV freeze, not used
        # to retune weights, budgets or select another sidecar.
        prior=n.contract(root)[0];stress={}
        for arm,net in models.items():
            stress[arm]=x.repair.stress(torch,net,prior,out,'stress_'+arm,device)
            n.p.save_json(out/'stress_regressions.json',stress)
    baseline=set(map(tuple,stress['f000']['failed']))
    assert len(baseline)==6 and stress['f000']['passed']==66,'Original41r stress reproduction failed'
    for arm,r in stress.items():r['new_failures_vs_f000']=sorted(set(map(tuple,r['failed']))-baseline)
    n.p.save_json(out/'stress_regressions.json',stress);n.p.save_json(out/'endpoint_preflight.json',parity)
    reproduction=reproduction_audit(root,allrows);n.p.save_json(out/'original45_reproduction.json',reproduction)
    assert n.weights_hash(frozen)==p['freeze']['logical_weights_sha256']
    report=dict(task='45b',valid=True,complete=True,exploratory=True,decision='FACTORIAL_COMPLETE_REQUIRES_SCOPED_AUDIT',
        thresholds_changed=False,abstraction_changed=False,frozen41r_unchanged=True,teacher_oracles_selectable=False,
        all_parameters_finite=all(bool(torch.isfinite(v).all()) for net in models.values() for v in net.parameters()),
        all_traces_finite=all(r['finite'] for r in allrows),endpoint_parity_passed=all(r['passed'] for r in parity),
        null_controls_passed=True,repair_freeze=freeze,training_updates_per_support_seed=2000,
        original45_reproduction=reproduction,
        panels={panel:{str(h):summary([r for r in allrows if r['panel']==panel and r['horizon_ms']==h]) for h in sorted({r['horizon_ms'] for r in allrows if r['panel']==panel})} for panel in p['panels']},
        stress_regressions=stress,speedup_claim=False,original41_gate_passed=False,independent47_accessed=False,
        candidate_promotion_authorized=False,limits=p['limits'])
    n.p.save_json(out/'final_report.json',report);return report
