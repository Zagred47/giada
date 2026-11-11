"""45c same-head kinetic refinement; fresh FIT/DEV, opened coupled regression.

Only existing readouts change. FP64 causal Jacobians are OFFLINE training;
all candidate inference uses the unchanged materialized FP32 learned class.
"""
import copy,hashlib,json,math,os,shutil,time,zipfile
from contextlib import contextmanager
import numpy as np
from . import task45b_rate_coverage as b
y,x,n,t=b.y,b.x,b.n,b.t
TARGETS={3:('Ca_LVAst',1,2),9:('Nap_Et2',4,8),10:('Ih',5,None)}
DEFINITIONS={'N00':(9,False,False,False),'N01':(9,False,True,False),'N10':(9,True,False,False),'N11':(9,True,True,False),
             'I0':(10,True,False,False),'I1':(10,True,False,True),'L0':(3,False,True,False),'L1':(3,True,True,False)}
ARMS=('none','C0','N00','N01','N10','N11','I0','I1','L0','L1','J','R41','R111')

def spec(root):
    p=json.loads((root/'experiments/task45c_preregistration_v1.json').read_text());prior,cfg,ca,base=n.contract(root)
    assert p['quality']==prior['quality'] and p['compartment_gates']==cfg['gates']
    assert p['coupled_arms']==list(ARMS) and p['fitting']['checkpoints']==[1,4]
    assert p['corrected_target']['primary_index']==3 and p['corrected_target']['negative_control_index']==7
    assert not p['thresholds_changed'] and not p['abstraction_changed']
    return p,cfg,ca,base

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_fixtures(root,source,out):
    plan=json.loads((root/'experiments/task45c_transport_v1.json').read_text());assert sha(source)==plan['source45b_sha256'] and source.stat().st_size==plan['source45b_bytes']
    out.mkdir(exist_ok=False)
    with zipfile.ZipFile(source) as z:
        assert set(plan['members'])<=set(z.namelist())
        for name,digest in plan['members'].items():
            target=(out/name).resolve();assert target.is_relative_to(out.resolve())
            data=z.read(name);assert hashlib.sha256(data).hexdigest()==digest
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    return plan

def readout_weights(torch,model,gate,joint):
    columns=([gate] if joint else [])+[17+gate]
    return torch.cat((model.parent.c[:,:,columns],model.parent.d[:,None,columns]),1).detach().cpu().numpy().astype('float64')

def materialize(torch,baseline,gate,joint,weights):
    assert not bool(baseline.mask[gate]),'45c target must be parent-owned, not a donor readout'
    model=copy.deepcopy(baseline);columns=([gate] if joint else [])+[17+gate]
    with torch.no_grad():
        for k,column in enumerate(columns):
            model.parent.c[:,:,column].copy_(torch.as_tensor(weights[:,:-1,k],device=model.a.device,dtype=torch.float32))
            model.parent.d[:,column].copy_(torch.as_tensor(weights[:,-1,k],device=model.a.device,dtype=torch.float32))
    for value in model.parameters():value.requires_grad_(False)
    return model.eval()

def histories(p,role):
    f=p['fitting'];seeds=f[role+'_seeds'];clock=np.arange(round(f['duration_ms']/.1))*.1
    vs=[];cas=[];initial=[];meta=[];multipliers=[]
    prior=None
    for family in f['families']:
        for seed in seeds:
            rng=np.random.default_rng(seed+1009*f['families'].index(family))
            rest=rng.uniform(-60,-40);v=np.full(len(clock),rest);ca=np.full_like(v,10**rng.uniform(-4.4,-3.6))
            if family=='doublepulse':
                v[clock<rng.uniform(12,25)]=rng.uniform(-30,-14);v[(clock>=85)&(clock<100)]=rng.uniform(-35,-18)
            elif family=='staircase':
                v[clock<20]=-22;v[(clock>=40)&(clock<90)]=rest+2;v[(clock>=90)&(clock<140)]=rest-2
                ca+=3e-4*np.minimum(np.clip((clock-100)/100,0,1),np.clip((700-clock)/300,0,1))
            elif family=='tail_hold':
                v[(clock>=40)&(clock<200)]=rng.uniform(-134,-105);v[(clock>=208)&(clock<216)]=-22
            elif family=='tail_ramp':
                ramp=(clock>=40)&(clock<80);v[ramp]=np.linspace(rest,-130,int(ramp.sum()));v[(clock>=80)&(clock<200)]=-130
                ramp=(clock>=200)&(clock<220);v[ramp]=np.linspace(-130,rest,int(ramp.sum()));v[(clock>=221)&(clock<229)]=-18
            else:raise ValueError(family)
            for history,start in enumerate((-95.,-42.)):
                uu,_=n.p.coefficients(np.array(start),np.array(ca[0]),.1)
                vs.append(v.astype('float32'));cas.append(ca.astype('float32'));initial.append(uu)
                meta.append(dict(family=family,input_seed=seed,history=history,nominal=family in ('doublepulse','staircase')))
                multipliers.append(np.array([.75,1.,1.25])[np.arange(11)%3])
    v=np.stack(vs,1);ca=np.stack(cas,1);initial=np.stack(initial)
    u,z=n.p.coefficients(v.astype('float64'),ca.astype('float64'),.1);state=initial.copy();truth=[]
    for tick in range(len(v)):state+=z[tick]*(u[tick]-state);truth.append(state.copy())
    truth=np.stack(truth);mult=np.stack(multipliers)
    current=n.p.readout(v,truth,mult,t.ionic.REVERSALS)[1]
    return dict(voltage=v,calcium=ca,initial=initial,target_u=u,target_z=z,truth=truth,target_current=current,mult=mult,meta=meta)

def features_and_raw(torch,model,v,ca,device):
    features=[];raw=[]
    with torch.inference_mode():
        for start in range(0,v.size,4096):
            vv=torch.tensor(v.reshape(-1)[start:start+4096],device=device)[None].expand(3,-1)
            cc=torch.tensor(ca.reshape(-1)[start:start+4096],device=device)[None].expand(3,-1)
            ru,rz=model.raw(vv,cc);features.append(b.features(torch,model,vv).cpu().numpy())
            raw.append(torch.stack((ru,rz),-1).cpu().numpy())
    ff=np.concatenate(features,1).reshape(3,*v.shape,-1).transpose(1,0,2,3).astype('float64')
    rr=np.concatenate(raw,1).reshape(3,*v.shape,18,2).transpose(1,0,2,3,4).astype('float64')
    return ff,rr

def prepare_role(torch,model,p,role,out,device):
    data=histories(p,role);data['features'],data['raw']=features_and_raw(torch,model,data['voltage'],data['calcium'],device)
    with torch.inference_mode():data['baseline_gates']=n.scan(torch,model,data['voltage'],data['calcium'],data['initial'],device)
    np.savez_compressed(out/(role+'_inputs.npz'),voltage=data['voltage'],calcium=data['calcium'],initial=data['initial'],multipliers=data['mult'])
    n.p.save_json(out/(role+'_manifest.json'),dict(meta=data['meta'],input_sha256=hashlib.sha256(data['voltage'].tobytes()+data['calcium'].tobytes()+data['initial'].tobytes()).hexdigest(),selection_role=role))
    return data

def coefficient_pool(torch,baseline,p,device):
    f=p['fitting'];rng=np.random.default_rng(f['coefficient_seed']);q=(np.arange(f['points'])+.5)/f['points']
    v=(-100+160*q).astype('float32');v[::2]=(-135+35*q[::2]).astype('float32');ca=np.full_like(v,1e-4)
    ff,raw=features_and_raw(torch,baseline,v[None],ca[None],device)
    u,z=n.p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    clipped_u=np.clip(u,1e-12,1-1e-12);clipped_z=np.clip(z,1e-12,1-1e-12)
    label=np.stack((np.log(clipped_u/(1-clipped_u)),np.log(clipped_z/(1-clipped_z))),-1)
    return dict(f=ff[0],raw=raw[0],label=label,censored=int(np.count_nonzero((u<=1e-12)|(u>=1-1e-12)|(z<=1e-12)|(z>=1-1e-12))))

def sigmoid(v):return np.exp(-np.logaddexp(0.,-v))

def channel_scale(data,gate):
    name,ch,partner=TARGETS[gate];v=data['voltage']
    scale=t.ionic.GBAR[ch]*data['mult'][None,:,ch]*(v-t.ionic.REVERSALS[ch])
    if partner is None:return np.broadcast_to(scale[:,None],(len(v),3,v.shape[1])).copy()
    return scale[:,None]*data['baseline_gates'][...,partner].astype('float64')**int(t.ionic.POWERS[ch])

def sensitivities(data,gate,joint,delta,sample_ticks):
    """Real-arithmetic causal Jacobian of SAME gate map, all history retained."""
    f=data['features'];width=f.shape[-1];raw=data['raw'][...,gate,:].copy()
    if joint:raw[...,0]+=np.einsum('tspk,sk->tsp',f,delta[:,:width])
    raw[...,1]+=np.einsum('tspk,sk->tsp',f,delta[:,width:] if joint else delta)
    u=sigmoid(raw[...,0]);z=sigmoid(raw[...,1]);state=np.broadcast_to(data['initial'][:,gate][None],u.shape[1:]).copy()
    jac=np.zeros((*state.shape,width*(2 if joint else 1)));selected=[];states=[];keep=set(map(int,sample_ticks))
    for tick in range(len(u)):
        old=state.copy();jac*=1-z[tick,...,None]
        if joint:jac[...,:width]+=z[tick,...,None]*(u[tick]*(1-u[tick]))[...,None]*f[tick]
        jac[...,width:] += (z[tick]*(1-z[tick])*(u[tick]-old))[...,None]*f[tick] if joint else 0.
        if not joint:jac+=(z[tick]*(1-z[tick])*(u[tick]-old))[...,None]*f[tick]
        state+=z[tick]*(u[tick]-state)
        if tick in keep:selected.append(jac.copy());states.append(state.copy())
    return np.stack(states),np.stack(selected)

def solve_refinement(data,pool,gate,joint,guard,crossing,theta,white,p):
    cfg=p['fitting'];width=pool['f'].shape[-1];dim=width*(2 if joint else 1)
    delta=np.einsum('sij,sj->si',white,theta);heads=[0,1] if joint else [1]
    matrices=[];residuals=[]
    for index,head in enumerate(heads):
        mat=np.zeros((3,pool['f'].shape[1],dim));mat[:,:,index*width:(index+1)*width]=pool['f']
        current=pool['raw'][...,gate,head]+np.einsum('snk,sk->sn',pool['f'],delta[:,index*width:(index+1)*width])
        matrices.append(mat/math.sqrt(mat.shape[1]));residuals.append((pool['label'][None,:,gate,head]-current)/math.sqrt(mat.shape[1]))
    need_path=guard or crossing
    if need_path:
        count=len(data['voltage']);ticks=np.unique(np.rint(np.geomspace(1,count,cfg['sample_ticks'])).astype(int)-1)
        events={};_,ch,_=TARGETS[gate];ref=data['target_current'][...,ch]
        for case in range(ref.shape[1]):
            signal=ref[:,case];distance=abs(signal-signal[0]);scale=max(float(abs(signal).max()),1e-7)
            metric=n.p.event_metrics(signal,signal,.1,1e-7,.001)
            if metric['status']=='evaluable':
                active=np.flatnonzero(distance>=.1*distance.max())
                for t0 in (int(active[0]),int(active[-1])):
                    slope=abs(float(distance[min(count-1,t0+1)]-distance[max(0,t0-1)]))/.2
                    for at in (max(0,t0-1),t0,min(count-1,t0+1)):events[(at,case)]=max(slope*.3,scale*1e-5)
        if crossing and events:ticks=np.unique(np.concatenate((ticks,np.array([k[0] for k in events]))))
        state,jac=sensitivities(data,gate,joint,delta,ticks);factor=channel_scale(data,gate)[ticks]
        target=data['truth'][ticks,:,gate][:,None];iscale=np.maximum(abs(ref).max(0),1e-7)
        # Training masks do not waive evaluator gates. Guarded Nap receives
        # nominal histories; calcium repair receives all declared histories.
        nominal=np.array([r['nominal'] for r in data['meta']]);chosen=nominal if gate==9 else np.ones_like(nominal)
        if guard:
            jj=jac[:,:,chosen].transpose(1,0,2,3).reshape(3,-1,dim)
            rr=(target-state)[:,:,chosen].transpose(1,0,2).reshape(3,-1)
            denom=math.sqrt(jj.shape[1]);matrices.append(jj/(.005*denom));residuals.append(rr/(.005*denom))
            jj=(jac*factor[...,None]/iscale[None,None,:,None]/.01)[:,:,chosen].transpose(1,0,2,3).reshape(3,-1,dim)
            rr=((ref[ticks,None]-state*factor)/iscale[None,None]/.01)[:,:,chosen].transpose(1,0,2).reshape(3,-1)
            matrices.append(jj/math.sqrt(jj.shape[1]));residuals.append(rr/math.sqrt(jj.shape[1]))
            if gate==9:
                prior=data['baseline_gates'][ticks,:,:,gate]*factor
                rr=((prior-state*factor)/iscale[None,None]/.01)[:,:,chosen].transpose(1,0,2).reshape(3,-1)
                matrices.append(jj*math.sqrt(cfg['imitation_weight']/jj.shape[1]));residuals.append(rr*math.sqrt(cfg['imitation_weight']/jj.shape[1]))
        if crossing and events:
            ids={int(t):i for i,t in enumerate(ticks)};aa=[];bb=[]
            for (tick,case),denom in events.items():
                at=ids[tick];aa.append(jac[at,:,case]*factor[at,:,case,None]/denom)
                bb.append((ref[tick,case]-state[at,:,case]*factor[at,:,case])/denom)
            jj=np.stack(aa,1);rr=np.stack(bb,1);matrices.append(jj/math.sqrt(len(aa)));residuals.append(rr/math.sqrt(len(aa)))
    A=np.concatenate(matrices,1);r=np.concatenate(residuals,1);AW=A@white
    lhs=AW.transpose(0,2,1)@AW+cfg['ridge']*np.eye(dim)[None]
    rhs=np.einsum('sni,sn->si',AW,r)-cfg['ridge']*theta
    step=np.linalg.solve(lhs,rhs[...,None])[...,0];norm=np.linalg.norm(step,axis=1)
    linear_residual=np.sqrt(np.mean((np.einsum('sni,si->sn',AW,step)-r)**2,axis=1))
    step*=np.minimum(1.,cfg['max_whitened_step_norm']/np.maximum(norm,1e-30))[:,None]
    assert np.isfinite(step).all();return theta+step,dict(rows=A.shape[1],unclipped_step_norm=norm.tolist(),unclipped_linear_residual_rmse=linear_residual.tolist(),censored_labels=pool['censored'])

def evaluate_role(torch,model,data,p,device):
    with torch.inference_mode():gates=n.scan(torch,model,data['voltage'],data['calcium'],data['initial'],device)
    rows=[]
    for si,seed in enumerate(p['seeds']):
        current=n.p.readout(data['voltage'],gates[:,si],data['mult'],t.ionic.REVERSALS)[1]
        for case,meta in enumerate(data['meta']):
            for h in p['fitting_horizons_ms']:
                ticks=round(h/.1)
                metric=n.p.evaluate_currents(data['target_current'][:ticks,case],current[:ticks,case],.1,p['quality'],data['truth'][:ticks,case],gates[:ticks,si,case])
                target={}
                for gate,(_,ch,_) in TARGETS.items():
                    error=(gates[:ticks,si,case,gate]-data['truth'][:ticks,case,gate]);c=metric['channels'][ch]
                    passed=c['passed'] and np.sqrt((error**2).mean())<=.005 and abs(error).max()<=.03
                    e=c['event'];severity=max(c['nrmse']/.01,c['nmax']/.05,c['signed_charge_normalized']/.02,c['absolute_charge_normalized']/.02,
                        np.sqrt((error**2).mean())/.005,float(abs(error).max())/.03,
                        (e.get('onset_error_ms') or 0)/.3,(e.get('offset_error_ms') or 0)/.3,(e.get('peak_error_ms') or 0)/.2)
                    target[str(gate)]=dict(passed=bool(passed),severity=float(severity),gate_max=float(abs(error).max()),gate_rmse=float(np.sqrt((error**2).mean())))
                rows.append(dict(seed=seed,case=case,horizon_ms=h,**meta,passed=metric['passed'],targets=target))
    return rows

def nominate(records,baseline,gate,order):
    base={(r['seed'],r['case'],r['horizon_ms']):r for r in baseline};eligible=[]
    for label,step,rows in records:
        regression=[r for r in rows if base[(r['seed'],r['case'],r['horizon_ms'])]['passed'] and not r['passed']]
        if regression:continue
        eligible.append((sum(r['targets'][str(gate)]['passed'] for r in rows),max(r['targets'][str(gate)]['severity'] for r in rows),step,order.index(label),label))
    if not eligible:return 'C0',0
    winner=min(eligible,key=lambda r:(-r[0],r[1],r[2],r[3]));return winner[-1],winner[2]

def fit(torch,baseline,p,out,device):
    fitdata=prepare_role(torch,baseline,p,'fit',out,device);dev=prepare_role(torch,baseline,p,'dev',out,device)
    assert not set(p['fitting']['fit_seeds'])&set(p['fitting']['dev_seeds'])
    pool=coefficient_pool(torch,baseline,p,device);base_dev=evaluate_role(torch,baseline,dev,p,device)
    n.p.save_json(out/'development_C0.json',base_dev);models={};records=[];saved={};diagnostics=[]
    ff=pool['f'];wh=[]
    for si in range(3):
        _,s,vh=np.linalg.svd(ff[si],full_matrices=False);wh.append(vh.T/np.maximum(s/np.sqrt(ff.shape[1]),p['fitting']['whitening_floor'])[None])
    whitening=np.stack(wh)
    for label,(gate,joint,guard,cross) in DEFINITIONS.items():
        w0=readout_weights(torch,baseline,gate,joint);width=w0.shape[1];dim=width*(2 if joint else 1)
        white=np.zeros((3,dim,dim))
        for head in range(2 if joint else 1):white[:,head*width:(head+1)*width,head*width:(head+1)*width]=whitening
        theta=np.zeros((3,dim))
        for step in range(1,max(p['fitting']['checkpoints'])+1):
            theta,info=solve_refinement(fitdata,pool,gate,joint,guard,cross,theta,white,p)
            diagnostics.append(dict(arm=label,step=step,**info))
            if step not in p['fitting']['checkpoints']:continue
            delta=np.einsum('sij,sj->si',white,theta).reshape(3,2 if joint else 1,width).transpose(0,2,1)
            weights=(w0+delta).astype('float32');assert np.isfinite(weights).all()
            model=materialize(torch,baseline,gate,joint,weights);saved[(label,step)]=weights
            torch.save(model.state_dict(),out/(label+'_'+str(step)+'.pt'))
            np.savez_compressed(out/(label+'_'+str(step)+'_refinement.npz'),theta=theta,whitening=white,weights=weights)
            rows=evaluate_role(torch,model,dev,p,device);records.append((label,step,rows));n.p.save_json(out/('development_'+label+'_'+str(step)+'.json'),rows)
            print('[GIADA45c] '+label+' refinement'+str(step)+' finite, fresh DEV complete',flush=True)
    chosen={};allrecords=[('C0',0,base_dev)]+records
    for label,(gate,joint,_,_) in DEFINITIONS.items():
        own=[r for r in records if r[0]==label];index={(r['seed'],r['case'],r['horizon_ms']):r for r in base_dev}
        safe=[r for r in own if not any(index[(q['seed'],q['case'],q['horizon_ms'])]['passed'] and not q['passed'] for q in r[2])]
        selected,step,rr=min(safe or own,key=lambda r:(-sum(q['targets'][str(gate)]['passed'] for q in r[2]),max(q['targets'][str(gate)]['severity'] for q in r[2]),r[1]))
        # Rejected candidates remain REAL treatment arms in the coupled audit.
        # Only composition nomination may fall back to C0; never collapse a
        # scientific contrast silently by substituting baseline weights.
        chosen[label]=dict(source=selected,step=step,DEV_nonregressing=bool(safe),composition_eligible=bool(safe))
        models[label]=materialize(torch,baseline,gate,joint,saved[(label,step)])
    combined=copy.deepcopy(baseline);components={}
    for gate,labels in ((9,['N00','N01','N10','N11']),(10,['I0','I1']),(3,['L0','L1'])):
        label,step=nominate([r for r in allrecords if r[0] in ['C0']+labels],base_dev,gate,['C0']+labels);components[str(gate)]=dict(arm=label,step=step)
        if label!='C0':
            joint=DEFINITIONS[label][1];candidate=materialize(torch,baseline,gate,joint,saved[(label,step)])
            with torch.no_grad():
                for col in ([gate] if joint else [])+[17+gate]:combined.parent.c[:,:,col].copy_(candidate.parent.c[:,:,col]);combined.parent.d[:,col].copy_(candidate.parent.d[:,col])
    models['J']=combined;combo_dev=evaluate_role(torch,combined,dev,p,device);n.p.save_json(out/'development_J.json',combo_dev)
    freeze=dict(valid=True,component_choices=components,per_arm_choices=chosen,selection_used_coupled=False,selection_used_old_stress=False,
        weights_sha256={k:n.weights_hash(v) for k,v in models.items()},all_parameters_finite=all(bool(torch.isfinite(v).all()) for model in models.values() for v in model.parameters()),
        refinement_diagnostics=diagnostics,least_squares_scope='Each refinement records the unconstrained linearized ridge residual and trust-capped step separately. Ideal affine projection is diagnostic only; actual quantized DEV is authoritative.')
    n.p.save_json(out/'nomination_freeze.json',freeze)
    del fitdata,pool
    return models,freeze

def batched_model(torch,baseline,models):
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.frozen=baseline;self.ensemble=copy.deepcopy(baseline.parent)
            source=[baseline.parent if arm=='none' else models[arm].parent for arm in ARMS]
            for key,param in self.ensemble.named_parameters():
                if param.ndim and param.shape[0]==3:param.data=torch.stack([dict(m.named_parameters())[key].detach() for m in source],1).reshape(3*len(ARMS),*param.shape[1:]).clone()
                param.requires_grad_(False)
            for key,value in list(self.ensemble.named_buffers()):
                if value.ndim and value.shape[0]==3:
                    owner,_,field=key.rpartition('.');module=self.ensemble.get_submodule(owner) if owner else self.ensemble
                    setattr(module,field,torch.stack([dict(m.named_buffers())[key] for m in source],1).reshape(3*len(ARMS),*value.shape[1:]).clone())
            enabled=[]
            for arm in ARMS:
                row=[]
                for gate in (3,9,10):
                    for kind in (0,1):
                        col=gate+17*kind;row.append(arm!='none' and not torch.equal(models[arm].parent.c[:,:,col],baseline.parent.c[:,:,col]) or arm!='none' and not torch.equal(models[arm].parent.d[:,col],baseline.parent.d[:,col]))
                enabled.append(row)
            self.register_buffer('enabled',torch.tensor(enabled,device=baseline.a.device,dtype=torch.bool))
        def __getattr__(self,name):
            if name in ('a','sa','sb','sc','sd','gbar'):return getattr(super().__getattr__('frozen'),name)
            return super().__getattr__(name)
        def raw(self,v,ca):
            u,z=self.frozen.raw(v,ca);count=v.shape[1]//len(ARMS);shape=(3,len(ARMS),count)
            eu,ez=self.ensemble.raw(v.reshape(3*len(ARMS),count),ca.reshape(3*len(ARMS),count));eu=eu.reshape(*shape,18);ez=ez.reshape(*shape,18)
            u=u.clone();z=z.clone();index=0
            for gate in (3,9,10):
                for src,new in ((u,eu),(z,ez)):
                    val=src[...,gate].reshape(shape);src[...,gate]=torch.where(self.enabled[None,:,None,index],new[...,gate],val).reshape(3,-1);index+=1
            return u,z
        def forward(self,v,ca,state):
            u,z=self.raw(v,ca);return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)
    return Adapter().eval()

@contextmanager
def configured(p):
    before=y.spec,y.ARMS,y.previous.design,y.channel_masks
    y.spec=lambda root:(p,*n.contract(root)[1:]);y.ARMS=ARMS;y.previous.design=b.design
    y.channel_masks=lambda count,arms:np.array([np.full((count,11),arm!='none',dtype=bool) for arm in arms])
    try:yield
    finally:y.spec,y.ARMS,y.previous.design,y.channel_masks=before

def packed_parity(torch,adapter,models,count,p,device):
    v=torch.linspace(-135,60,count,device=device)[None].expand(3,-1).repeat(1,len(ARMS));ca=torch.full_like(v,1e-4)
    state=torch.linspace(.02,.98,18,device=device)[None,None].expand(3,v.shape[1],18)
    packed=adapter(v,ca,state);checks=[]
    for ai,arm in enumerate(ARMS[1:],1):
        sl=slice(ai*count,(ai+1)*count);want=models[arm](v[:,sl],ca[:,sl],state[:,sl]);error=float((want-packed[:,sl]).abs().max())
        checks.append(dict(arm=arm,max_gate_map_error=error,passed=error<=p['quality']['occupancy_atol']))
    assert all(r['passed'] for r in checks),'45c packed/materialized mismatch';return checks

def initial_oracle_probes(torch,baseline,fixtures,out,device):
    """Read-only canonical replay of opened native V/Ca, not a selectable arm.

    Canonical replay is the oracle reference here, NOT a claim of exact NEURON
    substep-state reconstruction. No fitting/nomination uses these outputs.
    """
    rows=[]
    for panel,geo,case,node in [('original45','45_chain6',3,1),('safe_hyper','chain4',2,0)]:
        with np.load(fixtures/panel/('native_'+geo+'.npz')) as z:
            v=z['voltage'][:-1,case,node:node+1].astype('float32');ca=z['calcium'][:-1,case,node:node+1].astype('float32');initial=z['gates'][0,case,node]
        _,raw=features_and_raw(torch,baseline,v,ca,device);u=sigmoid(raw[...,0]);zz=sigmoid(raw[...,1])
        ur,zr=n.p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
        for gate in (3,7,9,10):
            truth=np.broadcast_to(initial[gate],(3,1)).copy();states=np.full((4,3,1),initial[gate]);maxima=np.zeros_like(states);reset=np.zeros_like(states)
            for tick in range(len(v)):
                target=truth+zr[tick,:,gate]*(ur[tick,:,gate]-truth)
                for k,(cu,cz) in enumerate(((u[tick,:,:,gate],zz[tick,:,:,gate]),(ur[tick,:,gate],zz[tick,:,:,gate]),(u[tick,:,:,gate],zr[tick,:,gate]),(ur[tick,:,gate],zr[tick,:,gate]))):
                    one=truth+cz*(cu-truth);reset[k]+=(one-target)**2;states[k]+=cz*(cu-states[k]);maxima[k]=np.maximum(maxima[k],abs(states[k]-target))
                truth=target
                if tick+1 in (1000,5000,10000):
                    for k,name in enumerate(('learned','oracle_inf','oracle_rate','oracle_both')):
                        rows.append(dict(panel=panel,geometry=geo,case=case,node=node,gate_index=gate,horizon_ms=(tick+1)*.1,control=name,
                            maximum_gate_error=maxima[k,:,0].tolist(),endpoint_gate_error=abs(states[k,:,0]-truth[:,0]).tolist(),teacher_reset_one_step_rmse=np.sqrt(reset[k,:,0]/(tick+1)).tolist(),selectable=False))
    n.p.save_json(out/'initial_oracle_probes.json',dict(rows=rows,reference='Canonical recurrence replay of saved native V/Ca, same initial gate states; not exact native substep states',nomination_used=False))

def crossing_audit(pred,ref,geometry,rows,mode):
    result=[]
    for case,row in enumerate(rows):
        for node in range(len(geometry['parents'])):
            for channel in (1,4,5):
                reference=ref['current'][:,0,0,case,node,channel];distance=abs(reference-reference[0]);threshold=.1*float(distance.max())
                eligible=n.p.event_metrics(reference,reference,.1,1e-7,.001)
                if eligible['status']!='evaluable':continue
                indices=np.flatnonzero(distance>=threshold);end=int(indices[-1]);slope=float(distance[min(len(distance)-1,end+1)]-distance[max(0,end-1)])/.2
                for si,seed in enumerate((17,29,43)):
                    for ai,arm in enumerate(ARMS[1:],1):
                        current=pred['current'][:,si,ai,case,node,channel];pd=abs(current-reference[0]);active=np.flatnonzero(pd>=threshold);residual=float(pd[end]-distance[end])
                        result.append(dict(geometry=geometry['id'],case=case,node=node,seed=seed,arm=arm,mode=mode,channel=t.ionic.CHANNELS[channel],
                            reference_offset_ms=end*.1,candidate_offset_ms=None if not len(active) else int(active[-1])*.1,
                            signed_shift_ms=None if not len(active) else (int(active[-1])-end)*.1,reference_slope=slope,distance_residual=residual,
                            tangent_shift_ms=None if slope==0 else -residual/slope,diagnostic_only=True,original_event=n.p.event_metrics(reference,current,.1,1e-7,.001)))
    return result

def evaluate(torch,root,source,fixtures,out,device='cuda'):
    p,cfg,ca,base=spec(root);cache=out.parent/'verified_source45c';x.repair.p.r.verify_source(root,source,cache)
    os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
    fixture_root=out.parent/'verified_native45c';verify_fixtures(root,fixtures,fixture_root)
    original_models,parent_control=n.load_models(torch,root,device);old=original_models['dual']
    with np.load(root/p['reference41r']['heads_path']) as z:ref41=x.repair.materialize(torch,old,dict(z))
    with np.load(root/p['baseline']['heads_path']) as z:baseline=b.materialize(torch,ref41,torch.tensor(z['weights'],device=device),old,(0,1,1))
    assert n.weights_hash(baseline)==p['baseline']['logical_weights_sha256'];original_hash=n.weights_hash(baseline)
    with torch.inference_mode():initial_oracle_probes(torch,baseline,fixture_root,out,device)
    models,freeze=fit(torch,baseline,p,out,device);models['C0']=baseline;models['R41']=ref41
    models['R111']=copy.deepcopy(ref41);models['R111'].load_state_dict(torch.load(fixture_root/'f111.pt',map_location=device,weights_only=True))
    for name,model in models.items():torch.save(model.state_dict(),out/(name+'.pt'))
    adapter=batched_model(torch,baseline,models);allrows=[];execution={};parity=[]
    # Reuse frozen calibration, but never copy large native source into outputs.
    shutil.copy2(fixture_root/'calibration_freeze.json',out/'calibration_freeze.json')
    with torch.inference_mode():
        for panel in b.panel_configs(root,p,out):
            label=panel['panel'];dest=out/label;dest.mkdir(exist_ok=True);floor=json.loads((fixture_root/label/'native_floor.json').read_text());rows=[]
            with configured(panel):
                for geo in panel['geometries']:
                    parity.extend(dict(panel=label,geometry=geo['id'],**r) for r in y.endpoint_preflight(torch,root,baseline,geo,panel,cfg,ca,base,device))
                    count=len(b.design(geo,panel)[0])*len(geo['parents']);parity.extend(dict(panel=label,geometry=geo['id'],**r) for r in packed_parity(torch,adapter,models,count,p,device))
                    with np.load(fixture_root/label/('native_'+geo['id']+'.npz')) as z:teacher=dict(z)
                    repeated,info=y.rollout(torch,None,geo,panel,cfg,ca,base,device,ARMS);ref={k:v[:,:,:1].copy() for k,v in repeated.items()};del repeated
                    execution[label+'/'+geo['id']+'/formula']=info;physical=b.design(geo,panel)[0]
                    for mode,forcing in [('imposed',ref),('autonomous',None)]:
                        pred,info=y.rollout(torch,adapter,geo,panel,cfg,ca,base,device,ARMS,forcing)
                        null=[dict(field=k,error=float(abs(pred[k][:,:,0]-ref[k][:,:,0]).max())) for k in ref]
                        assert all(r['error']<=y.previous.permutation_limit(p,r['field']) for r in null),'45c formula null mismatch'
                        info['null_control']=null;execution[label+'/'+geo['id']+'/'+mode]=info
                        for h in panel['prefix_horizons_ms']:
                            ticks=round(h/.1);measured=y.score(b.a.crop(pred,ticks),b.a.crop(ref,ticks),{k:v[:ticks+1] for k,v in teacher.items()},geo,physical,ARMS,mode,panel,cfg,floor['valid'])
                            controls={(r['case'],r['seed']):r for r in measured if r['arm']=='none'};inc={(r['case'],r['seed']):r for r in measured if r['arm']=='C0'}
                            for r in measured:
                                c=controls[(r['case'],r['seed'])];before=inc[(r['case'],r['seed'])]
                                r.update(panel=label,horizon_ms=h,reference_valid=bool(r['judgment_eligible'] and c['compartment_passed'] and c['current_passed']),
                                    regression_vs_f011=bool(before['current_passed'] and before['compartment_passed'] and not(r['current_passed'] and r['compartment_passed'])))
                            rows.extend(measured)
                        np.savez_compressed(dest/('traces_'+geo['id']+'_'+mode+'.npz'),voltage=pred['voltage'],calcium=pred['calcium'],
                            target_gates=pred['gates'][...,[3,7,9,10]],target_currents=pred['current'][...,[1,4,5]],
                            full_gates_C0_J=pred['gates'][:,:,[1,10]],full_currents_C0_J=pred['current'][:,:,[1,10]])
                        n.p.save_json(dest/('crossings_'+geo['id']+'_'+mode+'.json'),crossing_audit(pred,ref,geo,physical,mode))
                        n.p.save_json(dest/'results.json',rows);n.p.save_json(out/'execution.json',execution)
                        print('[GIADA45c] '+label+'/'+geo['id']+'/'+mode+' all13 arms saved',flush=True);del pred
                    np.savez_compressed(dest/('reference_'+geo['id']+'.npz'),**ref)
            allrows.extend(rows)
        stress={};prior=n.contract(root)[0]
        for name,net in models.items():
            stress[name]=x.repair.stress(torch,net,prior,out,'stress_'+name,device);n.p.save_json(out/'stress_regressions.json',stress)
    baseline_fail=set(map(tuple,stress['C0']['failed']));assert stress['C0']['passed']==66 and stress['R41']['passed']==66 and stress['R111']['passed']==64,'Frozen stress controls failed reproduction'
    for name,r in stress.items():r['new_failures_vs_f011']=sorted(set(map(tuple,r['failed']))-baseline_fail)
    assert n.weights_hash(baseline)==original_hash
    with np.load(cache/'native_shadow_traces.npz') as z:cost_data=dict(z)
    cost_cfg=n.old_native.x.calibration_config(n.contract(root)[0],cfg)
    costs=n.benchmark(torch,dict(dual=baseline,**{k:v for k,v in models.items() if k!='C0'}),parent_control,cost_data,(cost_cfg,ca,base),p,out,device)
    assert n.weights_hash(baseline)==original_hash
    summaries={}
    for panel in p['panels']:
        summaries[panel]={}
        for h in sorted({r['horizon_ms'] for r in allrows if r['panel']==panel}):
            summaries[panel][str(h)]={arm:{mode:dict(total=len(pool),eligible=sum(r['reference_valid'] for r in pool),passed=sum(r['reference_valid'] and r['current_passed'] and r['compartment_passed'] for r in pool),regressions=sum(r['reference_valid'] and r['regression_vs_f011'] for r in pool)) for mode in ('imposed','autonomous') for pool in [[r for r in allrows if r['panel']==panel and r['horizon_ms']==h and r['arm']==arm and r['mode']==mode]]} for arm in ARMS[1:]}
    n.p.save_json(out/'stress_regressions.json',stress);n.p.save_json(out/'endpoint_preflight.json',parity)
    report=dict(task='45c',valid=True,complete=True,decision='PRESERVATION_MATRIX_COMPLETE_REQUIRES_AUDIT',baseline='45b/f011',baseline_unchanged=True,
        thresholds_changed=False,abstraction_changed=False,all_parameters_finite=all(bool(torch.isfinite(v).all()) for model in models.values() for v in model.parameters()),
        all_traces_finite=all(r['finite'] for r in allrows),null_controls_passed=True,endpoint_parity_passed=all(r['passed'] for r in parity),
        nomination=freeze,panels=summaries,stress=stress,gate_identity=dict(primary='Ca_LVAst.h/index3',negative_control='NaTs2_t.h/index7'),
        original41_gate_passed=False,independent47_accessed=False,release_authorized=False,speedup_claim=False,cost_rows=len(costs),limits=p['limits'])
    n.p.save_json(out/'final_report.json',report);return report
