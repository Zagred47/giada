"""Learned calcium kinetic corrections, frozen J1, paired atomic scaling laws."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from . import task40_causal_reinsertion as t
j,p=t.j,t.p
CONFIG='experiments/task40b_preregistration_v1.json'

def arms():
    return [dict(name=f'B{k}',basis=64 if k==5 else 32,anchors=4096 if k in (3,4) else 1024,
                 paired=k in (2,4),equilibrium=k==6,m_only=k==7) for k in range(1,8)]

def mask(a):
    return [int(a['equilibrium'])]*4+[1,0 if a['m_only'] else 1,1,0 if a['m_only'] else 1]

def model_class(torch):
    class Model(j.model_class(torch)):
        def __init__(self,a,seeds):
            super().__init__('local',seeds)
            for x in self.parameters():x.requires_grad_(False)
            self.ca_definition=a
            self.ca_correction=torch.nn.Parameter(torch.zeros(len(seeds),a['basis'],8))
            self.register_buffer('ca_mask',torch.tensor(mask(a),dtype=torch.float32),persistent=False)
        def ca_features(self,v):
            n=self.ca_definition['basis'];centers=torch.linspace(-100,60,n,device=v.device,dtype=v.dtype)
            return (1-(v[...,None]-centers).abs()/(160/(n-1))).clamp_min(0)
        def raw(self,v,ca):
            u,z=super().raw(v,ca)
            extra=torch.bmm(self.ca_features(v).double(),self.ca_correction.double()).to(v.dtype)
            extra=.1*torch.tanh(extra/.1)*self.ca_mask
            return torch.cat((u[...,:4]+extra[...,:4],u[...,4:]),-1),torch.cat((z[...,:4]+extra[...,4:],z[...,4:]),-1)
    return Model

def make_model(torch,parent,a,seeds,device):
    model=model_class(torch)(a,seeds).to(device);state=model.state_dict();state.update(parent.state_dict());model.load_state_dict(state)
    return model

def atomic(seed,n):
    rng=np.random.default_rng(seed);v=rng.uniform(-100,60,n).astype('float32');ca=10**rng.uniform(-7,-2,n)
    eps=rng.choice([.05,.2,1.],n).astype('float32');pair=np.stack((v-eps,v+eps),-1).clip(-100,60)
    u,z=p.coefficients(pair.astype('float64'),np.broadcast_to(ca[:,None],pair.shape),.1)
    return dict(v=pair,ca=ca.astype('float32'),u=u[...,:4],z=z[...,:4],initial=rng.uniform(0,1,(n,4)))

def pool(torch,parent,data,device):
    v=torch.tensor(data['v'],device=device);ca=torch.tensor(data['ca'],device=device)
    with torch.no_grad():
        u,z=parent.raw(v.flatten()[None].expand(3,-1),ca[:,None].expand_as(v).flatten()[None].expand(3,-1))
    result=dict(base_u=u[...,:4].reshape(3,len(v),2,4).double(),base_z=z[...,:4].reshape(3,len(v),2,4).double())
    result.update({k:torch.tensor(data[k],device=device) for k in ('u','z','initial','v')})
    return result

def features(torch,v,n):
    centers=torch.linspace(-100,60,n,device=v.device,dtype=v.dtype)
    return (1-(v[...,None]-centers).abs()/(160/(n-1))).clamp_min(0).double()

def batch_pool(torch,pool,definitions):
    # 21 independent parameter sets; inactive padded basis entries have zero gradient.
    n=len(pool['v']);values={}
    for key in ('base_u','base_z'):values[key]=pool[key].repeat(len(definitions),1,1,1)
    for key in ('u','z','initial','v'):values[key]=pool[key][None].expand(3*len(definitions),*pool[key].shape)
    feat=[];masks=[]
    for a in definitions:
        f=features(torch,pool['v'],a['basis'])
        f=torch.nn.functional.pad(f,(0,64-a['basis']))
        feat.extend([f]*3);masks.extend([mask(a)]*3)
    values['features']=torch.stack(feat);values['mask']=torch.tensor(masks,device=pool['v'].device,dtype=torch.float64)
    values['member']=torch.arange(3*len(definitions),device=pool['v'].device)[:,None]
    values['ticks']=torch.tensor([1.,10.,80.,400.],device=pool['v'].device,dtype=torch.float64)
    values['gbar']=torch.tensor(t.ionic.GBAR[:2],device=pool['v'].device,dtype=torch.float64)
    return values

def terms(torch,w,data,idx):
    member=data['member'];f=data['features'][member,idx]
    extra=torch.bmm(f.reshape(len(w),-1,64),w.double()).float().double().reshape(len(w),idx.shape[1],2,8)
    # Match deployed FP32 tanh/addition, retaining FP64 loss arithmetic.
    extra=(.1*torch.tanh(extra.float()/.1)).double()*data['mask'][:,None,None]
    ru=(data['base_u'][member,idx]+extra[...,:4]).float().double()
    rz=(data['base_z'][member,idx]+extra[...,4:]).float().double()
    target_u=data['u'][member,idx];target_z=data['z'][member,idx]
    expected_u=torch.logit(target_u.clamp(1e-7,1-1e-7))
    coefficient=((ru-expected_u).square()+(torch.nn.functional.logsigmoid(rz)-target_z.clamp_min(1e-12).log()).square()).mean((1,2,3))
    ticks=data['ticks']
    x0=data['initial'][member,idx][:,:,None,None,:]
    state=x0-torch.expm1(ticks[None,None,None,:,None]*torch.nn.functional.logsigmoid(-rz)[...,None,:])*(torch.sigmoid(ru)[...,None,:]-x0)
    truth=x0-torch.expm1(ticks[None,None,None,:,None]*torch.log1p(-target_z)[...,None,:])*(target_u[...,None,:]-x0)
    def current(g):
        opening=torch.stack((g[...,0].square()*g[...,1],g[...,2].square()*g[...,3]),-1)
        vv=data['v'][member,idx][...,None,None].double()
        return opening*(vv-120.)*data['gbar']
    ii,rr=current(state),current(truth);delta=ii[:,:,1]-ii[:,:,0];ref=rr[:,:,1]-rr[:,:,0]
    scale=ref.abs().amax(2,keepdim=True).clamp_min(1e-7)
    paired=(10*torch.tanh((delta-ref)/(10*scale))).square().mean((1,2,3))
    return torch.stack((coefficient,paired),-1)

def train_step(torch,w,opt,data,idx,weights):
    opt.zero_grad(set_to_none=False);tt=terms(torch,w,data,idx);(tt*weights).sum().backward()
    norm=w.grad.double().flatten(1).square().sum(1).sqrt()
    w.grad.mul_((1/norm.clamp_min(1))[:,None,None].to(w.dtype));opt.step()
    return tt

def update_models(torch,models,w):
    with torch.no_grad():
        for k,m in enumerate(models):m.ca_correction.copy_(w[k*3:k*3+3,:m.ca_definition['basis']])

def preflight(torch,root,device):
    parent=t.load_models(torch,root,device)['J1'];aa=arms();data=batch_pool(torch,pool(torch,parent,atomic(40101,32),device),aa)
    w=torch.nn.Parameter(torch.zeros(21,64,8,device=device));idx=torch.arange(16,device=device)[None].expand(21,-1)
    weights=torch.ones(21,2,device=device);opt=torch.optim.Adam([w],lr=.001,eps=1e-8)
    tt=train_step(torch,w,opt,data,idx,weights);isolated=0.
    for member in (0,8,14,17,20):
        dd={k:(v if k in ('ticks','gbar') else (v[member:member+1] if k!='member' else torch.zeros(1,1,device=device,dtype=torch.long))) for k,v in data.items()}
        wi=torch.nn.Parameter(torch.zeros(1,64,8,device=device));oi=torch.optim.Adam([wi],lr=.001,eps=1e-8)
        train_step(torch,wi,oi,dd,idx[:1],weights[:1]);isolated=max(isolated,float((wi[0]-w[member]).abs().max()))
    assert isolated<=1e-6,isolated
    m=make_model(torch,parent,aa[0],[17,29,43],device)
    v=torch.linspace(-90,40,32,device=device)[None].expand(3,-1);ca=torch.full_like(v,1e-4);g=torch.full((3,32,18),.5,device=device)
    with torch.no_grad():
        zero=float((m(v,ca,g)-parent(v,ca,g)).abs().max());assert zero==0
        m.ca_correction.copy_(w[:3,:32]);before=parent.raw(v,ca);after=m.raw(v,ca)
        preserved=max(float((a[...,4:]-b[...,4:]).abs().max()) for a,b in zip(before,after));assert preserved==0
    inference_graph_error=0.
    if device=='cuda':
        with torch.no_grad():
            scanner=t.j.i.old.Scanner(torch,[m],2,4,device)
            vv=np.full((7,2),-60.,dtype='float32');cc=np.full_like(vv,1e-4);initial=np.full((2,18),.5,dtype='float32')
            actual=scanner.run(vv,cc,initial)[0];state=torch.tensor(initial,device=device)[None].expand(3,-1,-1);expected=[]
            for tick in range(7):
                state=m(torch.tensor(vv[tick],device=device)[None].expand(3,-1),torch.tensor(cc[tick],device=device)[None].expand(3,-1),state);expected.append(state.clone())
            inference_graph_error=float(np.max(np.abs(actual-torch.stack(expected).cpu().numpy())))
            assert inference_graph_error<=1e-5,inference_graph_error
    return dict(valid=True,optimizer_isolation_max_error=isolated,zero_correction_error=zero,inference_graph_error=inference_graph_error,
        noncalcium_preservation_error=preserved,finite=bool(torch.isfinite(tt).all()),device=device,
        source_sha256=p.digest(Path(__file__)),gpu=torch.cuda.get_device_name() if device=='cuda' else None)

def causal_evaluate(torch,root,output,models,spec,device):
    old,cfg,calcium,base=t.contract(root);data,meta=t.design(root,old);p.save_json(output/'cases.json',meta)
    golden=t.numpy_reference(data,cfg,calcium,base)
    natives=[dict(voltage=data['voltage'][k],calcium=data['calcium'][k]) for k in range(32)]
    native=t.t33.metrics({k:golden[k][:,:32][None] for k in ('voltage','calcium','gates')},natives)
    assert t.t33._floor_pass(native,cfg),'Native formula floor failed'
    p.save_json(output/'native_floor.json',native)
    reports={};traces={};references={}
    for mode in ('imposed','autonomous'):
        imposed={k:golden[k].copy() for k in ('voltage','calcium')}
        if mode=='imposed':
            for k in imposed:imposed[k][:,:32]=data[k][:32].T
        core=t.make_core(torch,cfg,calcium,base,data['area_um2'])
        rr=t.rollout(torch,core,data,device,imposed if mode=='imposed' else None,device=='cuda');ref={k:v[:,0] for k,v in rr.items()};references[mode]=ref
        floor=[]
        if mode=='autonomous':
            for case in range(32,88):
                origin=meta[case]['source_index']
                floor.append(p.evaluate_intervention(golden['current'][:,origin],golden['current'][:,case],ref['current'][:,origin].astype('float64'),ref['current'][:,case].astype('float64'),.1,old['current_quality']))
            assert all(r['passed'] for r in floor),'Differential formula floor failed'
            p.save_json(output/'differential_formula_floor.json',floor)
        for name,model in models.items():
            pp=t.rollout(torch,t.make_core(torch,cfg,calcium,base,data['area_um2'],model,t.FAMILIES['all']),data,device,imposed if mode=='imposed' else None,device=='cuda')
            ref_metric={k:v.copy() for k,v in ref.items()}
            if mode=='autonomous':
                for k in ('voltage','calcium'):ref_metric[k][:,:32]=data[k][:32].T
            measured=t.metrics(pp,ref_metric,data,meta,old['current_quality'],cfg['gates'],mode);inter=[]
            for s,seed in enumerate(spec['seeds']):
                for case in range(32,88):
                    origin=meta[case]['source_index'];mm=p.evaluate_intervention(ref['current'][:,origin],ref['current'][:,case],pp['current'][:,s,origin],pp['current'][:,s,case],.1,old['current_quality'])
                    inter.append(dict(case=case,seed=seed,metrics=mm))
            measured['interventions']=inter;reports[name+'/'+mode]=measured
            if mode=='autonomous':traces.update({name+'__'+k:v for k,v in pp.items()})
            p.save_json(output/'causal_results.json',reports)
            print(f'[GIADA Task40b] checkpoint causal {name}/{mode} trajectory={measured["passed_cases"]}/264 differential={sum(r["metrics"]["passed"] for r in inter)}/168',flush=True)
    np.savez_compressed(output/'autonomous_traces.npz',**traces)
    np.savez_compressed(output/'reference_traces.npz',**{name+'__'+k:v for name,row in references.items() for k,v in row.items()})
    original=json.loads((root/'experiments/results/task40_audited_v1/audit.json').read_text())
    reproduced={}
    for mode in ('imposed','autonomous'):
        flags={str((r['case'],r['seed'])):r['metrics']['passed'] for r in reports['B0/'+mode]['interventions']}
        expected=original['summary']['J1/all/'+mode]
        reproduced[mode]=flags==expected['flags'] and reports['B0/'+mode]['passed_cases']==expected['trajectory_passed']
    p.save_json(output/'B0_reproduction.json',dict(valid=all(reproduced.values()),modes=reproduced))
    assert all(reproduced.values()),'Frozen B0 does not reproduce Task40; model comparison suspended'
    return reports

def paired_contrasts(output,spec,reports):
    rows=json.loads((output/'recurrence_evaluation.json').read_text());scores={}
    def severity(mm):
        values=[]
        if mm.get('gate') is not None:values += [mm['gate']['rmse']/.005,mm['gate']['max_abs']/.03]
        for cc in mm['channels']:values += [cc['nrmse']/.01,cc['nmax']/.05,cc['signed_charge_normalized']/.02,cc['absolute_charge_normalized']/.02]
        return max(values)
    for name in ['B0']+[a['name'] for a in arms()]:
        scores[name]={}
        for domain in ('development','causal'):
            rr=[r for r in rows if r['model']=='joint_w'+name and r['role']=='development'] if domain=='development' else reports[name+'/autonomous']['interventions']
            scores[name][domain]={}
            for seed in spec['seeds']:
                selected=[r for r in rr if r['seed']==seed];values=[severity(r['metrics']) for r in selected]
                scores[name][domain][str(seed)]=dict(median=float(np.median(values)),worst=max(values),passed=sum(r['metrics']['passed'] for r in selected),total=len(selected))
    result=[]
    for contrast in spec['contrasts']:
        for domain in ('development','causal'):
            aa,bb=(scores[contrast[k]][domain] for k in ('control','treatment'))
            ratios=[bb[str(s)]['median']/max(aa[str(s)]['median'],1e-12) for s in spec['seeds']]
            worst=[bb[str(s)]['worst']/max(aa[str(s)]['worst'],1e-12) for s in spec['seeds']]
            result.append(dict(**contrast,domain=domain,ratios=ratios,worst_ratios=worst,
                supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05),severity_excludes_timing=True))
    p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=result,capacity_caveat=spec['capacity_caveat']))

def run(root,output,device='cuda',smoke=False):
    import torch
    assert device=='cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());assert spec['current_quality']==json.loads((root/t.CONFIG).read_text())['current_quality']
    for name,sha in spec['sources'].items():assert p.digest(root/name)==sha,name
    output.mkdir(exist_ok=True,parents=True);assert not (output/'checkpoint_0.pt').exists()
    p.save_json(output/'preregistration.json',spec);p.save_json(output/'preflight.json',preflight(torch,root,device))
    parent=t.load_models(torch,root,device)['J1'];defs=arms();models=[make_model(torch,parent,a,spec['seeds'],device) for a in defs]
    fit=atomic(spec['fit_seed'],4096);held=atomic(spec['selection_seed'],256)
    np.savez_compressed(output/'fit_atomic.npz',**fit);np.savez_compressed(output/'selection_atomic.npz',**held)
    fitpool=batch_pool(torch,pool(torch,parent,fit,device),defs);selpool=batch_pool(torch,pool(torch,parent,held,device),defs)
    w=torch.nn.Parameter(torch.zeros(21,64,8,device=device));opt=torch.optim.Adam([w],lr=spec['lr'],eps=1e-8,capturable=True)
    idx=torch.arange(spec['batch'],device=device)[None].expand(21,-1).clone()
    with torch.no_grad():baseline=terms(torch,w,selpool,torch.arange(256,device=device)[None].expand(21,-1)).clamp_min(1e-15)
    initial=terms(torch,w,fitpool,idx);norms=[]
    for k in range(2):norms.append(torch.autograd.grad(initial[:,k].sum(),w,retain_graph=k==0)[0].double().flatten(1).norm(dim=1))
    auxiliary=(.1*norms[0][3:6]/norms[1][3:6].clamp_min(1e-20)).clamp(1e-8,1e4).detach()
    weights=torch.zeros(21,2,device=device,dtype=torch.float64);weights[:,0]=1
    for number,a in enumerate(defs):
        if a['paired']:weights[number*3:number*3+3,1]=auxiliary
    p.save_json(output/'fit_loss_calibration.json',dict(weights=weights.tolist(),source='Initial B2 uniform FIT gradient norms; shared per seed across paired arms'))
    def step():return train_step(torch,w,opt,fitpool,idx,weights)
    # Vectorized eager training avoids the runtime's backward stream-capture fault.
    # Frozen inference continues to use the already verified CUDA Graph scanner.
    for _ in range(3):step()
    with torch.no_grad():
        w.zero_()
        for state in opt.state.values():
            for value in state.values():
                if torch.is_tensor(value):value.zero_()
    probe=torch.nn.Parameter(torch.zeros_like(w));probe_opt=torch.optim.Adam([probe],lr=spec['lr'],eps=1e-8,capturable=True)
    train_step(torch,probe,probe_opt,fitpool,idx,weights);step();torch.cuda.synchronize()
    graph_error=float((w-probe).abs().max());assert graph_error<=1e-6,graph_error
    with torch.no_grad():
        w.zero_()
        for state in opt.state.values():
            for value in state.values():
                if torch.is_tensor(value):value.zero_()
    p.save_json(output/'training_graph_preflight.json',dict(valid=True,cuda_graph_training=False,one_step_max_error=graph_error,independent_per_member_clipping=True,
        reason='Runtime backward stream capture incompatible;21members remain vectorized,with precomputed frozen parent coefficients.'))
    del probe,probe_opt
    clock=time.perf_counter()
    for _ in range(20):step()
    torch.cuda.synchronize();seconds_per_update=(time.perf_counter()-clock)/20
    with torch.no_grad():
        w.zero_()
        for state in opt.state.values():
            for value in state.values():
                if torch.is_tensor(value):value.zero_()
    p.save_json(output/'execution.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,training_members=21,
        backend='Vectorized eager training; independent Adam states and per-member clipping; frozen-parent coefficients precomputed',tf32=False,
        seconds_per_update=seconds_per_update,projected_training_seconds=seconds_per_update*spec['checkpoints'][-1],
        trainable_parameters_per_member={a['name']:a['basis']*sum(mask(a)) for a in defs},physical_padded_parameters=512,
        exposure_per_member_per_update=spec['batch'],thresholds_changed=False))
    curves=[];best=[None]*21;chosen=[None]*21;saved=[None]*21;rng=np.random.default_rng(spec['stream_seed']);start=time.perf_counter()
    checkpoints=[0,2] if smoke else spec['checkpoints']
    for update in range(checkpoints[-1]+1):
        if update in checkpoints:
            torch.save(dict(correction=w.detach().cpu(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,step=update),output/f'checkpoint_{update}.pt')
            p.save_json(output/'checkpoint_phase.json',dict(step=update,parameters_finite=bool(torch.isfinite(w).all()),phase='saved_before_selection'))
            print(f'[GIADA Task40b] checkpoint_saved {update}',flush=True)
            assert torch.isfinite(w).all(),'Nonfinite correction checkpoint preserved'
            with torch.no_grad():
                heldloss=terms(torch,w,selpool,torch.arange(256,device=device)[None].expand(21,-1));score=(heldloss/baseline).mean(-1)
                fitloss=terms(torch,w,fitpool,torch.arange(1024,device=device)[None].expand(21,-1))
            for member,value in enumerate(score.tolist()):
                if np.isfinite(value) and (best[member] is None or value<best[member]):best[member]=value;chosen[member]=update;saved[member]=w[member].detach().clone()
            curves.append(dict(step=update,selection_score=score.tolist(),selection_terms=heldloss.tolist(),fit_probe_terms=fitloss.tolist(),elapsed_seconds=time.perf_counter()-start))
            p.save_json(output/'training_curves.json',curves)
        if update==checkpoints[-1]:break
        common=rng.random(spec['batch']);indices=np.stack([np.floor(common*a['anchors']).astype('int64') for a in defs for _ in spec['seeds']])
        idx.copy_(torch.tensor(indices,device=device));step()
        if (update+1)%1000==0:print(f'[GIADA Task40b] update {update+1}/{checkpoints[-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    if smoke:
        result=dict(valid=True,decision='PREFLIGHT_ONLY',parameters_finite=bool(torch.isfinite(w).all()))
        p.save_json(output/'final_report.json',result);return result
    with torch.no_grad():
        w.copy_(torch.stack(saved));update_models(torch,models,w)
    torch.save({a['name']:m.state_dict() for a,m in zip(defs,models)},output/'frozen_checkpoint.pt')
    freeze=dict(arms=defs,seeds=spec['seeds'],chosen_steps=chosen,selection_scores=best,selection_source='Independent atomic selection set only; no Task40 paths or stress selection',
        checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'),stress_used_for_selection=False,independent_confirmation_accessed=False)
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    print('[GIADA Task40b] checkpoint selection frozen',flush=True)
    allmodels=dict(B0=parent,**{a['name']:m for a,m in zip(defs,models)})
    with torch.inference_mode():
        old=json.loads((root/j.i.old.CONFIG).read_text());evaluation=dict(json.loads((root/j.CONFIG).read_text()),architectures=list(allmodels))
        for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=old[key]
        quality=json.loads((root/p.CONFIG).read_text())
        development=j.i.old.evaluate(root,output,evaluation,list(allmodels.values()),torch,device,quality)
        reports=causal_evaluate(torch,root,output,allmodels,spec,device)
        paired_contrasts(output,spec,reports)
    passing=[name for name in allmodels if name!='B0' and development['joint_w'+name]['all_development_passed'] and
        all(reports[name+'/'+mode]['passed'] for mode in ('autonomous','imposed')) and
        all(r['metrics']['passed'] for r in reports[name+'/autonomous']['interventions'])]
    final=dict(valid=True,task='40b',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'CALCIUM_PRECISION_NO_GO',passing_arms=passing,
        development=development,causal_summary={k:dict(trajectory_passed=v['passed_cases'],total=v['total'],intervention_passed=sum(r['metrics']['passed'] for r in v['interventions']),intervention_total=168) for k,v in reports.items()},
        thresholds_changed=False,independent_confirmation=False,task41_authorized=False,stress_used_for_selection=False,
        elapsed_seconds=time.perf_counter()-start,limitations=spec['limits'])
    p.save_json(output/'final_report.json',final);return final
