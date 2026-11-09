"""Same-architecture 2x2 learnability experiment; teacher only supplies offline labels."""
import copy,json,os,time,zipfile,hashlib,shutil
from pathlib import Path
import numpy as np
from . import task41t_direct_timing as r
n=r.n
GATES=(8,9,12,13)
ARMS=('mean_heads','mean_features','margin_heads','margin_features','margin_features_coverage')
NAP_POWER=int(n.p.ionic.POWERS[4]);K_POWER=int(n.p.ionic.POWERS[7])

def affine_scan(torch,u,z,initial):
    """Differentiable associative scan; training only, original inference unchanged."""
    a=1-z.double();b=z.double()*u.double();offset=1
    while offset<len(a):
        b=torch.cat((b[:offset],b[offset:]+a[offset:]*b[:-offset]),0)
        a=torch.cat((a[:offset],a[offset:]*a[:-offset]),0)
        offset*=2
    return b+a*initial.double()[None]

def objective(torch,u,z,initial,scale,target,baseline,threshold,den,window,support,mode):
    gates=affine_scan(torch,u,z,initial)
    current=torch.stack((gates[...,0]**NAP_POWER*gates[...,1],gates[...,2]**K_POWER*gates[...,3]),-1)*scale[:,None]
    residual=(current-target[:,None])/den[None,None]
    mean=(residual.square()*window[:,None]).sum((0,2,3))/window.sum().clamp_min(1)
    active=(target-baseline[None]).abs()>=threshold[None]
    signed=(2*active.double()-1)[:,None]*((current-baseline[None,None]).abs()-threshold[None,None])/den[None,None]
    violation=torch.relu(.01-signed).square()*support[None,None]
    worst=violation.amax(0).amax((1,2))+.001*mean
    return (1-mode)*mean+mode*worst

def data_arrays(spec,seeds,coverage=False):
    prior=n.contract(Path(spec['repository_root']))[0] if 'repository_root' in spec else spec
    meta=n.stress_data(dict(prior,stress_seeds=seeds));clock=np.arange(round(prior['stress_duration_ms']/.1))*.1
    if coverage:
        for i,row in enumerate(meta):
            family=i//6;level=(i//2)%3;rest=(-65.,-50.,-40.,-35.)[family]
            vv=np.full(len(clock),rest);duration=(10.,50.,200.)[level];vv[clock<duration]=(-30.,-20.,-10.)[level]
            if family==0:vv[(clock>=duration+80)&(clock<duration+100)]=-25.
            if family==1:
                take=(clock>=duration)&(clock<duration+100);vv[take]=np.linspace((-30.,-20.,-10.)[level],rest,take.sum())
            if family==3:vv[(clock>=duration+50)&(clock<duration+150)]=rest+2
            row['v']=vv
            ca=float(row['ca'][0]);u0,_=n.p.coefficients(np.array(-85.),np.array(ca),.1)
            uu,zz=n.p.coefficients(np.array(-25.),np.array(ca),.1);state=uu+(u0-uu)*(1-zz)**1000
            uu,zz=n.p.coefficients(np.array(-65.),np.array(ca),.1)
            row['initial']=uu+(state-uu)*(1-zz)**(100 if row['history']==0 else 5000)
    v=np.stack([x['v'] for x in meta],1).astype('float32');ca=np.stack([x['ca'] for x in meta],1).astype('float32')
    initial=np.stack([x['initial'] for x in meta]);mult=np.array([prior['panels'][x['panel']] for x in meta])
    uu,zz=n.p.coefficients(v.astype('float64'),ca.astype('float64'),.1);state=initial.copy();truth=[]
    for t in range(len(v)):state+=zz[t]*(uu[t]-state);truth.append(state[:,GATES].copy())
    truth=np.stack(truth);channels=(4,7);scale=np.stack([n.p.ionic.GBAR[k]*mult[None,:,k]*(v-n.p.ionic.REVERSALS[k]) for k in channels],-1)
    target=np.stack((truth[...,0]**int(n.p.ionic.POWERS[4])*truth[...,1],truth[...,2]**int(n.p.ionic.POWERS[7])*truth[...,3]),-1)*scale
    baseline=target[0].copy();threshold=.1*np.abs(target-baseline).max(0);den=np.full(threshold.shape,1e-10);support=np.zeros_like(den);window=np.zeros_like(target)
    for case in range(v.shape[1]):
        for ch in range(2):
            signal=target[:,case,ch];ev=n.p.event_metrics(signal,signal,.1,prior['quality']['current_scale_floor_ma_cm2'])
            if ev['status']!='evaluable':continue
            support[case,ch]=1;active=np.flatnonzero(abs(signal-signal[0])>=threshold[case,ch]);slopes=[]
            for center in (int(active[0]),int(active[-1])):
                slopes.append(abs(signal[min(center+1,len(v)-1)]-signal[max(center-1,0)])/.2)
                for shift in (-100,-10,-1,0,1,10,100):window[max(0,min(len(v)-1,center+shift)),case,ch]=1
            den[case,ch]=max(min(slopes)*.3,1e-10)
    unique,inverse=np.unique(v,return_inverse=True)
    return dict(v=v,ca=ca,unique=unique,inverse=inverse.reshape(v.shape),initial=initial[:,GATES],scale=scale,target=target,baseline=baseline,threshold=threshold,den=den,window=window,support=support)

def tensor_data(torch,data,device):return {k:torch.tensor(v,device=device) for k,v in data.items()}

def set_trainable(torch,model,features):
    for p in model.parameters():p.requires_grad_(False)
    names=['parent.c','parent.d']+(['parent.a','parent.b'] if features else [])
    params=[]
    for name,value in model.named_parameters():
        if name in names:value.requires_grad_(True);params.append(value)
    assert len(params)==len(names)
    return params

def mask_and_clip(torch,model,params,limit):
    columns=(*GATES,*(g+17 for g in GATES))
    for name,value in model.named_parameters():
        if value.grad is not None and name in ('parent.c','parent.d'):
            mask=torch.zeros_like(value);mask[...,list(columns)]=1;value.grad.mul_(mask)
    norm=sum(p.grad.double().square().flatten(1).sum(1) for p in params if p.grad is not None).sqrt().clamp_min(1e-30)
    factor=(limit/norm).clamp_max(1)
    for p in params:
        if p.grad is not None:p.grad.mul_(factor.reshape(3,*([1]*(p.ndim-1))))
    return norm.detach()

def loss_for(torch,model,data,ids,lossfn,mode):
    v=data['unique'][None].expand(3,-1);ca=torch.full_like(v,1e-4)
    ru,rz=model.raw(v,ca);ix=data['inverse'][:,ids]
    u=torch.sigmoid(ru[...,list(GATES)])[:,ix].permute(1,0,2,3);z=torch.sigmoid(rz[...,list(GATES)])[:,ix].permute(1,0,2,3)
    initial=data['initial'][ids][None].expand(3,-1,-1)
    return lossfn(u,z,initial,data['scale'][:,ids],data['target'][:,ids],data['baseline'][ids],data['threshold'][ids],data['den'][ids],data['window'][:,ids],data['support'][ids],mode)

def export_checkpoint(out,step,models,optimizers,records,masters=None,rng_state=None,order=None):
    import torch
    payload=dict(step=step,models={k:v.state_dict() for k,v in models.items()},optimizers={k:v.state_dict() for k,v in optimizers.items()},training_records=records,
        masters={k:[p.detach().clone() for p in v] for k,v in (masters or {}).items()},rng_state=rng_state,minibatch_order=None if order is None else order.tolist())
    path=out/f'checkpoint_{step}.pt';torch.save(payload,path)
    public=Path('/kaggle/working')
    if public.is_dir():
        temporary=public/'checkpoint_backup.tmp.zip'
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:z.write(path,path.name)
        final=public/'checkpoint_backup.zip';os.replace(temporary,final)
        with final.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        receipt=public/'checkpoint_receipt.tmp.json';receipt.write_text(json.dumps(dict(step=step,sha256=digest,bytes=final.stat().st_size,archive=final.name)))
        os.replace(receipt,public/'checkpoint_receipt.json')

def train(torch,incumbent,prior,spec,out,device):
    print('[GIADA41u] preparing matched training paths and scan preflight',flush=True)
    datasets={False:tensor_data(torch,data_arrays(prior,spec['fit_path_seeds']),device),True:tensor_data(torch,data_arrays(prior,spec['fit_path_seeds'],True),device)}
    eager=lambda *args:objective(torch,*args)
    lossfn=eager;compile_report=dict(requested=spec['compile_scan'],enabled=False)
    # Verify the training-only scan and gradients before optimization.
    test_u=torch.full((64,3,2,4),.3,device=device,requires_grad=True);test_z=torch.full_like(test_u,.02,requires_grad=True);initial=torch.full((3,2,4),.7,device=device)
    actual=affine_scan(torch,test_u,test_z,initial);state=initial.double();seq=[]
    for tick in range(64):state=state+test_z[tick].double()*(test_u[tick].double()-state);seq.append(state)
    assert torch.allclose(actual,torch.stack(seq),atol=1e-12,rtol=1e-12)
    actual[-1].sum().backward();assert torch.isfinite(test_u.grad).all() and torch.isfinite(test_z.grad).all()
    if spec['compile_scan']:
        print('[GIADA41u] checking compiled scan forward/backward parity',flush=True)
        try:
            proposed=torch.compile(eager,fullgraph=True,dynamic=False)
            sample=datasets[False];ids=torch.arange(spec['batch_paths'],device=device)
            trial=copy.deepcopy(incumbent);set_trainable(torch,trial,True)
            for mode_value in (0.,1.):
                mode=torch.tensor(mode_value,device=device)
                a=loss_for(torch,trial,sample,ids,eager,mode);ga=torch.autograd.grad(a.sum(),trial.parent.c)[0]
                b=loss_for(torch,trial,sample,ids,proposed,mode);gb=torch.autograd.grad(b.sum(),trial.parent.c)[0]
                assert torch.allclose(a,b,rtol=1e-6,atol=1e-8) and torch.isfinite(gb).all()
                assert torch.allclose(ga,gb,rtol=1e-4,atol=1e-5),'Compiled loss gradient mismatch'
            lossfn=proposed;compile_report['enabled']=True
        except Exception as error:
            compile_report['fallback_reason']=repr(error)
            print('[GIADA41u] scan compile fallback to eager: '+type(error).__name__,flush=True)
    with torch.no_grad():
        sample=datasets[False];ids=torch.arange(spec['batch_paths'],device=device)
        ru,rz=incumbent.raw(sample['unique'][None].expand(3,-1),torch.full((3,len(sample['unique'])),1e-4,device=device))
        ix=sample['inverse'][:,ids];uu=torch.sigmoid(ru[...,list(GATES)])[:,ix].permute(1,0,2,3);zz=torch.sigmoid(rz[...,list(GATES)])[:,ix].permute(1,0,2,3)
        init=sample['initial'][ids][None].expand(3,-1,-1);parallel=affine_scan(torch,uu,zz,init).cpu().numpy()
        ua=uu.cpu().numpy();za=zz.cpu().numpy();state=init.float().cpu().numpy().copy();difference=0.
        for tick in range(len(ua)):
            state+=za[tick]*(ua[tick]-state);difference=max(difference,float(abs(state-parallel[tick]).max()))
    shape_probe=[]
    with torch.no_grad():
        for value in (-65.,-50.,-40.,-35.):
            reference=incumbent.raw(torch.full((3,1),value,device=device),torch.full((3,1),1e-4,device=device))
            for batch in (24,256):
                actual=incumbent.raw(torch.full((3,batch),value,device=device),torch.full((3,batch),1e-4,device=device))
                shape_probe.append(dict(voltage=value,batch=batch,max_target_logit_difference=max(float((a[:,:1,list(GATES)]-b[:,:,list(GATES)]).abs().max()) for a,b in zip(actual,reference))))
    n.p.save_json(out/'training_preflight.json',dict(valid=True,scan='FP64 training-only associative; canonical original FP32 inference unchanged',compile=compile_report,
        scan_vs_cpu_fp32_recurrence_max_gate_error=difference,comparison_scope='Same GPU FP32 coefficients, CPU sequential accumulation; diagnostic, not bitwise GPU validation',batch_shape_diagnostic=shape_probe))
    models={k:copy.deepcopy(incumbent) for k in ARMS};optimizers={};parameters={};masters={}
    for name,model in models.items():
        parameters[name]=set_trainable(torch,model,'features' in name)
        masters[name]=[torch.nn.Parameter(p.detach().double().clone()) for p in parameters[name]]
        optimizers[name]=torch.optim.Adam(masters[name],lr=spec['learning_rate'])
    anchor_v=torch.linspace(-100,60,256,device=device)[None].expand(3,-1);anchor_ca=torch.logspace(-7,-2,256,device=device)[None].expand(3,-1)
    with torch.no_grad():anchor=tuple(torch.sigmoid(x).detach() for x in incumbent.raw(anchor_v,anchor_ca))
    other=[g for g in range(18) if g not in GATES];records=[];rng=np.random.default_rng(spec['minibatch_seed']);order=np.arange(24)
    export_checkpoint(out,0,models,optimizers,records,masters,rng.bit_generator.state,order)
    print('[GIADA41u] checkpoint0 saved; starting paired5arm GPU optimization',flush=True)
    for step in range(1,max(spec['checkpoints'])+1):
        offset=((step-1)*spec['batch_paths'])%24
        if offset==0:order=rng.permutation(24)
        ids=torch.tensor(order[offset:offset+spec['batch_paths']],device=device)
        for name,model in models.items():
            opt=optimizers[name];opt.zero_grad(set_to_none=True)
            for p in parameters[name]:p.grad=None
            mode=torch.tensor(float(name.startswith('margin')),device=device)
            score=loss_for(torch,model,datasets[name.endswith('coverage')],ids,lossfn,mode)
            ru,rz=model.raw(anchor_v,anchor_ca)
            preserve=sum(((torch.sigmoid(value)[...,other]-wanted[...,other])/1e-4).square().mean((1,2)) for value,wanted in zip((ru,rz),anchor))
            total=score+spec['preservation_weight']*preserve;assert torch.isfinite(total).all()
            total.sum().backward();gradient_norm=mask_and_clip(torch,model,parameters[name],spec['gradient_clip'])
            for master,p in zip(masters[name],parameters[name]):
                assert p.grad is None or torch.isfinite(p.grad).all()
                master.grad=None if p.grad is None else p.grad.double()
            opt.step()
            with torch.no_grad():
                for master,p in zip(masters[name],parameters[name]):p.copy_(master.float())
            assert all(torch.isfinite(p).all() for p in parameters[name])
            if step%64==0:
                changes={key:float((value-incumbent.state_dict()[key]).abs().max()) for key,value in model.state_dict().items() if key in ('parent.a','parent.b','parent.c','parent.d')}
                records.append(dict(step=step,arm=name,fit_loss=score.detach().cpu().tolist(),other_channel_penalty=preserve.detach().cpu().tolist(),preclip_gradient_norm=gradient_norm.cpu().tolist(),max_parameter_change=changes,loss_measured_before_update=True))
        if step%64==0:
            n.p.save_json(out/'training_progress.json',records);print(f'[GIADA41u] updates {step}/{max(spec["checkpoints"])} all5arms finite',flush=True)
        if step in spec['checkpoints']:export_checkpoint(out,step,models,optimizers,records,masters,rng.bit_generator.state,order)
        if step==1:print('[GIADA41u] first paired update complete, all parameters finite',flush=True)
    for name,model in models.items():
        original=incumbent.state_dict()
        for key,value in model.state_dict().items():
            if key not in ('parent.c','parent.d')+(('parent.a','parent.b') if 'features' in name else ()):assert torch.equal(value,original[key]),(name,key)
        for column in range(34):
            if column not in (*GATES,*(g+17 for g in GATES)):
                assert torch.equal(model.parent.c[...,column],incumbent.parent.c[...,column]) and torch.equal(model.parent.d[...,column],incumbent.parent.d[...,column])
    # Selection only after ALL registered training has finished, no feedback to optimizer.
    development=dict(prior,stress_seeds=spec['development_seeds']);inc_dev=r.stress(torch,incumbent,development,out,'development_incumbent',device)
    basefail=set(map(tuple,inc_dev['failed']));chosen={};budgets=[]
    for name in ARMS:
        options=[dict(step=0,development=inc_dev)]
        for step in spec['checkpoints']:
            payload=torch.load(out/f'checkpoint_{step}.pt',map_location=device,weights_only=True)
            model=copy.deepcopy(incumbent);model.load_state_dict(payload['models'][name]);model.eval()
            for value in model.parameters():value.requires_grad_(False)
            dev=r.stress(torch,model,development,out,f'development_{name}_{step}',device)
            options.append(dict(step=step,development=dev));budgets.append(dict(arm=name,step=step,development=dev))
        eligible=[x for x in options if not(set(map(tuple,x['development']['failed']))-basefail)]
        winner=max(eligible,key=lambda x:(x['development']['passed'],-x['step']))
        model=copy.deepcopy(incumbent)
        if winner['step']:
            payload=torch.load(out/f'checkpoint_{winner["step"]}.pt',map_location=device,weights_only=True);model.load_state_dict(payload['models'][name])
        for value in model.parameters():value.requires_grad_(False)
        chosen[name]=model.eval();winner['weights_sha256']=n.weights_hash(model)
        n.p.save_json(out/f'selection_{name}.json',winner)
    n.p.save_json(out/'budget_development.json',dict(rows=budgets,opened41n_used=False,training_finished_before_development=True))
    return chosen

def causal_probes(torch,incumbent,prior,out,device):
    folder=out/'opened41n_incumbent'
    with np.load(folder/'fresh_stress_inputs.npz') as a:data=dict(a)
    with np.load(folder/'stress_dual.npz') as a:archived=a['gates']
    v=data['voltage'];uv,ix=np.unique(v,return_inverse=True);ix=ix.reshape(v.shape)
    with torch.no_grad():
        vv=torch.tensor(uv,device=device)[None].expand(3,-1);ru,rz=incumbent.raw(vv,torch.full_like(vv,1e-4))
        uu=torch.sigmoid(ru[...,list(GATES)]).cpu().numpy();zz=torch.sigmoid(rz[...,list(GATES)]).cpu().numpy()
    tu,tz=n.p.coefficients(v.astype('float64'),data['calcium'].astype('float64'),.1)
    target=n.p.readout(v,data['target_gates'],data['multipliers'],n.p.ionic.REVERSALS)[1];rows={}
    labels=['paired_control']+[f'{kind}_{gate}' for gate in GATES for kind in ('u','z')]+['reset_100','reset_600','reset_900']
    for label in labels:
        state=np.broadcast_to(data['initial'][:,GATES],(3,len(data['initial']),4)).copy();trace=[]
        for tick in range(len(v)):
            u=uu[:,ix[tick]].astype('float64');z=zz[:,ix[tick]].astype('float64')
            if label.startswith(('u_','z_')):
                kind,gate=label.split('_');position=GATES.index(int(gate))
                (u if kind=='u' else z)[:,:,position]=(tu if kind=='u' else tz)[tick,:,int(gate)]
            state+=z*(u-state)
            if label.startswith('reset_') and tick==int(label.split('_')[1])*10:state[:]=data['target_gates'][tick,:,GATES].T[None]
            trace.append(state.copy())
        trace=np.stack(trace);measured=[]
        for si,seed in enumerate(prior['seeds']):
            gates=archived[:,si].astype('float64');gates[...,list(GATES)]=trace[:,si]
            current=n.p.readout(v,gates,data['multipliers'],n.p.ionic.REVERSALS)[1]
            for ci in range(v.shape[1]):measured.append(dict(seed=seed,case=ci,metrics=n.p.evaluate_currents(target[:,ci],current[:,ci],.1,prior['quality'],data['target_gates'][:,ci],gates[:,ci])))
        rows[label]=dict(passed=sum(x['metrics']['passed'] for x in measured),total=len(measured),rows=measured)
    n.p.save_json(out/'causal_probes.json',dict(results=rows,teacher_oracles_selectable=False,precision='Paired FP64 offline recurrence of target gates, other gates from archived GPU. Compare to paired_control, not bitwise canonical inference.'))

def run(torch,root,source,archive,out,device='cuda'):
    out.mkdir(exist_ok=False);spec=json.loads((root/'experiments/task41u_preregistration_v1.json').read_text());manifest=json.loads((root/'experiments/results/task41u_protocol_v1/fit_manifest.json').read_text())
    assert n.p.digest(archive)==spec['recovery_sha256']
    _,prior,freeze,floors=r.p.r.verify_source(root,source,out.parent/'verified_original_41u')
    assert prior['quality']==spec['quality'];os.environ['GIADA_TASK41N_FROZEN']=str(out.parent/'verified_original_41u/frozen_source.zip.bin')
    nets,parent=n.load_models(torch,root,device);base=nets['dual'];base_hash=n.weights_hash(base)
    pre=n.parity(torch,nets,device,prior);assert pre['eligible']==freeze['eligible'];assert base_hash==manifest['baseline_weights_sha256']
    with np.load(root/'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz') as a:incumbent=r.materialize(torch,base,dict(a))
    assert n.weights_hash(incumbent)==manifest['incumbent_weights_sha256']
    n.p.save_json(out/'preregistration.json',spec);n.p.save_json(out/'gpu_preflight.json',pre)
    trained=train(torch,incumbent,prior,spec,out,device)
    models={'baseline':base,'incumbent':incumbent,**trained}
    assert list(models)==spec['arms']
    for name,model in models.items():
        assert all(torch.isfinite(x).all() for x in model.parameters())
        assert sum(x.numel() for x in model.parameters())==sum(x.numel() for x in base.parameters())
        torch.save(model.state_dict(),out/f'frozen_{name}.pt')
    development=dict(prior,stress_seeds=spec['development_seeds'])
    inc_rows=json.loads((out/'development_incumbent/fresh_stress_results.json').read_text())['dual']['rows']
    dev={'incumbent':dict(passed=sum(x['metrics']['passed'] for x in inc_rows),total=len(inc_rows),failed=[(x['model_seed'],x['family'],x['input_seed'],x['history'],x['panel']) for x in inc_rows if not x['metrics']['passed']])}
    dev['baseline']=r.stress(torch,base,development,out,'development_baseline',device)
    for name in ARMS:dev[name]=json.loads((out/f'selection_{name}.json').read_text())['development']
    base_dev=set(map(tuple,dev['incumbent']['failed']))
    eligible=[k for k in spec['selectable'] if not(set(map(tuple,dev[k]['failed']))-base_dev) and dev[k]['passed']>=dev['incumbent']['passed']]
    nominee=max(eligible,key=lambda k:(dev[k]['passed'],-spec['arms'].index(k)))
    n.p.save_json(out/'nomination_freeze.json',dict(nominee=nominee,eligible=eligible,development=dev,opened41n_used_for_selection=False))
    cache=out.parent/'verified_original_41u'
    with np.load(cache/'native_shadow_traces.npz') as a:data=dict(a)
    meta=json.loads((cache/'cases.json').read_text());_,cfg,calcium,physics_base=n.contract(root);cfg=n.old_native.x.calibration_config(prior,cfg)
    refs={}
    for policy in ('primary','voltage_solve64'):
        with np.load(cache/f'reference_autonomous_{policy}.npz') as a:refs[policy]=dict(a)
    imposed={k:data[k].T for k in ('voltage','calcium')};irefs={k:n.m.rollout(torch,None,data,(cfg,calcium,physics_base),device,k,imposed) for k in refs}
    with zipfile.ZipFile(archive) as z:
        old=json.loads(z.read('final_report.json'));oldrows=json.loads(z.read('fresh_stress_results.json'))['dual']['rows']
    oldfail={(x['model_seed'],x['family'],x['input_seed'],x['history'],x['panel']) for x in oldrows if not x['metrics']['passed']}
    baseline_responses={(106,s) for s in prior['seeds']}|{(122,s) for s in prior['seeds']};results={};opened={}
    for name,net in models.items():
        opened[name]=r.stress(torch,net,prior,out,'opened41n_'+name,device)
        if name=='incumbent':assert opened[name]['passed']==66,'Frozen41r incumbent reproduction failed'
        if name=='baseline':assert opened[name]['passed']==61 and set(map(tuple,opened[name]['failed']))==oldfail
        summaries={}
        if name=='baseline':
            summaries={f'{mode}/{ref}':old['summaries'][f'dual/{mode}/{ref}'] for mode in ('autonomous','imposed') for ref in refs}
        else:
            for mode,forcing,truths in [('autonomous',None,refs),('imposed',imposed,irefs)]:
                pred=n.m.rollout(torch,net,data,(cfg,calcium,physics_base),device,'voltage_solve64',forcing)
                assert all(np.isfinite(x).all() for x in pred.values())
                for ref,truth in truths.items():
                    measured=n.m.measure(pred,truth,data,meta,prior,cfg,mode,'41n');summaries[f'{mode}/{ref}']=n.m.summary(measured)
                    n.p.save_json(out/f'{name}_{mode}_{ref}.json',measured)
        incumbent_failed=set(map(tuple,opened['incumbent']['failed'])) if 'incumbent' in opened else oldfail
        newstress=set(map(tuple,opened[name]['failed']))-incumbent_failed;newdev=set(map(tuple,dev[name]['failed']))-base_dev
        primary=summaries['autonomous/primary'];newresponses=[x for x in primary['failed_responses'] if (x['case'],x['seed']) not in baseline_responses]
        acquired_ok=not newresponses and all(x['trajectory_passed']==x['trajectory_total'] for x in summaries.values()) and all(summaries[k]['passed'] for k in ('autonomous/voltage_solve64','imposed/primary','imposed/voltage_solve64'))
        results[name]=dict(development=dev[name],opened_stress=opened[name],summaries=summaries,new_stress_failures=sorted(newstress),new_development_failures=sorted(newdev),new_native_response_failures=newresponses,
            concrete_diagnostic_progress=opened[name]['passed']>66 and not newstress and not newdev and acquired_ok)
        n.p.save_json(out/'progress_results.json',results)
    causal_probes(torch,incumbent,prior,out,device)
    timed={'dual':base,**{k:v for k,v in models.items() if k!='baseline'}}
    costs=n.benchmark(torch,timed,parent,data,(cfg,calcium,physics_base),prior,out,device)
    assert n.weights_hash(base)==base_hash
    report=dict(valid=True,complete=True,task='41u',decision='NONLINEAR_RELAXATION_FACTORIAL_COMPLETE_REQUIRES_AUDIT',nominee=nominee,arms=results,
        original_fp32_floor_failed_checks=[218,234],baseline_unchanged=True,thresholds_changed=False,abstraction_changed=False,inference_parameter_count_unchanged=True,
        stochastic_training_updates_per_arm=max(spec['checkpoints']),deterministic_fits_prepared_locally=False,training_performed_on_gpu=True,teacher_at_inference=False,all_parameters_finite=True,cost_rows=len(costs),task42_authorized=False,speedup_claim=False,independent_confirmation=False,limits=spec['limits'])
    n.p.save_json(out/'final_report.json',report);return report
