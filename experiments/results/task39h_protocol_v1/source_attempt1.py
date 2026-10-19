"""Task39h: paired conservative updates, relaxation-bin loss and FIT recovery support."""
import copy
import hashlib
import itertools
import json
import time
import numpy as np
from . import task39g_objective_horizon as g
f,e,d,c,b,q,old,p=g.f,g.e,g.d,g.c,g.b,g.q,g.old,g.p
CONFIG='experiments/task39h_preregistration_v1.json'
model_class=g.model_class
evaluation_groups=g.evaluation_groups
development=g.development
RATE_EDGES=(-6.,-5.,-4.,-3.,-2.,-1.,-.3)

def arms():
    out=[dict(name=f'H{i}',architecture='wide',horizon=64,learning_rate=lr,z_only=z,auxiliary=aux,expanded=False)
         for i,(lr,z,aux) in enumerate(itertools.product((.001,.0001),(False,True),(False,True)))]
    for i,aux in enumerate((False,True),8):
        out.append(dict(name=f'H{i}',architecture='wide',horizon=64,learning_rate=.0001,z_only=False,auxiliary=aux,expanded=True))
    return out

def contrast_pairs(definitions=None):
    aa=definitions or arms();out=[]
    for axis in ('learning_rate','z_only','auxiliary'):
        for a in aa[:8]:
            if a[axis] != ({'learning_rate':.001,'z_only':False,'auxiliary':False}[axis]):continue
            z=next(z for z in aa[:8] if z[axis]!=a[axis] and all(z[k]==a[k] for k in ('learning_rate','z_only','auxiliary') if k!=axis))
            out.append((axis,a['name'],z['name']))
    return out+[('recovery_support','H4','H8'),('recovery_support','H5','H9')]

def load_parents(torch,root,spec,device,dtype):
    folder=root/spec['parent'];freeze=json.loads((folder/'selection_freeze.json').read_text());sha=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==sha
    assert p.digest(folder/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']==spec['parent_sha256']
    weights=torch.load(folder/'frozen_checkpoint.pt',map_location=device,weights_only=True);out=[]
    for name in spec['oracle_parent_arms']:
        a=next(a for a in freeze['arms'] if a['name']==name);key=f'h{a["horizon"]}'
        same=[v for v in freeze['arms'] if v['horizon']==a['horizon']];index=[v['name'] for v in same].index(name)*3
        model=f.model_class(torch)('wide',spec['seeds']).to(device)
        model.load_state_dict({k:v.clone() if k=='gbar' else v[index:index+3].clone() for k,v in weights[key].items()})
        out.append(model.to(dtype=dtype))
    return out

def gradient_masks(torch,model,definitions,seeds):
    mask={k:torch.ones_like(v) for k,v in model.named_parameters()}
    for i,a in enumerate(definitions):
        if not a['z_only']:continue
        sl=slice(i*len(seeds),(i+1)*len(seeds))
        for k in mask:mask[k][sl].zero_()
        mask['c'][sl,:,17:]=1;mask['d'][sl,17:]=1
        mask['sc'][sl,:,1]=1;mask['sd'][sl,1]=1
    return mask

def groups(torch,root,spec,device):
    parent=load_parents(torch,root,spec,device,torch.float32)[0];out=[]
    for lr in (.001,.0001):
        aa=[a for a in arms() if a['learning_rate']==lr];n=len(aa)*3
        model=model_class(torch)('wide',spec['seeds']*len(aa)).to(device)
        model.load_state_dict({k:v.clone() if k=='gbar' else v.repeat(len(aa),*([1]*(v.ndim-1))) for k,v in parent.state_dict().items()})
        reference=f.model_class(torch)('wide',spec['seeds']*len(aa)).to(device);reference.load_state_dict(model.state_dict())
        out.append(dict(key='lr001' if lr==.001 else 'lr0001',learning_rate=lr,definitions=aa,model=model,
                        reference=reference,flags=torch.ones(n,4,device=device),masks=gradient_masks(torch,model,aa,spec['seeds'])))
    return out

def bin_loss(torch,raw_z,target_z,scales,temperature=.001):
    error=((torch.nn.functional.logsigmoid(raw_z)-target_z.clamp_min(1e-30).log())/scales).square()
    bins=torch.bucketize(target_z.clamp_min(1e-30).log10().contiguous(),torch.tensor(RATE_EDGES,device=raw_z.device))
    membership=torch.nn.functional.one_hot(bins,8).to(error.dtype)
    count=membership.sum(1);means=(membership*error[...,None]).sum(1)/count.clamp_min(1)
    valid=count>0
    flat=means.flatten(1);mask=valid.flatten(1)
    # Finite sentinels avoid undefined gradients from all-infinite masked rows.
    value=temperature*(torch.logsumexp(torch.where(mask,flat/temperature,torch.full_like(flat,-1e9)),1)-mask.sum(1).clamp_min(1).log())
    return value,dict(bin_counts=count.detach(),gate_rmse=error.mean(1).sqrt().detach())

def terms(torch,cohort,s,scales,short=64):
    main=e.terms(torch,cohort,s,scales,64)
    _,rz=cohort['model'].raw(s['anchor'][...,0],s['anchor'][...,1])
    auxiliary,_=bin_loss(torch,rz,s['anchor'][...,89:107],scales[3])
    return torch.cat((main,auxiliary[:,None]),-1)

def objective(torch,cohort,s,scales,short=64):return (terms(torch,cohort,s,scales)*s['weights']).sum(-1)

def gradient_probe(torch,cohort,s,scales,step):
    values=terms(torch,cohort,s,scales);grads=[]
    for i in range(5):
        grad=torch.autograd.grad(values[:,i].sum(),tuple(cohort['model'].parameters()),retain_graph=i<4)
        grads.append(torch.cat([(v*cohort['masks'][k]).flatten(1) for (k,_),v in zip(cohort['model'].named_parameters(),grad)],-1).detach().double())
    stack=torch.stack(grads,1);norm=stack.norm(dim=-1);cos=torch.bmm(stack,stack.transpose(1,2))/(norm[:,:,None]*norm[:,None,:]).clamp_min(1e-20)
    with torch.no_grad():
        u,z,_=cohort['model'].coefficients(s['anchor'][...,0],s['anchor'][...,1])
        ru,rz,_=cohort['reference'].coefficients(s['anchor'][...,0],s['anchor'][...,1])
        _,diag=bin_loss(torch,cohort['model'].raw(s['anchor'][...,0],s['anchor'][...,1])[1],s['anchor'][...,89:107],scales[3])
        rate_error=z.clamp_min(1e-30).log()-s['anchor'][...,89:107].clamp_min(1e-30).log()
    return dict(step=step,cohort=cohort['key'],terms=['variable','held','u','log_z','worst_bin'],
        losses=values.detach().cpu().tolist(),term_norms=norm.cpu().tolist(),cosines=cos.cpu().tolist(),weights=s['weights'].cpu().tolist(),
        log_z_signed_bias_by_gate=rate_error.mean(1).cpu().tolist(),log_z_rmse_by_gate=rate_error.square().mean(1).sqrt().cpu().tolist(),
        u_drift_by_gate=(u-ru).square().mean(1).sqrt().cpu().tolist(),z_drift_by_gate=(z-rz).square().mean(1).sqrt().cpu().tolist(),
        occupied_rate_bins=(diag['bin_counts']>0).sum((1,2)).cpu().tolist())

def masked_update(torch,cohort,s,scales,opt,counter=None):
    opt.zero_grad(set_to_none=False);objective(torch,cohort,s,scales).sum().backward()
    for k,v in cohort['model'].named_parameters():v.grad.mul_(cohort['masks'][k])
    norm=torch.stack([v.grad.flatten(1).square().sum(1) for v in cohort['model'].parameters()]).sum(0).sqrt()
    if counter is not None:counter.add_((norm>1).float())
    for v in cohort['model'].parameters():v.grad.div_(norm.clamp_min(1).reshape(-1,*([1]*(v.ndim-1))))
    opt.step()

def optimizer_preflight(torch,cohort,s,scales,seeds):
    model=cohort['model'];initial={k:v.detach().clone() for k,v in model.named_parameters()}
    with torch.no_grad():before_u=model.coefficients(s['anchor'][...,0],s['anchor'][...,1])[0].clone()
    opt=torch.optim.Adam(model.parameters(),lr=cohort['learning_rate']);masked_update(torch,cohort,s,scales,opt);error=0.
    for j in range(model.a.shape[0]):
        a=cohort['definitions'][j//3];iso=model_class(torch)('wide',[seeds[j%3]]).to(s['batch'].device)
        with torch.no_grad():
            for k,v in iso.named_parameters():v.copy_(initial[k][j:j+1])
        gg=dict(model=iso,definitions=[a],flags=cohort['flags'][j:j+1],masks=gradient_masks(torch,iso,[a],[seeds[j%3]]))
        ss={k:v[j:j+1] for k,v in s.items()};oo=torch.optim.Adam(iso.parameters(),lr=cohort['learning_rate'])
        masked_update(torch,gg,ss,scales,oo)
        error=max(error,max(float((v[0]-dict(model.named_parameters())[k][j]).abs().max()) for k,v in iso.named_parameters()))
    frozen=0.
    for k,v in model.named_parameters():
        frozen=max(frozen,float(((v-initial[k])*(1-cohort['masks'][k])).abs().max()))
    with torch.no_grad():
        after_u=model.coefficients(s['anchor'][...,0],s['anchor'][...,1])[0]
        selected=[i*3+j for i,a in enumerate(cohort['definitions']) if a['z_only'] for j in range(3)]
        u_error=float((after_u[selected]-before_u[selected]).abs().max()) if selected else 0.
    with torch.no_grad():
        for k,v in model.named_parameters():v.copy_(initial[k]);v.grad=None
    assert error<=1e-5 and frozen==0 and u_error==0,(error,frozen,u_error)
    return dict(valid=True,members=model.a.shape[0],max_abs_error=error,frozen_parameter_error=frozen,frozen_u_output_error=u_error)

def windows(root,role,families,count,length,seed):
    assert role in ('fit','fit_extra')
    data,info=q.windows(root,role,families,count,length,seed)
    return data,info

def recovery_fit(root,output,spec,quality):
    folder=output.parent/'recovery_fit';folder.mkdir(exist_ok=False)
    manifest=[];windows=[];anchors=[]
    for role in ('fit','fit_extra'):
        rng=np.random.default_rng(spec['expanded_seeds'][role]);nodes=spec['recovery_nodes']
        v=-100+160*(rng.permutation(nodes)+rng.random(nodes))/nodes;ca=10**(-7+5*(rng.permutation(nodes)+rng.random(nodes))/nodes)
        u,z=p.coefficients(v,ca,.1);chosen=[]
        for gate in range(18):
            ordered=np.argsort(z[:,gate],kind='stable')
            chosen.extend([int(ordered[round(frac*(nodes-1))]) for frac in (.05,.5)])
        selected=np.asarray(chosen);count=len(selected);ticks=spec['recovery_ticks']
        voltage=np.broadcast_to(v[selected],(ticks,count)).copy();calcium=np.broadcast_to(ca[selected],(ticks,count)).copy()
        initial=rng.uniform(0,1,(count,18));initial[::2]=rng.integers(0,2,(count//2,18))
        mult=rng.uniform(.25,4,(count,11));rev=p.ionic.REVERSALS+rng.uniform(-5,5,(count,11))
        gt,ct,source_ca=p.recurrent_reference(voltage,calcium,initial,mult,rev,.1)
        signatures=[]
        for j in range(count):
            hh=hashlib.sha256()
            for value in (voltage[:,j],calcium[:,j],initial[j],mult[j],rev[j]):hh.update(np.asarray(value,dtype='<f4').tobytes())
            signatures.append(hh.hexdigest())
        assert len(set(signatures))==count
        path=folder/f'{role}__recovery_grid.npz'
        np.savez_compressed(path,voltage=voltage.astype('float32'),bridge_ca=calcium.astype('float32'),initial_gates=initial.astype('float32'),
            multipliers=mult.astype('float32'),reversals=rev.astype('float32'),gate_targets=gt.astype('float32'),current_targets=ct.astype('float32'),source_ca=source_ca.astype('float32'))
        total=spec['windows_per_family']*len(quality['families']);cases=rng.integers(count,size=total)
        starts=np.floor(np.exp(rng.uniform(0,np.log(ticks-255),total))-1).astype(int)
        starts[np.arange(total)%4==0]=rng.integers(ticks-255,size=(np.arange(total)%4==0).sum())
        starts[np.arange(total)%4==1]=0
        ix=starts[:,None]+np.arange(256);ci=cases[:,None];pre=gt[np.maximum(ix-1,0),ci].copy()
        pre[starts==0,0]=initial[cases[starts==0]]
        data=np.concatenate((voltage[ix,ci,None],calcium[ix,ci,None],pre,gt[ix,ci],
            np.broadcast_to(mult[cases,None],(total,256,11)),np.broadcast_to(rev[cases,None],(total,256,11)),ct[ix,ci]),-1).astype('float32')
        windows.append(data);anchors.append(c.anchor_pool(data,spec['expanded_seeds'][role]+2))
        manifest.append(dict(role=role,seed=spec['expanded_seeds'][role],trajectories=count,ticks=ticks,selected_nodes=selected.tolist(),
            target_gate=np.repeat(np.arange(18),2).tolist(),nodes_voltage=v[selected].tolist(),nodes_calcium=ca[selected].tolist(),
            node_u=u[selected].tolist(),node_z=z[selected].tolist(),input_signatures=signatures,
            sha256=p.digest(path),file=path.name,starts=starts.tolist(),cases=cases.tolist(),
            window_sha256=hashlib.sha256(data.tobytes()).hexdigest()))
        print(f'[GIADA Task39h] recovery FIT {role} generated',flush=True)
    assert not set(manifest[0]['input_signatures'])&set(manifest[1]['input_signatures'])
    original_signatures=set()
    for previous in quality['roles']:
        for family in quality['families']:
            with np.load(root/spec['dataset']/f'{previous}__{family}.npz') as zz:
                for j in range(zz['voltage'].shape[1]):
                    hh=hashlib.sha256()
                    for key in ('voltage','bridge_ca','initial_gates','multipliers','reversals'):
                        value=zz[key][:,j] if key in ('voltage','bridge_ca') else zz[key][j]
                        hh.update(np.asarray(value,dtype='<f4').tobytes())
                    original_signatures.add(hh.hexdigest())
    assert not original_signatures&{s for item in manifest for s in item['input_signatures']}
    p.save_json(output/'recovery_fit_manifest.json',dict(recipe=spec['expanded_support'],files=manifest,roles=['fit','fit_extra'],selection_used=False))
    return windows,np.concatenate(anchors)

def residual_oracle_audit(torch,root,output,spec,quality,device):
    """Post-hoc localization on frozen failed cases; never controls training arms."""
    base=json.loads((output/'frozen_oracle_attribution.json').read_text())
    targets=[r for r in base if r['precision']=='float32' and r['intervention']=='learned' and not r['metrics']['passed']]
    models=load_parents(torch,root,spec,device,torch.float32);rows=[]
    for row in targets:
        parent=spec['oracle_parent_arms'].index(row['parent_arm']);seed=spec['seeds'].index(row['seed']);case=row['case']
        with np.load(root/spec['dataset']/f'development__{row["family"]}.npz') as zz:a={k:zz[k] for k in zz.files}
        voltage=a['voltage'][:,case];ca=a['bridge_ca'][:,case];target=a['gate_targets'][:,case];T=len(voltage)
        with torch.no_grad():
            u=[];z=[]
            for start in range(0,T,2048):
                v=torch.tensor(a['voltage'][start:start+2048].reshape(-1),device=device)[None].expand(3,-1)
                cc=torch.tensor(a['bridge_ca'][start:start+2048].reshape(-1),device=device)[None].expand(3,-1)
                uu,zz,_=models[parent].coefficients(v,cc)
                u.append(uu.reshape(3,-1,a['voltage'].shape[1],18)[seed,:,case].cpu().numpy())
                z.append(zz.reshape(3,-1,a['voltage'].shape[1],18)[seed,:,case].cpu().numpy())
        u=np.concatenate(u);z=np.concatenate(z);tu,tz=p.coefficients(voltage.astype(float),ca.astype(float),.1);tu=tu.astype('float32');tz=tz.astype('float32')
        failed=[j for j,m in enumerate(row['metrics']['channels']) if not m['passed']]
        gates=sorted({gate for channel in failed for gate in range(p.OFFSETS[channel],p.OFFSETS[channel+1])})
        modes=[('learned',None,None),('reset8',None,None),('reset64',None,None)]
        for gate in gates:
            for kind in ('u','z'):modes.append((f'{kind}_gate{gate}',kind,gate))
        for kind in ('u','z'):
            for phase in ('rise','plateau','recovery'):modes.append((f'{kind}_{phase}',kind,phase))
        peak=int(np.argmax(np.abs(a['current_targets'][:,case,failed[0]]-a['current_targets'][0,case,failed[0]])))
        excursion=np.abs(a['current_targets'][:,case,failed[0]]-a['current_targets'][0,case,failed[0]])
        amplitude=float(excursion.max());plateau=excursion>=.99*amplitude
        masks={'plateau':plateau,'rise':(np.arange(T)<peak)&~plateau,'recovery':(np.arange(T)>peak)&~plateau}
        state=np.broadcast_to(a['initial_gates'][case],(len(modes),18)).astype('float32').copy();prediction=np.empty((T,len(modes),18),dtype='float32')
        channel_mask=np.zeros(18,dtype=bool);channel_mask[gates]=True
        for tick in range(T):
            uu=np.broadcast_to(u[tick],state.shape).copy();zz=np.broadcast_to(z[tick],state.shape).copy()
            for j,(name,kind,where) in enumerate(modes):
                if name.startswith('reset') and tick and tick%int(name[5:])==0:state[j]=target[tick-1]
                if kind is None:continue
                select=np.arange(18)==where if isinstance(where,int) else channel_mask&masks[where][tick]
                if kind=='u':uu[j,select]=tu[tick,select]
                else:zz[j,select]=tz[tick,select]
            state=state+zz*(uu-state);prediction[tick]=state
        for j,(name,_,_) in enumerate(modes):
            _,current=p.readout(voltage,prediction[:,j],a['multipliers'][case],a['reversals'][case])
            metric=p.evaluate_currents(a['current_targets'][:,case],current,.1,quality['quality'],target,prediction[:,j],a['multipliers'][case]==0)
            if name=='learned':assert metric==row['metrics'],'Frozen residual oracle reproduction mismatch'
            rows.append(dict(parent=row['parent_arm'],seed=row['seed'],family=row['family'],case=case,intervention=name,
                corrected_gates=gates,metrics=metric,selection_eligible=False,phase_peak_index=peak,
                event=p.event_metrics(a['current_targets'][:,case,failed[0]],current[:,failed[0]],.1,1e-7)))
        del prediction
        p.save_json(output/'residual_oracle_attribution.json',rows)
        print(f'[GIADA Task39h] residual oracle {row["parent_arm"]}/{row["family"]}/{row["seed"]}/{case}',flush=True)
    return rows

def synthetic_preflight(torch,device):
    target=torch.full((2,64,18),.01,device=device);raw=torch.logit(target).requires_grad_()
    loss,_=bin_loss(torch,raw,target,torch.ones(18,device=device));loss.sum().backward()
    assert bool(torch.isfinite(raw.grad).all()) and float(loss.abs().max())<1e-7
    damaged=raw.detach().clone();damaged[:,0,0]+=.2
    loss2,_=bin_loss(torch,damaged,target,torch.ones(18,device=device))
    assert bool((loss2>loss.detach()).all())
    return dict(valid=True,exact_loss=loss.detach().cpu().tolist(),localized_error_loss=loss2.cpu().tolist(),empty_bins_finite=True)

def capture_training(torch,groups,static,scales,spec,objective_fn=None):
    loss_fn=objective_fn or objective
    originals=[{k:x.detach().clone() for k,x in g['model'].named_parameters()} for g in groups]
    opts=[torch.optim.Adam(g['model'].parameters(),lr=g['learning_rate'],capturable=True) for g in groups]
    clips=[torch.zeros(g['model'].a.shape[0],device='cuda') for g in groups]
    def update():
        for g,s,opt,counter in zip(groups,static,opts,clips):
            masked_update(torch,g,s,scales,opt,counter)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):update()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):update()
    def reset():
        with torch.no_grad():
            for g,init,opt in zip(groups,originals,opts):
                for k,x in g['model'].named_parameters():x.copy_(init[k]);x.grad.zero_()
                for st in opt.state.values():
                    for x in st.values():
                        if torch.is_tensor(x):x.zero_()
            for x in clips:x.zero_()
    reset();update();update();expected=[{k:x.detach().clone() for k,x in g['model'].named_parameters()} for g in groups]
    reset();graph.replay();graph.replay();torch.cuda.synchronize()
    err=max(float((x-e[k]).abs().max()) for g,e in zip(groups,expected) for k,x in g['model'].named_parameters())
    assert all(bool(torch.isfinite(x).all()) for g in groups for x in g['model'].parameters())
    assert err<=spec['numeric_atol'],err
    reset();start=time.perf_counter()
    for _ in range(20):graph.replay()
    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/20;reset()
    assert seconds*spec['checkpoints'][-1]<spec['max_projected_training_seconds'],'Fit-only performance preflight exceeded registered runtime cap'
    # Adam state is allocated during eager warmup, OUTSIDE the private graph
    # pool. CUDA Graph records addresses, not ownership of those tensors.
    # Keep optimizers alive for exactly the lifetime of the replayable graph.
    graph._giada_optimizers=opts
    return graph,clips,dict(valid=True,two_update_max_error=err,seconds_per_update=seconds,
        projected_training_seconds=seconds*spec['checkpoints'][-1],optimizer_reset=True,optimizer_state_lifetime_retained=True,backend='two member-vectorized cohorts in one CUDA Graph')


def run(root,output,device='cuda'):
    import torch
    assert device=='cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());quality=json.loads((root/p.CONFIG).read_text());assert p.digest(root/p.CONFIG)==spec['quality_config_sha256']
    output.mkdir(parents=True,exist_ok=True);assert not (output/'final_report.json').exists();p.save_json(output/'preregistration.json',spec)
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()):assert p.digest(root/spec['dataset']/row['file'])==row['sha256']
    p.save_json(output/'synthetic_loss_preflight.json',synthetic_preflight(torch,device))
    d.oracle_audit(torch,root,output,spec,quality,device,parent_loader=load_parents,timing_fn=f.timing_audit)
    residual_oracle_audit(torch,root,output,spec,quality,device)
    extra_windows,extra_anchors=recovery_fit(root,output,spec,quality);families=list(quality['families']);data=[];support=[]
    for source in (root/spec['dataset'],):
        aa=[]
        for role,extra in [('fit',0),('fit_extra',1009)]:
            a,info=windows(source,role,families,spec['windows_per_family'],256,spec['data_seed']+extra);aa.append(a);support+=info
        data.append(aa)
    data.append(extra_windows)
    anchors=c.anchor_pool(np.concatenate([a[...,:71] for a in data[0]]),spec['data_seed']+2,True)
    original_calib=json.loads((root/spec['parent']/'fit_loss_calibration.json').read_text());calib=np.asarray(original_calib['parent_weights'],dtype='float32')
    # Exact39f normalization and frozen FIT calibration, shared across every arm.
    normal=json.loads((root/spec['normalizers']).read_text());us=np.asarray(original_calib['u_scales']);zs=np.asarray(original_calib['log_z_scales'])
    scales=[torch.tensor(v,device=device,dtype=torch.float32) for v in (normal['gate'],normal['current'],us,zs)]
    dev,held=development(root,spec,quality)
    pools=[[torch.tensor(a,device=device) for a in aa] for aa in data];at=torch.tensor(anchors,device=device);new_at=torch.tensor(extra_anchors,device=device);cohorts=groups(torch,root,spec,device);defs=[a for g in cohorts for a in g['definitions']];count=30;static=[]
    map_error=0.
    with torch.no_grad():
        deployed=evaluation_groups(torch,cohorts,spec['seeds'],device)[0]['model'];n=count
        v=torch.linspace(-95,55,7,device=device)[None].expand(n,-1);ca=torch.logspace(-7,-2,7,device=device)[None].expand(n,-1);state=torch.full((n,7,18),.4,device=device)
        trained=torch.cat([group['model'](v[:group['model'].a.shape[0]],ca[:group['model'].a.shape[0]],state[:group['model'].a.shape[0]]) for group in cohorts])
        map_error=float((trained-deployed(v,ca,state)).abs().max())
    assert map_error<=spec['numeric_atol'],map_error
    p.save_json(output/'training_deployment_preflight.json',dict(valid=True,one_step_max_error=map_error,atol=spec['numeric_atol'],training_affine_accumulation='float64',inference='standard float32;used for selection and full evaluation'))
    for group in cohorts:
        n=group['model'].a.shape[0]
        weights=[[.5,.5,float(calib[j,0]),float(calib[j,1]),0.] for arm in group['definitions'] for j in range(3)]
        static.append(dict(batch=pools[0][0][:2].reshape(8,64,71)[None].expand(n,-1,-1,-1).clone(),
            anchor=at[:64][None].expand(n,-1,-1).clone(),ticks=torch.full((n,64,1),1000.,device=device),weights=torch.tensor(weights,device=device)))
    probe=gradient_probe(torch,cohorts[0],static[0],scales,0);norm=np.asarray(probe['term_norms'])[:3];cos=np.asarray(probe['cosines'])[:3]
    main=np.asarray(static[0]['weights'].cpu())[:3,:4]*norm[:,:4]
    combined=np.sqrt(np.maximum(np.einsum('bi,bij,bj->b',main,cos[:,:4,:4],main),1e-30))
    auxiliary_weights=np.clip(combined/np.maximum(norm[:,4],1e-12),.001,1e6)
    for group,s in zip(cohorts,static):
        for i,arm in enumerate(group['definitions']):
            if arm['auxiliary']:s['weights'][i*3:(i+1)*3,4]=torch.tensor(auxiliary_weights,device=device,dtype=torch.float32)
    p.save_json(output/'window_support.json',dict(windows=support,anchor_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),expanded_fraction=.5,target_ticks_per_update=512))
    p.save_json(output/'fit_loss_calibration.json',dict(parent_weights=calib.tolist(),weights={group['key']:s['weights'].cpu().tolist() for group,s in zip(cohorts,static)},auxiliary_weights=auxiliary_weights.tolist(),calibration_probe=probe,source='Initial original FIT all-parameter gradients only;identical per-seed auxiliary weight across treatments',u_scales=us.tolist(),log_z_scales=zs.tolist()))
    p.save_json(output/'composition_preflight.json',c.composition_preflight(torch,cohorts,device))
    p.save_json(output/'optimizer_preflight.json',[optimizer_preflight(torch,g,s,scales,spec['seeds']) for g,s in zip(cohorts,static)])
    old.recurrence_preflight(output,spec,f.split_models(torch,cohorts,spec['seeds'],device),torch,device)
    graph,clips,execution=capture_training(torch,cohorts,static,scales,spec,objective_fn=objective)
    execution.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,members=count,backend='two learning-rate cohorts;masked independent Adam in one CUDA graph',parameters_per_member=2404,training_affine_accumulation='float64;outputs/parameters/states/Adam float32',evaluation_inference_dtype='float32',thresholds_changed=False)
    p.save_json(output/'execution.json',execution);print(f'[GIADA Task39h] preflights passed; projected fit {execution["projected_training_seconds"]/60:.1f}min',flush=True)
    fixed=[{k:v.clone() for k,v in s.items()} for s in static];fixed_probes=[]
    rng=np.random.default_rng(spec['data_seed']+299);best=[None]*count;saved=[None]*count;chosen=[None]*count;common_best=np.full(count,np.inf);common_chosen=[None]*count;curves=[];probes=[];invalid=set();start=time.perf_counter();exposure=np.zeros((2,len(families),2),dtype=np.int64)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            tmp=output/f'checkpoint_{step}.tmp';torch.save({g['key']:g['model'].state_dict() for g in cohorts},tmp);tmp.replace(output/f'checkpoint_{step}.pt')
            tmp=output/f'optimizer_{step}.tmp';torch.save(dict(optimizers=[o.state_dict() for o in graph._giada_optimizers],numpy_rng=rng.bit_generator.state,step=step,clipping_counts=[x.cpu() for x in clips]),tmp);tmp.replace(output/f'optimizer_{step}.pt')
            p.save_json(output/'checkpoint_phase.json',dict(step=step,phase='before_selection',parameters_finite={g['key']:all(bool(torch.isfinite(v).all()) for v in g['model'].parameters()) for g in cohorts}))
            print(f'[GIADA Task39h] checkpoint_saved {step} before_selection',flush=True)
            common,parts,ranks,panel=e.checkpoint_scores(torch,evaluation_groups(torch,cohorts,spec['seeds'],device),dev,held,normal,spec,device,quality);j=0
            for g in cohorts:
                for k in range(g['model'].a.shape[0]):
                    if ranks[j] is None:invalid.add(j);j+=1;continue
                    if best[j] is None or ranks[j]<best[j]:
                        best[j]=ranks[j];chosen[j]=step;saved[j]={name:v[k].detach().cpu().clone() for name,v in g['model'].named_parameters()}
                    if common[j]<common_best[j]:common_best[j]=common[j];common_chosen[j]=step
                    j+=1
            assert all(v is not None for v in saved),'Initial finite selection preflight failed'
            curves.append(dict(step=step,common_development_score=common.tolist(),panel=panel,elapsed_seconds=time.perf_counter()-start,clipping_counts=[v.cpu().tolist() for v in clips]));p.save_json(output/'training_curves.json',e.diagnostic_json(curves))
            p.save_json(output/'numerical_health.json',dict(invalid_members=sorted(invalid),current_invalid=[j for j,r in enumerate(ranks) if r is None],invalid_checkpoints_are_never_selected=True))
            for g,s in zip(cohorts,static):probes.append(gradient_probe(torch,g,s,scales,step))
            for g,s in zip(cohorts,fixed):fixed_probes.append(gradient_probe(torch,g,s,scales,step))
            p.save_json(output/'fixed_fit_gradient_probes.json',e.diagnostic_json(fixed_probes))
            p.save_json(output/'gradient_probes.json',e.diagnostic_json(probes));print(f'[GIADA Task39h] checkpoint {step} median_common={np.median(common):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
        if step==spec['checkpoints'][-1]:break
        ix=rng.integers(len(data[0][0]),size=2);extra=rng.random(2)<.5;use_new=rng.random(2)<.5;ti=torch.tensor(ix,device=device);ee=torch.tensor(extra,device=device)[:,None,None]
        original=torch.where(ee,pools[0][1][ti],pools[0][0][ti]);new=torch.where(ee,pools[1][1][ti],pools[1][0][ti]);expanded_batch=torch.where(torch.tensor(use_new,device=device)[:,None,None],new,original)
        ai=rng.integers(len(anchors),size=64);di=rng.integers(6,size=64);tt=torch.tensor(np.asarray(spec['held_ticks'])[di],device=device,dtype=torch.float32)[None,:,None]
        for g,s in zip(cohorts,static):
            h=g['definitions'][0]['horizon'];mask=torch.tensor([a['expanded'] for a in g['definitions'] for _ in spec['seeds']],device=device)[:,None,None,None]
            s['batch'].copy_(torch.where(mask,expanded_batch.reshape(512//h,h,71)[None],original.reshape(512//h,h,71)[None]));s['anchor'].copy_(torch.where(mask[:,0],torch.where(torch.tensor(rng.random(64)<.5,device=device)[None,:,None],new_at[ai % len(extra_anchors)][None],at[ai][None]),at[ai][None]));s['ticks'].copy_(tt)
        graph.replay()
        np.add.at(exposure[0],(ix//spec['windows_per_family'],extra.astype(int)),256)
        np.add.at(exposure[1],(ix[use_new]//spec['windows_per_family'],extra[use_new].astype(int)),256)
        if (step+1)%2000==0:print(f'[GIADA Task39h] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-start;j=0
    with torch.no_grad():
        for g in cohorts:
            for k in range(g['model'].a.shape[0]):
                for name,v in g['model'].named_parameters():v[k].copy_(saved[j][name])
                j+=1
    torch.save({g['key']:g['model'].state_dict() for g in cohorts},output/'frozen_checkpoint.pt')
    freeze=dict(arms=defs,seeds=spec['seeds'],chosen_steps=chosen,ranks=best,common_rule_steps=common_chosen,development_only=True,stress_used_for_selection=False,independent_confirmation_accessed=False,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'))
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    p.save_json(output/'exposure.json',dict(family_role_ticks=exposure.tolist(),new_data_ticks_for_expanded_arms=int(exposure[1].sum()),variable_targets_per_member=512*spec['checkpoints'][-1],anchor_targets_per_member=64*spec['checkpoints'][-1]))
    base=json.loads((root/old.CONFIG).read_text());evaluation=dict(spec,architectures=[a['name'] for a in defs]+['frozen39g_'+n for n in spec['oracle_parent_arms']])
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=base[key]
    assert p.digest(root/base['baseline_parent'])==base['baseline_parent_sha256'] and p.digest(root/base['baseline_checkpoint'])==base['baseline_sha256']
    summary=old.evaluate(root,output,evaluation,f.split_models(torch,cohorts,spec['seeds'],device)+load_parents(torch,root,spec,device,torch.float32),torch,device,quality)
    rows=json.loads((output/'recurrence_evaluation.json').read_text());scores={}
    for a in defs:
        scores[a['name']]={}
        for seed in spec['seeds']:
            rr=[r for r in rows if r['model']=='joint_w'+a['name'] and r['role']=='development' and r['seed']==seed];vals=[q.severity(r['metrics']) for r in rr]
            scores[a['name']][str(seed)]=dict(median=float(np.median(vals)),worst=max(vals),passed=sum(r['metrics']['passed'] for r in rr),timing_failed=sum(any(cc['event']['status']=='missing_event' or (cc['event']['status']=='evaluable' and (cc['event']['peak_error_ms']>.2+1e-9 or max(cc['event']['onset_error_ms'],cc['event']['offset_error_ms'])>.3+1e-9)) for cc in r['metrics']['channels']) for r in rr))
    contrasts=[]
    for axis,a,z in contrast_pairs():
        ratios=[scores[z][str(s)]['median']/max(scores[a][str(s)]['median'],1e-12) for s in spec['seeds']];worst=[scores[z][str(s)]['worst']/max(scores[a][str(s)]['worst'],1e-12) for s in spec['seeds']]
        contrasts.append(dict(axis=axis,control=a,treatment=z,paired_ratios=ratios,worst_ratios=worst,supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=contrasts))
    c.cost_benchmark(torch,cohorts,spec['seeds'],device,output,model_factory=f.model_class,splitter=f.split_models,task_label='39h')
    common_dir=output/'common_selection_diagnostic';common_dir.mkdir();checkpoints={step:torch.load(output/f'checkpoint_{step}.pt',map_location=device,weights_only=True) for step in set(common_chosen)};j=0
    with torch.no_grad():
        for g in cohorts:
            for k in range(g['model'].a.shape[0]):
                for name,v in g['model'].named_parameters():v[k].copy_(checkpoints[common_chosen[j]][g['key']][name][k])
                j+=1
    cs=old.evaluate(root,common_dir,evaluation,f.split_models(torch,cohorts,spec['seeds'],device)+load_parents(torch,root,spec,device,torch.float32),torch,device,quality)
    p.save_json(output/'common_selection_diagnostic.json',dict(summaries=cs,selected_steps=common_chosen,selection_eligible=False))
    unstable={defs[j//3]['name'] for j in invalid};passing=[a['name'] for a in defs if summary['joint_w'+a['name']]['all_development_passed'] and a['name'] not in unstable]
    execution.update(training_wall_seconds=training,total_wall_seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated());p.save_json(output/'execution.json',execution)
    final=dict(valid=True,task='39h',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',passing_arms=passing,numerically_unstable_arms=sorted(unstable),summaries=summary,thresholds_changed=False,independent_confirmation_accessed=False,speedup_claimed=False,limits=spec['limits']);p.save_json(output/'final_report.json',final);return final
