"""Task39g: warm paired loss anatomy, credit horizon and independent FIT support."""
import copy
import hashlib
import json
import time
import numpy as np
from . import task39f_temporal_recovery as f

e,d,c,b,q,old,p=f.e,f.d,f.c,f.b,f.q,f.old,f.p
CONFIG='experiments/task39g_preregistration_v1.json'

def model_class(torch):
    """Same FP32 parameters/map; high-precision affine accumulation in training."""
    class Training(f.model_class(torch)):
        def raw(self,v,ca):
            vv=((v+40)/80).double();cc=((ca.clamp_min(1e-9).log10()+4.5)/2.5).double()
            raw=torch.bmm(torch.nn.functional.silu(torch.bmm(vv[...,None],self.a.double())+self.b.double()),self.c.double())+self.d.double()[:,None]
            sk=torch.bmm(torch.nn.functional.silu(torch.bmm(cc[...,None],self.sa.double())+self.sb.double()),self.sc.double())+self.sd.double()[:,None]
            raw=raw.to(v.dtype);sk=sk.to(v.dtype)
            return torch.cat((raw[...,:16],sk[...,0:1],raw[...,16:17]),-1),torch.cat((raw[...,17:33],sk[...,1:2],raw[...,33:34]),-1)
    return Training

def evaluation_groups(torch,cohorts,seeds,device):
    definitions=[a for group in cohorts for a in group['definitions']]
    m=f.model_class(torch)('wide',seeds*len(definitions)).to(device)
    states=[group['model'].state_dict() for group in cohorts]
    m.load_state_dict({k:states[0][k].clone() if k=='gbar' else torch.cat([s[k] for s in states],0) for k in states[0]})
    return [dict(key='common_fp32_evaluation',model=m,definitions=definitions,flags=torch.cat([group['flags'] for group in cohorts]))]

def arms():
    return [dict(name=f'A{i}',architecture='wide',horizon=64 if i<6 else 256,
        loss=['none','full','full','precision','precision','aligned','none','aligned','none','aligned'][i],
        factor=[0,1,.01,1,.01,.01,0,.01,0,.01][i],expanded=i>=8) for i in range(10)]

def contrast_pairs(definitions=None):
    return [('weight','A1','A2'),('weight','A3','A4'),('derivative','A1','A3'),('derivative','A2','A4'),
        ('alignment_package','A4','A5'),('horizon','A0','A6'),('horizon','A5','A7'),
        ('independent_support','A6','A8'),('independent_support','A7','A9'),
        ('aligned_auxiliary','A0','A5'),('aligned_auxiliary','A6','A7')]

def load_parents(torch,root,spec,device,dtype):
    folder=root/spec['parent'];freeze=json.loads((folder/'selection_freeze.json').read_text())
    claimed=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==claimed
    assert p.digest(folder/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']==spec['parent_sha256']
    weights=torch.load(folder/'frozen_checkpoint.pt',map_location=device,weights_only=True);out=[]
    for name in spec['oracle_parent_arms']:
        definition=next(a for a in freeze['arms'] if a['name']==name);arch=definition['architecture']
        same=[a for a in freeze['arms'] if a['architecture']==arch];index=next(i for i,a in enumerate(same) if a['name']==name)*3
        model=f.model_class(torch)(arch,spec['seeds']).to(device)
        model.load_state_dict({k:v.clone() if k=='gbar' else v[index:index+3].clone() for k,v in weights[arch].items()})
        out.append(model.to(dtype=dtype))
    return out

def groups(torch,root,spec,device):
    parent=load_parents(torch,root,spec,device,torch.float32)[0];assert parent.architecture=='wide'
    result=[]
    for horizon in (64,256):
        aa=[a for a in arms() if a['horizon']==horizon];n=len(aa)*len(spec['seeds'])
        model=model_class(torch)('wide',spec['seeds']*len(aa)).to(device)
        model.load_state_dict({k:v.clone() if k=='gbar' else v.repeat(len(aa),*([1]*(v.ndim-1))) for k,v in parent.state_dict().items()})
        result.append(dict(key=f'h{horizon}',model=model,definitions=aa,flags=torch.ones(n,4,device=device)))
    return result

def temporal_components(torch,pred,ref):
    scale=ref.abs().amax(2,keepdim=True).clamp_min(1e-7)
    residual=(pred-ref)/scale
    slope=torch.diff(ref,dim=2,prepend=ref[:,:,:1]).abs()/scale/.1
    error=residual/slope.clamp_min(.001)/.3
    bounded=5*torch.tanh(error/5)
    excursion=(ref-ref[:,:,:1]).abs()/scale;amplitude=excursion.amax(2,keepdim=True)
    band=torch.exp(-((excursion-.1*amplitude)/(.05*amplitude).clamp_min(.001)).square())
    precision=((.5+.5*band)*bounded.square()).mean((1,2,3))
    derivative=(torch.diff(pred,dim=2)-torch.diff(ref,dim=2))/scale/.1
    return precision,.25*derivative.square().mean((1,2,3))

def event_metadata(current,floor=1e-7):
    """Full FIT episode labels only. Columns baseline,scale,amplitude,eligible,slope."""
    base=current[:1];scale=np.abs(current).max(0,keepdims=True).clip(floor)
    excursion=np.abs(current-base);amplitude=excursion.max(0,keepdims=True);peak=excursion.argmax(0,keepdims=True)
    eligible=(amplitude>=.25*scale)&(amplitude>=10*floor)&(peak>0)&(peak<len(current)-1)
    slope=np.abs(np.diff(excursion,axis=0,prepend=excursion[:1]))/scale/.1
    return np.concatenate([np.broadcast_to(a,current.shape) for a in (base,scale,amplitude,eligible)]+[slope],-1)

def aligned_loss(torch,pred,ref,metadata):
    base,scale,amplitude,eligible,slope=metadata.split(11,-1)
    scale=scale.clamp_min(1e-7);amplitude=amplitude/scale
    excursion=(ref-base).abs()/scale;residual=(pred-ref)/scale
    crossing=torch.exp(-((excursion-.1*amplitude)/(.05*amplitude).clamp_min(.001)).square())*eligible
    error=residual/slope.clamp_min(.001)/.3
    precision=crossing*(5*torch.tanh(error/5)).square()
    peak=torch.exp(-((excursion-amplitude)/(.1*amplitude).clamp_min(.001)).square())*eligible
    shape=peak*(5*torch.tanh(residual/.05/5)).square()
    # No off-event inverse-slope term; amplitude remains protected by main losses.
    return (precision+.25*shape).mean((1,2,3))

def terms(torch,g,s,scales,short=64):
    main,current,ref=e.terms(torch,g,s,scales,s['batch'].shape[2],return_trajectory=True)
    precision,derivative=temporal_components(torch,current,ref)
    aligned=aligned_loss(torch,current,ref,s['batch'][...,71:126])
    return torch.cat((main,precision[:,None],derivative[:,None],aligned[:,None]),-1)

def objective(torch,g,s,scales,short=64):return (terms(torch,g,s,scales,short)*s['weights']).sum(-1)

def gradient_probe(torch,g,s,scales,step):
    values=terms(torch,g,s,scales);grads=[]
    for i in range(7):
        grad=torch.autograd.grad(values[:,i].sum(),tuple(g['model'].parameters()),retain_graph=i<6)
        grads.append(torch.cat([v.flatten(1) for v in grad],-1).detach().double())
    stack=torch.stack(grads,1);norm=stack.norm(dim=-1)
    cos=torch.bmm(stack,stack.transpose(1,2))/(norm[:,:,None]*norm[:,None,:]).clamp_min(1e-20)
    with torch.no_grad():
        _,current,ref=e.terms(torch,g,s,scales,s['batch'].shape[2],return_trajectory=True)
        scale=ref.abs().amax(2,keepdim=True).clamp_min(1e-7)
        slope=torch.diff(ref,dim=2,prepend=ref[:,:,:1]).abs()/scale/.1
        error=((current-ref)/scale)/slope.clamp_min(.001)/.3
        saturated=(error.abs()>15).float().mean((1,2,3)).cpu().tolist()
        floored=(slope<.001).float().mean((1,2,3)).cpu().tolist()
        eligible=s['batch'][...,104:115].mean((1,2,3)).cpu().tolist()
    return dict(step=step,cohort=g['key'],losses=values.detach().cpu().tolist(),term_norms=norm.cpu().tolist(),cosines=cos.cpu().tolist(),weights=s['weights'].cpu().tolist(),terms=['variable','held','u','log_z','old_precision','old_derivative','episode_aligned'],old_precision_saturation_fraction=saturated,old_slope_floor_fraction=floored,full_episode_event_eligible_fraction=eligible)

def optimizer_preflight(torch,g,s,scales,seeds):
    model=g['model'];initial={k:v.detach().clone() for k,v in model.named_parameters()}
    opt=torch.optim.Adam(model.parameters(),lr=.001);objective(torch,g,s,scales).sum().backward();old.clip(torch,model);opt.step();error=0.
    for j in range(model.a.shape[0]):
        iso=model_class(torch)('wide',[seeds[j%len(seeds)]]).to(s['batch'].device)
        with torch.no_grad():
            for k,v in iso.named_parameters():v.copy_(initial[k][j:j+1])
        gg=dict(g,model=iso,flags=g['flags'][j:j+1]);ss={k:v[j:j+1] for k,v in s.items()}
        oo=torch.optim.Adam(iso.parameters(),lr=.001);objective(torch,gg,ss,scales).sum().backward();old.clip(torch,iso);oo.step()
        error=max(error,max(float((v[0]-dict(model.named_parameters())[k][j]).abs().max()) for k,v in iso.named_parameters()))
    with torch.no_grad():
        for k,v in model.named_parameters():v.copy_(initial[k]);v.grad=None
    assert error<=1e-5,error
    return dict(valid=True,members=model.a.shape[0],max_abs_error=error)

def windows(root,role,families,count,length,seed):
    """Same assignments and71 inputs as39f control, with55 offline loss labels."""
    assert role in ('fit','fit_extra')
    rng=np.random.default_rng(seed);out=[];report=[]
    for family in families:
        with np.load(root/f'{role}__{family}.npz') as z:a={k:z[k] for k in z.files}
        ticks,cases=a['voltage'].shape
        events=(np.abs(np.diff(a['voltage'],axis=0,prepend=a['voltage'][:1]))>1)|(np.abs(np.diff(np.log10(a['bridge_ca'].clip(1e-9)),axis=0,prepend=np.log10(a['bridge_ca'][:1].clip(1e-9))))>.05)
        events[:25]=True;events[-25:]=True;candidates=np.argwhere(events)
        starts=rng.integers(0,ticks-length+1,count);case=rng.integers(0,cases,count)
        selected=candidates[rng.integers(0,len(candidates),count//2)]
        starts[:count//2]=(selected[:,0]-rng.integers(0,length,count//2)).clip(0,ticks-length);case[:count//2]=selected[:,1]
        ix=starts[:,None]+np.arange(length);ci=np.broadcast_to(case[:,None],ix.shape)
        pre=a['gate_targets'][np.maximum(ix-1,0),ci].copy()
        for j in np.flatnonzero(starts==0):pre[j,0]=a['initial_gates'][case[j]]
        meta=event_metadata(a['current_targets'])
        data=np.concatenate((a['voltage'][ix,ci,None],a['bridge_ca'][ix,ci,None],pre,a['gate_targets'][ix,ci],
            np.broadcast_to(a['multipliers'][case,None],(count,length,11)),np.broadcast_to(a['reversals'][case,None],(count,length,11)),a['current_targets'][ix,ci],meta[ix,ci]),-1)
        out.append(data);report.append(dict(role=role,family=family,windows=count,independent_trajectories=cases,starts=starts.tolist(),cases=case.tolist(),assignment_sha256=hashlib.sha256(np.stack((starts,case)).tobytes()).hexdigest(),metadata_source='whole FIT episode;not model input'))
    return np.concatenate(out).astype('float32'),report

def expanded_fit(root,output,spec,quality):
    """Independent seeded teacher data; only FIT roles, no development mining."""
    folder=output.parent/'expanded_fit';folder.mkdir(exist_ok=False)
    cfg=copy.deepcopy(quality);manifest=[];all_signatures=set()
    def stored_signature(data,j):
        h=hashlib.sha256()
        for key in ('voltage','bridge_ca','initial_gates','multipliers','reversals'):
            value=data[key][:,j] if key in ('voltage','bridge_ca') else data[key][j]
            h.update(np.asarray(value,dtype='<f4').tobytes())
        return h.hexdigest()
    cfg['roles']={role:dict(seed=spec['expanded_seeds'][role],trajectories_per_family=12) for role in ('fit','fit_extra')}
    for role in cfg['roles']:
        for family in cfg['families']:
            data,meta=p.make_paths(cfg,role,family)
            gates,current,source=p.recurrent_reference(data['voltage'],data['bridge_ca'],data['initial_gates'],data['multipliers'],data['reversals'],.1)
            for j,row in enumerate(meta):row['stored_float32_input_sha256']=stored_signature(data,j)
            signatures={x['stored_float32_input_sha256'] for x in meta};assert len(signatures)==len(meta) and not signatures&all_signatures;all_signatures.update(signatures)
            # Compare actual original inputs in the same float32 representation.
            for previous in quality['roles']:
                with np.load(root/spec['dataset']/f'{previous}__{family}.npz') as z:original={k:z[k] for k in ('voltage','bridge_ca','initial_gates','multipliers','reversals')}
                assert not signatures&{stored_signature(original,j) for j in range(original['voltage'].shape[1])}
            path=folder/f'{role}__{family}.npz'
            np.savez_compressed(path,**{k:v.astype('float32') for k,v in data.items()},gate_targets=gates.astype('float32'),current_targets=current.astype('float32'),source_ca=source.astype('float32'))
            manifest.append(dict(file=path.name,sha256=p.digest(path),bytes=path.stat().st_size,trajectories=meta))
            print(f'[GIADA Task39g] independent FIT {role}/{family}',flush=True)
    p.save_json(output/'expanded_fit_manifest.json',dict(config=cfg,files=manifest,raw_location=str(folder),raw_in_checkpoint_bundles=False,regeneration='make_paths+recurrent_reference from pinned code;all inputs/seeds/hashes retained;generated NPZ timestamps are not semantic identity'))
    return folder

def development(root,spec,quality):
    blocks=[]
    for k,family in enumerate(quality['families']):
        a,_=q.windows(root/spec['dataset'],'development',[family],2,512,spec['data_seed']+4000+k);blocks.append(a)
    for family in ('same_v_history','slow_recovery'):
        with np.load(root/spec['dataset']/f'development__{family}.npz') as z:a={k:z[k] for k in z.files}
        ix=np.arange(len(a['voltage'])-spec['development_long_ticks'],len(a['voltage']));pre=a['gate_targets'][np.maximum(ix-1,0)]
        block=np.concatenate((a['voltage'][ix,...,None],a['bridge_ca'][ix,...,None],pre,a['gate_targets'][ix],np.broadcast_to(a['multipliers'],(len(ix),2,11)),np.broadcast_to(a['reversals'],(len(ix),2,11)),a['current_targets'][ix]),-1)
        blocks.append(block.transpose(1,0,2).astype('float32'))
    held=c.anchor_pool(np.tile(blocks[0],(spec['development_anchor_nodes']//len(blocks[0]),1,1)),spec['data_seed']+5000,True)
    return blocks,held

def synthetic_preflight(torch,device):
    t=torch.arange(256,device=device)*.1;signal=torch.exp(-((t-12)/2).square())*1e-3
    ref=signal[None,None,:,None].expand(1,1,-1,11).clone();meta=event_metadata(ref[0,0].cpu().numpy()[:,None])[:,0]
    meta=torch.tensor(meta,device=device)[None,None]
    results={}
    for name,pred in [('exact',ref.clone()),('delay_01',torch.roll(ref,1,2)),('delay_04',torch.roll(ref,4,2)),('half_amplitude',ref*.5),('offset',ref+1e-5)]:
        pred=pred.detach().requires_grad_();loss=aligned_loss(torch,pred,ref,meta);loss.sum().backward()
        assert bool(torch.isfinite(loss).all()) and bool(torch.isfinite(pred.grad).all())
        results[name]=float(loss.item())
    assert results['exact']==0 and results['delay_04']>results['delay_01']>0 and results['half_amplitude']>0 and results['offset']>0
    flat=torch.zeros_like(ref);flatmeta=torch.tensor(event_metadata(np.zeros((256,1,11))),device=device,dtype=ref.dtype).permute(1,0,2)[None]
    assert float(aligned_loss(torch,flat,flat,flatmeta))==0
    precision,derivative=temporal_components(torch,ref+.00001,ref)
    assert torch.allclose(precision+derivative,f.temporal_loss(torch,ref+.00001,ref),rtol=1e-6,atol=1e-6)
    return dict(valid=True,losses=results,old_loss_decomposition_equal=True,offline_metadata_not_in_model=True)

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
    expanded=expanded_fit(root,output,spec,quality);families=list(quality['families']);data=[];support=[]
    for source in (root/spec['dataset'],expanded):
        aa=[]
        for role,extra in [('fit',0),('fit_extra',1009)]:
            a,info=windows(source,role,families,spec['windows_per_family'],256,spec['data_seed']+extra);aa.append(a);support+=info
        data.append(aa)
    anchors=c.anchor_pool(np.concatenate([a[...,:71] for a in data[0]]),spec['data_seed']+2,True)
    original_calib=json.loads((root/spec['parent']/'fit_loss_calibration.json').read_text());calib=np.asarray(original_calib['per_seed_weights'],dtype='float32')
    # Exact39f normalization and frozen FIT calibration, shared across every arm.
    empirical=c.anchor_pool(np.concatenate([a[...,:71] for a in data[0]]),spec['data_seed']+2)
    normal=json.loads((root/spec['normalizers']).read_text());us=np.std(empirical[:,71:89],0).clip(.05);zs=np.std(np.log(empirical[:,89:107].clip(1e-30)),0).clip(.5)
    scales=[torch.tensor(v,device=device,dtype=torch.float32) for v in (normal['gate'],normal['current'],us,zs)]
    dev,held=development(root,spec,quality)
    pools=[[torch.tensor(a,device=device) for a in aa] for aa in data];at=torch.tensor(anchors,device=device);cohorts=groups(torch,root,spec,device);defs=[a for g in cohorts for a in g['definitions']];count=30;static=[]
    map_error=0.
    with torch.no_grad():
        deployed=evaluation_groups(torch,cohorts,spec['seeds'],device)[0]['model'];n=count
        v=torch.linspace(-95,55,7,device=device)[None].expand(n,-1);ca=torch.logspace(-7,-2,7,device=device)[None].expand(n,-1);state=torch.full((n,7,18),.4,device=device)
        trained=torch.cat([group['model'](v[:group['model'].a.shape[0]],ca[:group['model'].a.shape[0]],state[:group['model'].a.shape[0]]) for group in cohorts])
        map_error=float((trained-deployed(v,ca,state)).abs().max())
    assert map_error<=spec['numeric_atol'],map_error
    p.save_json(output/'training_deployment_preflight.json',dict(valid=True,one_step_max_error=map_error,atol=spec['numeric_atol'],training_affine_accumulation='float64',inference='standard float32;used for selection and full evaluation'))
    for g in cohorts:
        n=g['model'].a.shape[0];h=g['definitions'][0]['horizon'];weights=[]
        for a in g['definitions']:
            for j in range(3):
                lam=float(calib[j,2])*a['factor'];weights.append([.5,.5,float(calib[j,0]),float(calib[j,1]),lam if a['loss'] in ('full','precision') else 0,lam if a['loss']=='full' else 0,lam if a['loss']=='aligned' else 0])
        static.append(dict(batch=pools[0][0][:2].reshape(512//h,h,126)[None].expand(n,-1,-1,-1).clone(),anchor=at[:64][None].expand(n,-1,-1).clone(),ticks=torch.full((n,64,1),1000.,device=device),weights=torch.tensor(weights,device=device)))
    p.save_json(output/'window_support.json',dict(windows=support,metadata_layout='71baseline,82scale,93amplitude,104eligible,115slope;all55fields offline loss labels',anchor_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),expanded_fraction=.5,target_ticks_per_update=512))
    p.save_json(output/'fit_loss_calibration.json',dict(parent_weights=calib.tolist(),weights={g['key']:s['weights'].cpu().tolist() for g,s in zip(cohorts,static)},source='frozen39f FIT calibration,not recalibrated on development',u_scales=us.tolist(),log_z_scales=zs.tolist()))
    p.save_json(output/'composition_preflight.json',c.composition_preflight(torch,cohorts,device))
    p.save_json(output/'optimizer_preflight.json',[optimizer_preflight(torch,g,s,scales,spec['seeds']) for g,s in zip(cohorts,static)])
    old.recurrence_preflight(output,spec,f.split_models(torch,cohorts,spec['seeds'],device),torch,device)
    graph,clips,execution=d.capture_training(torch,cohorts,static,scales,spec,objective_fn=objective)
    execution.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,members=count,backend='two horizon-vectorized cohorts in one CUDA graph',parameters_per_member=2404,training_affine_accumulation='float64;outputs/parameters/states/Adam float32',evaluation_inference_dtype='float32',thresholds_changed=False)
    p.save_json(output/'execution.json',execution);print(f'[GIADA Task39g] preflights passed; projected fit {execution["projected_training_seconds"]/60:.1f}min',flush=True)
    fixed=[{k:v.clone() for k,v in s.items()} for s in static];fixed_probes=[]
    rng=np.random.default_rng(spec['data_seed']+199);best=[None]*count;saved=[None]*count;chosen=[None]*count;common_best=np.full(count,np.inf);common_chosen=[None]*count;curves=[];probes=[];invalid=set();start=time.perf_counter();exposure=np.zeros((2,len(families),2),dtype=np.int64)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            tmp=output/f'checkpoint_{step}.tmp';torch.save({g['key']:g['model'].state_dict() for g in cohorts},tmp);tmp.replace(output/f'checkpoint_{step}.pt')
            tmp=output/f'optimizer_{step}.tmp';torch.save(dict(optimizers=[o.state_dict() for o in graph._giada_optimizers],numpy_rng=rng.bit_generator.state,step=step,clipping_counts=[x.cpu() for x in clips]),tmp);tmp.replace(output/f'optimizer_{step}.pt')
            p.save_json(output/'checkpoint_phase.json',dict(step=step,phase='before_selection',parameters_finite={g['key']:all(bool(torch.isfinite(v).all()) for v in g['model'].parameters()) for g in cohorts}))
            print(f'[GIADA Task39g] checkpoint_saved {step} before_selection',flush=True)
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
            p.save_json(output/'gradient_probes.json',e.diagnostic_json(probes));print(f'[GIADA Task39g] checkpoint {step} median_common={np.median(common):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
        if step==spec['checkpoints'][-1]:break
        ix=rng.integers(len(data[0][0]),size=2);extra=rng.random(2)<.5;use_new=rng.random(2)<.5;ti=torch.tensor(ix,device=device);ee=torch.tensor(extra,device=device)[:,None,None]
        original=torch.where(ee,pools[0][1][ti],pools[0][0][ti]);new=torch.where(ee,pools[1][1][ti],pools[1][0][ti]);expanded_batch=torch.where(torch.tensor(use_new,device=device)[:,None,None],new,original)
        ai=rng.integers(len(anchors),size=64);di=rng.integers(6,size=64);tt=torch.tensor(np.asarray(spec['held_ticks'])[di],device=device,dtype=torch.float32)[None,:,None]
        for g,s in zip(cohorts,static):
            h=g['definitions'][0]['horizon'];mask=torch.tensor([a['expanded'] for a in g['definitions'] for _ in spec['seeds']],device=device)[:,None,None,None]
            s['batch'].copy_(torch.where(mask,expanded_batch.reshape(512//h,h,126)[None],original.reshape(512//h,h,126)[None]));s['anchor'].copy_(at[ai][None]);s['ticks'].copy_(tt)
        graph.replay()
        np.add.at(exposure[0],(ix//spec['windows_per_family'],extra.astype(int)),256)
        np.add.at(exposure[1],(ix[use_new]//spec['windows_per_family'],extra[use_new].astype(int)),256)
        if (step+1)%2000==0:print(f'[GIADA Task39g] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
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
    base=json.loads((root/old.CONFIG).read_text());evaluation=dict(spec,architectures=[a['name'] for a in defs]+['frozen39f_'+n for n in spec['oracle_parent_arms']])
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
    c.cost_benchmark(torch,cohorts,spec['seeds'],device,output,model_factory=f.model_class,splitter=f.split_models,task_label='39g')
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
    final=dict(valid=True,task='39g',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',passing_arms=passing,numerically_unstable_arms=sorted(unstable),summaries=summary,thresholds_changed=False,independent_confirmation_accessed=False,speedup_claimed=False,limits=spec['limits']);p.save_json(output/'final_report.json',final);return final
