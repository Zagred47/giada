"""Task39f: frozen39e attribution and paired temporal/support/capacity matrix."""
import hashlib
import itertools
import json
import time
import numpy as np
from . import task39e_tail_support_heads as e
d,c,b,q,old,p=e.d,e.c,e.b,e.q,e.old,e.p
CONFIG='experiments/task39f_preregistration_v1.json'

def arms():
    return [dict(name=f'{arch}_t{t}_s{s}',architecture=arch,temporal=t,support=s)
            for arch in ('base','wide') for t,s in itertools.product((0,1),repeat=2)]

def contrast_pairs(defs):
    pairs=[]
    for axis in ('temporal','support','architecture'):
        for a in defs:
            if a[axis] != ('base' if axis=='architecture' else 0):continue
            z=next(z for z in defs if z[axis]!=a[axis] and all(z[k]==a[k] for k in ('temporal','support','architecture') if k!=axis))
            pairs.append((axis,a['name'],z['name']))
    return pairs

def model_class(torch):
    class Model(c.model_class(torch)):
        def __init__(self,architecture,seeds):
            super().__init__(architecture,seeds)
            if architecture=='wide':
                aa=[];cc=[]
                for seed in seeds:
                    gen=torch.Generator().manual_seed(seed+2718)
                    aa.append(torch.randn(1,8,generator=gen));cc.append(torch.randn(8,2,generator=gen)*.01)
                self.sa=torch.nn.Parameter(torch.cat((self.sa.detach(),torch.stack(aa)),-1))
                self.sb=torch.nn.Parameter(torch.cat((self.sb.detach(),torch.zeros_like(self.sb)), -1))
                self.sc=torch.nn.Parameter(torch.cat((self.sc.detach(),torch.stack(cc)),1))
    return Model

def groups(torch,defs,seeds,device):
    out=[]
    for arch in ('base','wide'):
        aa=[a for a in defs if a['architecture']==arch];n=len(aa)*len(seeds)
        if not aa:continue
        out.append(dict(key=arch,definitions=aa,model=model_class(torch)(arch,seeds*len(aa)).to(device),
            flags=torch.ones(n,4,device=device),temporal=torch.tensor([a['temporal'] for a in aa for _ in seeds],device=device),
            support=torch.tensor([a['support'] for a in aa for _ in seeds],device=device).bool()))
    return out

def split_models(torch,cohorts,seeds,device):
    out=[]
    for g in cohorts:
        for i,a in enumerate(g['definitions']):
            model=model_class(torch)(a['architecture'],seeds).to(device)
            model.load_state_dict({k:v.clone() if k=='gbar' else v[i*len(seeds):(i+1)*len(seeds)].clone() for k,v in g['model'].state_dict().items()})
            out.append(model)
    return out

def temporal_loss(torch,pred,ref):
    """FIT-derived slow-slope precision plus derivative error, no event relabeling."""
    scale=ref.abs().amax(2,keepdim=True).clamp_min(1e-7)
    residual=(pred-ref)/scale
    slope=torch.diff(ref,dim=2,prepend=ref[:,:,:1]).abs()/scale/.1
    # A slope floor bounds conditioning; tanh bounds each local time residual.
    time_residual=residual/slope.clamp_min(.001)/.3
    bounded=5*torch.tanh(time_residual/5)
    rr=(ref-ref[:,:,:1]).abs()/scale
    amplitude=rr.amax(2,keepdim=True)
    band=torch.exp(-((rr-.1*amplitude)/(.05*amplitude).clamp_min(.001)).square())
    precision=((.5+.5*band)*bounded.square()).mean((1,2,3))
    derivative=(torch.diff(pred,dim=2)-torch.diff(ref,dim=2))/scale/.1
    return precision+.25*derivative.square().mean((1,2,3))

def terms(torch,g,s,scales,short=64):
    values,current,ref=e.terms(torch,g,s,scales,short,return_trajectory=True)
    return torch.cat((values,temporal_loss(torch,current,ref)[:,None]),-1)

def objective(torch,g,s,scales,short=64):return (terms(torch,g,s,scales,short)*s['weights']).sum(-1)

def gradient_probe(torch,g,s,scales,step):
    values=terms(torch,g,s,scales);grads=[]
    for i in range(5):
        grad=torch.autograd.grad(values[:,i].sum(),tuple(g['model'].parameters()),retain_graph=i<4)
        grads.append(torch.cat([v.flatten(1) for v in grad],-1).detach().double())
    stack=torch.stack(grads,1);norm=stack.norm(dim=-1)
    cosine=torch.bmm(stack,stack.transpose(1,2))/(norm[:,:,None]*norm[:,None,:]).clamp_min(1e-20)
    return dict(step=step,cohort=g['key'],losses=values.detach().cpu().tolist(),term_norms=norm.cpu().tolist(),cosines=cosine.cpu().tolist(),weights=s['weights'].cpu().tolist()),norm

def optimizer_preflight(torch,g,s,scales,seeds):
    m=g['model'];initial={k:v.detach().clone() for k,v in m.named_parameters()}
    opt=torch.optim.Adam(m.parameters(),lr=.001);objective(torch,g,s,scales).sum().backward();old.clip(torch,m);opt.step();error=0.
    for j in range(m.a.shape[0]):
        iso=model_class(torch)(m.architecture,[seeds[j%len(seeds)]]).to(s['batch'].device)
        with torch.no_grad():
            for k,v in iso.named_parameters():v.copy_(initial[k][j:j+1])
        gg=dict(g,model=iso,flags=g['flags'][j:j+1]);ss={k:v[j:j+1] for k,v in s.items()}
        oo=torch.optim.Adam(iso.parameters(),lr=.001);objective(torch,gg,ss,scales).sum().backward();old.clip(torch,iso);oo.step()
        error=max(error,max(float((v[0]-dict(m.named_parameters())[k][j]).abs().max()) for k,v in iso.named_parameters()))
    with torch.no_grad():
        for k,v in m.named_parameters():v.copy_(initial[k]);v.grad=None
    assert error<=1e-5,error
    return dict(valid=True,members=m.a.shape[0],max_abs_error=error)

def recovery_windows(root,role,families,count,length,seed):
    """FIT only: equal family counts, quarter uniform/transition/log-lag/crossing."""
    assert role in ('fit','fit_extra')
    rng=np.random.default_rng(seed);out=[];support=[]
    for family in families:
        with np.load(root/f'{role}__{family}.npz') as z:a={k:z[k] for k in z.files}
        ticks,cases=a['voltage'].shape;starts=rng.integers(0,ticks-length+1,count);case=rng.integers(0,cases,count)
        changes=(np.abs(np.diff(a['voltage'],axis=0,prepend=a['voltage'][:1]))>1)|(np.abs(np.diff(np.log10(a['bridge_ca'].clip(1e-9)),axis=0,prepend=np.log10(a['bridge_ca'][:1].clip(1e-9))))>.05)
        changes[0]=True;events=np.argwhere(changes)
        currents=a['current_targets'];amp=np.abs(currents-currents[:1]).max(0)
        active=np.abs(currents-currents[:1])>=.1*amp[None]
        crossing=np.argwhere(np.any(active[1:]!=active[:-1],axis=-1));crossing[:,0]+=1
        modes=np.arange(count)%4
        for mode in (1,2,3):
            ids=np.flatnonzero(modes==mode);candidates=crossing if mode==3 and len(crossing) else events
            selected=candidates[rng.integers(len(candidates),size=len(ids))]
            lag=np.exp(rng.uniform(0,np.log(max(ticks,2)),len(ids))).astype(int) if mode==2 else np.zeros(len(ids),dtype=int)
            starts[ids]=(selected[:,0]+lag-rng.integers(length,size=len(ids))).clip(0,ticks-length);case[ids]=selected[:,1]
        ix=starts[:,None]+np.arange(length);ci=np.broadcast_to(case[:,None],ix.shape)
        previous=a['gate_targets'][np.maximum(ix-1,0),ci].copy()
        for j in np.flatnonzero(starts==0):previous[j,0]=a['initial_gates'][case[j]]
        data=np.concatenate((a['voltage'][ix,ci,None],a['bridge_ca'][ix,ci,None],previous,a['gate_targets'][ix,ci],np.broadcast_to(a['multipliers'][case,None],(count,length,11)),np.broadcast_to(a['reversals'][case,None],(count,length,11)),currents[ix,ci]),-1)
        out.append(data);support.append(dict(role=role,family=family,windows=count,modes=np.bincount(modes,minlength=4).tolist(),unique_windows=len(set(zip(starts.tolist(),case.tolist()))),assignment_sha256=hashlib.sha256(np.stack((starts,case)).tobytes()).hexdigest(),starts=starts.tolist(),cases=case.tolist(),fit_only=True))
    return np.concatenate(out).astype('float32'),support

def load_parents(torch,root,spec,device,dtype):
    folder=root/spec['parent'];freeze=json.loads((folder/'selection_freeze.json').read_text());claimed=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==claimed
    assert p.digest(folder/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']==spec['parent_sha256']
    weights=torch.load(folder/'frozen_checkpoint.pt',map_location=device,weights_only=True);out=[]
    for name in spec['oracle_parent_arms']:
        a=next(a for a in freeze['arms'] if a['name']==name)
        aa=[a2 for a2 in freeze['arms'] if a2['architecture']==a['architecture'] and a2['long']==a['long']]
        idx=next(i for i,a2 in enumerate(aa) if a2['name']==name)*3
        model=e.model_class(torch)(a['architecture'],spec['seeds']).to(device)
        model.load_state_dict({k:v.clone() if k=='gbar' else v[idx:idx+3].clone() for k,v in weights[f'{a["architecture"]}_long{a["long"]}'].items()})
        out.append(model.to(dtype=dtype))
    return out

def timing_audit(reference,prediction,dt=.1):
    rows=d.timing_audit(reference,prediction,dt)
    for row in rows:
        j=list(p.ionic.CHANNELS).index(row['channel']);r=reference[:,j];a=prediction[:,j];base=r[0]
        amp=np.abs(r-base).max();threshold=.1*amp;rr=np.abs(r-base);pp=np.abs(a-base)
        ri=np.flatnonzero(rr>=threshold);pi=np.flatnonzero(pp>=threshold)
        crossings=[]
        for label,k in [('onset',int(ri[0])),('offset',int(ri[-1]))]:
            slope=float(np.gradient(rr.astype(float),dt)[k]);residual=float(pp[k]-rr[k])
            crossings.append(dict(kind=label,reference_index=k,prediction_index=int(pi[0 if label=='onset' else -1]) if len(pi) else None,signed_shift_ms=(int(pi[0 if label=='onset' else -1])-k)*dt if len(pi) else None,reference_slope=slope,amplitude_residual=residual,linear_shift_prediction_ms=-residual/slope if abs(slope)>1e-20 else None))
        phase=[]
        grid=np.arange(len(r),dtype=float)
        for fraction in (.25,.5,.75):
            phase.append(dict(fraction=fraction,event=p.event_metrics(np.interp(grid+fraction,grid,r),np.interp(grid+fraction,grid,a),dt,1e-7)))
        row.update(crossings=crossings,reference_excursions=int(np.count_nonzero(np.diff(np.r_[False,rr>=threshold].astype(int))==1)),prediction_excursions=int(np.count_nonzero(np.diff(np.r_[False,pp>=threshold].astype(int))==1)),grid_phase_diagnostics=phase,diagnostic_only=True)
    return rows

def run(root,output,device='cuda'):
    import torch
    assert device=='cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());quality=json.loads((root/p.CONFIG).read_text())
    assert p.digest(root/p.CONFIG)==spec['quality_config_sha256']
    output.mkdir(parents=True,exist_ok=True);assert not (output/'final_report.json').exists()
    p.save_json(output/'preregistration.json',spec)
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()):assert p.digest(root/spec['dataset']/row['file'])==row['sha256']
    d.oracle_audit(torch,root,output,spec,quality,device,parent_loader=load_parents,timing_fn=timing_audit)
    families=list(quality['families']);data=[];targeted=[];support=[]
    for role,extra in [('fit',0),('fit_extra',1009)]:
        a,info=q.windows(root/spec['dataset'],role,families,spec['windows_per_family'],256,spec['data_seed']+extra)
        data.append(a);support+=info
        a,info=recovery_windows(root/spec['dataset'],role,families,spec['windows_per_family'],256,spec['data_seed']+extra)
        targeted.append(a);support+=info
    empirical=c.anchor_pool(np.concatenate(data),spec['data_seed']+2)
    anchor=c.anchor_pool(np.concatenate(data),spec['data_seed']+2,True)
    normal=json.loads((root/spec['normalizers']).read_text());us=np.std(empirical[:,71:89],0).clip(.05);zs=np.std(np.log(empirical[:,89:107].clip(1e-30)),0).clip(.5)
    scales=[torch.tensor(a,device=device,dtype=torch.float32) for a in (normal['gate'],normal['current'],us,zs)]
    dev=[]
    for k,family in enumerate(families):
        a,info=q.windows(root/spec['dataset'],'development',[family],2,512,spec['data_seed']+4000+k);dev.append(a);support+=info
    for family in ('same_v_history','slow_recovery'):
        with np.load(root/spec['dataset']/f'development__{family}.npz') as z:a={k:z[k] for k in z.files}
        ix=np.arange(len(a['voltage'])-spec['development_long_ticks'],len(a['voltage']));pre=a['gate_targets'][np.maximum(ix-1,0)]
        block=np.concatenate((a['voltage'][ix,...,None],a['bridge_ca'][ix,...,None],pre,a['gate_targets'][ix],np.broadcast_to(a['multipliers'],(len(ix),2,11)),np.broadcast_to(a['reversals'],(len(ix),2,11)),a['current_targets'][ix]),-1)
        dev.append(block.transpose(1,0,2).astype('float32'))
    held=c.anchor_pool(np.tile(dev[0],(spec['development_anchor_nodes']//len(dev[0]),1,1)),spec['data_seed']+5000,True)
    pools=[[torch.tensor(a,device=device) for a in rows] for rows in (data,targeted)]
    anchors=torch.tensor(anchor,device=device);cohorts=groups(torch,arms(),spec['seeds'],device);defs=[a for g in cohorts for a in g['definitions']];count=24
    static=[]
    for g in cohorts:
        n=g['model'].a.shape[0]
        static.append(dict(batch=pools[0][0][:2].reshape(8,64,-1)[None].expand(n,-1,-1,-1).clone(),anchor=anchors[:64][None].expand(n,-1,-1).clone(),ticks=torch.full((n,64,1),1000.,device=device),weights=torch.ones(n,5,device=device)))
    rec,norm=gradient_probe(torch,cohorts[0],static[0],scales,0)
    weights=(.5*(norm[:3,0]+norm[:3,1])[:,None]/norm[:3,2:].clamp_min(1e-12)).clamp(.001,1000).float()
    for g,s in zip(cohorts,static):
        s['weights'][:,:2]=.5;s['weights'][:,2:]=weights.repeat(len(s['weights'])//3,1);s['weights'][:,4]*=g['temporal']
    p.save_json(output/'fit_loss_calibration.json',dict(records=rec,per_seed_weights=weights.cpu().tolist(),source='initial base_t0_s0 FIT only;fixed for all arms'))
    p.save_json(output/'window_support.json',dict(windows=support,anchor_sha256=hashlib.sha256(anchor.tobytes()).hexdigest(),targets_per_update=512,anchors_per_update=64))
    p.save_json(output/'composition_preflight.json',c.composition_preflight(torch,cohorts,device))
    p.save_json(output/'optimizer_preflight.json',[optimizer_preflight(torch,g,s,scales,spec['seeds']) for g,s in zip(cohorts,static)])
    old.recurrence_preflight(output,spec,split_models(torch,cohorts,spec['seeds'],device),torch,device)
    graph,clips,execution=d.capture_training(torch,cohorts,static,scales,spec,objective_fn=objective)
    execution.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,members=count,backend='two vectorized width cohorts in one CUDA Graph',parameters_per_member={g['key']:sum(v.numel() for v in g['model'].parameters())//g['model'].a.shape[0] for g in cohorts},thresholds_changed=False)
    p.save_json(output/'execution.json',execution)
    print(f'[GIADA Task39f] preflights passed; projected fit {execution["projected_training_seconds"]/60:.1f}min',flush=True)
    rng=np.random.default_rng(spec['data_seed']+99);best=[None]*count;saved=[None]*count;chosen=[None]*count
    common_best=np.full(count,np.inf);common_chosen=[None]*count
    curves=[];probes=[];invalid=set();start=time.perf_counter();exposure=np.zeros((2,len(families),2),dtype=np.int64)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            tmp=output/f'checkpoint_{step}.tmp';torch.save({g['key']:g['model'].state_dict() for g in cohorts},tmp);tmp.replace(output/f'checkpoint_{step}.pt')
            tmp=output/f'optimizer_{step}.tmp';torch.save(dict(optimizers=[o.state_dict() for o in graph._giada_optimizers],numpy_rng=rng.bit_generator.state,step=step,clipping_counts=[x.cpu() for x in clips]),tmp);tmp.replace(output/f'optimizer_{step}.pt')
            p.save_json(output/'checkpoint_phase.json',dict(step=step,phase='before_selection',parameters_finite={g['key']:all(bool(torch.isfinite(v).all()) for v in g['model'].parameters()) for g in cohorts}))
            print(f'[GIADA Task39f] checkpoint_saved {step} before_selection',flush=True)
            common,parts,ranks,panel=e.checkpoint_scores(torch,cohorts,dev,held,normal,spec,device,quality)
            j=0
            for g in cohorts:
                for k in range(g['model'].a.shape[0]):
                    if ranks[j] is None:invalid.add(j);j+=1;continue
                    if best[j] is None or ranks[j]<best[j]:
                        best[j]=ranks[j];chosen[j]=step;saved[j]={name:v[k].detach().cpu().clone() for name,v in g['model'].named_parameters()}
                    if common[j]<common_best[j]:common_best[j]=common[j];common_chosen[j]=step
                    j+=1
            assert all(v is not None for v in saved),'Initial finite selection preflight failed'
            curves.append(dict(step=step,common_development_score=common.tolist(),panel=panel,elapsed_seconds=time.perf_counter()-start,clipping_counts=[x.cpu().tolist() for x in clips]))
            p.save_json(output/'training_curves.json',e.diagnostic_json(curves));p.save_json(output/'numerical_health.json',dict(invalid_members=sorted(invalid),current_invalid=[j for j,r in enumerate(ranks) if r is None],invalid_checkpoints_are_never_selected=True))
            for g,s in zip(cohorts,static):probes.append(gradient_probe(torch,g,s,scales,step)[0])
            p.save_json(output/'gradient_probes.json',e.diagnostic_json(probes))
            print(f'[GIADA Task39f] checkpoint {step} median_common={np.median(common):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
        if step==spec['checkpoints'][-1]:break
        ix=rng.integers(len(data[0]),size=2);extra=rng.random(2)<.5;ti=torch.tensor(ix,device=device);ee=torch.tensor(extra,device=device)[:,None,None]
        batches=[torch.where(ee,pools[k][1][ti],pools[k][0][ti]).reshape(8,64,-1) for k in (0,1)]
        ai=rng.integers(len(anchor),size=64);di=rng.integers(6,size=64);tt=torch.tensor(np.asarray(spec['held_ticks'])[di],device=device,dtype=torch.float32)[None,:,None]
        for g,s in zip(cohorts,static):
            s['batch'].copy_(torch.where(g['support'][:,None,None,None],batches[1][None],batches[0][None]));s['anchor'].copy_(anchors[ai][None]);s['ticks'].copy_(tt)
        graph.replay()
        for coverage in (0,1):np.add.at(exposure[coverage],(ix//spec['windows_per_family'],extra.astype(int)),256)
        if (step+1)%2000==0:print(f'[GIADA Task39f] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-start;j=0
    with torch.no_grad():
        for g in cohorts:
            for k in range(g['model'].a.shape[0]):
                for name,v in g['model'].named_parameters():v[k].copy_(saved[j][name])
                j+=1
    torch.save({g['key']:g['model'].state_dict() for g in cohorts},output/'frozen_checkpoint.pt')
    freeze=dict(arms=defs,seeds=spec['seeds'],chosen_steps=chosen,ranks=best,common_rule_steps=common_chosen,development_only=True,stress_used_for_selection=False,independent_confirmation_accessed=False,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'))
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    p.save_json(output/'exposure.json',dict(coverage_family_role_ticks=exposure.tolist(),variable_targets_per_member=512*spec['checkpoints'][-1],anchor_targets_per_member=64*spec['checkpoints'][-1]))
    base=json.loads((root/old.CONFIG).read_text());evaluation=dict(spec,architectures=[a['name'] for a in defs]+['frozen39e_'+n for n in spec['oracle_parent_arms']])
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=base[key]
    assert p.digest(root/base['baseline_parent'])==base['baseline_parent_sha256'] and p.digest(root/base['baseline_checkpoint'])==base['baseline_sha256']
    summary=old.evaluate(root,output,evaluation,split_models(torch,cohorts,spec['seeds'],device)+load_parents(torch,root,spec,device,torch.float32),torch,device,quality)
    rows=json.loads((output/'recurrence_evaluation.json').read_text());scores={}
    for a in defs:
        scores[a['name']]={}
        for seed in spec['seeds']:
            rr=[r for r in rows if r['model']=='joint_w'+a['name'] and r['role']=='development' and r['seed']==seed];vals=[q.severity(r['metrics']) for r in rr]
            scores[a['name']][str(seed)]=dict(median=float(np.median(vals)),worst=max(vals),passed=sum(r['metrics']['passed'] for r in rr))
    contrasts=[]
    for axis,a,z in contrast_pairs(defs):
        ratios=[scores[z][str(s)]['median']/max(scores[a][str(s)]['median'],1e-12) for s in spec['seeds']];worst=[scores[z][str(s)]['worst']/max(scores[a][str(s)]['worst'],1e-12) for s in spec['seeds']]
        contrasts.append(dict(axis=axis,control=a,treatment=z,paired_ratios=ratios,worst_ratios=worst,supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=contrasts))
    c.cost_benchmark(torch,cohorts,spec['seeds'],device,output,model_factory=model_class,splitter=split_models,task_label='39f')
    common_dir=output/'common_selection_diagnostic';common_dir.mkdir()
    checkpoints={step:torch.load(output/f'checkpoint_{step}.pt',map_location=device,weights_only=True) for step in set(common_chosen)}
    j=0
    with torch.no_grad():
        for g in cohorts:
            for k in range(g['model'].a.shape[0]):
                for name,v in g['model'].named_parameters():v[k].copy_(checkpoints[common_chosen[j]][g['key']][name][k])
                j+=1
    common_summary=old.evaluate(root,common_dir,evaluation,split_models(torch,cohorts,spec['seeds'],device)+load_parents(torch,root,spec,device,torch.float32),torch,device,quality)
    p.save_json(output/'common_selection_diagnostic.json',dict(summaries=common_summary,selected_steps=common_chosen,selection_eligible=False,limits='Same saved trajectories;diagnostic old selection rule cannot replace primary freeze.'))
    unstable={defs[j//3]['name'] for j in invalid};passing=[a['name'] for a in defs if summary['joint_w'+a['name']]['all_development_passed'] and a['name'] not in unstable]
    execution.update(training_wall_seconds=training,total_wall_seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated());p.save_json(output/'execution.json',execution)
    final=dict(valid=True,task='39f',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',passing_arms=passing,numerically_unstable_arms=sorted(unstable),summaries=summary,thresholds_changed=False,independent_confirmation_accessed=False,speedup_claimed=False,limits=spec['limits'])
    p.save_json(output/'final_report.json',final);return final
