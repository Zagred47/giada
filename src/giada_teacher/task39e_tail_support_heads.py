"""Task39e: paired tail objective, coefficient support, heads and long exposure.

All teacher coefficients are OFFLINE labels/oracles; candidate inference uses
learned u/z and18 bounded states. No canonical kinetics are restored at runtime.
"""
import hashlib
import itertools
import json
import time
import numpy as np
from . import task39d_current_timing as d
c,b,q,old,p=d.c,d.b,d.q,d.old,d.p
CONFIG='experiments/task39e_preregistration_v1.json'

def arms():
    out=[dict(name=f'{"split" if h else "shared"}_o{o}_s{s}_h{h}',architecture='split' if h else 'base',
              tail=o,stratified=s,heads=h,long=0) for o,s,h in itertools.product((0,1),repeat=3)]
    out += [dict(name='long_000',architecture='base',tail=0,stratified=0,heads=0,long=1),
            dict(name='long_111',architecture='split',tail=1,stratified=1,heads=1,long=1)]
    return sorted(out,key=lambda x:(x['architecture'],x['long'],x['name']))

def contrast_pairs(defs):
    out=[];main=[a for a in defs if not a['long']]
    for axis in ('tail','stratified','heads'):
        for a in main:
            if a[axis]:continue
            z=next(z for z in main if z[axis] and all(z[k]==a[k] for k in ('tail','stratified','heads') if k!=axis))
            out.append((axis,a['name'],z['name']))
    for z in (a for a in defs if a['long']):
        a=next(a for a in main if all(a[k]==z[k] for k in ('tail','stratified','heads')))
        out.append(('long_exposure',a['name'],z['name']))
    return out

def model_class(torch):
    class Model(c.model_class(torch)):
        def __init__(self,architecture,seeds):
            super().__init__('base',seeds);self.architecture=architecture
            if architecture=='split':
                # 30 features per head, parameter budget1208 vs shared1220 (-0.98%).
                self.a=torch.nn.Parameter(self.a[:,:,:30].detach().clone())
                self.b=torch.nn.Parameter(self.b[:,:,:30].detach().clone())
                self.c=torch.nn.Parameter(self.c[:,:30,:17].detach().clone())
                self.d=torch.nn.Parameter(self.d[:,:17].detach().clone())
                joint=c.model_class(torch)('base',seeds)
                self.za=torch.nn.Parameter(joint.a[:,:,:30].detach().clone())
                self.zb=torch.nn.Parameter(joint.b[:,:,:30].detach().clone())
                self.zc=torch.nn.Parameter(joint.c[:,:30,17:].detach().clone())
                self.zd=torch.nn.Parameter(joint.d[:,17:].detach().clone())
        def raw(self,v,ca):
            if self.architecture!='split':return super().raw(v,ca)
            vv=(v+40)/80;cc=(ca.clamp_min(1e-9).log10()+4.5)/2.5
            u=torch.bmm(torch.nn.functional.silu(torch.bmm(vv[...,None],self.a)+self.b),self.c)+self.d[:,None]
            z=torch.bmm(torch.nn.functional.silu(torch.bmm(vv[...,None],self.za)+self.zb),self.zc)+self.zd[:,None]
            sk=torch.bmm(torch.nn.functional.silu(torch.bmm(cc[...,None],self.sa)+self.sb),self.sc)+self.sd[:,None]
            return torch.cat((u[...,:16],sk[...,0:1],u[...,16:]),-1),torch.cat((z[...,:16],sk[...,1:2],z[...,16:]),-1)
    return Model

def groups(torch,defs,seeds,device):
    out=[]
    for arch,long in itertools.product(('base','split'),(0,1)):
        aa=[a for a in defs if a['architecture']==arch and a['long']==long]
        if not aa:continue
        out.append(dict(key=f'{arch}_long{long}',definitions=aa,long=long,
            model=model_class(torch)(arch,seeds*len(aa)).to(device),
            flags=torch.tensor([[a[k] for k in ('tail','stratified','heads','long')] for a in aa for _ in seeds],device=device)))
    return out

def split_models(torch,cohorts,seeds,device):
    out=[]
    for g in cohorts:
        for i,a in enumerate(g['definitions']):
            m=model_class(torch)(a['architecture'],seeds).to(device)
            m.load_state_dict({k:x.clone() if k=='gbar' else x[i*len(seeds):(i+1)*len(seeds)].clone() for k,x in g['model'].state_dict().items()})
            out.append(m)
    return out

def censored_logit_error(torch,raw,target,epsilon=1e-6):
    clipped=target.clamp(epsilon,1-epsilon);label=torch.logit(clipped)
    err=raw-label
    err=torch.where(target<=epsilon,torch.relu(err),torch.where(target>=1-epsilon,-torch.relu(-err),err))
    return (err/6).square()

def terms(torch,g,s,scales,short=64,return_trajectory=False):
    model=g['model'];batch,anchor,ticks=s['batch'],s['anchor'],s['ticks'];gs,cs,us,zs=scales
    flags=g['flags'];tail=flags[:,0].bool()
    # Every sequence starts once; no hidden teacher resets inside either horizon.
    state=d.rollout_training(torch,model,batch,torch.ones_like(tail),short)
    current=b.captured_readout(torch,batch[...,0],state,batch[...,38:49],batch[...,49:60],model.gbar)
    ref=batch[...,60:71];local=ref.abs().amax(2,keepdim=True).clamp_min(1e-7)
    global_error=((current-ref)/cs).square().mean((1,2,3));local_error=((current-ref)/local).square().mean((1,2,3))
    variable=((state-batch[...,20:38])/gs).square().mean((1,2,3))+torch.where(tail,.5*(global_error+local_error),global_error)
    u,z,ld=model.coefficients(anchor[...,0],anchor[...,1]);target=c.held_targets(torch,anchor,ticks)
    pred=c.compose(torch,u,ld,anchor[...,2:20],ticks)
    cur=b.captured_readout(torch,anchor[...,0],pred,anchor[...,38:49],anchor[...,49:60],model.gbar)
    true=b.captured_readout(torch,anchor[...,0],target,anchor[...,38:49],anchor[...,49:60],model.gbar)
    initial=b.captured_readout(torch,anchor[...,0],anchor[...,2:20],anchor[...,38:49],anchor[...,49:60],model.gbar)
    ls=torch.maximum(initial.abs(),true.abs()).clamp_min(1e-7)
    ge=((cur-true)/cs).square().mean((1,2));le=((cur-true)/ls).square().mean((1,2))
    held=((pred-target)/gs).square().mean((1,2))+torch.where(tail,.5*(ge+le),ge)
    ru,rz=model.raw(anchor[...,0],anchor[...,1]);ordinary=((u-anchor[...,71:89])/us).square().mean((1,2))
    censored=censored_logit_error(torch,ru,anchor[...,71:89]).mean((1,2))
    inf=torch.where(tail,.5*(ordinary+censored),ordinary)
    rate=((torch.nn.functional.logsigmoid(rz)-anchor[...,89:107].clamp_min(1e-30).log())/zs).square().mean((1,2))
    result=torch.stack((variable,held,inf,rate),-1)
    return (result,current,ref) if return_trajectory else result

def objective(torch,g,s,scales,short=64):return (terms(torch,g,s,scales,short)*s['weights']).sum(-1)

def gradient_probe(torch,g,s,scales,step):
    values=terms(torch,g,s,scales);grads=[]
    for i in range(4):
        grad=torch.autograd.grad(values[:,i].sum(),tuple(g['model'].parameters()),retain_graph=i<3)
        grads.append(torch.cat([x.flatten(1) for x in grad],-1).detach())
    stack=torch.stack(grads,1);norm=stack.norm(dim=-1)
    # Diagnostics only: FP32 dot products can overflow even for finite gradients.
    # Preserve original norm return (FIT loss calibration) bit-for-bit.
    diagnostic=stack.double();dn=diagnostic.norm(dim=-1)
    cosine=torch.bmm(diagnostic,diagnostic.transpose(1,2))/(dn[:,:,None]*dn[:,None,:]).clamp_min(1e-20)
    return dict(step=step,cohort=g['key'],losses=values.detach().cpu().tolist(),term_norms=dn.cpu().tolist(),
                cosines=cosine.cpu().tolist(),weights=s['weights'].cpu().tolist()),norm

def optimizer_preflight(torch,g,s,scales,seeds):
    m=g['model'];initial={k:x.detach().clone() for k,x in m.named_parameters()}
    opt=torch.optim.Adam(m.parameters(),lr=.001);objective(torch,g,s,scales).sum().backward();old.clip(torch,m);opt.step();error=0.
    for j in range(m.a.shape[0]):
        iso=model_class(torch)(m.architecture,[seeds[j%len(seeds)]]).to(s['batch'].device)
        with torch.no_grad():
            for k,x in iso.named_parameters():x.copy_(initial[k][j:j+1])
        gg=dict(g,model=iso,flags=g['flags'][j:j+1]);ss={k:v[j:j+1] for k,v in s.items()}
        oo=torch.optim.Adam(iso.parameters(),lr=.001);objective(torch,gg,ss,scales).sum().backward();old.clip(torch,iso);oo.step()
        error=max(error,max(float((x[0]-dict(m.named_parameters())[k][j]).abs().max()) for k,x in iso.named_parameters()))
    with torch.no_grad():
        for k,x in m.named_parameters():x.copy_(initial[k]);x.grad=None
    assert error<=1e-5,error
    return dict(valid=True,cohort=g['key'],members=m.a.shape[0],max_abs_error=error)


def diagnostic_json(value):
    """Nonfinite diagnostics are explicit missing values, never passing scores."""
    if isinstance(value,dict):return {k:diagnostic_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [diagnostic_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value

def checkpoint_scores(torch,cohorts,dev,held,normal,spec,device,quality):
    common,parts=c.selection(torch,cohorts,dev,held,normal,spec,device)
    ranks=[];detail=[];offset=0
    for g in cohorts:
        model=g['model'];n=model.a.shape[0];scores=[[] for _ in range(n)]
        scanner=old.Scanner(torch,[model],spec['development_windows_per_family'],spec['recurrence_chunk'],device)
        for index,block in enumerate(dev):
            gates=scanner.run(block[...,0].T,block[...,1].T,block[:,0,2:20])[0].transpose(1,2,0,3)
            for member in range(n):
                _,cur=p.readout(block[...,0],gates[member],block[...,38:49],block[...,49:60])
                for case in range(len(block)):
                    metric=p.evaluate_currents(block[case,:,60:71],cur[case],.1,quality['quality'],block[case,:,20:38],gates[member,case],block[case,0,38:49]==0)
                    if not metric.get('finite',False) or metric.get('gate') is None or not metric['gate']['finite']:
                        scores[member].append((1,None));continue
                    errors=[metric['gate']['rmse']/.005,metric['gate']['max_abs']/.03]
                    for cc in metric['channels']:
                        errors += [cc['nrmse']/.01,cc['nmax']/.05,abs(cc['signed_charge_normalized'])/.02,cc['absolute_charge_normalized']/.02]
                        ev=cc['event']
                        if ev['status']=='missing_event':errors.append(100.)
                        elif ev['status']=='evaluable':errors += [ev['peak_error_ms']/.2,ev['onset_error_ms']/.3,ev['offset_error_ms']/.3]
                    scores[member].append((int(not metric['passed']),max(errors)))
        for member,rr in enumerate(scores):
            invalid=any(x[1] is None for x in rr) or not np.isfinite(common[offset+member])
            rank=None if invalid else (sum(x[0] for x in rr),max(x[1] for x in rr),float(common[offset+member]))
            ranks.append(rank);detail.append(dict(cohort=g['key'],member=member,rank=rank,panel_cases=len(rr),numerically_invalid=bool(invalid),nonfinite_cases=sum(x[1] is None for x in rr),selection_eligible=not invalid))
        offset+=n
    return common,parts,ranks,detail

def run(root,output,device='cuda'):
    import torch
    assert device=='cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());quality=json.loads((root/p.CONFIG).read_text())
    output.mkdir(parents=True,exist_ok=True);assert not (output/'final_report.json').exists()
    p.save_json(output/'preregistration.json',spec)
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()):assert p.digest(root/spec['dataset']/row['file'])==row['sha256']
    d.oracle_audit(torch,root,output,spec,quality,device)
    families=list(quality['families']);data=[];support=[]
    for role,extra in [('fit',0),('fit_extra',1009)]:
        a,info=q.windows(root/spec['dataset'],role,families,spec['windows_per_family'],spec['long_ticks'],spec['data_seed']+extra)
        data.append(a);support+=info
    empirical=c.anchor_pool(np.concatenate(data),spec['data_seed']+2)
    stratified=c.anchor_pool(np.concatenate(data),spec['data_seed']+2,True)
    normal=json.loads((root/spec['normalizers']).read_text());us=np.std(empirical[:,71:89],0).clip(.05)
    zs=np.std(np.log(empirical[:,89:107].clip(1e-30)),0).clip(.5)
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
    pools=[torch.tensor(a,device=device) for a in data];anchors=[torch.tensor(a,device=device) for a in (empirical,stratified)]
    cohorts=groups(torch,arms(),spec['seeds'],device);defs=[a for g in cohorts for a in g['definitions']];count=len(defs)*len(spec['seeds']);static=[]
    for g in cohorts:
        n=g['model'].a.shape[0];batch=pools[0][:2]
        if not g['long']:batch=batch.reshape(8,64,-1)
        at=torch.where(g['flags'][:,1,None,None].bool(),anchors[1][:64][None],anchors[0][:64][None])
        static.append(dict(batch=batch[None].expand(n,-1,-1,-1).clone(),anchor=at.clone(),ticks=torch.full((n,64,1),1000.,device=device),weights=torch.ones(n,4,device=device)))
    rec,norm=gradient_probe(torch,cohorts[0],static[0],scales,0)
    weights=(.5*(norm[:3,0]+norm[:3,1])[:,None]/norm[:3,2:].clamp_min(1e-12)).clamp(.001,1000)
    for s in static:s['weights'][:,:2]=.5;s['weights'][:,2:]=weights.repeat(len(s['weights'])//3,1)
    p.save_json(output/'fit_loss_calibration.json',dict(records=rec,per_seed_weights=weights.cpu().tolist(),source='initial shared000 FIT only; common and fixed'))
    p.save_json(output/'window_support.json',dict(windows=support,empirical_sha256=hashlib.sha256(empirical.tobytes()).hexdigest(),stratified_sha256=hashlib.sha256(stratified.tobytes()).hexdigest(),targets_per_update=512,anchors_per_update=64))
    p.save_json(output/'composition_preflight.json',c.composition_preflight(torch,cohorts,device))
    p.save_json(output/'optimizer_preflight.json',[optimizer_preflight(torch,g,s,scales,spec['seeds']) for g,s in zip(cohorts,static)])
    old.recurrence_preflight(output,spec,split_models(torch,cohorts,spec['seeds'],device),torch,device)
    graph,clips,execution=d.capture_training(torch,cohorts,static,scales,spec,objective_fn=objective)
    execution.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,members=count,backend='four member-vectorized cohorts; one CUDA Graph',parameters_per_member={g['model'].architecture:sum(x.numel() for x in g['model'].parameters())//g['model'].a.shape[0] for g in cohorts},thresholds_changed=False)
    p.save_json(output/'execution.json',execution)
    print(f'[GIADA Task39e] preflights passed; projected fit {execution["projected_training_seconds"]/60:.1f}min',flush=True)
    rng=np.random.default_rng(spec['data_seed']+99);best=[None]*count;saved=[None]*count;chosen=[None]*count;common_best=np.full(count,np.inf);common_chosen=[None]*count
    curves=[];probes=[];invalid_members=set();start=time.perf_counter();family_exposure=np.zeros((len(families),2),dtype=np.int64)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            # Preserve the actual weights BEFORE any diagnostic/evaluation can fail.
            pending=output/f'checkpoint_{step}.tmp'
            torch.save({g['key']:g['model'].state_dict() for g in cohorts},pending)
            pending.replace(output/f'checkpoint_{step}.pt')
            torch.save(dict(optimizers=[o.state_dict() for o in graph._giada_optimizers],numpy_rng=rng.bit_generator.state,step=step,clipping_counts=[x.cpu() for x in clips]),output/f'optimizer_{step}.pt')
            p.save_json(output/'checkpoint_phase.json',dict(step=step,phase='before_selection',parameters_finite={g['key']:all(bool(torch.isfinite(x).all()) for x in g['model'].parameters()) for g in cohorts}))
            print(f'[GIADA Task39e] checkpoint_saved {step} before_selection',flush=True)
            common,parts,ranks,panel=checkpoint_scores(torch,cohorts,dev,held,normal,spec,device,quality)
            j=0
            for g in cohorts:
                for k in range(g['model'].a.shape[0]):
                    if ranks[j] is None:
                        invalid_members.add(j);j+=1;continue
                    if best[j] is None or ranks[j]<best[j]:
                        best[j]=ranks[j];chosen[j]=step;saved[j]={name:x[k].detach().cpu().clone() for name,x in g['model'].named_parameters()}
                    if common[j]<common_best[j]:common_best[j]=common[j];common_chosen[j]=step
                    j+=1
            curves.append(dict(step=step,common_development_score=common.tolist(),panel=panel,elapsed_seconds=time.perf_counter()-start,clipping_counts=[x.cpu().tolist() for x in clips]))
            p.save_json(output/'training_curves.json',diagnostic_json(curves))
            p.save_json(output/'numerical_health.json',dict(invalid_members=sorted(invalid_members),current_invalid=[j for j,r in enumerate(ranks) if r is None],invalid_checkpoints_are_never_selected=True))
            p.save_json(output/'checkpoint_phase.json',dict(step=step,phase='gradient_probe',common_finite=bool(np.isfinite(common).all())))
            for g,s in zip(cohorts,static):probes.append(gradient_probe(torch,g,s,scales,step)[0])
            p.save_json(output/'gradient_probes.json',diagnostic_json(probes))
            print(f'[GIADA Task39e] checkpoint {step} median_common={np.median(common):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
        if step==spec['checkpoints'][-1]:break
        idx=rng.integers(len(data[0]),size=2);extra=rng.random(2)<.5
        ti=torch.tensor(idx,device=device);ee=torch.tensor(extra,device=device)[:,None,None]
        batch=torch.where(ee,pools[1][ti],pools[0][ti]);ai=rng.integers(len(empirical),size=64);di=rng.integers(6,size=64)
        tt=torch.tensor(np.asarray(spec['held_ticks'])[di],device=device,dtype=torch.float32)[None,:,None]
        for g,s in zip(cohorts,static):
            s['batch'].copy_((batch if g['long'] else batch.reshape(8,64,-1))[None])
            s['anchor'].copy_(torch.where(g['flags'][:,1,None,None].bool(),anchors[1][ai][None],anchors[0][ai][None]));s['ticks'].copy_(tt)
        graph.replay();np.add.at(family_exposure,(idx//spec['windows_per_family'],extra.astype(int)),256)
        if (step+1)%2000==0:print(f'[GIADA Task39e] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-start;j=0
    with torch.no_grad():
        for g in cohorts:
            for k in range(g['model'].a.shape[0]):
                for name,x in g['model'].named_parameters():x[k].copy_(saved[j][name])
                j+=1
    torch.save({g['key']:g['model'].state_dict() for g in cohorts},output/'frozen_checkpoint.pt')
    freeze=dict(arms=defs,seeds=spec['seeds'],cohort_order=[g['key'] for g in cohorts],chosen_steps=chosen,ranks=best,common_rule_steps=common_chosen,development_only=True,stress_used_for_selection=False,independent_confirmation_accessed=False,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'))
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    p.save_json(output/'exposure.json',dict(variable_ticks_family_role=family_exposure.tolist(),variable_targets_per_member=512*spec['checkpoints'][-1],anchor_targets_per_member=64*spec['checkpoints'][-1]))
    base=json.loads((root/old.CONFIG).read_text());evaluation=dict(spec,architectures=[a['name'] for a in defs])
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=base[key]
    assert p.digest(root/base['baseline_parent'])==base['baseline_parent_sha256'] and p.digest(root/base['baseline_checkpoint'])==base['baseline_sha256']
    summary=old.evaluate(root,output,evaluation,split_models(torch,cohorts,spec['seeds'],device),torch,device,quality)
    rows=json.loads((output/'recurrence_evaluation.json').read_text());scores={}
    for a in defs:
        scores[a['name']]={}
        for seed in spec['seeds']:
            rr=[r for r in rows if r['model']=='joint_w'+a['name'] and r['role']=='development' and r['seed']==seed]
            vals=[q.severity(r['metrics']) for r in rr]
            scores[a['name']][str(seed)]=dict(median=float(np.median(vals)),worst=max(vals),passed=sum(r['metrics']['passed'] for r in rr))
    contrasts=[]
    for axis,a,z in contrast_pairs(defs):
        ratios=[scores[z][str(s)]['median']/max(scores[a][str(s)]['median'],1e-12) for s in spec['seeds']]
        worst=[scores[z][str(s)]['worst']/max(scores[a][str(s)]['worst'],1e-12) for s in spec['seeds']]
        contrasts.append(dict(axis=axis,control=a,treatment=z,paired_ratios=ratios,worst_ratios=worst,supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=contrasts))
    c.cost_benchmark(torch,cohorts,spec['seeds'],device,output,model_factory=model_class,splitter=split_models,task_label='39e')
    # Counterfactual selection uses the SAME saved trajectories, no refit/stress selection.
    common_dir=output/'common_selection_diagnostic';common_dir.mkdir()
    checkpoints={step:torch.load(output/f'checkpoint_{step}.pt',map_location=device,weights_only=True) for step in set(common_chosen)}
    j=0
    with torch.no_grad():
        for g in cohorts:
            for k in range(g['model'].a.shape[0]):
                for name,x in g['model'].named_parameters():x[k].copy_(checkpoints[common_chosen[j]][g['key']][name][k])
                j+=1
    common_summary=old.evaluate(root,common_dir,evaluation,split_models(torch,cohorts,spec['seeds'],device),torch,device,quality)
    p.save_json(output/'common_selection_diagnostic.json',dict(summaries=common_summary,selected_steps=common_chosen,selection_eligible=False,
        limits='Prespecified counterfactual selection of the same saved weights; no refit. Primary rule remains unchanged regardless of diagnostic result.'))
    unstable_arms={defs[j//len(spec['seeds'])]['name'] for j in invalid_members}
    passing=[a['name'] for a in defs if summary['joint_w'+a['name']]['all_development_passed'] and a['name'] not in unstable_arms]
    execution.update(training_wall_seconds=training,total_wall_seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated());p.save_json(output/'execution.json',execution)
    final=dict(valid=True,task='39e',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',passing_arms=passing,numerically_unstable_arms=sorted(unstable_arms),invalid_checkpoint_members=sorted(invalid_members),summaries=summary,thresholds_changed=False,independent_confirmation_accessed=False,speedup_claimed=False,limits=spec['limits'])
    p.save_json(output/'final_report.json',final);return final
