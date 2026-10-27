"""Task41l: same operator, factorial repair; compression intentionally parked."""
import copy
import hashlib
import json
import time
import numpy as np
from . import task41k_learning_alignment as k
from . import task41l_support as support
j=k.j
p,t,c,d,f=j.p,j.t,j.c,j.d,j.f
CONFIG='experiments/task41l_preregistration_v1.json'
verify_freeze=j.verify_freeze
path_prediction=k.path_prediction
feedback_prediction=k.feedback_prediction
path_errors=k.path_errors
feedback_errors=k.feedback_errors


def arms():
    return [dict(name=f'L_A{a}D{dd}P{pp}',representation='chain',width=64,task='repair',
                 amplitude=a,coverage=dd,protection=pp,lr=1e-5 if a==0 else 1e-6)
            for a in range(2) for dd in range(2) for pp in range(2)]


def contract(root):
    spec=json.loads((root/CONFIG).read_text());old,cfg,calcium,base=k.contract(root)
    assert spec['arms']==arms() and spec['seeds']==[17,29,43] and spec['dt_ms']==.1
    for key in ('quality','compartment_gates','cost'):assert spec[key]==old[key],key
    assert spec['checkpoints']==[0,16,64,256,1024,4096]
    assert not set(spec['new_seeds'].values())&set(old['new_seeds'].values())
    for name,digest in spec['sources'].items():assert p.digest(root/name)==digest,name
    return spec,cfg,calcium,base


def install_scanner():
    from .task41k_recovery import managed_scanner_class
    module=c.j.i.old
    if not getattr(module.Scanner,'_task41l_managed',False):
        module.Scanner=managed_scanner_class(module.Scanner);module.Scanner._task41l_managed=True


def setup(torch,root,out,device,spec,cfg,calcium,base,small=False):
    controls=j.controls(torch,root,device);parent=controls['H_nominee']
    models={a['name']:j.model_class(torch)(parent,a).to(device) for a in arms()}
    assert all(sum(x.numel() for x in m.parameters())//3==4004 for m in models.values())
    data,_=j.atomic_pool(torch,parent,spec['new_seeds']['fit'],16 if small else spec['anchors'],device)
    selection,_=j.atomic_pool(torch,parent,spec['new_seeds']['selection'],16 if small else spec['selection_anchors'],device)
    j.f.ee.differentiable_constants();physics=(cfg,calcium,base)
    pools=support.nested_pools(torch,spec,physics,device,small)
    sp=j.path_pool(torch,spec,device,selection=True,count=2 if small else None)
    local=dict(spec,selection_feedback_anchors=2) if small else spec
    sf=j.feedback_pool(torch,local,*physics,device,selection=True,small=small)
    k.prepare_targets(torch,parent,sp,sf,physics)
    samplers={};records={};cutoffs=None
    for level,(paths,fb) in pools.items():
        errors=k.prepare_targets(torch,parent,paths,fb,physics)
        pair=[support.family_samplers(e,pool['family'].cpu().numpy(),None if cutoffs is None else cutoffs[i]) for i,(e,pool) in enumerate(zip(errors,(paths,fb)))]
        if cutoffs is None:cutoffs=[row[1] for row in pair]
        samplers[level]=[row[0] for row in pair]
        records[str(level)]=dict(errors=[e.tolist() for e in errors],cutoffs=cutoffs,samplers=samplers[level],families=[pool['family'].tolist() for pool in (paths,fb)])
    p.save_json(out/'fit_targeting.json',dict(fit_only=True,levels=records,nested=True,cutoffs_calibrated_on_1x_only=True,uniform_family_probabilities=True))
    return controls,models,data,selection,pools,sp,sf,samplers


def update(torch,models,opts,pools,batches,physics,step_number):
    diagnostics={}
    for definition in arms():
        name=definition['name'];model=models[name];opt=opts[name]
        params=j.parameters(model);opt.zero_grad(set_to_none=False)
        lr=definition['lr']*.1**(min(step_number-1,1024)/1024)
        for group in opt.param_groups:group['lr']=lr
        before=[x.detach().clone() for x in params]
        start=time.perf_counter()
        total,diag=support.objective(torch,model,*pools[definition['coverage']],batches[definition['coverage']],physics,definition['protection'])
        total.sum().backward()
        assert bool(torch.isfinite(total).all()),name
        assert all(bool(torch.isfinite(x.grad).all()) for x in params if x.grad is not None),name
        j.clip_members(torch,params);opt.step()
        if params[0].is_cuda:torch.cuda.synchronize()
        norm=sum((x.detach()-old).double().reshape(3,-1).square().sum(1) for x,old in zip(params,before)).sqrt()
        diagnostics[name]=dict(diag,lr=lr,actual_adam_update_norm=norm.tolist(),seconds=time.perf_counter()-start)
    return diagnostics


def adam_probe(torch,model,opt,paths,fb,batch,physics,protect,next_lr=None):
    """Replay the real next Adam direction on a clone; never mutate the run."""
    before=copy.deepcopy(model.state_dict());optimizer_before=copy.deepcopy(opt.state_dict())
    probe=copy.deepcopy(model);params=j.parameters(probe)
    po=torch.optim.Adam(params);po.load_state_dict(copy.deepcopy(optimizer_before));po.zero_grad(set_to_none=False)
    if next_lr is not None:
        for group in po.param_groups:group['lr']=next_lr
    groups,names=support.group_losses(torch,probe,paths,fb,batch[2],batch[3],physics)
    channels=list(c.ionic.CHANNELS);channel_groups=[['Ca_HVA'],['Nap_Et2'],['K_Tst'],[n for n in channels if n not in ('Ca_HVA','Nap_Et2','K_Tst')]]
    by_group=[];flat=[];block_norm=[];block_gradients=[]
    for selected in channel_groups:
        indices=[i for i,name in enumerate(names) if name.split('/')[-1] in selected]
        grad=torch.autograd.grad(groups[:,indices].mean(1).sum(),params,retain_graph=True,allow_unused=True)
        pieces=[torch.zeros_like(par) if g is None else g for par,g in zip(params,grad)]
        flat.append(torch.cat([g.reshape(3,-1).double() for g in pieces],1))
        block_norm.append([g.reshape(3,-1).double().norm(dim=1).tolist() for g in pieces])
        block_gradients.append([g.reshape(3,-1).double() for g in pieces])
        by_group.append(selected)
    cosine=[]
    for a in flat:
        cosine.append([((a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)).clamp_min(1e-30)).tolist() for b in flat])
    block_cosines=[]
    for block in range(len(params)):
        vv=[grad[block] for grad in block_gradients]
        block_cosines.append([[((a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)).clamp_min(1e-30)).tolist() for b in vv] for a in vv])
    regime_gradients=[]
    for regime in range(15):
        grad=torch.autograd.grad(groups[:,regime*11:(regime+1)*11].mean(1).sum(),params,retain_graph=True,allow_unused=True)
        vector=torch.cat([(torch.zeros_like(par) if g is None else g).reshape(3,-1).double() for par,g in zip(params,grad)],1)
        regime_gradients.append(dict(regime=names[regime*11].split('/')[0],norm=vector.norm(dim=1).tolist(),channel_cosines=[((vector*g).sum(1)/(vector.norm(dim=1)*g.norm(dim=1)).clamp_min(1e-30)).tolist() for g in flat]))
    del groups
    total,_=support.objective(torch,probe,paths,fb,batch,physics,protect);total.sum().backward()
    j.clip_members(torch,params);po.step();after=copy.deepcopy(probe.state_dict())
    effects=[]
    with torch.no_grad():
        for fraction in (0.,.25,.5,1.):
            probe.load_state_dict({name:old+fraction*(after[name]-old) if old.is_floating_point() else old for name,old in before.items()})
            values,_=support.group_losses(torch,probe,paths,fb,batch[2],batch[3],physics)
            loss,_=support.objective(torch,probe,paths,fb,batch,physics,protect)
            effects.append(dict(fraction=fraction,objective=loss.tolist(),groups=values.tolist()))
    assert all(torch.equal(model.state_dict()[key],value) for key,value in before.items())
    def same(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[key],b[key]) for key in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    assert same(opt.state_dict(),optimizer_before)
    return dict(probe_learning_rate=po.param_groups[0]['lr'],group_names=names,channel_groups=by_group,gradient_cosines=cosine,parameter_block_names=[name for name,x in model.named_parameters() if x.requires_grad],gradient_block_norms=block_norm,gradient_block_cosines=block_cosines,regime_gradients=regime_gradients,finite_step_effects=effects,candidate_and_optimizer_unchanged=True,selection_eligible=False)

def run(root,out,device='cuda',smoke=False,resume=False,stop_after=None,evaluation_only=False):
    import torch
    from . import task41l_evaluation as ev
    spec,cfg,calcium,base=contract(root);install_scanner();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert device=='cuda' and torch.cuda.is_available();out.mkdir(parents=True,exist_ok=True)
    if resume:assert json.loads((out/'preregistration.json').read_text())==spec
    else:p.save_json(out/'preregistration.json',spec)
    if evaluation_only:
        assert resume;freeze=j.verify_freeze(out);cc=j.controls(torch,root,device)
        models={a['name']:j.model_class(torch)(cc['H_nominee'],a).to(device) for a in arms()}
        saved=torch.load(out/'frozen_checkpoint.pt',map_location=device,weights_only=True)
        for name,m in models.items():m.load_state_dict(saved[name]);m.eval()
        def forbidden(*args,**kwargs):raise RuntimeError('No training in evaluation-only phase')
        torch.optim.Adam.step=forbidden;ev.install_cache(out)
        with torch.inference_mode():report=ev.evaluate(torch,root,out,cc,models,spec,cfg,calcium,base,device)
        j.verify_freeze(out);report.update(training_updates_performed=0,selection_recomputed=False);return report
    cc,models,data,selection,pools,sp,sf,samplers=setup(torch,root,out,device,spec,cfg,calcium,base,smoke)
    opts={name:torch.optim.Adam(j.parameters(m),lr=next(a['lr'] for a in arms() if a['name']==name),eps=1e-8) for name,m in models.items()}
    rng=np.random.default_rng(spec['new_seeds']['stream']);physics=(cfg,calcium,base)
    if smoke:
        batches=support.minibatches(torch,rng,pools,samplers,spec,device)
        probe=adam_probe(torch,models[arms()[0]['name']],opts[arms()[0]['name']],*pools[0],batches[0],physics,False)
        sequential=copy.deepcopy(models)
        timed_update=update(torch,models,opts,pools,batches,physics,1);j.finite(torch,models,opts)
        print('[GIADA Task41l] all8 objectives finite; checking seed isolation',flush=True)
        seed_error=0.
        for definition in arms():
            name=definition['name'];level=definition['coverage']
            for seed in range(3):
                independent=copy.deepcopy(sequential[name]);params=j.parameters(independent)
                optimizer=torch.optim.Adam(params,lr=definition['lr'],eps=1e-8)
                loss,_=support.objective(torch,independent,*pools[level],batches[level],physics,definition['protection'])
                loss[seed].backward();j.clip_members(torch,params);optimizer.step()
                seed_error=max(seed_error,max(float((a[seed]-b[seed]).abs().max()) for a,b in zip(independent.parameters(),models[name].parameters())))
        assert seed_error<=spec['computational_parity_atol']
        copies=copy.deepcopy(models);co={n:torch.optim.Adam(j.parameters(m)) for n,m in copies.items()}
        torch.save(dict(models={n:m.state_dict() for n,m in models.items()},optimizer={n:o.state_dict() for n,o in opts.items()}),out/'preflight_resume.pt')
        disk=torch.load(out/'preflight_resume.pt',map_location=device,weights_only=True)
        for n in co:copies[n].load_state_dict(disk['models'][n]);co[n].load_state_dict(disk['optimizer'][n])
        update(torch,models,opts,pools,batches,physics,2);update(torch,copies,co,pools,batches,physics,2)
        error=max(float((a-b).abs().max()) for n in models for a,b in zip(models[n].parameters(),copies[n].parameters()))
        assert error<=spec['computational_parity_atol']
        result=dict(valid=True,all_parameters_finite=True,resume_parameter_error=error,all_eight_objectives_exercised=True,
                    per_arm_first_update=timed_update,estimated_update_only_hours_4096=sum(r['seconds'] for r in timed_update.values())*4096/3600,
                    sequential_vectorized_parameter_error=seed_error,serialized_resume_verified=True,probe_isolated=probe['candidate_and_optimizer_unchanged'],scientific_training_performed=False,
                    protocol_sha256=p.digest(root/CONFIG),thresholds_changed=False,members=24)
        p.save_json(out/'preflight.json',result);return result
    dev=d.development(spec['new_seeds']['development'])
    with torch.no_grad():parent_rows=c.shadow_evaluate(torch,[cc['H_nominee']],dev,spec['quality'],device)
    first=0;curves=[];best={k:[None]*3 for k in models};saved={k:[None]*3 for k in models};chosen={k:[None]*3 for k in models}
    if resume:
        ck=max(out.glob('checkpoint_*.pt'),key=lambda x:int(x.stem.split('_')[-1]));s=torch.load(ck,map_location=device,weights_only=False)
        assert s['protocol_sha256']==p.digest(root/CONFIG) and s['torch_version']==torch.__version__ and s['gpu']==torch.cuda.get_device_name()
        for name,m in models.items():m.load_state_dict(s['models'][name]);opts[name].load_state_dict(s['optimizer'][name])
        first=s['step'];rng.bit_generator.state=s['rng'];best=s['best'];saved=s['selected_weights'];chosen=s['chosen_steps'];curves=json.loads((out/'training_curves.json').read_text())
    else:
        np.savez_compressed(out/'development_inputs.npz',**dev)
    start=time.perf_counter();exposure={n:dict(path=[0,0],action=[0]*13,protection_path=[0,0],protection_action=[0]*13,seconds=0.) for n in models}
    if resume:exposure=s['exposure']
    for step_number in range(first,spec['checkpoints'][-1]+1):
        if step_number in spec['checkpoints'] and not (resume and step_number==first):
            j.finite(torch,models,opts)
            with torch.no_grad():
                measured=c.shadow_evaluate(torch,list(models.values()),dev,spec['quality'],device)
                points={n:j.coefficient_loss(torch,m,selection,torch.arange(spec['selection_anchors'],device=device),arms()[i]['task']=='compression') for i,(n,m) in enumerate(models.items())}
                # Small, disjoint heldout dynamics pool (16 paths/8 anchors).
                auxiliary={};teacher_error={}
                for n,m in models.items():
                    pi=torch.arange(len(sp['v']),device=device);fi=torch.arange(len(sf['v']),device=device)
                    pr=path_prediction(torch,m,sp,pi);fr=feedback_prediction(torch,m,sf,fi,physics)
                    auxiliary[n]=path_errors(torch,pr,sp,pi,reduce=True)+feedback_errors(torch,fr,sf,fi).mean(1)
                    teacher_error[n]=path_errors(torch,pr,sp,pi,True,reduce=True)+feedback_errors(torch,fr,sf,fi,True).mean(1)
            # Probe uses a separate deterministic RNG, not the training stream.
            probe_rng=np.random.default_rng(spec['new_seeds']['probe']+step_number)
            probe_batches=support.minibatches(torch,probe_rng,pools,samplers,spec,device)
            probes={}
            for definition in arms():
                name=definition['name'];level=definition['coverage']
                probes[name]=adam_probe(torch,models[name],opts[name],*pools[level],probe_batches[level],physics,definition['protection'],definition['lr']*.1**(min(step_number,1024)/1024))
            p.save_json(out/f'optimizer_probe_{step_number}.json',probes)
            ranks={}
            for ai,(name,m) in enumerate(models.items()):
                ranks[name]=[]
                for s in range(3):
                    if arms()[ai]['task']=='compression':
                        current=(float(teacher_error[name][s]),float(points[name][s]))
                    else:current=j.rank(measured[ai*3+s],parent_rows[s],auxiliary[name][s],points[name][s],spec['quality'])
                    ranks[name].append(current)
                    if best[name][s] is None or current<best[name][s]:
                        best[name][s]=current;chosen[name][s]=step_number
                        saved[name][s]={key:value[s:s+1].detach().cpu().clone() if value.ndim and value.shape[0]==3 else value.detach().cpu().clone() for key,value in m.state_dict().items()}
            curves.append(dict(step=step_number,ranks=ranks,coefficient={n:v.tolist() for n,v in points.items()},canonical_dynamic={n:v.tolist() for n,v in auxiliary.items()},parent_dynamic={n:v.tolist() for n,v in teacher_error.items()},elapsed_seconds=time.perf_counter()-start,exposure=copy.deepcopy(exposure)))
            p.save_json(out/'training_curves.json',curves);p.save_json(out/f'development_{step_number}.json',measured)
            torch.save(dict(step=step_number,exposure=exposure,models={n:m.state_dict() for n,m in models.items()},optimizer={n:o.state_dict() for n,o in opts.items()},rng=rng.bit_generator.state,best=best,selected_weights=saved,chosen_steps=chosen,protocol_sha256=p.digest(root/CONFIG),torch_version=torch.__version__,gpu=torch.cuda.get_device_name()),out/f'checkpoint_{step_number}.pt')
            print(f'[GIADA Task41l] checkpoint {step_number} finite=True',flush=True)
        if stop_after==step_number and step_number<spec['checkpoints'][-1]:return dict(valid=True,decision='TRAINING_STAGE_COMPLETE',updates_completed=step_number,training_complete=False,thresholds_changed=False)
        if step_number==spec['checkpoints'][-1]:break
        batches=support.minibatches(torch,rng,pools,samplers,spec,device)
        diagnostics=update(torch,models,opts,pools,batches,physics,step_number+1)
        for definition in arms():
            name=definition['name'];level=definition['coverage'];paths,fb=pools[level];pi,fi,ppi,ffi=batches[level]
            for field,pool,ids in [('path',paths,pi),('action',fb,fi),('protection_path',paths,ppi),('protection_action',fb,ffi)]:
                for group in pool['family'][ids].tolist():exposure[name][field][group]+=1
            exposure[name]['seconds']+=diagnostics[name]['seconds']
        if (step_number+1)%16==0:p.save_json(out/'latest_update.json',dict(step=step_number+1,arms=diagnostics))
        if (step_number+1)%32==0:print(f'[GIADA Task41l] update {step_number+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    frozen={n:j.merge_selected(torch,saved[n],m.state_dict()) for n,m in models.items()}
    torch.save(frozen,out/'frozen_checkpoint.pt')
    repair=[a['name'] for a in arms() if a['task']=='repair'];nominee=min(repair,key=lambda n:tuple(max(best[n][s][k] for s in range(3)) for k in range(5))+(repair.index(n),))
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen,ranks=best,nominee=nominee,checkpoint_sha256=p.digest(out/'frozen_checkpoint.pt'),opened_regressions_used_for_selection=False,independent_confirmation=False,parent='H_nominee',compression_not_eligible_for_repair_nomination=True)
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(out/'selection_freeze.json',freeze)
    return dict(valid=True,decision='FROZEN_TRAINING_COMPLETE',updates_completed=spec['checkpoints'][-1],training_complete=True,thresholds_changed=False)


