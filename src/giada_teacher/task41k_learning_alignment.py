"""Separate repair from compression; frozen scientific predicates from 41j."""
import copy
import hashlib
import json
import time
import numpy as np
from . import task41j_operator_reset as j

p,t,c,d,f=j.p,j.t,j.c,j.d,j.f
CONFIG='experiments/task41k_preregistration_v1.json'
verify_freeze=j.verify_freeze


def arms():
    return [dict(name='K_point',representation='chain',width=64,task='repair',objective='point',targeted=False,retain=False),
            dict(name='K_oldmix',representation='chain',width=64,task='repair',objective='oldmix',targeted=False,retain=False),
            *[dict(name='K_dyn_'+('target' if target else 'uniform')+('_retain' if retain else ''),representation='chain',width=64,task='repair',objective='dynamic',targeted=target,retain=retain) for target in (False,True) for retain in (False,True)],
            dict(name='K_copy_coeff',representation='compact',width=64,task='compression',objective='teacher_coeff',targeted=False,retain=False),
            dict(name='K_copy_dynamic',representation='compact',width=64,task='compression',objective='teacher_dynamic',targeted=False,retain=False)]


def contract(root):
    spec=json.loads((root/CONFIG).read_text());old,cfg,calcium,base=j.contract(root)
    assert spec['arms']==arms() and spec['seeds']==[17,29,43] and spec['dt_ms']==.1
    for key in ('quality','compartment_gates','cost'):assert spec[key]==old[key],key
    assert not set(spec['new_seeds'].values())&set(old['new_seeds'].values())
    for name,digest in spec['sources'].items():assert p.digest(root/name)==digest,name
    return spec,cfg,calcium,base


def path_prediction(torch,model,data,ids):
    get=lambda k:j.expand(torch,data[k][ids]);v=get('v');ca=get('ca')[...,None].expand_as(v)
    u,z=model.raw(v,ca);states=j.affine_scan(torch,torch.sigmoid(u.double()),torch.sigmoid(z.double()),get('initial'))
    times=get('times').long();T=v.shape[-1]
    before=states.gather(2,times.clamp_max(T-1)[...,None].expand(3,len(ids),times.shape[-1],18))
    ru,rz=model.raw(get('rest'),get('ca'));ticks=(times+1-T).clamp_min(0)
    decay=torch.exp(torch.nn.functional.logsigmoid(-rz.double())[:,:,None]*ticks[...,None])
    inf=torch.sigmoid(ru.double());tail=inf[:,:,None]+(states[:,:,-1,None]-inf[:,:,None])*decay
    gates=torch.where((times<T)[...,None],before,tail)
    return t.readout(torch,get('query_v'),gates,get('mult')[:,:,None])[1]


def feedback_prediction(torch,model,data,ids,physics):
    get=lambda k:j.expand(torch,data[k][ids]).reshape(3,len(ids)*2,*data[k].shape[2:])
    v,ca,g=(get(k) for k in ('v','ca','g'));mult,inj,syn=(get(k) for k in ('mult','inj','syn'))
    core=t.make_core(torch,*physics,100*np.pi,j.TrainingView(model),t.FAMILIES['all']);rows=[]
    for tick in range(40):
        v,ca,g,current=core(v,ca,g,mult,inj,syn)
        pair=current.reshape(3,len(ids),2,11);rows.append(pair[:,:,1]-pair[:,:,0])
    return torch.stack(rows,1)


def path_errors(torch,pred,data,ids,teacher=False,reduce=False):
    target=data['parent'][:,ids] if teacher else data['target'][ids][None]
    error=pred-target
    ordinary=(error/data['scale'][ids][None,:,None]).square().mean((2,3))
    diagonal=error[:,:,8:].diagonal(dim1=2,dim2=3)
    numerator=((diagonal/data['timing_scale'][ids][None]).square().clamp_max(1e6)*data['valid'][ids][None]).sum(-1)
    counts=data['valid'][ids].sum(-1)
    if reduce:return ordinary.mean(1)+.01*numerator.sum(1)/counts.sum().clamp_min(1)
    timing=numerator/counts.clamp_min(1)[None]
    return ordinary+.01*timing


def feedback_errors(torch,pred,data,ids,teacher=False):
    target=data['parent'][:,:,ids] if teacher else data['target'][:40,ids][None]
    return ((pred-target)/data['scale'][ids][None,None]).square().mean((1,3))


def prepare_targets(torch,parent,paths,fb,physics):
    """Parent targets and targeting weights depend on FIT only, never tests."""
    parent=j.model_class(torch)(parent,arms()[0])
    with torch.no_grad():
        paths['parent']=torch.cat([path_prediction(torch,parent,paths,torch.arange(i,min(i+8,len(paths['v'])),device=paths['v'].device)) for i in range(0,len(paths['v']),8)],1)
        fb['parent']=torch.cat([feedback_prediction(torch,parent,fb,torch.arange(i,min(i+4,len(fb['v'])),device=fb['v'].device),physics) for i in range(0,len(fb['v']),4)],2)
        pi=torch.arange(len(paths['v']),device=paths['v'].device);fi=torch.arange(len(fb['v']),device=fb['v'].device)
        pe=path_errors(torch,paths['parent'],paths,pi);fe=feedback_errors(torch,fb['parent'],fb,fi)
        # Training-only replay masks: lower-half FIT discrepancy per seed.
        # These are NOT claims of full-trajectory qualification or changed gates.
        paths['retain_mask']=(pe<=pe.median(1,keepdim=True).values).double()
        fb['retain_mask']=(fe<=fe.median(1,keepdim=True).values).double()
    return pe.mean(0).cpu().numpy(),fe.mean(0).cpu().numpy()


def targeted_cdf(errors):
    scores=np.asarray(errors);assert np.isfinite(scores).all()
    top=np.argsort(scores,kind='stable')[-max(1,len(scores)//4):]
    weights=np.full(len(scores),.5/len(scores));weights[top]+=.5/len(top)
    return np.cumsum(weights)


def paired_ids(torch,rng,n,count,cdf,device):
    u=rng.random(count)
    return (torch.tensor(np.minimum((u*n).astype('int64'),n-1),device=device),
            torch.tensor(np.searchsorted(cdf,u).clip(0,n-1),device=device))


def dynamic_losses(torch,model,paths,fb,pi,fi,physics,teacher=False,retain=False):
    pp=path_prediction(torch,model,paths,pi);fp=feedback_prediction(torch,model,fb,fi,physics)
    loss=path_errors(torch,pp,paths,pi,teacher,reduce=True)+feedback_errors(torch,fp,fb,fi,teacher).mean(1)
    replay=torch.zeros_like(loss)
    if retain:
        pe=path_errors(torch,pp,paths,pi,True);fe=feedback_errors(torch,fp,fb,fi,True)
        pm=paths['retain_mask'][:,pi];fm=fb['retain_mask'][:,fi]
        replay=(pe*pm).sum(1)/pm.sum(1).clamp_min(1)+(fe*fm).sum(1)/fm.sum(1).clamp_min(1)
    return loss,replay


def update(torch,models,opts,data,ids,paths,fb,pair_pi,pair_fi,physics,index,probe=False):
    diagnostics={}
    for definition in arms():
        name=definition['name'];model=models[name];params=j.parameters(model);opt=opts[name];opt.zero_grad(set_to_none=False)
        objective=definition['objective'];target=definition['targeted'];pi=pair_pi[int(target)];fi=pair_fi[int(target)]
        if objective in ('point','teacher_coeff') or (objective=='oldmix' and index%16):
            loss=j.coefficient_loss(torch,model,data,ids,objective=='teacher_coeff');loss.sum().backward()
        elif objective=='oldmix':
            point=j.coefficient_loss(torch,model,data,ids)
            gp=torch.autograd.grad(point.sum(),params,allow_unused=True)
            dyn,_=dynamic_losses(torch,model,paths,fb,pi,fi,physics)
            ga=torch.autograd.grad(dyn.sum(),params,allow_unused=True)
            pn=sum(x.double().reshape(3,-1).square().sum(1) for x in gp if x is not None).sqrt()
            an=sum(x.double().reshape(3,-1).square().sum(1) for x in ga if x is not None).sqrt()
            w=(.1*pn/an.clamp_min(1e-20)).clamp_max(1e6).detach()
            for par,a,b in zip(params,gp,ga):
                grad=(torch.zeros_like(par) if a is None else a)+(torch.zeros_like(par) if b is None else b*w.reshape(3,*([1]*(par.ndim-1))).to(par))
                par.grad=grad
            loss=point+dyn*w
        else:
            loss,replay=dynamic_losses(torch,model,paths,fb,pi,fi,physics,objective=='teacher_dynamic',definition['retain'])
            loss=loss+replay # preregistered unit replay weight, no after-the-fact tuning
            loss.sum().backward()
        assert torch.isfinite(loss).all(),name
        assert all(torch.isfinite(par.grad).all() for par in params if par.grad is not None),name
        j.clip_members(torch,params);opt.step()
        if probe:diagnostics[name]=loss.detach().tolist()
    return diagnostics


def gradient_probe(torch,parent,data,ids,paths,fb,pi,fi,physics):
    model=j.model_class(torch)(parent,arms()[0]);params=j.parameters(model)
    point=j.coefficient_loss(torch,model,data,ids);gp=torch.autograd.grad(point.sum(),params,allow_unused=True)
    dyn,_=dynamic_losses(torch,model,paths,fb,pi,fi,physics);ga=torch.autograd.grad(dyn.sum(),params,allow_unused=True)
    flat=lambda grad:torch.cat([(torch.zeros_like(par) if g is None else g).reshape(3,-1).double() for par,g in zip(params,grad)],1)
    a,b=flat(gp),flat(ga);an=a.norm(dim=1);bn=b.norm(dim=1)
    directions={'point':a/an[:,None].clamp_min(1e-30),'dynamic':b/bn[:,None].clamp_min(1e-30)}
    original=copy.deepcopy(model.state_dict());moves=[]
    for objective,g in directions.items():
        for amount in (1e-5,1e-4,1e-3):
            model.load_state_dict(original);offset=0
            with torch.no_grad():
                for par in params:
                    n=par[0].numel();par.sub_(amount*g[:,offset:offset+n].reshape_as(par).to(par));offset+=n
                q=j.coefficient_loss(torch,model,data,ids);v,_=dynamic_losses(torch,model,paths,fb,pi,fi,physics)
            moves.append(dict(direction=objective,parameter_step_norm=amount,coefficient_loss=q.tolist(),dynamic_loss=v.tolist()))
    return dict(coefficient_norm=an.tolist(),dynamic_norm=bn.tolist(),cosine=((a*b).sum(1)/(an*bn).clamp_min(1e-30)).tolist(),
                base_coefficient=point.tolist(),base_dynamic=dyn.tolist(),finite_differences=moves,scientific_candidate_retained=False)


def setup(torch,root,out,device,spec,cfg,calcium,base,small=False):
    controls=j.controls(torch,root,device);parent=controls['H_nominee'];models={a['name']:j.model_class(torch)(parent,a).to(device) for a in arms()}
    data,raw=j.atomic_pool(torch,parent,spec['new_seeds']['fit'],16 if small else spec['anchors'],device)
    selection,_=j.atomic_pool(torch,parent,spec['new_seeds']['selection'],16 if small else spec['selection_anchors'],device)
    paths=j.path_pool(torch,spec,device,count=2 if small else None);sp=j.path_pool(torch,spec,device,selection=True,count=2 if small else None)
    j.f.ee.differentiable_constants()
    local=dict(spec)
    if small:local.update(feedback_anchors=2,selection_feedback_anchors=2)
    fb=j.feedback_pool(torch,local,cfg,calcium,base,device,small=small)
    sf=j.feedback_pool(torch,local,cfg,calcium,base,device,selection=True,small=small)
    errors=prepare_targets(torch,parent,paths,fb,(cfg,calcium,base));prepare_targets(torch,parent,sp,sf,(cfg,calcium,base))
    return controls,models,data,selection,paths,sp,fb,sf,errors


def run(root,out,device='cuda',smoke=False,resume=False,stop_after=None,evaluation_only=False):
    import torch
    from . import task41k_evaluation as ev
    spec,cfg,calcium,base=contract(root);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
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
    cc,models,data,selection,paths,sp,fb,sf,errors=setup(torch,root,out,device,spec,cfg,calcium,base,smoke)
    opts={name:torch.optim.Adam(j.parameters(m),lr=spec['lr'],eps=1e-8) for name,m in models.items()}
    rng=np.random.default_rng(spec['new_seeds']['stream']);physics=(cfg,calcium,base)
    cdfs=[targeted_cdf(e) for e in errors]
    if smoke:
        ids=torch.arange(2,device=device);pi=(ids,ids);fi=(torch.arange(1,device=device),)*2
        # Validate identical repair starts and exact teacher loss before mutation.
        v=data['v'][ids].reshape(1,-1).expand(3,-1);ca=data['ca'][ids].reshape(1,-1).expand_as(v)
        assert all(torch.equal(a,b) for a,b in zip(cc['H_nominee'].raw(v,ca),models['K_point'].raw(v,ca)))
        parent_dyn,_=dynamic_losses(torch,models['K_point'],paths,fb,pi[0],fi[0],physics,True)
        assert float(parent_dyn.max())<spec['computational_parity_atol']**2
        update(torch,models,opts,data,ids,paths,fb,pi,fi,physics,16)
        j.finite(torch,models,opts)
        # Serialization/resumption and paired update equality for every objective.
        copies=copy.deepcopy(models);co={n:torch.optim.Adam(j.parameters(m),lr=spec['lr']) for n,m in copies.items()}
        for n in co:co[n].load_state_dict(copy.deepcopy(opts[n].state_dict()))
        update(torch,models,opts,data,ids,paths,fb,pi,fi,physics,17);update(torch,copies,co,data,ids,paths,fb,pi,fi,physics,17)
        err=max(float((a-b).abs().max()) for n in models for a,b in zip(models[n].parameters(),copies[n].parameters()))
        assert err<=spec['computational_parity_atol']
        result=dict(valid=True,all_parameters_finite=True,resume_parameter_error=err,all_eight_objectives_exercised=True,exact_parent_teacher_targets=True,
                    scientific_training_performed=False,protocol_sha256=p.digest(root/CONFIG),thresholds_changed=False,members=24)
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
        p.save_json(out/'fit_targeting.json',dict(path_errors=errors[0].tolist(),feedback_errors=errors[1].tolist(),cdf=[a.tolist() for a in cdfs],fit_only=True))
        probe=gradient_probe(torch,cc['H_nominee'],data,torch.arange(8,device=device),paths,fb,torch.arange(4,device=device),torch.arange(2,device=device),physics)
        p.save_json(out/'gradient_alignment_probe.json',probe)
    start=time.perf_counter()
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
            curves.append(dict(step=step_number,ranks=ranks,coefficient={n:v.tolist() for n,v in points.items()},canonical_dynamic={n:v.tolist() for n,v in auxiliary.items()},parent_dynamic={n:v.tolist() for n,v in teacher_error.items()},elapsed_seconds=time.perf_counter()-start))
            p.save_json(out/'training_curves.json',curves);p.save_json(out/f'development_{step_number}.json',measured)
            torch.save(dict(step=step_number,models={n:m.state_dict() for n,m in models.items()},optimizer={n:o.state_dict() for n,o in opts.items()},rng=rng.bit_generator.state,best=best,selected_weights=saved,chosen_steps=chosen,protocol_sha256=p.digest(root/CONFIG),torch_version=torch.__version__,gpu=torch.cuda.get_device_name()),out/f'checkpoint_{step_number}.pt')
            print(f'[GIADA Task41k] checkpoint {step_number} finite=True',flush=True)
        if stop_after==step_number and step_number<spec['checkpoints'][-1]:return dict(valid=True,decision='TRAINING_STAGE_COMPLETE',updates_completed=step_number,training_complete=False,thresholds_changed=False)
        if step_number==spec['checkpoints'][-1]:break
        rate=spec['lr']*(spec['lr_min_ratio']**(step_number/spec['checkpoints'][-1]))
        for opt in opts.values():opt.param_groups[0]['lr']=rate
        ids=torch.tensor(rng.integers(0,spec['anchors'],spec['batch']),device=device)
        pi=paired_ids(torch,rng,len(paths['v']),spec['tail_batch'],cdfs[0],device)
        fi=paired_ids(torch,rng,len(fb['v']),spec['feedback_batch'],cdfs[1],device)
        update(torch,models,opts,data,ids,paths,fb,pi,fi,physics,step_number+1)
        if (step_number+1)%32==0:print(f'[GIADA Task41k] update {step_number+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    frozen={n:j.merge_selected(torch,saved[n],m.state_dict()) for n,m in models.items()}
    torch.save(frozen,out/'frozen_checkpoint.pt')
    repair=[a['name'] for a in arms() if a['task']=='repair'];nominee=min(repair,key=lambda n:tuple(max(best[n][s][k] for s in range(3)) for k in range(5))+(repair.index(n),))
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen,ranks=best,nominee=nominee,checkpoint_sha256=p.digest(out/'frozen_checkpoint.pt'),opened_regressions_used_for_selection=False,independent_confirmation=False,parent='H_nominee',compression_not_eligible_for_repair_nomination=True)
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(out/'selection_freeze.json',freeze)
    return dict(valid=True,decision='FROZEN_TRAINING_COMPLETE',updates_completed=spec['checkpoints'][-1],training_complete=True,thresholds_changed=False)
