"""Task41e: isolated vectorized learning, shared controls and auditable budgets."""
import copy
import hashlib
import json
import time
import numpy as np
from . import task41e_transfer_model as m
from . import task41e_transfer_data as data_module
p, t, c = m.p, m.t, m.c


def member_indices(definitions, predicate):
    return [3*k+s for k,a in enumerate(definitions) if predicate(a) for s in range(3)]


def feedback_groups(torch, parent, model, definitions, device):
    groups=[]
    for ticks in (4,40):
        selected=[a for a in definitions if a['feedback'] and a['ticks']==ticks]
        indices=member_indices(definitions,lambda a:a['feedback'] and a['ticks']==ticks)
        if not indices:
            continue
        index=torch.tensor(indices,device=device)
        view=m.MemberView(torch,parent,model,selected,index,device,scenarios=True)
        groups.append(dict(ticks=ticks,definitions=selected,indices=indices,index=index,view=view))
    return groups


def feedback_terms(torch, model, groups, data, supports, batch_indices, cfg, calcium, base):
    loss_by_member={}
    for group in groups:
        count=len(group['indices']); batch=len(batch_indices)
        ids=torch.stack([supports[a['support']][batch_indices] for a in group['definitions'] for _ in range(3)])
        def expanded(key):
            value=data[key][ids]
            value=value[:,:,None].expand(count,batch,2,*value.shape[2:])
            return value.reshape(count,batch*4,*value.shape[4:])
        vv,cc,gg=(expanded(k) for k in ('v','ca','g'))
        mult,inj,syn=(expanded(k) for k in ('mult','inj','syn'))
        core=t.make_core(torch,cfg,calcium,base,100*np.pi,group['view'],t.FAMILIES['all'])
        ticks=group['ticks'];target=data['target'][:ticks]
        scale=target.abs().amax(0).clamp_min(1e-7)[ids]
        loss=torch.zeros(count,2,device=vv.device,dtype=torch.float64)
        for tick in range(ticks):
            vv,cc,gg,current=core(vv,cc,gg,mult,inj,syn)
            current=current[...,list(m.CHANNELS)].reshape(count,batch,2,2,2).double()
            delta=current[:,:,:,1]-current[:,:,:,0]
            error=((delta-target[tick][ids][:,:,None])/scale[:,:,None]).square()
            loss+=error.mean((1,3))
        joint=group['view'].joint
        # A joint rollout is duplicated only to keep one static sample layout.
        # Average those duplicates; independent heads each get their own loss.
        aggregate=torch.where(joint,loss.mean(1),loss.sum(1))/ticks
        for local,global_index in enumerate(group['indices']):
            loss_by_member[global_index]=aggregate[local]
    zero=model.transfer_correction.sum()*0
    return torch.stack([loss_by_member.get(k,zero) for k in range(len(model.transfer_correction))])


def deployed_feedback(torch,parent,model,definitions,data,ids,cfg,calcium,base,device,ticks=40):
    view=m.MemberView(torch,parent,model,definitions,torch.arange(len(model.transfer_correction),device=device),device)
    count=len(model.transfer_correction);batch=len(ids)
    def expand(key):
        value=data[key][ids][None].expand(count,*data[key][ids].shape)
        return value.reshape(count,2*batch,*value.shape[3:])
    vv,cc,gg=(expand(k) for k in ('v','ca','g'))
    mult,inj,syn=(expand(k) for k in ('mult','inj','syn'))
    core=t.make_core(torch,cfg,calcium,base,100*np.pi,view,t.FAMILIES['all'])
    scale=data['target'][:ticks,ids].abs().amax(0).clamp_min(1e-7)
    loss=torch.zeros(count,device=device,dtype=torch.float64)
    for tick in range(ticks):
        vv,cc,gg,current=core(vv,cc,gg,mult,inj,syn)
        pair=current[...,list(m.CHANNELS)].reshape(count,batch,2,2).double()
        delta=pair[:,:,1]-pair[:,:,0]
        loss+=((delta-data['target'][tick,ids])/scale).square().mean((1,2))
    return loss/ticks


def protection_terms(torch, model, data, idx):
    members=len(model.transfer_correction)
    seed=torch.arange(members,device=idx.device)%3
    v=data['v'][idx][None].expand(members,-1)
    extra=m.correction(torch,v,model.transfer_correction)
    raw=m.add_correction(torch,data['base_u'][seed][:,idx],extra)
    rz=data['base_z'][seed][:,idx].double()
    initial=data['initial'][idx][None].expand(members,-1,-1)
    state=initial[:,:,None]-torch.expm1(data['ticks'][None,None,:,None]*torch.nn.functional.logsigmoid(-rz)[:,:,None])*(torch.sigmoid(raw).double()[:,:,None]-initial[:,:,None])
    multipliers=torch.ones(*state.shape[:-1],11,device=v.device,dtype=state.dtype)
    _,current=t.readout(torch,v.double()[:,:,None].expand(*state.shape[:-1]),state,multipliers)
    error=(current[...,list(m.CHANNELS)]-data['target'][idx][None])/data['scale'][idx][None,:,None]
    return error.square().mean((1,2,3))


def point_gradient(torch,model,atomic,idx,censored,norm):
    point,secant=m.atomic_terms(torch,model.transfer_correction,atomic,idx,censored)
    grad=torch.autograd.grad((point/norm).sum(),model.transfer_correction)[0]
    return point.detach(),secant.detach(),grad


def auxiliary_gradients(torch,model,groups,fb,supports,fi,tail,ti,cfg,calcium,base):
    feedback=feedback_terms(torch,model,groups,fb,supports,fi,cfg,calcium,base)
    fg=torch.autograd.grad(feedback.sum(),model.transfer_correction)[0]
    protection=protection_terms(torch,model,tail,ti)
    pg=torch.autograd.grad(protection.sum(),model.transfer_correction)[0]
    return feedback.detach(),protection.detach(),fg,pg


def initialize_balance(torch,model,groups,atomic,idx,censored,fb,supports,fi,tail,ti,cfg,calcium,base,spec):
    point,_=m.atomic_terms(torch,model.transfer_correction,atomic,idx,censored)
    norm=point.detach().clamp_min(1e-12)
    _,_,g0=point_gradient(torch,model,atomic,idx,censored,norm)
    _,_,gf,gp=auxiliary_gradients(torch,model,groups,fb,supports,fi,tail,ti,cfg,calcium,base)
    n0,nf,np_=(g.double().norm(dim=-1) for g in (g0,gf,gp))
    dynamic=torch.tensor([a['balance']=='dynamic' for a in model.definitions for _ in range(3)],device=idx.device)
    active=torch.tensor([a['feedback'] for a in model.definitions for _ in range(3)],device=idx.device)[:,None]
    protected=torch.tensor([a['protection'] for a in model.definitions for _ in range(3)],device=idx.device)[:,None]
    weights_f=m.balance_weights(torch,n0,nf,None,spec,dynamic)*active
    weights_p=m.balance_weights(torch,n0,np_,None,spec,dynamic)*protected
    state=dict(feedback=weights_f,protection=weights_p,
               feedback_ema=torch.stack((n0,nf),-1),protection_ema=torch.stack((n0,np_),-1),
               dynamic=dynamic,active=active,protected=protected)
    return norm,state


def clip_heads(torch,gradient):
    norm=gradient.double().norm(dim=-1)
    gradient.mul_((1/norm.clamp_min(1))[...,None].to(gradient.dtype))
    return norm


def atomic_step(torch,model,opt,atomic,idx,censored,norm):
    opt.zero_grad(set_to_none=False)
    point,_=m.atomic_terms(torch,model.transfer_correction,atomic,idx,censored)
    (point/norm).sum().backward()
    clip_heads(torch,model.transfer_correction.grad)
    opt.step()


def feedback_step(torch,model,opt,groups,atomic,idx,censored,norm,fb,supports,fi,tail,ti,cfg,calcium,base,spec,state):
    point,secant,g0=point_gradient(torch,model,atomic,idx,censored,norm)
    response,protection,gf,gp=auxiliary_gradients(torch,model,groups,fb,supports,fi,tail,ti,cfg,calcium,base)
    n0,nf,np_=(g.double().norm(dim=-1) for g in (g0,gf,gp))
    for key,aux_norm in (('feedback',nf),('protection',np_)):
        state[key].copy_(m.balance_weights(torch,n0,aux_norm,state[key],spec,state['dynamic'],state[key+'_ema']))
    state['feedback'].mul_(state['active']);state['protection'].mul_(state['protected'])
    gradient=g0+gf*state['feedback'][...,None]+gp*state['protection'][...,None]
    assert bool(torch.isfinite(gradient).all()),'Nonfinite gradient before Adam'
    opt.zero_grad(set_to_none=False)
    model.transfer_correction.grad.copy_(gradient)
    clipped=clip_heads(torch,model.transfer_correction.grad)
    before=model.transfer_correction.detach().clone()
    opt.step()
    assert bool(torch.isfinite(model.transfer_correction).all()),'Nonfinite correction after Adam'
    cosine=(g0.double()*gf.double()).sum(-1)/(n0*nf).clamp_min(1e-30)
    return dict(point=point,secant=secant,response=response,protection=protection,
                point_gradient=n0,feedback_gradient=nf,protection_gradient=np_,
                weighted_feedback_gradient=nf*state['feedback'],weighted_protection_gradient=np_*state['protection'],
                feedback_point_cosine=cosine,unclipped_norm=clipped,
                parameter_update_norm=(model.transfer_correction.detach()-before).double().norm(dim=-1))


def capture_atomic(torch,model,opt,atomic,idx,censored,norm):
    original=model.transfer_correction.detach().clone()
    for _ in range(3):
        atomic_step(torch,model,opt,atomic,idx,censored,norm)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        atomic_step(torch,model,opt,atomic,idx,censored,norm)
    with torch.no_grad():
        model.transfer_correction.copy_(original)
        for state in opt.state.values():
            for value in state.values():
                if torch.is_tensor(value):value.zero_()
    return graph


def channel_severity(metric,q,channels):
    values=[]
    for k in channels:
        row=metric['channels'][k]
        values.extend([row['nrmse']/q['current_nrmse_max'],row['nmax']/q['current_nmax_max'],
                       row['signed_charge_normalized']/q['signed_charge_error_max'],
                       row['absolute_charge_normalized']/q['absolute_charge_error_max']])
        event=row['event']
        if event['status']=='missing_event':values.append(1e12)
        elif event['status']=='evaluable':
            values.extend([event['peak_error_ms']/q['peak_timing_max_ms'],
                           event['onset_error_ms']/q['onset_offset_max_ms'],event['offset_error_ms']/q['onset_offset_max_ms']])
    return max(values)


def ranks(rows,parent_rows,response,atomic,q):
    target_failed=sum(channel_severity(r,q,m.CHANNELS)>1 for r in rows)
    target_worst=max(channel_severity(r,q,m.CHANNELS) for r in rows)
    global_failed=sum(not r['passed'] for r in rows)
    global_worst=max(c.severity(r,q) for r in rows)
    # First protect every individual previously passing current/channel gate,
    # then rank the intended target. Unchanged channels stay in final gates.
    introduced=sum(channel_severity(r,q,(k,))>1 and channel_severity(b,q,(k,))<=1
                   for r,b in zip(rows,parent_rows) for k in range(11))
    introduced+=sum(not r['gate']['passed'] and b['gate']['passed'] for r,b in zip(rows,parent_rows))
    primary=(introduced,target_failed,target_worst,float(response),float(atomic),global_failed,global_worst)
    secondary=(global_failed,global_worst,float(response),float(atomic))
    return primary,secondary


def serializable(value):
    if hasattr(value,'detach'):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    return value


def preflight(torch,root,output,device,spec):
    _,cfg,calcium,base=m.contract(root)
    parent=m.d.prior.load_models(torch,root,device)['P110']
    model=m.make_model(torch,parent,m.arms(),device)
    atomic,_=m.atomic_data(torch,parent,spec,device,count=32)
    censored=torch.tensor([a['objective']=='censored' for a in m.arms() for _ in range(3)],device=device)
    idx=torch.arange(8,device=device)[None].expand(36,-1).clone()
    fb,supports,_,_=data_module.prepare_feedback(torch,spec,cfg,calcium,base,device,small=True)
    groups=feedback_groups(torch,parent,model,m.arms(),device)
    fi=torch.arange(2,device=device);tail=data_module.held_tail_data(torch,parent,spec,device,16);ti=torch.arange(4,device=device)
    norm,state=initialize_balance(torch,model,groups,atomic,idx,censored,fb,supports,fi,tail,ti,cfg,calcium,base,spec)
    v=torch.linspace(-95,55,31,device=device)[None].expand(36,-1);ca=torch.full_like(v,1e-4)
    with torch.no_grad():
        before=parent.raw(v[:3],ca[:3]);actual=model.raw(v,ca)
        zero=max(float((actual[j*3:j*3+3]-b).abs().max()) for actual,b in zip(actual,before) for j in range(12))
    assert zero==0,'Zero-initialization must reproduce P110 exactly'
    opt=torch.optim.Adam([model.transfer_correction],lr=spec['lr'],eps=1e-8,capturable=device=='cuda')
    atomic_step(torch,model,opt,atomic,idx,censored,norm)
    # Isolate Adam and per-head clipping using the exact shared atomic objective.
    isolation=0.
    for member in (0,8,17,35):
        w=torch.nn.Parameter(torch.zeros(1,2,64,device=device))
        ds={k:v[member:member+1] for k,v in atomic.items()}
        oo=torch.optim.Adam([w],lr=spec['lr']);oo.zero_grad()
        point,_=m.atomic_terms(torch,w,ds,idx[:1],censored[member:member+1])
        (point/norm[member:member+1]).sum().backward();clip_heads(torch,w.grad);oo.step()
        isolation=max(isolation,float((w[0]-model.transfer_correction[member]).abs().max()))
    assert isolation<1e-6
    metrics=feedback_step(torch,model,opt,groups,atomic,idx,censored,norm,fb,supports,fi,tail,ti,cfg,calcium,base,spec,state)
    assert all(bool(torch.isfinite(v).all()) for v in metrics.values())
    with torch.no_grad():
        after=model.raw(v,ca)
        untouched=[k for k in range(18) if k not in m.GATES]
        assert torch.equal(actual[0][...,untouched],after[0][...,untouched])
        assert torch.equal(actual[1],after[1])
    graph_error=None
    if device=='cuda':
        reference=copy.deepcopy(model)
        a=torch.optim.Adam([model.transfer_correction],lr=spec['lr'],capturable=True)
        b=torch.optim.Adam([reference.transfer_correction],lr=spec['lr'],capturable=True)
        graph=capture_atomic(torch,model,a,atomic,idx,censored,norm)
        for _ in range(3):
            graph.replay();atomic_step(torch,reference,b,atomic,idx,censored,norm)
        torch.cuda.synchronize()
        graph_error=float((model.transfer_correction-reference.transfer_correction).abs().max())
        assert graph_error<=spec['computational_parity_atol']
    # Measure both actual feedback horizons and the resident atomic update.
    benchmark={}
    for ticks in (4,40):
        chosen=[g for g in groups if g['ticks']==ticks]
        start=time.perf_counter()
        for _ in range(2):
            loss=feedback_terms(torch,model,chosen,fb,supports,fi,cfg,calcium,base)
            torch.autograd.grad(loss.sum(),model.transfer_correction)
        if device=='cuda':torch.cuda.synchronize()
        benchmark[str(ticks)]=(time.perf_counter()-start)/2
    probe=m.extract(torch,model,0,root,device)
    core=t.make_core(torch,cfg,calcium,base,100*np.pi,probe,t.FAMILIES['all'])
    inputs=[v[:3,:2],ca[:3,:2],torch.full((3,2,18),.4,device=device),
            torch.ones(3,2,11,device=device),torch.zeros(3,2,device=device),torch.zeros(3,2,4,device=device)]
    old=t._rates
    try:
        t._rates=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Canonical kinetic fallback'))
        with torch.no_grad():
            aa=t.Stepper(torch,core,inputs);bb=t.Stepper(torch,core,inputs,device=='cuda')
            parity=max(float((a-b).abs().max()) for _ in range(3) for a,b in zip(aa.tick(),bb.tick()))
    finally:t._rates=old
    assert parity<=spec['computational_parity_atol']
    report=dict(valid=True,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,
                torch=torch.__version__,members=36,new_weights_per_seed=128,
                zero_initialization_error=zero,optimizer_isolation_max_error=isolation,
                training_graph_max_error=graph_error,physical_graph_max_error=parity,
                all_parameters_finite=all(bool(torch.isfinite(a).all()) for a in model.parameters()),
                canonical_fallback=False,untargeted_coefficients_exact=True,
                feedback_seconds_per_gradient=benchmark,
                source_sha256=p.digest(root/'src/giada_teacher/task41e_transfer_training.py'),
                protocol_sha256=p.digest(root/m.CONFIG))
    p.save_json(output/'preflight.json',report)
    return report


def run(root,output,device='cuda',smoke=False):
    import torch
    spec,cfg,calcium,base=m.contract(root)
    assert torch.cuda.is_available() and device=='cuda'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'checkpoint_0.pt').exists()
    p.save_json(output/'preregistration.json',spec)
    preflight(torch,root,output,device,spec)
    print('[GIADA Task41e] preflights passed',flush=True)
    if smoke:
        result=dict(valid=True,decision='GPU_PREFLIGHT_ONLY',task='41e')
        p.save_json(output/'final_report.json',result);return result
    parent=m.d.prior.load_models(torch,root,device)['P110']
    definitions=m.arms();model=m.make_model(torch,parent,definitions,device)
    atomic,raw=m.atomic_data(torch,parent,spec,device)
    selection,_=m.atomic_data(torch,parent,spec,device,selection=True)
    np.savez_compressed(output/'fit_atomic.npz',**raw)
    print('[GIADA Task41e] materializing independent FIT histories',flush=True)
    fb,supports,_,_=data_module.prepare_feedback(torch,spec,cfg,calcium,base,device,output)
    sf,ss,_,_=data_module.prepare_feedback(torch,spec,cfg,calcium,base,device,output,selection=True)
    tail=data_module.held_tail_data(torch,parent,spec,device,spec['protection_anchors'])
    groups=feedback_groups(torch,parent,model,definitions,device)
    censored=torch.tensor([a['objective']=='censored' for a in definitions for _ in range(3)],device=device)
    idx=torch.arange(spec['batch'],device=device)[None].expand(36,-1).clone()
    fi=torch.arange(spec['feedback_batch'],device=device)
    ti=torch.arange(spec['protection_batch'],device=device)
    norm,state=initialize_balance(torch,model,groups,atomic,idx,censored,fb,supports,fi,tail,ti,cfg,calcium,base,spec)
    p.save_json(output/'fit_calibration.json',dict(normalizers=norm.tolist(),balance=serializable(state)))
    opt=torch.optim.Adam([model.transfer_correction],lr=spec['lr'],eps=1e-8,capturable=True)
    graph=capture_atomic(torch,model,opt,atomic,idx,censored,norm)
    dev=m.d.development(spec['new_seeds']['development']);np.savez_compressed(output/'development_inputs.npz',**dev)
    parent_rows=c.shadow_evaluate(torch,[parent],dev,spec['quality'],device)
    stream=np.random.default_rng(spec['new_seeds']['stream'])
    fbrng=np.random.default_rng(spec['new_seeds']['feedback_stream'])
    tailrng=np.random.default_rng(spec['new_seeds']['protection_stream'])
    si=torch.arange(spec['selection_anchors'],device=device)[None].expand(36,-1)
    sfi=ss['broad']
    best=[None]*36;global_best=[None]*36;chosen=[None]*36;global_chosen=[None]*36
    saved=[None]*36;global_saved=[None]*36;curves=[];clock=time.perf_counter();diagnostics={}
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            assert all(bool(torch.isfinite(a).all()) for a in model.parameters())
            checkpoint=dict(step=step,correction=model.transfer_correction.detach().cpu(),optimizer=opt.state_dict(),
                            balance_state=state,stream=stream.bit_generator.state,feedback_rng=fbrng.bit_generator.state,
                            protection_rng=tailrng.bit_generator.state,parent_sha256=spec['parent_sha256'])
            torch.save(checkpoint,output/f'checkpoint_{step}.pt')
            print(f'[GIADA Task41e] checkpoint_saved {step}',flush=True)
            models=[m.extract(torch,model,k,root,device) for k in range(12)]
            with torch.no_grad():
                point,secant=m.atomic_terms(torch,model.transfer_correction,selection,si,censored)
                response=deployed_feedback(torch,parent,model,definitions,sf,sfi,cfg,calcium,base,device)
                measured=c.shadow_evaluate(torch,models,dev,spec['quality'],device)
            rr=[];gg=[]
            for j,rows in enumerate(measured):
                rank,global_rank=ranks(rows,parent_rows[j%3],response[j],(point[j]/norm[j]).sum(),spec['quality'])
                rr.append(rank);gg.append(global_rank)
                if best[j] is None or rank<best[j]:
                    best[j],chosen[j],saved[j]=rank,step,model.transfer_correction[j].detach().clone()
                if global_best[j] is None or global_rank<global_best[j]:
                    global_best[j],global_chosen[j],global_saved[j]=global_rank,step,model.transfer_correction[j].detach().clone()
            feedback_updates=step//spec['feedback_every']
            budgets=[dict(name=a['name'],atomic_updates=step,feedback_windows=feedback_updates*spec['feedback_batch'] if a['feedback'] else 0,
                          feedback_ticks=feedback_updates*spec['feedback_batch']*a['ticks'] if a['feedback'] else 0,
                          independent_scenarios=2 if a['feedback'] else 0) for a in definitions]
            curves.append(dict(step=step,primary_ranks=rr,global_ranks=gg,point=point.tolist(),secant=secant.tolist(),
                               deployed_response=response.tolist(),balance=serializable(state),gradients=serializable(diagnostics),
                               elapsed_seconds=time.perf_counter()-clock,budgets=budgets))
            p.save_json(output/'training_curves.json',curves);p.save_json(output/f'development_{step}.json',measured)
            print(f'[GIADA Task41e] checkpoint development {step} target-passing={sum(r[1]==0 for r in rr)}/36',flush=True)
        if step==spec['checkpoints'][-1]:break
        idx.copy_(torch.tensor(stream.integers(0,spec['anchors'],spec['batch']),device=device)[None].expand(36,-1))
        if (step+1)%spec['feedback_every']==0:
            fi.copy_(torch.tensor(fbrng.integers(0,len(supports['broad']),spec['feedback_batch']),device=device))
            ti.copy_(torch.tensor(tailrng.integers(0,spec['protection_anchors'],spec['protection_batch']),device=device))
            diagnostics=feedback_step(torch,model,opt,groups,atomic,idx,censored,norm,fb,supports,fi,tail,ti,cfg,calcium,base,spec,state)
        else:
            graph.replay()
        if (step+1)%1000==0:
            print(f'[GIADA Task41e] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-clock)/60:.1f}min',flush=True)
    primary_weights=torch.stack(saved);secondary_weights=torch.stack(global_saved)
    trained={a['name']:m.extract(torch,model,k,root,device,primary_weights) for k,a in enumerate(definitions)}
    nominee=min(range(12),key=lambda k:tuple(max(best[k*3+s][i] for s in range(3)) for i in range(len(best[0])))+(k,))
    trained['global_selection_same_arm']=m.extract(torch,model,nominee,root,device,secondary_weights)
    torch.save({name:model.state_dict() for name,model in trained.items()},output/'frozen_checkpoint.pt')
    torch.save(dict(primary=primary_weights.cpu(),secondary=secondary_weights.cpu()),output/'all_selection_weights.pt')
    freeze=dict(arms=definitions,seeds=spec['seeds'],chosen_steps=chosen,global_chosen_steps=global_chosen,
                primary_ranks=best,global_ranks=global_best,nominee=definitions[nominee]['name'],
                checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'),all_weights_sha256=p.digest(output/'all_selection_weights.pt'),
                opened_regressions_accessed=False,independent_confirmation=False,primary_selection=spec['selection'])
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()
    p.save_json(output/'selection_freeze.json',freeze)
    print('[GIADA Task41e] checkpoint selection frozen',flush=True)
    from .task41e_transfer_evaluation import evaluate
    with torch.inference_mode():
        result=evaluate(torch,root,output,trained,spec,device)
    result['training_and_evaluation_seconds']=time.perf_counter()-clock
    p.save_json(output/'final_report.json',result)
    return result
