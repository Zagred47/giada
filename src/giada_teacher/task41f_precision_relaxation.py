"""41f: learned equilibrium/relaxation corrections with matched controls.

All candidates keep the same 18 recurrent states and physical interface.
Canonical kinetics are training labels and explicit diagnostic oracles only.
"""
import copy
import hashlib
import io
import json
import sys
from pathlib import Path
import time
import zipfile
import numpy as np
from . import task41e_transfer_model as prior
from . import task41e_transfer_training as et
from . import task41e_transfer_data as ed
from . import task41e_transfer_evaluation as ee

p,t,c,d,x=prior.p,prior.t,prior.c,prior.d,prior.x
CONFIG='experiments/task41f_preregistration_v1.json'
GATES=(0,1,8)  # Ca_HVA.m, Ca_HVA.h, Nap_Et2.m
CHANNELS=(0,4)


def arms():
    return [dict(name=f'{kind}_{support}_{objective}',kind=kind,basis=basis,
                 support=support,objective=objective,feedback=objective=='response')
            for kind,basis in [('U64',64),('U128',128),('UZ64',64)]
            for support in ('broad','rare') for objective in ('point','response')]


def contrasts():
    result=[]
    for support in ('broad','rare'):
        for objective in ('point','response'):
            result += [('capacity',f'U64_{support}_{objective}',f'U128_{support}_{objective}'),
                       ('relaxation_equal_parameters',f'U128_{support}_{objective}',f'UZ64_{support}_{objective}')]
    for kind in ('U64','U128','UZ64'):
        for objective in ('point','response'):
            result.append(('support',f'{kind}_broad_{objective}',f'{kind}_rare_{objective}'))
        for support in ('broad','rare'):
            result.append(('objective',f'{kind}_{support}_point',f'{kind}_{support}_response'))
    return result


def contract(root):
    spec=json.loads((root/CONFIG).read_text())
    old,cfg,calcium,base=prior.contract(root)
    assert spec['quality']==old['quality'] and spec['compartment_gates']==old['compartment_gates']
    assert spec['cost']==old['cost'] and spec['arms']==arms()
    assert spec['contrasts']==[list(r) for r in contrasts()]
    for path,sha in spec['sources'].items():assert p.digest(root/path)==sha,path
    assert not set(spec['new_seeds'].values()) & set(old['new_seeds'].values())
    return spec,cfg,calcium,base


def active_mask(torch,definitions,device):
    mask=torch.zeros(3*len(definitions),6,128,device=device)
    for j,a in enumerate(definitions):
        mask[j*3:j*3+3,:3,:a['basis']]=1
        if a['kind']=='UZ64':mask[j*3:j*3+3,3:,:a['basis']]=1
    return mask


def features(torch,v,definitions):
    small=torch.nn.functional.pad(d.feature(torch,v,64),(0,64))
    wide=d.feature(torch,v,128)
    choose=torch.tensor([a['basis']==128 for a in definitions for _ in range(3)],device=v.device)
    return torch.where(choose.reshape(-1,*([1]*(v.ndim))),wide,small)


def extras(torch,v,weights,definitions,mask):
    if len(definitions)==1:
        a=definitions[0];n=a['basis'];heads=6 if a['kind']=='UZ64' else 3
        value=torch.bmm(d.feature(torch,v,n).double(),weights[:,:heads,:n].transpose(1,2).double()).to(v.dtype)
        value=.1*torch.tanh(value/.1)
        return value if heads==6 else torch.cat((value,torch.zeros_like(value)),-1)
    f=features(torch,v,definitions)
    # A compact view is used for deployed single-arm models. Training retains
    # one padded tensor so every seed/arm is updated in the same GPU operation.
    value=torch.bmm(f.double(),(weights*mask).transpose(1,2).double()).to(v.dtype)
    return .1*torch.tanh(value/.1)


def add(torch,u,z,extra):
    um={g:k for k,g in enumerate(GATES)}
    return tuple(torch.cat([raw[...,g:g+1]+extra[...,um[g]+offset:um[g]+offset+1]
        if g in um else raw[...,g:g+1] for g in range(18)],-1)
        for raw,offset in ((u,0),(z,3)))


def model_class(torch):
    class Model(c.model_class(torch)):
        def __init__(self,definitions):
            super().__init__(dict(basis=64),[17,29,43]*len(definitions))
            for v in self.parameters():v.requires_grad_(False)
            self.definitions=definitions
            self.f_correction=torch.nn.Parameter(torch.zeros(3*len(definitions),6,128))
            self.register_buffer('active',active_mask(torch,definitions,'cpu'))
        def raw(self,v,ca):
            u,z=super().raw(v,ca)
            return add(torch,u,z,extras(torch,v,self.f_correction,self.definitions,self.active))
    return Model


def make_model(torch,parent,definitions,device):
    model=model_class(torch)(definitions).to(device=device,dtype=next(parent.parameters()).dtype)
    state=model.state_dict()
    for name,value in prior.repeat_parent(torch,parent,definitions,device).state_dict().items():state[name]=value
    model.load_state_dict(state)
    return model


def extract(torch,model,index,parent,device,weights=None):
    result=make_model(torch,parent,[model.definitions[index]],device)
    with torch.no_grad():result.f_correction.copy_((model.f_correction if weights is None else weights)[index*3:index*3+3])
    result.eval()
    return result


def frozen_controls(torch,root,spec,device):
    controls=prior.frozen_controls(torch,root,spec,device)
    assert p.digest(root/spec['task41e_checkpoint'])==spec['task41e_checkpoint_sha256']
    saved=torch.load(root/spec['task41e_checkpoint'],map_location=device,weights_only=True)
    for name in ('Q0','H10'):
        definition=next(a for a in prior.arms() if a['name']==name)
        model=prior.make_model(torch,controls['P110'],[definition],device)
        model.load_state_dict(saved[name]);controls['E_'+name]=model
    return controls


def channel_severity(metric,q,channels):
    values=[]
    for k in channels:
        row=metric['channels'][k]
        value=et.channel_severity(metric,q,(k,))
        # Reuse the canonical decision, including its existing clock tolerance.
        values.append(min(value,1.) if row['passed'] else max(value,1.+1e-12))
    return max(values)


def ranks(rows,parent_rows,response,atomic,q):
    introduced=sum(not r['channels'][k]['passed'] and b['channels'][k]['passed']
        for r,b in zip(rows,parent_rows) for k in range(11))
    introduced+=sum(not r['gate']['passed'] and b['gate']['passed'] for r,b in zip(rows,parent_rows))
    failed=sum(any(not r['channels'][k]['passed'] for k in CHANNELS) for r in rows)
    worst=max(channel_severity(r,q,CHANNELS) for r in rows)
    global_failed=sum(not r['passed'] for r in rows)
    global_worst=max(min(c.severity(r,q),1.) if r['passed'] else max(c.severity(r,q),1.+1e-12) for r in rows)
    return (introduced,failed,worst,float(response),float(atomic),global_failed,global_worst)


def atomic_data(torch,parent,spec,device,selection=False,count=None):
    n=count or spec['selection_anchors' if selection else 'anchors']
    seed=spec['new_seeds']['selection' if selection else 'fit']
    raw=d.anchors(seed,n*2)
    # Both arms retain a broad half; enrichment uses only FIT coefficients.
    # Slow relaxation, voltage sensitivity and unsaturated coefficients are
    # ranked together; opened protocols never enter this pool.
    us,zs=raw['u'][:,1,list(GATES)],raw['z'][:,1,list(GATES)]
    score=(us*(1-us)/(zs+.0001)).max(-1)
    score+=abs(raw['u'][:,2,list(GATES)]-raw['u'][:,0,list(GATES)]).max(-1)*100
    rng=np.random.default_rng(seed+700001)
    broad=rng.permutation(2*n)[:n]
    fixed=list(broad[:n//2]);used=set(fixed)
    rare=fixed+[int(k) for k in np.argsort(-score,kind='stable') if int(k) not in used][:n-len(fixed)]
    ids={'broad':np.asarray(broad),'rare':np.asarray(rare)}
    defs=arms();out={k:[] for k in ('features','base','target','support_ids')}
    for a in defs:
        chosen=ids['rare' if a['support']=='rare' and not selection else 'broad']
        v=torch.tensor(raw['v'][chosen],device=device)
        ca=torch.tensor(raw['ca'][chosen],device=device)
        with torch.no_grad():u,z=parent.raw(v.reshape(1,-1).expand(3,-1),ca.reshape(1,-1).expand(3,-1))
        out['base'].append(torch.cat((u[...,list(GATES)],z[...,list(GATES)]),-1).reshape(3,n,3,6))
        targets=np.concatenate((raw['u'][chosen][...,list(GATES)],raw['z'][chosen][...,list(GATES)]),-1)
        out['target'].append(torch.tensor(targets,device=device)[None].expand(3,-1,-1,-1))
        f=torch.nn.functional.pad(d.feature(torch,v,a['basis']),(0,128-a['basis']))
        out['features'].append(f[None].expand(3,-1,-1,-1))
        out['support_ids'].append(torch.tensor(chosen,device=device)[None].expand(3,-1))
    return {k:torch.cat(v) for k,v in out.items()},dict(**raw,broad_ids=ids['broad'],rare_ids=ids['rare'])


def atomic_terms(torch,weights,mask,data,idx):
    member=torch.arange(len(weights),device=weights.device)[:,None]
    f=data['features'][member,idx]
    extra=torch.einsum('mbkp,mop->mbko',f.double(),(weights*mask).double()).float()
    raw=data['base'][member,idx]+.1*torch.tanh(extra/.1)
    target=data['target'][member,idx]
    ru=prior.clipped_or_censored_error(torch,raw[...,:3],target[...,:3],True)
    rz=torch.nn.functional.logsigmoid(raw[...,3:].double())-target[...,3:].clamp_min(1e-30).log()
    return torch.cat((ru,rz),-1).square().mean((1,2))*mask.sum(-1).clamp_max(1)


def projection_probe(torch,atomic,selection,model,output):
    rows=[]
    # Least-squares coefficient projection is an optimization diagnostic only.
    # It is never installed in a candidate or used by checkpoint selection.
    for arm_index in (0,2,4,6,8,10):
        definition=model.definitions[arm_index]
        for si,seed in enumerate((17,29,43)):
            member=arm_index*3+si;n=definition['basis']
            order=np.random.default_rng(751001).permutation(atomic['features'].shape[1])
            for count in (1024,4096):
                count=min(count,len(order));ids=order[:count]
                xx=atomic['features'][member,ids,1,:n].double().cpu()
                base=atomic['base'][member,ids,1].double().cpu()
                target=atomic['target'][member,ids,1].double().cpu()
                du=-prior.clipped_or_censored_error(torch,base[:,:3],target[:,:3],True)
                dz=torch.logit(target[:,3:].clamp(1e-30,1-1e-15))-base[:,3:]
                desired=torch.cat((du,dz),-1)
                linear=.1*torch.atanh((desired/.1).clamp(-.999999,.999999))
                weights=torch.linalg.lstsq(xx,linear,driver='gelsy').solution
                sx=selection['features'][member,:,1,:n].double().cpu()
                sb=selection['base'][member,:,1].double().cpu();st=selection['target'][member,:,1].double().cpu()
                projected=sb+.1*torch.tanh((sx@weights)/.1)
                errors=torch.cat((prior.clipped_or_censored_error(torch,projected[:,:3],st[:,:3],True),
                    torch.nn.functional.logsigmoid(projected[:,3:])-st[:,3:].clamp_min(1e-30).log()),-1)
                enabled=model.active[member].sum(-1).clamp_max(1).cpu()
                rows.append(dict(arm=definition['name'],seed=seed,FIT_points=count,
                    heldout_coefficient_mse_per_head=(errors.square().mean(0)*enabled).tolist(),
                    affine_rank=int(torch.linalg.matrix_rank(xx)),selection_eligible=False))
    p.save_json(output/'projection_data_resolution_probe.json',dict(rows=rows,
        scope='Bounded coefficient approximation only; no SGD scaling claim, recurrence qualification or candidate promotion.'))


def objective_probe(torch,model,view,atomic,norm,fb,supports,sf,ss,cfg,calcium,base):
    device=model.f_correction.device
    idx=torch.arange(8,device=device)[None].expand(len(model.f_correction),-1);fi=torch.arange(2,device=device)
    point=(atomic_terms(torch,model.f_correction,model.active,atomic,idx)/norm).sum()
    pg=torch.autograd.grad(point,model.f_correction)[0]
    response=feedback_terms(torch,model,view,fb,supports,fi,cfg,calcium,base)
    rg=torch.autograd.grad(response.sum(),model.f_correction)[0]
    original=model.f_correction.detach().clone();rows={}
    try:
        for name,gradient in [('unchanged',torch.zeros_like(pg)),('point_descent',pg),('response_descent',rg)]:
            direction=gradient/gradient.double().norm(dim=-1,keepdim=True).clamp_min(1e-30).to(gradient.dtype)
            with torch.no_grad():
                model.f_correction.copy_(original-.0001*direction)
                rows[name]=dict(point=(atomic_terms(torch,model.f_correction,model.active,atomic,idx)/norm).sum(-1).tolist(),
                    response_FIT=feedback_terms(torch,model,view,fb,supports,fi,cfg,calcium,base).tolist(),
                    response_holdout=feedback_terms(torch,model,view,sf,ss,fi,cfg,calcium,base,True).tolist())
    finally:
        with torch.no_grad():model.f_correction.copy_(original)
    return dict(epsilon=.0001,per_head_unit_direction=True,optimizer_and_training_weights_unchanged=True,rows=rows)


class FeedbackView:
    def __init__(self,torch,parent,model):
        self.torch,self.model=torch,model
        self.parent=prior.repeat_parent(torch,parent,model.definitions,next(model.parameters()).device).double()
    def __getattr__(self,k):return getattr(self.parent,k)
    def raw(self,v,ca):
        u,z=self.parent.raw(v,ca)
        return add(self.torch,u,z,extras(self.torch,v,self.model.f_correction,self.model.definitions,self.model.active))
    def __call__(self,v,ca,g):
        u,z=self.raw(v,ca)
        return g+self.torch.sigmoid(z)*(self.torch.sigmoid(u)-g)


def feedback_data(torch,spec,cfg,calcium,base,device,output=None,selection=False,small=False):
    effective=dict(spec)
    if selection:
        effective['new_seeds']={k:v+100000 for k,v in spec['new_seeds'].items()}
        effective['feedback_histories']=spec['selection_feedback_histories']
    raw=ed.history_pool(torch,effective,cfg,calcium,base,device,small)
    labels=ed.reference_windows(torch,raw,spec,cfg,calcium,base,device)
    u,z=p.coefficients(raw['v'][:,0],raw['ca'][:,0],.1)
    labels['slow_tail']=abs((raw['g'][:,0]-u)*np.maximum(1-z,0)**400)[:,list(GATES)].max(-1)
    count=min(len(raw['v']),spec['selection_feedback_anchors' if selection else 'feedback_anchors'])
    sup=ed.select_support(raw,labels,effective,count,selection)
    supports={'broad':torch.tensor(sup['broad'],device=device),'rare':torch.tensor(sup['stratified'],device=device)}
    values=ed.tensor_inputs(torch,raw,device,torch.float64)
    values['target']=torch.tensor(labels['delta'][...,list(CHANNELS)],device=device)
    if output:
        prefix='selection' if selection else 'fit'
        np.savez_compressed(output/(prefix+'_feedback.npz'),**raw,**labels,**{k+'_ids':v.cpu().numpy() for k,v in supports.items()})
        p.save_json(output/(prefix+'_support.json'),dict(states=len(raw['v']),independent_histories=int(raw['history'].max()+1),counts={k:len(v) for k,v in supports.items()},dtype='FP64 canonical labels and feedback dynamics',seed=effective['new_seeds']['feedback']))
    return values,supports


def feedback_terms(torch,model,view,fb,supports,idx,cfg,calcium,base,selection=False):
    members=len(model.f_correction);batch=len(idx)
    ids=torch.stack([supports['broad' if selection else a['support']][idx] for a in model.definitions for _ in range(3)])
    expand=lambda k:fb[k][ids].reshape(members,batch*2,*fb[k].shape[2:])
    v,ca,g=(expand(k) for k in ('v','ca','g'))
    mult,inj,syn=(expand(k) for k in ('mult','inj','syn'))
    core=t.make_core(torch,cfg,calcium,base,100*np.pi,view,t.FAMILIES['all'])
    target=fb['target'][:spec_ticks(model)]
    scale=target.abs().amax(0).clamp_min(1e-7)[ids]
    loss=torch.zeros(members,device=v.device,dtype=torch.float64)
    for tick in range(len(target)):
        v,ca,g,current=core(v,ca,g,mult,inj,syn)
        pair=current[...,list(CHANNELS)].reshape(members,batch,2,2)
        delta=pair[:,:,1]-pair[:,:,0]
        loss+=((delta-target[tick][ids])/scale).square().mean((1,2))
    return loss/len(target)


def spec_ticks(model):return 40


def clip(torch,gradient):
    norm=gradient.double().norm(dim=-1)
    gradient.mul_((1/norm.clamp_min(1))[...,None].to(gradient.dtype))


def atomic_step(torch,model,opt,atomic,idx,norm):
    opt.zero_grad(set_to_none=False)
    (atomic_terms(torch,model.f_correction,model.active,atomic,idx)/norm).sum().backward()
    clip(torch,model.f_correction.grad);opt.step()


def auxiliary_step(torch,model,opt,view,atomic,idx,norm,fb,supports,fi,cfg,calcium,base,spec,state):
    point=atomic_terms(torch,model.f_correction,model.active,atomic,idx)/norm
    gp=torch.autograd.grad(point.sum(),model.f_correction)[0]
    response=feedback_terms(torch,model,view,fb,supports,fi,cfg,calcium,base)
    gr=torch.autograd.grad(response.sum(),model.f_correction)[0]
    np_,nr=gp.double().norm(dim=-1),gr.double().norm(dim=-1)
    state['ema'].mul_(.9).add_(torch.stack((np_,nr),-1),alpha=.1)
    a,b=state['ema'].unbind(-1)
    weight=(spec['feedback_ratio']*a/b.clamp_min(1e-20)).clamp(0,1e6)
    weight=torch.where((a>1e-16)&(b>1e-16),weight,torch.zeros_like(weight))
    enabled=torch.tensor([a['feedback'] for a in model.definitions for _ in range(3)],device=gp.device)[:,None]
    gradient=gp+gr*(weight*enabled)[...,None]
    assert torch.isfinite(gradient).all()
    opt.zero_grad(set_to_none=False);model.f_correction.grad.copy_(gradient);clip(torch,model.f_correction.grad)
    before=model.f_correction.detach().clone();opt.step()
    assert torch.isfinite(model.f_correction).all()
    return dict(point=point.detach().tolist(),response=response.detach().tolist(),
        point_gradient=np_.tolist(),response_gradient=nr.tolist(),weighted_ratio=(nr*weight*enabled/np_.clamp_min(1e-30)).tolist(),
        cosine=((gp.double()*gr.double()).sum(-1)/(np_*nr).clamp_min(1e-30)).tolist(),
        update_norm=(model.f_correction-before).detach().double().norm(dim=-1).tolist())


def preflight(torch,root,output,device,spec):
    _,cfg,calcium,base=contract(root)
    parent=d.prior.load_models(torch,root,device)['P110']
    model=make_model(torch,parent,arms(),device)
    v=torch.linspace(-90,40,13,device=device)[None].expand(36,-1);ca=torch.full_like(v,1e-4)
    with torch.no_grad():
        expected=parent.raw(v[:3],ca[:3]);actual=model.raw(v,ca)
        parity=max(float((a[j*3:j*3+3]-b).abs().max()) for a,b in zip(actual,expected) for j in range(12))
    assert parity==0
    atomic,_=atomic_data(torch,parent,spec,device,count=16)
    idx=torch.arange(8,device=device)[None].expand(36,-1).clone()
    norm=atomic_terms(torch,model.f_correction,model.active,atomic,idx).detach().clamp_min(1e-12)
    opt=torch.optim.Adam([model.f_correction],lr=spec['lr'],capturable=device=='cuda')
    atomic_step(torch,model,opt,atomic,idx,norm)
    assert bool(torch.equal(model.f_correction*(1-model.active),torch.zeros_like(model.f_correction)))
    isolation=0.
    for member in (0,14,35):
        w=torch.nn.Parameter(torch.zeros(1,6,128,device=device));oo=torch.optim.Adam([w],lr=spec['lr'])
        ds={k:v[member:member+1] for k,v in atomic.items()};mask=model.active[member:member+1]
        oo.zero_grad();(atomic_terms(torch,w,mask,ds,idx[:1])/norm[member:member+1]).sum().backward();clip(torch,w.grad);oo.step()
        isolation=max(isolation,float((w[0]-model.f_correction[member]).abs().max()))
    assert isolation<=1e-6
    fb,supports=feedback_data(torch,spec,cfg,calcium,base,device,small=True)
    view=FeedbackView(torch,parent,model)
    start=time.perf_counter()
    response=feedback_terms(torch,model,view,fb,supports,torch.arange(2,device=device),cfg,calcium,base)
    grad=torch.autograd.grad(response.sum(),model.f_correction)[0]
    if device=='cuda':torch.cuda.synchronize()
    assert bool(torch.isfinite(grad).all())
    feedback_seconds=time.perf_counter()-start
    graph_error=None
    if device=='cuda':
        ref=copy.deepcopy(model)
        aa=torch.optim.Adam([model.f_correction],lr=spec['lr'],capturable=True)
        bb=torch.optim.Adam([ref.f_correction],lr=spec['lr'],capturable=True)
        graph=capture(torch,model,aa,atomic,idx,norm)
        for _ in range(3):graph.replay();atomic_step(torch,ref,bb,atomic,idx,norm)
        torch.cuda.synchronize();graph_error=float((model.f_correction-ref.f_correction).abs().max())
        assert graph_error<=1e-5
    report=dict(valid=True,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,
        members=36,all_parameters_finite=all(bool(torch.isfinite(v).all()) for v in model.parameters()),
        active_weights_per_seed=[int(model.active[j*3].sum()) for j in range(12)],
        zero_parent_error=parity,optimizer_isolation_error=isolation,graph_error=graph_error,
        feedback_seconds_per_gradient=feedback_seconds,protocol_sha256=p.digest(root/CONFIG),
        source_sha256=p.digest(root/'src/giada_teacher/task41f_precision_relaxation.py'),thresholds_changed=False)
    p.save_json(output/'preflight.json',report)
    return report


def capture(torch,model,opt,atomic,idx,norm):
    initial=model.f_correction.detach().clone()
    for _ in range(3):atomic_step(torch,model,opt,atomic,idx,norm)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):atomic_step(torch,model,opt,atomic,idx,norm)
    with torch.no_grad():
        model.f_correction.copy_(initial)
        for state in opt.state.values():
            for value in state.values():
                if torch.is_tensor(value):value.zero_()
    return graph


def run(root,output,device='cuda',smoke=False):
    import torch
    from . import task41f_evaluation as evaluation
    spec,cfg,calcium,base=contract(root)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert device=='cuda' and torch.cuda.is_available()
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'checkpoint_0.pt').exists()
    p.save_json(output/'preregistration.json',spec)
    preflight(torch,root,output,device,spec)
    print('[GIADA Task41f] preflights passed',flush=True)
    if smoke:return dict(valid=True,decision='GPU_PREFLIGHT_ONLY')
    controls=frozen_controls(torch,root,spec,device)
    with torch.inference_mode():evaluation.frozen_audit(torch,root,output,controls,spec,cfg,calcium,base,device)
    ee.differentiable_constants()
    parent=controls['P110'];model=make_model(torch,parent,arms(),device)
    atomic,raw=atomic_data(torch,parent,spec,device)
    selection,_=atomic_data(torch,parent,spec,device,selection=True)
    np.savez_compressed(output/'fit_atomic.npz',**raw)
    projection_probe(torch,atomic,selection,model,output)
    fb,supports=feedback_data(torch,spec,cfg,calcium,base,device,output)
    sf,ss=feedback_data(torch,spec,cfg,calcium,base,device,output,selection=True)
    view=FeedbackView(torch,parent,model)
    idx=torch.arange(spec['batch'],device=device)[None].expand(36,-1).clone()
    norm=atomic_terms(torch,model.f_correction,model.active,atomic,idx).detach().clamp_min(1e-12)
    opt=torch.optim.Adam([model.f_correction],lr=spec['lr'],capturable=True)
    graph=capture(torch,model,opt,atomic,idx,norm)
    state=dict(ema=torch.zeros(36,6,2,device=device,dtype=torch.float64))
    dev=d.development(spec['new_seeds']['development']);np.savez_compressed(output/'development_inputs.npz',**dev)
    parent_rows=c.shadow_evaluate(torch,[parent],dev,spec['quality'],device)
    rng=np.random.default_rng(spec['new_seeds']['stream']);frng=np.random.default_rng(spec['new_seeds']['feedback_stream'])
    best=[None]*36;chosen=[None]*36;saved=[None]*36;curves=[];diagnostics={};start=time.perf_counter()
    si=torch.arange(spec['selection_anchors'],device=device)[None].expand(36,-1)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            assert all(bool(torch.isfinite(v).all()) for v in model.parameters())
            torch.save(dict(step=step,correction=model.f_correction.detach().cpu(),optimizer=opt.state_dict(),
                balance=state,rng=rng.bit_generator.state,feedback_rng=frng.bit_generator.state),output/f'checkpoint_{step}.pt')
            print(f'[GIADA Task41f] checkpoint_saved {step}',flush=True)
            models=[extract(torch,model,k,parent,device) for k in range(12)]
            with torch.no_grad():
                point=atomic_terms(torch,model.f_correction,model.active,selection,si)/norm
                response=feedback_terms(torch,model,view,sf,ss,torch.arange(spec['selection_feedback_anchors'],device=device),cfg,calcium,base,True)
                measured=c.shadow_evaluate(torch,models,dev,spec['quality'],device)
            rr=[]
            for j,rows in enumerate(measured):
                rank=ranks(rows,parent_rows[j%3],response[j],point[j].sum(),spec['quality']);rr.append(rank)
                if best[j] is None or rank<best[j]:best[j],chosen[j],saved[j]=rank,step,model.f_correction[j].detach().clone()
            curves.append(dict(step=step,ranks=rr,point=point.tolist(),response=response.tolist(),gradients=diagnostics,
                elapsed_seconds=time.perf_counter()-start,atomic_examples=step*spec['batch'],
                feedback_windows=(step//spec['feedback_every'])*spec['feedback_batch'],feedback_ticks=40))
            p.save_json(output/'training_curves.json',curves);p.save_json(output/f'development_{step}.json',measured)
            if step in (0,spec['checkpoints'][-1]):
                p.save_json(output/f'objective_probe_{step}.json',objective_probe(torch,model,view,atomic,norm,fb,supports,sf,ss,cfg,calcium,base))
            print(f'[GIADA Task41f] checkpoint development {step} target-passing={sum(r[1]==0 for r in rr)}/36',flush=True)
        if step==spec['checkpoints'][-1]:break
        idx.copy_(torch.tensor(rng.integers(0,spec['anchors'],spec['batch']),device=device)[None].expand(36,-1))
        if (step+1)%spec['feedback_every']==0:
            fi=torch.tensor(frng.integers(0,len(supports['broad']),spec['feedback_batch']),device=device)
            diagnostics=auxiliary_step(torch,model,opt,view,atomic,idx,norm,fb,supports,fi,cfg,calcium,base,spec,state)
        else:graph.replay()
        if (step+1)%1000==0:print(f'[GIADA Task41f] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    weights=torch.stack(saved)
    trained={a['name']:extract(torch,model,k,parent,device,weights) for k,a in enumerate(arms())}
    nominee=min(range(12),key=lambda k:tuple(max(best[k*3+s][i] for s in range(3)) for i in range(len(best[0])))+(k,))
    torch.save({k:v.state_dict() for k,v in trained.items()},output/'frozen_checkpoint.pt')
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen,ranks=best,nominee=arms()[nominee]['name'],
        checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'),opened_regressions_accessed=True,
        opened_regressions_used_for_selection=False,diagnostic_access_before_training=True,
        selection='Preregistered canonical-predicate ranks; independent new development only.',independent_confirmation=False)
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()
    p.save_json(output/'selection_freeze.json',freeze)
    print('[GIADA Task41f] checkpoint selection frozen',flush=True)
    with torch.inference_mode():result=evaluation.evaluate(torch,root,output,controls,trained,spec,cfg,calcium,base,device)
    result['training_and_evaluation_seconds']=time.perf_counter()-start
    p.save_json(output/'final_report.json',result)
    return result
