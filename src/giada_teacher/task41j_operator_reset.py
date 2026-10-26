"""Joint relearning versus compact ionic operators, at the SAME abstraction.

18 occupancies, causal V/Ca, dt=.1ms, bounded affine update and the original
readout/membrane/synapse equations are retained. Canonical rates are labels only.
No previous scientific source or acceptance predicate is modified.
"""
import copy
import hashlib
import json
import time
import numpy as np
from . import task41i_selective_repair as prior

p,t,c,d,f=prior.p,prior.t,prior.c,prior.d,prior.f
CONFIG='experiments/task41j_preregistration_v1.json'


def arms():
    return [dict(name='J_chain_joint',representation='chain',width=64),
            dict(name='J_compact64',representation='compact',width=64),
            dict(name='J_compact128',representation='compact',width=128)]


def contract(root):
    spec=json.loads((root/CONFIG).read_text())
    old,cfg,calcium,base=prior.contract(root)
    assert spec['arms']==arms() and spec['seeds']==[17,29,43] and spec['dt_ms']==.1
    for key in ('quality','compartment_gates','cost'):assert spec[key]==old[key],key
    assert not set(spec['new_seeds'].values())&set(old['new_seeds'].values())
    for name,sha in spec['sources'].items():assert p.digest(root/name)==sha,name
    return spec,cfg,calcium,base


def controls(torch,root,device):
    old=prior.contract(root)[0]
    return prior.frozen_controls(torch,root,old,device)


def model_class(torch):
    class Operator(torch.nn.Module):
        def __init__(self,parent,definition):
            super().__init__();self.definition=definition;self.definitions=[definition]
            self.architecture='shared';self.is_compact=definition['representation']=='compact'
            if not self.is_compact:
                self.inner=copy.deepcopy(parent)
                for par in self.inner.parameters():par.requires_grad_(True)
            else:
                # Same voltage-only dependence for 17 gates; calcium-only SK.
                width=definition['width'];rows=[]
                for seed in (17,29,43):
                    gen=torch.Generator().manual_seed(seed+710000)
                    rows.append(torch.randn(1,width,generator=gen))
                for name in ('a','b','c','d','sa','sb','sc','sd'):
                    value=getattr(parent,name).detach().clone()
                    if name in ('a','b','c') and width!=64:
                        shape=list(value.shape);axis=2 if name in ('a','b') else 1;shape[axis]=width
                        extended=torch.zeros(shape,device=value.device,dtype=value.dtype)
                        if name=='a':extended.copy_(torch.stack(rows).to(value))
                        if axis==2:extended[...,:64]=value
                        else:extended[:,:64]=value
                        value=extended
                    self.register_parameter(name,torch.nn.Parameter(value))
                self.register_buffer('gbar',parent.gbar.detach().clone())
        def __getattr__(self,name):
            try:return super().__getattr__(name)
            except AttributeError:
                if name in ('a','sa','sb','sc','sd','gbar'):
                    return getattr(super().__getattr__('inner'),name)
                raise
        def raw(self,v,ca):
            shape=v.shape;vv=v.reshape(3,-1).float();cc=ca.reshape(3,-1).float()
            if not self.is_compact:u,z=self.inner.raw(vv,cc)
            else:
                vv=(vv+40)/80;cc=(cc.clamp_min(1e-9).log10()+4.5)/2.5
                raw=torch.bmm(torch.nn.functional.silu(torch.bmm(vv[...,None],self.a)+self.b),self.c)+self.d[:,None]
                sk=torch.bmm(torch.nn.functional.silu(torch.bmm(cc[...,None],self.sa)+self.sb),self.sc)+self.sd[:,None]
                u=torch.cat((raw[...,:16],sk[...,:1],raw[...,16:17]),-1)
                z=torch.cat((raw[...,17:33],sk[...,1:2],raw[...,33:34]),-1)
            return u.reshape(*shape,18).to(v.dtype),z.reshape(*shape,18).to(v.dtype)
        def forward(self,v,ca,state):
            u,z=self.raw(v,ca)
            return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)
        def coefficients(self,v,ca):
            u,z=self.raw(v,ca)
            return torch.sigmoid(u),torch.sigmoid(z),torch.nn.functional.logsigmoid(-z)
    return Operator


class TrainingView:
    """FP32 coefficient network; differentiable FP64 reference training solve."""
    def __init__(self,model):self.model=model
    def __getattr__(self,name):
        val=getattr(self.model,name)
        return val.double() if name in ('sa','sb','sc','sd') else val
    def raw(self,v,ca):return self.model.raw(v,ca)
    def __call__(self,v,ca,state):return self.model(v,ca,state)


def atomic_pool(torch,parent,seed,count,device):
    raw=d.anchors(seed,count)
    # Store exact targets for both equilibrium AND update, not parent residuals.
    data={k:torch.tensor(raw[k],device=device) for k in ('v','ca','u','z')}
    with torch.no_grad():
        v=data['v'].reshape(1,-1).expand(3,-1);ca=data['ca'].reshape(1,-1).expand(3,-1)
        u,z=parent.raw(v,ca)
        data['teacher']=torch.stack((torch.sigmoid(u.double()),torch.sigmoid(z.double())),-2).reshape(3,count,3,2,18)
    return data,raw


def expand(torch,x):return x[None].expand(3,*x.shape)


def coefficient_loss(torch,model,data,ids,distill=False):
    v=expand(torch,data['v'][ids]);ca=expand(torch,data['ca'][ids])
    logits=torch.stack(model.raw(v,ca),-2).double()
    target=data['teacher'][:,ids] if distill else expand(torch,torch.stack((data['u'][ids],data['z'][ids]),-2))
    # Training censoring is not a metric tolerance. Keep slow rates down to1e-12.
    lo=1e-12;safe=target.clamp(lo,1-lo)
    error=logits-torch.logit(safe)
    error=torch.where(target<=lo,error.clamp_min(0),torch.where(target>=1-lo,error.clamp_max(0),error))
    point=error.square().mean((1,2,3,4))
    prob=torch.sigmoid(logits)
    secant=((prob[:,:,2]-prob[:,:,0])-(target[:,:,2]-target[:,:,0])).square().mean((1,2,3))
    return point+.1*secant


def affine_scan(torch,u,z,initial):
    """Parallel composition; a training accelerator, NOT a new inference rule."""
    a=1-z;b=z*u;stride=1
    while stride<u.shape[-2]:
        olda,oldb=a,b
        a=torch.cat((olda[...,:stride,:],olda[...,stride:,:]*olda[...,:-stride,:]),-2)
        b=torch.cat((oldb[...,:stride,:],oldb[...,stride:,:]+olda[...,stride:,:]*oldb[...,:-stride,:]),-2)
        stride*=2
    return a*initial[...,None,:]+b


def path_pool(torch,spec,device,selection=False,count=None):
    n=count or spec['selection_paths' if selection else 'fit_paths'];T=spec['path_prefix_ticks'];L=12000
    rng=np.random.default_rng(spec['new_seeds']['selection' if selection else 'fit']+19001)
    rest=rng.uniform(-85,-30,n);rest[:n//2]=rng.uniform(-50,-35,n//2)
    ca=10**rng.uniform(-5,-3.5,n);peak=rng.uniform(-31,-14,n);stop=rng.integers(130,231,n)
    mode=rng.integers(0,2,n);clock=np.arange(T)
    v=np.where(mode[:,None],peak[:,None]+(rest[:,None]-peak[:,None])*np.clip((clock[None]-stop[:,None])/300,0,1),rest[:,None]).astype('float32')
    initial,_=p.coefficients(rng.uniform(-95,35,n),ca,.1);mult=rng.uniform(.5,2,(n,11));rows=[]
    for first in range(0,n,16):
        sl=slice(first,min(first+16,n));nn=len(rest[sl]);state=initial[sl].copy();prefix=[]
        u,z=p.coefficients(v[sl].T.astype('float64'),np.broadcast_to(ca[sl],(T,nn)),.1)
        for k in range(T):state+=z[k]*(u[k]-state);prefix.append(state.copy())
        prefix=np.stack(prefix,1);ru,rz=p.coefficients(rest[sl],ca[sl],.1)
        tail=ru[:,None]+(state[:,None]-ru[:,None])*(1-rz[:,None])**np.arange(1,L-T+1)[None,:,None]
        truth=np.concatenate((prefix,tail),1);voltage=np.concatenate((v[sl],np.broadcast_to(rest[sl,None],(nn,L-T))),1)
        currents=p.readout(voltage,truth,mult[sl,None],c.ionic.REVERSALS)[1]
        baseline=currents[:,0];amplitude=np.abs(currents-baseline[:,None]).max(1);threshold=.1*amplitude
        above=np.abs(currents-baseline[:,None])>=threshold[:,None]
        crossing=np.where(above,np.arange(L)[None,:,None],-1).max(1).clip(0,L-1)
        fixed=np.broadcast_to([0,9,79,T-1,T+399,3999,7999,L-1],(nn,8));times=np.concatenate((fixed,crossing),1)
        slope=np.empty((nn,11));valid=np.empty((nn,11),bool)
        for head in range(11):
            end=crossing[:,head];low=np.maximum(end-1,0);high=np.minimum(end+1,L-1);ix=np.arange(nn)
            slope[:,head]=(np.abs(currents[ix,high,head]-baseline[:,head])-np.abs(currents[ix,low,head]-baseline[:,head]))/np.maximum((high-low)*.1,.1)
            valid[:,head]=(threshold[:,head]>1e-8)&(end>0)&(end<L-1)
        ix=np.arange(nn)[:,None]
        rows.append(dict(v=v[sl],ca=ca[sl],initial=initial[sl],rest=rest[sl],mult=mult[sl],times=times,
                         target=currents[ix,times],query_v=voltage[ix,times],scale=np.maximum(np.abs(currents).max(1),1e-7),
                         timing_scale=np.maximum(np.abs(slope)*spec['quality']['onset_offset_max_ms'],1e-10),valid=valid))
    return {k:torch.tensor(np.concatenate([row[k] for row in rows]),device=device) for k in rows[0]}


def path_loss(torch,model,data,ids):
    get=lambda k:expand(torch,data[k][ids]);v=get('v');ca=get('ca')[...,None].expand_as(v)
    u,z=model.raw(v,ca);states=affine_scan(torch,torch.sigmoid(u.double()),torch.sigmoid(z.double()),get('initial'))
    times=get('times').long();T=v.shape[-1];at=(times.clamp_max(T-1))[...,None].expand(3,len(ids),times.shape[-1],18)
    before=states.gather(2,at);ru,rz=model.raw(get('rest'),get('ca'));inf=torch.sigmoid(ru.double())
    ticks=(times+1-T).clamp_min(0)
    decay=torch.exp(torch.nn.functional.logsigmoid(-rz.double())[:,:,None]*ticks[...,None])
    tail=inf[:,:,None]+(states[:,:,-1,None]-inf[:,:,None])*decay
    gates=torch.where((times<T)[...,None],before,tail)
    pred=t.readout(torch,get('query_v'),gates,get('mult')[:,:,None])[1]
    error=pred-get('target');ordinary=(error/get('scale')[:,:,None]).square().mean((1,2,3))
    diagonal=error[:,:,8:].diagonal(dim1=2,dim2=3)
    timing=((diagonal/get('timing_scale')).square().clamp_max(1e6)*get('valid')).sum((1,2))/get('valid').sum((1,2)).clamp_min(1)
    return ordinary+.01*timing


def feedback_pool(torch,spec,cfg,calcium,base,device,selection=False,small=False):
    ed=f.ed;effective=dict(spec)
    if selection:
        effective['new_seeds']={k:v+100000 for k,v in spec['new_seeds'].items()};effective['feedback_histories']=spec['selection_feedback_histories']
    raw=ed.history_pool(torch,effective,cfg,calcium,base,device,small)
    labels=ed.reference_windows(torch,raw,spec,cfg,calcium,base,device)
    n=min(len(raw['v']),spec['selection_feedback_anchors' if selection else 'feedback_anchors'])
    order=np.random.default_rng(effective['new_seeds']['support']).permutation(len(raw['v']))[:n]
    vals=ed.tensor_inputs(torch,{k:raw[k][order] for k in ('v','ca','g','mult','inj','syn')},device,torch.float64)
    vals['target']=torch.tensor(labels['delta'][:,order],device=device)
    vals['scale']=vals['target'].abs().amax(0).clamp_min(1e-7)
    return vals


def feedback_loss(torch,model,data,ids,physics):
    get=lambda k:expand(torch,data[k][ids]).reshape(3,len(ids)*2,*data[k].shape[2:])
    v,ca,g=(get(k) for k in ('v','ca','g'));mult,inj,syn=(get(k) for k in ('mult','inj','syn'))
    core=t.make_core(torch,*physics,100*np.pi,TrainingView(model),t.FAMILIES['all'])
    target=data['target'][:40,ids];scale=data['scale'][ids];loss=0
    for tick in range(40):
        v,ca,g,current=core(v,ca,g,mult,inj,syn)
        pair=current.reshape(3,len(ids),2,11);delta=pair[:,:,1]-pair[:,:,0]
        loss=loss+((delta-target[tick])/scale).square().mean((1,2))
    return loss/40


def parameters(model):return [x for x in model.parameters() if x.requires_grad]


def clip_members(torch,params):
    total=sum(x.grad.double().reshape(3,-1).square().sum(-1) for x in params if x.grad is not None)
    factor=total.sqrt().clamp_min(1).reciprocal()
    for par in params:
        if par.grad is not None:par.grad.mul_(factor.reshape(3,*([1]*(par.ndim-1))).to(par))


def step(torch,models,opts,data,ids,distill=False,paths=None,pids=None,fb=None,fids=None,physics=None):
    diagnostics={}
    for name,model in models.items():
        params=parameters(model);opts[name].zero_grad(set_to_none=False)
        point=coefficient_loss(torch,model,data,ids,distill)
        if paths is None:point.sum().backward()
        else:
            gp=torch.autograd.grad(point.sum(),params,allow_unused=True)
            aux=path_loss(torch,model,paths,pids)+feedback_loss(torch,model,fb,fids,physics)
            ga=torch.autograd.grad(aux.sum(),params,allow_unused=True)
            pn=sum((a.double().reshape(3,-1).square().sum(-1) if a is not None else torch.zeros(3,device=ids.device)) for a in gp).sqrt()
            an=sum((a.double().reshape(3,-1).square().sum(-1) if a is not None else torch.zeros(3,device=ids.device)) for a in ga).sqrt()
            weight=(.1*pn/an.clamp_min(1e-20)).clamp_max(1e6).detach()
            for par,a,b in zip(params,gp,ga):
                if a is None and b is None:continue
                grad=(torch.zeros_like(par) if a is None else a)+(torch.zeros_like(par) if b is None else b*weight.reshape(3,*([1]*(par.ndim-1))).to(par))
                assert torch.isfinite(grad).all(),name
                if par.grad is None:par.grad=grad.clone()
                else:par.grad.copy_(grad)
            diagnostics[name]=dict(point=point.detach().tolist(),aux=aux.detach().tolist(),point_norm=pn.tolist(),aux_norm=an.tolist(),weight=weight.tolist())
        clip_members(torch,params);opts[name].step()
    return diagnostics


def capture(torch,models,opts,data,ids,distill=False):
    weights={k:copy.deepcopy(m.state_dict()) for k,m in models.items()}
    original={k:copy.deepcopy(o.state_dict()) for k,o in opts.items()}
    # Adam state must already exist. All captured buffers keep their addresses.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step(torch,models,opts,data,ids,distill)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):step(torch,models,opts,data,ids,distill)
    torch.cuda.current_stream().wait_stream(stream)
    for name,model in models.items():
        model.load_state_dict(weights[name]);restore_optimizer(torch,opts[name],original[name])
    return graph


def restore_optimizer(torch,opt,saved):
    current=opt.state_dict()
    with torch.no_grad():
        for gid,group in enumerate(current['param_groups']):
            actual_group=opt.param_groups[gid]
            for pid,par in zip(group['params'],actual_group['params']):
                for key,value in saved['state'][pid].items():
                    if torch.is_tensor(value):opt.state[par][key].copy_(value)
        for actual,s in zip(opt.param_groups,saved['param_groups']):
            if torch.is_tensor(actual['lr']):actual['lr'].fill_(float(s['lr']))
            else:actual['lr']=s['lr']


def finite(torch,models,opts=None):
    assert all(bool(torch.isfinite(x).all()) for m in models.values() for x in m.parameters())
    if opts:
        assert all(bool(torch.isfinite(x).all()) for o in opts.values() for s in o.state.values() for x in s.values() if torch.is_tensor(x))


def rank(rows,parent_rows,aux,point,q):
    introduced=sum(f.channel_severity(r,q,(k,))>1 and f.channel_severity(b,q,(k,))<=1 for r,b in zip(rows,parent_rows) for k in range(11))
    introduced+=sum(not r['gate']['passed'] and b['gate']['passed'] for r,b in zip(rows,parent_rows))
    return (sum(not r['passed'] for r in rows),introduced,max(c.severity(r,q) for r in rows),float(aux),float(point))


def preflight(torch,root,out,device,spec):
    parent=controls(torch,root,device)['H_nominee'];models={a['name']:model_class(torch)(parent,a).to(device) for a in arms()}
    v=torch.linspace(-95,55,17,device=device)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
    with torch.no_grad():
        a=parent.raw(v,ca);b=models['J_chain_joint'].raw(v,ca)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        u,z=map(torch.sigmoid,b);assert (u>=0).all() and (u<=1).all() and (z>=0).all() and (z<=1).all()
    data,_=atomic_pool(torch,parent,spec['new_seeds']['fit'],16,device);ids=torch.arange(4,device=device)
    opts={k:torch.optim.Adam(parameters(m),lr=torch.tensor(spec['lr'],device=device),eps=1e-8,capturable=device=='cuda') for k,m in models.items()}
    step(torch,models,opts,data,ids)
    graph_error=None
    if device=='cuda':
        graph=capture(torch,models,opts,data,ids);copies=copy.deepcopy(models)
        copts={k:torch.optim.Adam(parameters(m),lr=torch.tensor(spec['lr'],device=device),eps=1e-8,capturable=True) for k,m in copies.items()}
        for k in copts:copts[k].load_state_dict(copy.deepcopy(opts[k].state_dict()))
        graph.replay();step(torch,copies,copts,data,ids)
        graph_error=max(float((a-b).abs().max()) for name in models for a,b in zip(models[name].parameters(),copies[name].parameters()))
        assert graph_error<=spec['computational_parity_atol'],graph_error
    paths=path_pool(torch,spec,device,count=2)
    for model in models.values():
        ll=path_loss(torch,model,paths,torch.arange(2,device=device))
        grad=torch.autograd.grad(ll.sum(),parameters(model),allow_unused=True)
        assert all(torch.isfinite(a).all() for a in grad if a is not None)
        assert any(a is not None and a.abs().max()>0 for a in grad)
    if device=='cuda':
        f.ee.differentiable_constants();_,cfg,calcium,base=contract(root)
        fb=feedback_pool(torch,spec,cfg,calcium,base,device,small=True)
        step(torch,models,opts,data,ids,paths=paths,pids=torch.arange(2,device=device),fb=fb,fids=torch.arange(1,device=device),physics=(cfg,calcium,base))
    finite(torch,models,opts)
    result=dict(valid=True,device=device,members=9,graph_error=graph_error,all_parameters_finite=True,
                joint_chain_initially_exact=True,all_equilibria_and_rates_trainable=True,path_gradients_finite=True,
                auxiliary_update_executed=device=='cuda',parameters_per_seed={k:sum(x.numel() for x in m.parameters())//3 for k,m in models.items()},
                protocol_sha256=p.digest(root/CONFIG),thresholds_changed=False,scientific_training_performed=False)
    p.save_json(out/'preflight.json',result);return result


def verify_freeze(out):
    freeze=json.loads((out/'selection_freeze.json').read_text());payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    assert p.digest(out/'frozen_checkpoint.pt')==freeze['checkpoint_sha256'] and not freeze['opened_regressions_used_for_selection']
    return freeze


def run(root,out,device='cuda',smoke=False,resume=False,stop_after=None,evaluation_only=False):
    import torch
    from . import task41j_evaluation as ev
    spec,cfg,calcium,base=contract(root);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert device=='cuda' and torch.cuda.is_available();out.mkdir(parents=True,exist_ok=True)
    if resume:assert json.loads((out/'preregistration.json').read_text())==spec
    else:
        assert not (out/'checkpoint_0.pt').exists();p.save_json(out/'preregistration.json',spec)
    if smoke:return preflight(torch,root,out,device,spec)
    cc=controls(torch,root,device);parent=cc['H_nominee']
    models={a['name']:model_class(torch)(parent,a).to(device) for a in arms()}
    if evaluation_only:
        assert resume;freeze=verify_freeze(out);saved=torch.load(out/'frozen_checkpoint.pt',map_location=device,weights_only=True)
        for name,m in models.items():m.load_state_dict(saved[name]);m.eval()
        def forbidden(*a,**kw):raise RuntimeError('Training forbidden during frozen evaluation')
        torch.optim.Adam.step=forbidden
        ev.install_cache(out)
        with torch.inference_mode():report=ev.evaluate(torch,root,out,cc,models,spec,cfg,calcium,base,device)
        verify_freeze(out);report.update(training_updates_performed=0,selection_recomputed=False)
        return report
    data,raw=atomic_pool(torch,parent,spec['new_seeds']['fit'],spec['anchors'],device)
    selected,_=atomic_pool(torch,parent,spec['new_seeds']['selection'],spec['selection_anchors'],device)
    paths=path_pool(torch,spec,device);stail=path_pool(torch,spec,device,selection=True)
    f.ee.differentiable_constants();fb=feedback_pool(torch,spec,cfg,calcium,base,device);sf=feedback_pool(torch,spec,cfg,calcium,base,device,selection=True)
    if not resume:
        np.savez_compressed(out/'fit_atomic.npz',**raw)
        np.savez_compressed(out/'fit_paths.npz',**{k:v.cpu().numpy() for k,v in paths.items()})
        np.savez_compressed(out/'selection_paths.npz',**{k:v.cpu().numpy() for k,v in stail.items()})
    dev=d.development(spec['new_seeds']['development']);np.savez_compressed(out/'development_inputs.npz',**dev)
    with torch.no_grad():parent_rows=c.shadow_evaluate(torch,[parent],dev,spec['quality'],device)
    opts={k:torch.optim.Adam(parameters(m),lr=torch.tensor(spec['lr'],device=device),eps=1e-8,capturable=True) for k,m in models.items()}
    rng=np.random.default_rng(spec['new_seeds']['stream']);ids=torch.arange(spec['batch'],device=device)
    # Allocate optimizer buffers without a training update or parameter mutation.
    for name,m in models.items():
        for par in parameters(m):
            opts[name].state[par].update(step=torch.zeros((),device=device),exp_avg=torch.zeros_like(par),exp_avg_sq=torch.zeros_like(par))
    graph=capture(torch,models,opts,data,ids)
    warm_graph=capture(torch,models,opts,data,ids,distill=True) if not resume else None
    best={k:[None]*3 for k in models};saved={k:[None]*3 for k in models};chosen={k:[None]*3 for k in models};curves=[];first=0;warm=0;diag={}
    if resume:
        checkpoint=max(out.glob('checkpoint_*.pt'),key=lambda x:int(x.stem.split('_')[-1]));state=torch.load(checkpoint,map_location=device,weights_only=False)
        assert state['protocol_sha256']==p.digest(root/CONFIG) and state['torch_version']==torch.__version__ and state['gpu']==torch.cuda.get_device_name()
        for name,m in models.items():m.load_state_dict(state['models'][name]);restore_optimizer(torch,opts[name],state['optimizer'][name])
        rng.bit_generator.state=state['rng'];best=state['best'];saved=state['selected_weights'];chosen=state['chosen_steps'];first=state['step'];warm=state['warmup_updates']
        curves=json.loads((out/'training_curves.json').read_text())
        print(f'[GIADA Task41j] exact optimizer/RNG resume at {first}; warmup={warm}',flush=True)
    start=time.perf_counter()
    if not resume:
        diagnostics=[]
        for w in range(spec['distillation_updates']):
            ids.copy_(torch.tensor(rng.integers(0,spec['anchors'],spec['batch']),device=device))
            warm_graph.replay()
            if w==0 or (w+1)%512==0:
                finite(torch,models,opts)
                with torch.no_grad():errors={name:coefficient_loss(torch,m,selected,torch.arange(spec['selection_anchors'],device=device),True).tolist() for name,m in models.items()}
                diagnostics.append(dict(step=w+1,heldout_teacher_error=errors));p.save_json(out/'distillation_diagnostics.json',diagnostics)
                print(f'[GIADA Task41j] distillation {w+1}/{spec["distillation_updates"]}',flush=True)
        warm=spec['distillation_updates']
        # Fresh Adam for canonical refinement; warmup effort is separately logged.
        for opt in opts.values():
            for state in opt.state.values():
                for value in state.values():value.zero_()
    for step_number in range(first,spec['checkpoints'][-1]+1):
        if step_number in spec['checkpoints'] and not (resume and step_number==first):
            finite(torch,models,opts)
            with torch.no_grad():
                measured=c.shadow_evaluate(torch,list(models.values()),dev,spec['quality'],device)
                points={k:coefficient_loss(torch,m,selected,torch.arange(spec['selection_anchors'],device=device)) for k,m in models.items()}
                auxiliary={k:path_loss(torch,m,stail,torch.arange(spec['selection_paths'],device=device))+feedback_loss(torch,m,sf,torch.arange(min(8,len(sf['v'])),device=device),(cfg,calcium,base)) for k,m in models.items()}
            rr={}
            for ai,(name,m) in enumerate(models.items()):
                rows=measured[3*ai:3*ai+3];ranks=[]
                for seed_index in range(3):
                    current=rank(rows[seed_index],parent_rows[seed_index],auxiliary[name][seed_index],points[name][seed_index],spec['quality']);ranks.append(current)
                    if best[name][seed_index] is None or current<best[name][seed_index]:
                        best[name][seed_index]=current;chosen[name][seed_index]=step_number
                        saved[name][seed_index]={key:value[seed_index:seed_index+1].detach().cpu().clone() if value.ndim and value.shape[0]==3 else value.detach().cpu().clone() for key,value in m.state_dict().items()}
                rr[name]=ranks
            curves.append(dict(step=step_number,ranks=rr,point={k:v.tolist() for k,v in points.items()},auxiliary={k:v.tolist() for k,v in auxiliary.items()},gradients=diag,elapsed_seconds=time.perf_counter()-start))
            p.save_json(out/'training_curves.json',curves);p.save_json(out/f'development_{step_number}.json',measured)
            torch.save(dict(step=step_number,models={k:m.state_dict() for k,m in models.items()},optimizer={k:o.state_dict() for k,o in opts.items()},rng=rng.bit_generator.state,
                best=best,selected_weights=saved,chosen_steps=chosen,warmup_updates=warm,protocol_sha256=p.digest(root/CONFIG),torch_version=torch.__version__,gpu=torch.cuda.get_device_name()),out/f'checkpoint_{step_number}.pt')
            print(f'[GIADA Task41j] checkpoint_saved {step_number}',flush=True)
        if stop_after==step_number and step_number<spec['checkpoints'][-1]:return dict(valid=True,decision='TRAINING_STAGE_COMPLETE',updates_completed=step_number,training_complete=False,thresholds_changed=False)
        if step_number==spec['checkpoints'][-1]:break
        rate=spec['lr']*(spec['lr_min_ratio']**(step_number/spec['checkpoints'][-1]))
        for opt in opts.values():opt.param_groups[0]['lr'].fill_(rate)
        ids.copy_(torch.tensor(rng.integers(0,spec['anchors'],spec['batch']),device=device))
        if (step_number+1)%spec['aux_every']==0:
            diag=step(torch,models,opts,data,ids,paths=paths,pids=torch.tensor(rng.integers(0,spec['fit_paths'],spec['tail_batch']),device=device),fb=fb,
                      fids=torch.tensor(rng.integers(0,len(fb['v']),spec['feedback_batch']),device=device),physics=(cfg,calcium,base))
        else:graph.replay()
        if (step_number+1)%1000==0:
            finite(torch,models,opts);print(f'[GIADA Task41j] update {step_number+1}/60000 finite=True elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    selected_weights={}
    for name,m in models.items():
        state={key:torch.cat([saved[name][s][key] for s in range(3)],0) if value.ndim and value.shape[0]==3 else saved[name][0][key] for key,value in m.state_dict().items()}
        m.load_state_dict(state);selected_weights[name]=state;m.eval()
    nominee=min(models,key=lambda name:tuple(max(best[name][s][k] for s in range(3)) for k in range(5))+(list(models).index(name),))
    torch.save(selected_weights,out/'frozen_checkpoint.pt')
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen,ranks=best,nominee=nominee,checkpoint_sha256=p.digest(out/'frozen_checkpoint.pt'),opened_regressions_used_for_selection=False,independent_confirmation=False,parent='H_nominee')
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(out/'selection_freeze.json',freeze)
    if stop_after is not None:return dict(valid=True,decision='FROZEN_TRAINING_COMPLETE',updates_completed=60000,training_complete=True,thresholds_changed=False)
    ev.install_cache(out)
    with torch.inference_mode():return ev.evaluate(torch,root,out,cc,models,spec,cfg,calcium,base,device)
