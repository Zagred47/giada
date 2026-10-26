"""Selective learned Ca/Nap equilibrium repair; unchanged 18-state inference.

Canonical functions are FIT labels/oracles only. The two loss definitions are
contrasted on identical paths, query times, initial states and minibatches.
"""
import copy
import hashlib
import json
import time
import types
import numpy as np
from . import task41h_path_repair as h

p,t,c,d,f,g = h.p,h.t,h.c,h.d,h.f,h.g
CONFIG='experiments/task41i_preregistration_v1.json'
GATES=(0,8)
CHANNELS=(0,4)


def arms():
    return [dict(name=f'I{nap}{loss}{basis}',nap=nap,official_loss=loss,adaptive=basis,
                 basis=128,optimizer='anneal',support='broad',feedback=True)
            for nap in (0,1) for loss in (0,1) for basis in (0,1)]


def contrasts():
    return [(key,a['name'],b['name']) for key in ('nap','official_loss','adaptive')
            for a in arms() for b in arms() if a[key]==0 and b[key]==1
            and all(a[k]==b[k] for k in ('nap','official_loss','adaptive') if k!=key)]


def contract(root):
    spec=json.loads((root/CONFIG).read_text(encoding='utf-8'))
    old,cfg,ca,base=h.contract(root)
    for key in ('quality','compartment_gates','cost'):assert spec[key]==old[key],key
    assert spec['arms']==arms() and spec['dt_ms']==.1 and spec['seeds']==[17,29,43]
    assert not set(spec['new_seeds'].values())&set(old['new_seeds'].values())
    for path,sha in spec['sources'].items():assert p.digest(root/path)==sha,path
    return spec,cfg,ca,base


def frozen_controls(torch,root,spec,device):
    old=h.frozen_controls(torch,root,spec,device)
    definition=next(a for a in h.arms() if a['name']==spec['task41h_nominee'])
    parent=h.make_model(torch,old['E_Q0'],[definition],device)
    state=torch.load(root/spec['task41h_checkpoint'],map_location=device,weights_only=True)
    parent.load_state_dict(state[spec['task41h_nominee']]);parent.eval()
    return {**{key:old[key] for key in ('P110','E_Q0','E_H10')},'H_nominee':parent}


def fit_knots(raw,n=128):
    """One shared FIT-only allocation per head; no opened-case voltages."""
    v=raw['v'][:,1].astype('float64');result=[]
    for gate in GATES:
        sensitivity=np.abs(raw['u'][:,2,gate]-raw['u'][:,0,gate])
        slow=(1-raw['z'][:,1,gate])**400
        score=sensitivity*(1+slow)
        score=np.minimum(score,np.quantile(score,.99))
        score=.5/len(v)+.5*score/max(float(score.sum()),1e-30)
        order=np.argsort(v);cdf=np.cumsum(score[order]);cdf/=cdf[-1]
        q=np.linspace(0,1,n)
        adaptive=.5*np.linspace(-100,60,n)+.5*np.interp(q,cdf,v[order])
        adaptive[0],adaptive[-1]=-100.,60.
        assert np.all(np.diff(adaptive)>0)
        result.append(adaptive)
    return np.asarray(result)


def feature(torch,v,knots):
    """Nonuniform hat basis, with endpoint supports matching the old hats."""
    shape=(len(knots),)+((1,)*(v.ndim-1))+(knots.shape[-1],)
    centers=knots.reshape(shape)
    widths=knots[:,1:]-knots[:,:-1]
    left=torch.cat((widths[:,:1],widths),-1).reshape(shape)
    right=torch.cat((widths,widths[:,-1:]),-1).reshape(shape)
    distance=v[...,None]-centers
    return (1-torch.where(distance<0,-distance/left,distance/right)).clamp_min(0)


def model_class(torch):
    class Model(torch.nn.Module):
        def __init__(self,parent,definitions,knots):
            super().__init__();self.parent=parent;self.definitions=definitions
            for par in parent.parameters():par.requires_grad_(False)
            members=3*len(definitions)
            self.correction=torch.nn.Parameter(torch.zeros(members,2,128,device=knots.device))
            self.register_buffer('knots',knots)
            mask=torch.ones_like(self.correction)
            for index,a in enumerate(definitions):mask[3*index:3*index+3,1]*=a['nap']
            self.register_buffer('mask',mask);self.execution='dense'
        def __getattr__(self,key):
            try:return super().__getattr__(key)
            except AttributeError:
                if key in ('a','sa','sb','sc','sd','gbar'):
                    return getattr(super().__getattr__('parent'),key)
                raise
        def delta(self,v):
            values=[]
            for head in range(2):
                ff=feature(torch,v,self.knots[:,head].to(v.dtype))
                ww=(self.correction*self.mask)[:,head]
                shape=(len(ww),)+((1,)*(v.ndim-1))+(128,)
                value=(ff.double()*ww.reshape(shape).double()).sum(-1).to(v.dtype)
                values.append(.1*torch.tanh(value/.1))
            return torch.stack(values,-1)
        def raw(self,v,ca):
            u,z=self.parent.raw(v,ca);extra=self.delta(v)
            return torch.cat([u[...,gate:gate+1]+extra[...,GATES.index(gate):GATES.index(gate)+1]
                              if gate in GATES else u[...,gate:gate+1] for gate in range(18)],-1),z
        def forward(self,v,ca,state):
            u,z=self.raw(v,ca)
            return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)
    return Model


def make_model(torch,parent,definitions,adaptive,device):
    definition=parent.definitions[0]
    repeated=h.model_class(torch)([definition]*len(definitions)).to(device)
    state={}
    for key,value in parent.state_dict().items():
        state[key]=value.repeat(len(definitions),*([1]*(value.ndim-1))) if value.ndim and value.shape[0]==3 and key!='gbar' else value
    repeated.load_state_dict(state)
    uniform=np.broadcast_to(np.linspace(-100,60,128),(2,128)).copy()
    kk=np.stack([adaptive if a['adaptive'] else uniform for a in definitions for _ in range(3)])
    return model_class(torch)(repeated,definitions,torch.tensor(kk,device=device,dtype=torch.float32)).to(device)


def extract(torch,model,index,parent,adaptive,device,weights=None):
    result=make_model(torch,parent,[model.definitions[index]],adaptive,device)
    with torch.no_grad():result.correction.copy_((model.correction if weights is None else weights)[3*index:3*index+3])
    return result.eval()


def atomic_pool(torch,parent,raw,device):
    vv=torch.tensor(raw['v'],device=device);cc=torch.tensor(raw['ca'],device=device)
    with torch.no_grad():u,z=parent.raw(vv.reshape(1,-1).expand(3,-1),cc.reshape(1,-1).expand(3,-1))
    return dict(v=vv,base=u.reshape(3,len(vv),3,18)[...,list(GATES)].double(),
                target=torch.tensor(raw['u'][...,list(GATES)],device=device,dtype=torch.float64))


def atomic_loss(torch,model,data,ids):
    members=len(model.correction);seed=torch.arange(members,device=ids.device)%3
    v=data['v'][ids]
    base=data['base'][seed[:,None],ids]
    raw=base+model.delta(v)
    target=data['target'][ids]
    # Always-censored branch of the original objective, without constructing a
    # CPU Boolean and copying it inside a captured CUDA graph.
    residual=raw.double()-torch.logit(target.double().clamp(1e-7,1-1e-7))
    residual=torch.where(target<=1e-7,residual.clamp_min(0),
                         torch.where(target>=1-1e-7,residual.clamp_max(0),residual))
    return residual.square().mean((1,2,3)),residual.square().mean((1,2))


def path_pool(torch,parent,spec,device,selection=False,count=None):
    """Full-path event labels; exact affine recurrent weights at query times.

    Queries include crossings of BOTH definitions for BOTH channels and fixed
    times. Old/new losses see exactly the same observations, not different data.
    """
    n=count or spec['selection_paths' if selection else 'fit_paths'];T=spec['path_prefix_ticks'];L=12000
    seed=spec['new_seeds']['selection' if selection else 'fit']+19001
    rng=np.random.default_rng(seed);rest=rng.uniform(-85,-30,n)
    rest[:n//2]=rng.uniform(-50,-35,n//2)
    ca=10**rng.uniform(-5,-3.5,n);peak=rng.uniform(-31,-14,n);stop=rng.integers(130,231,n)
    modes=rng.integers(0,2,n);clock=np.arange(T)
    v=np.where(modes[:,None],peak[:,None]+(rest[:,None]-peak[:,None])*np.clip((clock[None]-stop[:,None])/300,0,1),rest[:,None]).astype('float32')
    initial,_=p.coefficients(rng.uniform(-95,35,n),ca,.1);mult=rng.uniform(.5,2,(n,11))
    collected={key:[] for key in ('prefix_u','weights','initial_factor','base_query','query_v','query_ca','target','scale','timing_scale','valid','prefix_v','times','crossings','reference_base','threshold')}
    for first in range(0,n,16):
        sl=slice(first,min(first+16,n));nn=len(rest[sl]);vc=v[sl].T;cc=np.broadcast_to(ca[sl],vc.shape).astype('float32')
        cu,cz=p.coefficients(vc.astype('float64'),cc.astype('float64'),.1)
        state=initial[sl].copy();prefix=[];parent_state=np.broadcast_to(state,(3,nn,18)).copy();pu=[];pz=[];parent_prefix=[]
        with torch.no_grad():
            pv=torch.tensor(vc.T.reshape(-1),device=device)[None].expand(3,-1);pc=torch.tensor(cc.T.reshape(-1),device=device)[None].expand(3,-1)
            bu,bz=parent.raw(pv,pc);bu=bu.double().cpu().numpy().reshape(3,nn,T,18);bz=bz.double().cpu().numpy().reshape(3,nn,T,18)
            rv=torch.tensor(rest[sl],device=device,dtype=torch.float32)[None].expand(3,-1);rc=torch.tensor(ca[sl],device=device,dtype=torch.float32)[None].expand(3,-1)
            ru,rz=parent.raw(rv,rc);ru,rz=ru.double().cpu().numpy(),rz.double().cpu().numpy()
        inf=1/(1+np.exp(-bu));rate=1/(1+np.exp(-bz));rest_inf=1/(1+np.exp(-ru));rest_rate=1/(1+np.exp(-rz))
        for tick in range(T):
            state+=cz[tick]*(cu[tick]-state);prefix.append(state.copy())
            parent_state+=rate[:,:,tick]*(inf[:,:,tick]-parent_state);parent_prefix.append(parent_state.copy())
        prefix=np.stack(prefix,1);parent_prefix=np.stack(parent_prefix,2)
        u,z=p.coefficients(rest[sl],ca[sl],.1);tail_ticks=np.arange(1,L-T+1)
        tail=u[:,None]+(state[:,None]-u[:,None])*(1-z[:,None])**tail_ticks[None,:,None]
        truth=np.concatenate((prefix,tail),1)
        full_v=np.concatenate((v[sl],np.broadcast_to(rest[sl,None],(nn,L-T))),1)
        currents=p.readout(full_v,truth,mult[sl,None],c.ionic.REVERSALS)[1][...,list(CHANNELS)]
        steady=p.readout(rest[sl],u,mult[sl],c.ionic.REVERSALS)[1][...,list(CHANNELS)]
        baseline=currents[:,0];threshold=.1*np.abs(currents-baseline[:,None]).max(1)
        new_above=np.abs(currents-baseline[:,None])>=threshold[:,None]
        new_cross=np.where(new_above,np.arange(L)[None,:,None],-1).max(1)
        old_base=steady;old_threshold=.1*np.abs(currents[:,T-1]-steady)
        old_above=np.abs(currents-old_base[:,None])>=old_threshold[:,None]
        old_above[:,:T-1]=False
        old_cross=np.where(old_above,np.arange(L)[None,:,None],-1).max(1)
        crossing=np.stack((old_cross,new_cross),1).clip(0,L-1)
        fixed=np.broadcast_to(np.array([0,9,79,T-1,T+399,3999,7999,L-1]),(nn,8))
        times=np.concatenate((fixed,crossing.reshape(nn,4)),1);Q=times.shape[1]
        rows=np.arange(nn)[:,None];target=currents[rows,times];base_query=np.empty((3,nn,Q,18));weights=np.zeros((3,nn,Q,T,2));initial_factor=np.empty((3,nn,Q,2))
        for qi in range(Q):
            ticks=times[:,qi];prefix_count=np.minimum(ticks+1,T)
            tail_count=np.maximum(ticks+1-T,0)
            parent_q=rest_inf+(parent_state-rest_inf)*(1-rest_rate)**tail_count[None,:,None]
            parent_q=np.where((ticks<T)[None,:,None],parent_prefix[:,np.arange(nn),np.minimum(ticks,T-1)],parent_q)
            base_query[:,:,qi]=parent_q
            survival=np.ones((3,nn,2))
            for tick in range(T-1,-1,-1):
                enabled=(tick<prefix_count)[None,:,None]
                weights[:,:,qi,tick]=np.where(enabled,rate[:,:,tick,list(GATES)]*survival,0)
                survival*=np.where(enabled,1-rate[:,:,tick,list(GATES)],1)
            dec=(1-rest_rate[...,list(GATES)])**tail_count[None,:,None]
            weights[:,:,qi]*=dec[:,:,None]
            initial_factor[:,:,qi]=survival*dec
        # Query-time residual also changes equilibrium during the constant tail.
        qv=full_v[rows,times];qca=np.broadcast_to(ca[sl,None],qv.shape)
        slope=np.empty((nn,2,2));valid=np.empty((nn,2,2),bool)
        for method in range(2):
            bb=old_base if method==0 else baseline;th=old_threshold if method==0 else threshold
            for head in range(2):
                end=crossing[:,method,head];lo=np.maximum(end-1,0);hi=np.minimum(end+1,L-1)
                slope[:,method,head]=(np.abs(currents[np.arange(nn),hi,head]-bb[:,head])-np.abs(currents[np.arange(nn),lo,head]-bb[:,head]))/np.maximum((hi-lo)*.1,.1)
                valid[:,method,head]=(th[:,head]>1e-8)&(end<L-1)&(end>0)
        values=dict(prefix_u=bu[...,list(GATES)],weights=weights,initial_factor=initial_factor,
                    base_query=base_query,query_v=qv,query_ca=qca,target=target,
                    scale=np.maximum(np.abs(currents-baseline[:,None]).max(1),1e-7),
                    timing_scale=np.maximum(np.abs(slope)*spec['quality']['onset_offset_max_ms'],1e-10),valid=valid,
                    prefix_v=v[sl],times=times,crossings=crossing,reference_base=baseline,threshold=threshold)
        # Coefficients needed for the equilibrium contribution in the tail.
        values.update(rest_u=ru,rest_z=rz,initial=initial[sl],tail_count=np.maximum(times+1-T,0),rest=rest[sl],ca=ca[sl],mult=mult[sl])
        for key,val in values.items():collected.setdefault(key,[]).append(val)
    seed_keys={'prefix_u','weights','initial_factor','base_query','rest_u','rest_z'}
    return {key:torch.tensor(np.concatenate(values,axis=1 if key in seed_keys else 0),device=device) for key,values in collected.items()}


def path_loss(torch,model,data,ids,method=None,return_predictions=False):
    M=len(model.correction);seed=torch.arange(M,device=ids.device)%3
    get=lambda key:data[key][seed[:,None],ids[None]] if key in ('prefix_u','weights','initial_factor','base_query','rest_u','rest_z') else data[key][ids][None].expand(M,*data[key][ids].shape)
    pv=get('prefix_v');extra=model.delta(pv)
    prefix=torch.sigmoid(get('prefix_u')+extra)
    gates=(get('weights')*prefix[:,:,None]).sum(-2)+get('initial_factor')*get('initial')[...,list(GATES)][:,:,None]
    rest=get('rest');ru=get('rest_u');rz=get('rest_z')
    ri=torch.sigmoid(ru[...,list(GATES)]+model.delta(rest))
    decay=(1-torch.sigmoid(rz[...,list(GATES)]))[:,:,None]**get('tail_count')[...,None]
    gates=gates+ri[:,:,None]*(1-decay)
    other=get('base_query');full=torch.cat([gates[...,GATES.index(k):GATES.index(k)+1] if k in GATES else other[...,k:k+1] for k in range(18)],-1)
    pred=t.readout(torch,get('query_v'),full,get('mult')[:,:,None])[1][...,list(CHANNELS)]
    if return_predictions:return pred,full
    error=pred-get('target');ordinary=(error/get('scale')[:,:,None]).square().mean((1,2,3))
    methods=torch.tensor([a['official_loss'] for a in model.definitions for _ in range(3)],device=ids.device)
    if method is not None:methods.fill_(method)
    # Same query union for both arms: only which labelled margin is weighted differs.
    timing=[]
    for head in range(2):
        query=8+methods*2+head
        selected=error.gather(2,query[:,None,None,None].expand(M,len(ids),1,2)).squeeze(2)[...,head]
        scales=get('timing_scale').gather(2,methods[:,None,None,None].expand(M,len(ids),1,2)).squeeze(2)[...,head]
        valid=get('valid').gather(2,methods[:,None,None,None].expand(M,len(ids),1,2)).squeeze(2)[...,head]
        term=(selected/scales).square().clamp_max(1e6)
        timing.append((term*valid).sum(-1)/valid.sum(-1).clamp_min(1))
    return ordinary+.01*torch.stack(timing,-1).mean(-1)


class FeedbackView:
    def __init__(self,torch,model):
        self.torch,self.model=torch,model;self.parent=copy.deepcopy(model.parent).double()
    def __getattr__(self,key):return getattr(self.parent,key)
    def raw(self,v,ca):
        u,z=self.parent.raw(v,ca);extra=self.model.delta(v)
        return self.torch.cat([u[...,k:k+1]+extra[...,GATES.index(k):GATES.index(k)+1] if k in GATES else u[...,k:k+1] for k in range(18)],-1),z
    def __call__(self,v,ca,state):
        u,z=self.raw(v,ca);return state+self.torch.sigmoid(z)*(self.torch.sigmoid(u)-state)


def feedback(torch,model,view,data,support,ids,cfg,calcium,base):
    # The inherited function only needs member count and definitions on this proxy.
    proxy=types.SimpleNamespace(f_correction=model.correction,definitions=model.definitions)
    return f.feedback_terms(torch,proxy,view,data,support,ids,cfg,calcium,base)


def loss_step(torch,model,opt,data,ids,norm,preservation,pids,tail=None,tids=None,view=None,fb=None,support=None,fi=None,physics=None):
    point,heads=atomic_loss(torch,model,data,ids)
    keep=model.delta(preservation['v'][pids]).square().mean((1,2,3))
    point=point/norm+.01*keep/norm
    if tail is None:
        opt.zero_grad();point.sum().backward();diag={}
    else:
        gp=torch.autograd.grad(point.sum(),model.correction)[0]
        aux=path_loss(torch,model,tail,tids)+feedback(torch,model,view,fb,support,fi,*physics)
        ga=torch.autograd.grad(aux.sum(),model.correction)[0]
        pn,an=gp.double().norm(dim=(-1,-2)),ga.double().norm(dim=(-1,-2))
        scale=(.1*pn/an.clamp_min(1e-20)).clamp_max(1e6)
        grad=(gp+scale[:,None,None]*ga)*model.mask
        assert torch.isfinite(grad).all()
        if model.correction.grad is None:model.correction.grad=grad.clone()
        else:model.correction.grad.copy_(grad)
        diag=dict(point=point.tolist(),aux=aux.tolist(),point_norm=pn.tolist(),aux_norm=an.tolist(),scale=scale.tolist(),cosine=((gp.double()*ga.double()).sum((-1,-2))/(pn*an).clamp_min(1e-30)).tolist())
    f.clip(torch,model.correction.grad);opt.step();return diag


def capture(torch,model,opt,*args):
    initial=model.correction.detach().clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):loss_step(torch,model,opt,*args)
    torch.cuda.current_stream().wait_stream(stream);opt.zero_grad(set_to_none=True);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):loss_step(torch,model,opt,*args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.no_grad():model.correction.copy_(initial);opt.reset()
    return graph


def ranks(rows,parent,aux,atomic,q):
    return f.ranks(rows,parent,aux,atomic,q)


def restore_optimizer(torch,opt,state):
    # Copy in place: captured CUDA graphs retain the original buffer addresses.
    with torch.no_grad():
        for dst,src in ((opt.m,state['m']),(opt.v,state['v']),(opt.step_count,state['step']),(opt.lr,state['lr'])):dst.copy_(src)


def metric_visibility(q):
    clock=np.arange(1000);pulse=np.where(clock>=50,-.001*np.exp(-np.maximum(clock-50,0)/100),0.)
    target=np.zeros((1000,11));target[:,0]=pulse;target[:,4]=pulse
    before=target.copy();after=target.copy()
    before[:,0]=np.roll(pulse,6);before[:,4]=np.roll(pulse,21)
    after[:,4]=before[:,4]
    a=p.evaluate_currents(target,before,.1,q);b=p.evaluate_currents(target,after,.1,q)
    ca_a=f.channel_severity(a,q,(0,));ca_b=f.channel_severity(b,q,(0,))
    global_a=f.channel_severity(a,q,CHANNELS);global_b=f.channel_severity(b,q,CHANNELS)
    assert ca_b<ca_a and global_a==global_b
    exact=p.evaluate_currents(target,target,.1,q);assert exact['passed']
    return dict(valid=True,positive_Ca_metric_changes=True,negative_identical_trace_passes=True,
                unchanged_Nap_can_mask_global_max=True,Ca_severity_before=ca_a,Ca_severity_after=ca_b)


def preflight(torch,root,out,device,spec):
    parent=frozen_controls(torch,root,spec,device)['H_nominee']
    raw=d.anchors(spec['new_seeds']['fit'],64);knots=fit_knots(raw)
    model=make_model(torch,parent,arms(),knots,device)
    v=torch.linspace(-95,55,17,device=device)[None].expand(24,-1);ca=torch.full_like(v,1e-4)
    with torch.no_grad():
        actual=model.raw(v,ca);wanted=parent.raw(v[:3],ca[:3])
        assert all(torch.equal(a[3*j:3*j+3],b) for a,b in zip(actual,wanted) for j in range(8))
        model.correction.normal_(0,.001);model.correction.mul_(model.mask)
        changed=model.raw(v,ca)
        assert torch.equal(actual[1],changed[1])
        for k in range(18):
            if k not in GATES:assert torch.equal(actual[0][...,k],changed[0][...,k])
        assert torch.equal(actual[0][:12,:,8],changed[0][:12,:,8])
        assert not torch.equal(actual[0][12:,:,8],changed[0][12:,:,8])
    pool=atomic_pool(torch,parent,raw,device);ids=torch.arange(8,device=device)[None].expand(24,-1).clone()
    norm=atomic_loss(torch,model,pool,ids)[0].detach().clamp_min(1e-12)
    opt=g.Adam(torch,model.correction,arms(),spec);reference=copy.deepcopy(model);oo=g.Adam(torch,reference.correction,arms(),spec)
    if device=='cuda':
        graph=capture(torch,model,opt,pool,ids,norm,pool,ids)
        graph.replay();loss_step(torch,reference,oo,pool,ids,norm,pool,ids)
        graph_error=float((model.correction-reference.correction).abs().max());assert graph_error<1e-5
    else:loss_step(torch,model,opt,pool,ids,norm,pool,ids);graph_error=None
    tail=path_pool(torch,parent,spec,device,count=4)
    ll=path_loss(torch,model,tail,torch.arange(4,device=device))
    grad=torch.autograd.grad(ll.sum(),model.correction)[0]
    assert torch.isfinite(grad).all() and (grad*(1-model.mask)==0).all()
    visibility=metric_visibility(spec['quality'])
    assert all(torch.isfinite(x).all() for x in model.parameters())
    report=dict(valid=True,members=24,all_parameters_finite=True,zero_initial_function_error=0.,nap_reachable=True,protected_coefficients_unchanged=True,
                graph_error=graph_error,path_gradient_finite=True,protocol_sha256=p.digest(root/CONFIG),device=device,
                metric_visibility=visibility,thresholds_changed=False,scientific_training_performed=False)
    p.save_json(out/'preflight.json',report);return report


def run(root,out,device='cuda',smoke=False,resume=False,stop_after=None,evaluation_only=False):
    import torch
    from . import task41i_evaluation as ev
    spec,cfg,calcium,base=contract(root)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert device=='cuda' and torch.cuda.is_available()
    out.mkdir(parents=True,exist_ok=True)
    if resume:
        assert json.loads((out/'preregistration.json').read_text())==spec
    else:
        assert not (out/'checkpoint_0.pt').exists()
        p.save_json(out/'preregistration.json',spec);preflight(torch,root,out,device,spec)
    print('[GIADA Task41i] preflights passed',flush=True)
    if smoke:return dict(valid=True,decision='GPU_PREFLIGHT_ONLY')
    controls=frozen_controls(torch,root,spec,device);parent=controls['H_nominee']
    if evaluation_only:
        assert resume
        freeze=json.loads((out/'selection_freeze.json').read_text())
        payload={key:value for key,value in freeze.items() if key!='freeze_sha256'}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
        assert p.digest(out/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
        knots=np.asarray(json.loads((out/'FIT_knots.json').read_text())['knots'])
        saved=torch.load(out/'frozen_checkpoint.pt',map_location=device,weights_only=True);trained={}
        for a in arms():
            net=make_model(torch,parent,[a],knots,device);net.load_state_dict(saved[a['name']]);trained[a['name']]=net.eval()
        def forbidden(*args,**kwargs):raise RuntimeError('Training forbidden during frozen evaluation')
        g.Adam.step=forbidden
        ev.install_resume_cache(out)
        with torch.inference_mode():report=ev.evaluate(torch,root,out,controls,trained,spec,cfg,calcium,base,device)
        assert p.digest(out/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
        report.update(selection_recomputed=False,training_updates_performed=0)
        p.save_json(out/'final_report.json',report);return report
    raw=d.anchors(spec['new_seeds']['fit'],spec['anchors']);knots=fit_knots(raw)
    p.save_json(out/'FIT_knots.json',dict(knots=knots.tolist(),source_seed=spec['new_seeds']['fit'],opened_cases_used=False))
    model=make_model(torch,parent,arms(),knots,device)
    data=atomic_pool(torch,parent,raw,device)
    selected=atomic_pool(torch,parent,d.anchors(spec['new_seeds']['selection'],spec['selection_anchors']),device)
    keep=atomic_pool(torch,parent,d.anchors(spec['new_seeds']['fit']+200000,4096),device)
    tail=path_pool(torch,parent,spec,device);stail=path_pool(torch,parent,spec,device,selection=True)
    np.savez_compressed(out/'fit_path_labels.npz',**{k:v.cpu().numpy() for k,v in tail.items() if k not in ('weights',)})
    np.savez_compressed(out/'selection_path_labels.npz',**{k:v.cpu().numpy() for k,v in stail.items() if k not in ('weights',)})
    fb,support=f.feedback_data(torch,spec,cfg,calcium,base,device,out);sf,ss=f.feedback_data(torch,spec,cfg,calcium,base,device,out,selection=True)
    view=FeedbackView(torch,model);f.ee.differentiable_constants()
    ids=torch.arange(spec['batch'],device=device)[None].expand(24,-1).clone();pids=ids.clone()
    norm=atomic_loss(torch,model,data,ids)[0].detach().clamp_min(1e-12)
    opt=g.Adam(torch,model.correction,arms(),spec);graph=capture(torch,model,opt,data,ids,norm,keep,pids)
    ev.projection_probe(torch,model,data,selected,out)
    dev=d.development(spec['new_seeds']['development']);np.savez_compressed(out/'development_inputs.npz',**dev)
    parent_rows=c.shadow_evaluate(torch,[parent],dev,spec['quality'],device)
    rng=np.random.default_rng(spec['new_seeds']['stream']);best=[None]*24;saved=[None]*24;chosen=[None]*24;curves=[];diag={};start=time.perf_counter();first_step=0
    if resume:
        paths=list(out.glob('checkpoint_*.pt'));assert paths
        checkpoint=max(paths,key=lambda path:int(path.stem.split('_')[-1]))
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        assert state['protocol_sha256']==p.digest(root/CONFIG) and state['torch_version']==torch.__version__ and state['gpu']==torch.cuda.get_device_name()
        assert np.array_equal(state['knots'],knots)
        with torch.no_grad():
            model.correction.copy_(state['correction'])
        restore_optimizer(torch,opt,state['optimizer'])
        rng.bit_generator.state=state['rng'];best=state['best'];saved=[value.to(device) for value in state['selected_weights']];chosen=state['chosen_steps'];first_step=state['step']
        curves=json.loads((out/'training_curves.json').read_text())
        assert curves[-1]['step']==first_step and float(opt.step_count)==first_step
        print(f'[GIADA Task41i] resumed exact optimizer/RNG at {first_step}',flush=True)
    si=torch.arange(spec['selection_anchors'],device=device)[None].expand(24,-1)
    for step in range(first_step,spec['checkpoints'][-1]+1):
        if step in spec['checkpoints'] and not (resume and step==first_step):
            assert all(torch.isfinite(x).all() for x in model.parameters())
            models=[extract(torch,model,k,parent,knots,device) for k in range(8)]
            with torch.no_grad():
                point=atomic_loss(torch,model,selected,si)[0]/norm
                tails=path_loss(torch,model,stail,torch.arange(spec['selection_paths'],device=device),method=1)
                response=feedback(torch,model,view,sf,ss,torch.arange(spec['selection_feedback_anchors'],device=device),cfg,calcium,base)
                measured=c.shadow_evaluate(torch,models,dev,spec['quality'],device)
            rr=[];per_head=[]
            for j,rows in enumerate(measured):
                rank=ranks(rows,parent_rows[j%3],tails[j]+response[j],point[j],spec['quality']);rr.append(rank)
                per_head.append([max(f.channel_severity(r,spec['quality'],(channel,)) for r in rows) for channel in CHANNELS])
                if best[j] is None or rank<best[j]:best[j],saved[j],chosen[j]=rank,model.correction[j].detach().clone(),step
            curves.append(dict(step=step,ranks=rr,per_head_severity=per_head,point=point.tolist(),official_tail=tails.tolist(),feedback=response.tolist(),gradients=diag,elapsed_seconds=time.perf_counter()-start))
            p.save_json(out/'training_curves.json',curves);p.save_json(out/f'development_{step}.json',measured)
            torch.save(dict(step=step,correction=model.correction.detach().cpu(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,knots=knots,
                            best=best,selected_weights=[value.detach().cpu() for value in saved],chosen_steps=chosen,
                            protocol_sha256=p.digest(root/CONFIG),torch_version=torch.__version__,gpu=torch.cuda.get_device_name()),out/f'checkpoint_{step}.pt')
            print(f'[GIADA Task41i] checkpoint_saved {step}',flush=True)
        if stop_after is not None and step==stop_after and step<spec['checkpoints'][-1]:
            return dict(valid=True,decision='TRAINING_STAGE_COMPLETE',updates_completed=step,training_complete=False,thresholds_changed=False,scientific_go_claimed=False)
        if step==spec['checkpoints'][-1]:break
        opt.rate(step);ids.copy_(torch.tensor(rng.integers(0,spec['anchors'],spec['batch']),device=device)[None].expand(24,-1))
        pids.copy_(torch.tensor(rng.integers(0,4096,spec['batch']),device=device)[None].expand(24,-1))
        if (step+1)%16==0:
            diag=loss_step(torch,model,opt,data,ids,norm,keep,pids,tail,torch.tensor(rng.integers(0,spec['fit_paths'],spec['tail_batch']),device=device),view,fb,support,torch.tensor(rng.integers(0,len(support['broad']),spec['feedback_batch']),device=device),(cfg,calcium,base))
        else:graph.replay()
        if (step+1)%1000==0:
            assert torch.isfinite(model.correction).all() and torch.isfinite(opt.m).all() and torch.isfinite(opt.v).all()
            assert (model.correction*(1-model.mask)==0).all()
            print(f'[GIADA Task41i] update {step+1}/60000 finite=True elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    weights=torch.stack(saved);trained={a['name']:extract(torch,model,k,parent,knots,device,weights) for k,a in enumerate(arms())}
    nominee=min(range(8),key=lambda k:tuple(max(best[3*k+s][j] for s in range(3)) for j in range(len(best[0])))+(k,))
    torch.save({k:v.state_dict() for k,v in trained.items()},out/'frozen_checkpoint.pt')
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen,ranks=best,nominee=arms()[nominee]['name'],checkpoint_sha256=p.digest(out/'frozen_checkpoint.pt'),opened_regressions_used_for_selection=False,independent_confirmation=False,parent='H_nominee')
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(out/'selection_freeze.json',freeze)
    print('[GIADA Task41i] checkpoint selection frozen',flush=True)
    if stop_after is not None:
        return dict(valid=True,decision='FROZEN_TRAINING_COMPLETE',updates_completed=60000,training_complete=True,thresholds_changed=False,scientific_go_claimed=False)
    with torch.inference_mode():report=ev.evaluate(torch,root,out,controls,trained,spec,cfg,calcium,base,device)
    p.save_json(out/'final_report.json',report);return report
