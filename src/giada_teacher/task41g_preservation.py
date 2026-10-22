"""Task41g: paired equilibrium repair, preservation and numerical attribution."""
import copy
import hashlib
import json
import time
import types
import numpy as np
from . import task41f_precision_relaxation as f
from . import task41f_evaluation as fe

p,t,c,d,x,prior=f.p,f.t,f.c,f.d,f.x,f.prior
CONFIG='experiments/task41g_preregistration_v1.json'
GATES,CHANNELS=f.GATES,f.CHANNELS


def arms():
    return [dict(name=f'{heads}_{optimizer}_{objective}_{support}',heads=heads,
                 optimizer=optimizer,objective=objective,support=support,
                 basis=64,kind='U64',feedback=objective=='sensitive')
            for support in ('broad','tail') for heads in ('M','ALL')
            for optimizer in ('constant','anneal') for objective in ('point','sensitive')
            if support=='broad' or heads=='M']


def contrasts():
    rows=[];aa=arms()
    for axis in ('heads','optimizer','objective','support'):
        levels={'heads':('M','ALL'),'optimizer':('constant','anneal'),
                'objective':('point','sensitive'),'support':('broad','tail')}[axis]
        for a in aa:
            if a[axis]!=levels[0]:continue
            for b in aa:
                if b[axis]==levels[1] and all(a[k]==b[k] for k in
                    ('heads','optimizer','objective','support') if k!=axis):
                    rows.append((axis,a['name'],b['name']))
    return rows


def contract(root):
    spec=json.loads((root/CONFIG).read_text());old,cfg,calcium,base=f.contract(root)
    for key in ('quality','compartment_gates','cost'):assert spec[key]==old[key],key
    assert spec['arms']==arms() and spec['contrasts']==[list(r) for r in contrasts()]
    assert spec['seeds']==[17,29,43] and spec['dt_ms']==.1
    assert not set(spec['new_seeds'].values()) & set(old['new_seeds'].values())
    for name,sha in spec['sources'].items():assert p.digest(root/name)==sha,name
    return spec,cfg,calcium,base


def model_class(torch):
    class Model(prior.model_class(torch)):
        def __init__(self,definitions,seeds=(17,29,43)):
            super().__init__(definitions,list(seeds))
            for value in self.parameters():value.requires_grad_(False)
            n=len(seeds)*len(definitions)
            self.f_correction=torch.nn.Parameter(torch.zeros(n,6,128))
            active=torch.zeros(n,6,128)
            for j,a in enumerate(definitions):
                active[j*len(seeds):(j+1)*len(seeds),:1 if a['heads']=='M' else 3,:64]=1
            self.register_buffer('active',active)
        def raw(self,v,ca):
            u,z=super().raw(v,ca)
            return f.add(torch,u,z,f.extras(torch,v,self.f_correction*self.active,self.definitions,self.active))
    return Model


def make_model(torch,parent,definitions,device):
    result=model_class(torch)(definitions).to(device=device,dtype=next(parent.parameters()).dtype)
    state=result.state_dict()
    for name,value in parent.state_dict().items():
        if name in ('active','f_correction'):continue
        state[name]=value.repeat(len(definitions),*([1]*(value.ndim-1))) if value.ndim and value.shape[0]==3 and name!='gbar' else value
    result.load_state_dict(state)
    return result


def extract(torch,model,index,parent,device,weights=None):
    result=make_model(torch,parent,[model.definitions[index]],device)
    with torch.no_grad():result.f_correction.copy_((weights if weights is not None else model.f_correction)[index*3:index*3+3])
    return result.eval()


def frozen_controls(torch,root,spec,device):
    controls=f.frozen_controls(torch,root,spec,device)
    saved=torch.load(root/spec['task41f_checkpoint'],map_location=device,weights_only=True)
    definition=next(a for a in f.arms() if a['name']==spec['task41f_nominee'])
    old=f.make_model(torch,controls['P110'],[definition],device)
    old.load_state_dict(saved[spec['task41f_nominee']]);controls['F_nominee']=old
    return controls


def atomic_data(torch,parent,spec,device,selection=False,count=None):
    # Reuse the audited label path with this explicit set of definitions.
    fn=types.FunctionType(f.atomic_data.__code__,dict(f.__dict__,arms=arms))
    data,raw=fn(torch,parent,spec,device,selection,count)
    if not selection:
        score=tail_score(raw)
        n=data['target'].shape[1];order=np.random.default_rng(spec['new_seeds']['fit']+700001).permutation(2*n)
        keep=list(order[:n//2]);used=set(keep)
        ids=np.array(keep+[int(k) for k in np.argsort(-score,kind='stable') if int(k) not in used][:n-len(keep)])
        raw['tail_ids']=ids
        for ai,a in enumerate(arms()):
            if a['support']!='tail':continue
            vv=torch.tensor(raw['v'][ids],device=device);cc=torch.tensor(raw['ca'][ids],device=device)
            with torch.no_grad():u,z=parent.raw(vv.reshape(1,-1).expand(3,-1),cc.reshape(1,-1).expand(3,-1))
            sl=slice(3*ai,3*ai+3)
            data['base'][sl]=torch.cat((u[...,list(GATES)],z[...,list(GATES)]),-1).reshape(3,n,3,6)
            data['target'][sl]=torch.tensor(np.concatenate((raw['u'][ids][...,list(GATES)],raw['z'][ids][...,list(GATES)]),-1),device=device)[None]
            data['features'][sl]=torch.nn.functional.pad(d.feature(torch,vv,64),(0,64))[None]
            data['support_ids'][sl]=torch.tensor(ids,device=device)[None]
    return data,raw


def tail_score(raw):
    # FIT-only voltage sensitivity weighted by long m relaxation.
    u=raw['u'][:,1,0];z=raw['z'][:,1,0]
    sensitivity=np.abs(raw['u'][:,2,0]-raw['u'][:,0,0])
    return sensitivity*(1-z)**400/(z+1e-5)+u*(1-u)*(1-z)**1600


def tail_data(torch,parent,spec,device,count=None,selection=False):
    n=count or spec['tail_anchors'];seed=spec['new_seeds']['selection' if selection else 'fit']+19001
    rng=np.random.default_rng(seed)
    v=rng.uniform(-85,-30,n);ca=10**rng.uniform(-5,-3.5,n)
    initial,_=p.coefficients(rng.uniform(-15,35,n),ca,.1)
    u,z=p.coefficients(v,ca,.1)
    mult=rng.uniform(.5,2,(n,11))
    def truth(ticks):
        g=u+(initial-u)*np.exp(np.log1p(-np.minimum(z,1-1e-15))*ticks[:,None])
        return p.readout(v,g,mult,c.ionic.REVERSALS)[1][:,list(CHANNELS)]
    i0=truth(np.zeros(n));steady=p.readout(v,u,mult,c.ionic.REVERSALS)[1][:,list(CHANNELS)]
    amplitude=np.abs(i0-steady);threshold=.1*amplitude[:,0]
    lo=np.zeros(n);hi=np.full(n,4000.)
    valid=(np.abs(truth(hi)[:,0]-steady[:,0])<threshold)&(amplitude[:,0]>1e-6)
    for _ in range(14):
        mid=(lo+hi)/2;above=np.abs(truth(mid)[:,0]-steady[:,0])>=threshold
        lo=np.where(above,mid,lo);hi=np.where(above,hi,mid)
    crossing=np.rint((lo+hi)/2).clip(1,3999)
    ticks=np.stack([np.full(n,k) for k in (10,80,400,1600)]+[crossing],1)
    target=np.stack([truth(ticks[:,j]) for j in range(5)],1)
    slope=(truth(crossing+1)[:,0]-truth(crossing-1)[:,0])/.2
    scale=np.maximum(amplitude,1e-7)
    # Denominator expresses a 0.3ms error near the *labelled* crossing. It is
    # a differentiable training surrogate, never the official acceptance test.
    timing_scale=np.maximum(np.abs(slope)*spec['quality']['onset_offset_max_ms'],1e-10)
    vv=torch.tensor(v,device=device,dtype=torch.float32);cc=torch.tensor(ca,device=device,dtype=torch.float32)
    with torch.no_grad():ru,rz=parent.raw(vv[None].expand(3,-1),cc[None].expand(3,-1))
    return dict(v=vv,ca=cc,base_u=ru,base_z=rz,initial=torch.tensor(initial,device=device),
        ticks=torch.tensor(ticks,device=device),target=torch.tensor(target,device=device),
        mult=torch.tensor(mult,device=device),scale=torch.tensor(scale,device=device),
        timing_scale=torch.tensor(timing_scale,device=device),valid=torch.tensor(valid,device=device),
        features=torch.nn.functional.pad(d.feature(torch,vv,64),(0,64)))


def tail_loss(torch,model,tail,ids):
    members=len(model.f_correction);n=len(ids)
    uu=tail['base_u'][:,ids].repeat(len(arms()),1,1).double()
    zz=tail['base_z'][:,ids].repeat(len(arms()),1,1).double()
    vv=tail['v'][ids][None].expand(members,-1).double()
    extra=f.extras(torch,vv,model.f_correction,model.definitions,model.active)
    uu,zz=f.add(torch,uu,zz,extra)
    inf,rate=torch.sigmoid(uu),torch.sigmoid(zz)
    ticks=tail['ticks'][ids]
    gates=inf[:,:,None]+(tail['initial'][ids][None,:,None]-inf[:,:,None])*torch.exp(
        torch.log1p(-rate.clamp_max(1-1e-15))[:,:,None]*ticks[None,:,:,None])
    current=t.readout(torch,vv[:,:,None].expand(-1,-1,5),gates,tail['mult'][ids][None,:,None])[1][...,list(CHANNELS)]
    err=current-tail['target'][ids][None]
    ordinary=(err/tail['scale'][ids][None,:,None]).square().mean((1,2,3))
    crossing=(err[:,:,-1,0]/tail['timing_scale'][ids][None]).square().clamp_max(1e6)
    timing=(crossing*tail['valid'][ids]).sum(-1)/tail['valid'][ids].sum().clamp_min(1)
    return ordinary+.01*timing


class Adam:
    """Independent Adam moments and member-specific LR, CUDA graph compatible."""
    def __init__(self,torch,weights,definitions,spec):
        self.torch,self.weights,self.definitions,self.spec=torch,weights,definitions,spec
        self.m=torch.zeros_like(weights);self.v=torch.zeros_like(weights)
        self.step_count=torch.zeros((),device=weights.device,dtype=torch.float64)
        self.lr=torch.full((len(weights),1,1),spec['lr'],device=weights.device,dtype=weights.dtype)
    def rate(self,step):
        values=[self.spec['lr'] if a['optimizer']=='constant' else self.spec['lr']*
            max(self.spec['lr_min_ratio'],(.5+.5*np.cos(np.pi*min(step/self.spec['checkpoints'][-1],1)))**2)
            for a in self.definitions for _ in range(3)]
        self.lr.copy_(self.torch.tensor(values,device=self.weights.device,dtype=self.weights.dtype)[:,None,None])
    def zero_grad(self,set_to_none=False):
        if set_to_none:self.weights.grad=None
        elif self.weights.grad is not None:self.weights.grad.zero_()
    def step(self):
        torch=self.torch;g=self.weights.grad
        with torch.no_grad():
            self.step_count.add_(1);self.m.mul_(.9).add_(g,alpha=.1);self.v.mul_(.999).addcmul_(g,g,value=.001)
            m=self.m/(1-.9**self.step_count);v=self.v/(1-.999**self.step_count)
            self.weights.add_(-self.lr*m/(v.sqrt()+1e-8))
    def reset(self):
        self.m.zero_();self.v.zero_();self.step_count.zero_()
    def state_dict(self):return dict(m=self.m,v=self.v,step=self.step_count,lr=self.lr)


def point_loss(torch,model,data,ids,norm,preservation,pids):
    point=f.atomic_terms(torch,model.f_correction,model.active,data,ids)/norm
    member=torch.arange(len(model.f_correction),device=ids.device)[:,None]
    features=preservation['features'][member,pids]
    extra=torch.einsum('mbkp,mop->mbko',features.double(),(model.f_correction*model.active).double())
    keep=(.1*torch.tanh(extra/.1)).square().mean((1,2))/norm
    return point.sum(-1)+.01*keep.sum(-1),point


def point_step(torch,model,opt,data,ids,norm,preservation,pids):
    opt.zero_grad();loss,_=point_loss(torch,model,data,ids,norm,preservation,pids)
    loss.sum().backward();f.clip(torch,model.f_correction.grad);opt.step()


def capture(torch,model,opt,data,ids,norm,preservation,pids):
    initial=model.f_correction.detach().clone()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):point_step(torch,model,opt,data,ids,norm,preservation,pids)
    torch.cuda.current_stream().wait_stream(stream);opt.zero_grad(set_to_none=True)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):point_step(torch,model,opt,data,ids,norm,preservation,pids)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.no_grad():model.f_correction.copy_(initial);opt.reset()
    return graph


class FeedbackView:
    def __init__(self,torch,parent,model):
        self.torch,self.model=torch,model
        self.parent=make_model(torch,parent,model.definitions,next(model.parameters()).device).double()
        for a in self.parent.parameters():a.requires_grad_(False)
    def __getattr__(self,k):return getattr(self.parent,k)
    def raw(self,v,ca):
        u,z=self.parent.raw(v,ca)
        return f.add(self.torch,u,z,f.extras(self.torch,v,self.model.f_correction,self.model.definitions,self.model.active))
    def __call__(self,v,ca,g):
        u,z=self.raw(v,ca);return g+self.torch.sigmoid(z)*(self.torch.sigmoid(u)-g)


def feedback_terms(torch,model,view,fb,support,ids,cfg,calcium,base):
    # Both supports have equal count. Tail enrichment is independently recorded.
    proxy=copy.copy(model);proxy.definitions=[dict(a,support='broad' if a['support']=='broad' else 'rare') for a in model.definitions]
    return f.feedback_terms(torch,proxy,view,fb,support,ids,cfg,calcium,base)


def auxiliary_step(torch,model,opt,data,ids,norm,preservation,pids,tail,tids,view,fb,support,fi,cfg,calcium,base):
    point,heads=point_loss(torch,model,data,ids,norm,preservation,pids)
    gp=torch.autograd.grad(point.sum(),model.f_correction)[0]
    aux=tail_loss(torch,model,tail,tids)+feedback_terms(torch,model,view,fb,support,fi,cfg,calcium,base)
    ga=torch.autograd.grad(aux.sum(),model.f_correction)[0]
    pn,an=gp.double().norm(dim=(-1,-2)),ga.double().norm(dim=(-1,-2))
    ratio=(.1*pn/an.clamp_min(1e-20)).clamp_max(1e6)
    enabled=torch.tensor([a['objective']=='sensitive' for a in model.definitions for _ in range(3)],device=gp.device)
    grad=(gp+ga*(ratio*enabled)[:,None,None]).to(model.f_correction.dtype)
    assert torch.isfinite(grad).all()
    opt.zero_grad()
    if model.f_correction.grad is None:model.f_correction.grad=grad.clone()
    else:model.f_correction.grad.copy_(grad)
    f.clip(torch,model.f_correction.grad);opt.step()
    return dict(point=point.tolist(),tail_and_feedback=aux.tolist(),point_gradient=pn.tolist(),auxiliary_gradient=an.tolist(),
        cosine=((gp.double()*ga.double()).sum((-1,-2))/(pn*an).clamp_min(1e-30)).tolist(),normalized_heads=heads.tolist())


def preflight(torch,root,output,device,spec):
    _,cfg,calcium,base=contract(root);controls=f.frozen_controls(torch,root,spec,device);parent=controls['E_Q0']
    model=make_model(torch,parent,arms(),device)
    v=torch.linspace(-90,40,17,device=device)[None].expand(36,-1);ca=torch.full_like(v,1e-4)
    with torch.no_grad():
        actual=model.raw(v,ca);wanted=parent.raw(v[:3],ca[:3])
        parity=max(float((aa[j*3:j*3+3]-bb).abs().max()) for aa,bb in zip(actual,wanted) for j in range(12))
    assert parity==0
    atomic,_=atomic_data(torch,parent,spec,device,count=16)
    keep,_=atomic_data(torch,parent,dict(spec,new_seeds={**spec['new_seeds'],'fit':spec['new_seeds']['fit']+200000}),device,count=16)
    ids=torch.arange(8,device=device)[None].expand(36,-1).clone()
    norm=f.atomic_terms(torch,model.f_correction,model.active,atomic,ids).detach().clamp_min(1e-12)
    opt=Adam(torch,model.f_correction,arms(),spec);ref=copy.deepcopy(model)
    oo=Adam(torch,ref.f_correction,arms(),spec)
    if device=='cuda':
        graph=capture(torch,model,opt,atomic,ids,norm,keep,ids)
        for step in range(3):
            opt.rate(step);oo.rate(step);graph.replay();point_step(torch,ref,oo,atomic,ids,norm,keep,ids)
        torch.cuda.synchronize();graph_error=float((model.f_correction-ref.f_correction).detach().abs().max());assert graph_error<=1e-5
    else:point_step(torch,model,opt,atomic,ids,norm,keep,ids);graph_error=None
    with torch.no_grad():
        assert bool((model.f_correction*(1-model.active)==0).all())
        assert bool((model.f_correction[:,3:]==0).all())
    tail=tail_data(torch,parent,spec,device,count=16)
    loss=tail_loss(torch,model,tail,torch.arange(4,device=device))
    grad=torch.autograd.grad(loss.sum(),model.f_correction)[0];assert torch.isfinite(grad).all();del loss,grad
    fe.json_native(dict(valid=tail['valid'].cpu().numpy()))
    report=dict(valid=True,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,
        zero_Q0_error=parity,graph_error=graph_error,all_parameters_finite=all(bool(torch.isfinite(v).all()) for v in model.parameters()),
        members=36,active_weights_per_seed=[int(model.active[j*3].sum()) for j in range(12)],
        tail_gradient_finite=True,z_frozen=True,protocol_sha256=p.digest(root/CONFIG),thresholds_changed=False)
    p.save_json(output/'preflight.json',report);return report


def run(root,output,device='cuda',smoke=False):
    import torch
    from . import task41g_evaluation as ev
    spec,cfg,calcium,base=contract(root)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert device=='cuda' and torch.cuda.is_available()
    output.mkdir(parents=True,exist_ok=True);assert not (output/'checkpoint_0.pt').exists()
    p.save_json(output/'preregistration.json',spec);preflight(torch,root,output,device,spec)
    print('[GIADA Task41g] preflights passed',flush=True)
    if smoke:return dict(valid=True,decision='GPU_PREFLIGHT_ONLY')
    controls=frozen_controls(torch,root,spec,device)
    with torch.inference_mode():ev.frozen_audit(torch,root,output,controls,spec,cfg,calcium,base,device)
    f.ee.differentiable_constants();parent=controls['E_Q0'];model=make_model(torch,parent,arms(),device)
    atomic,raw=atomic_data(torch,parent,spec,device);np.savez_compressed(output/'fit_atomic.npz',**raw)
    selection,_=atomic_data(torch,parent,spec,device,selection=True)
    keep,_=atomic_data(torch,parent,dict(spec,new_seeds={**spec['new_seeds'],'fit':spec['new_seeds']['fit']+200000}),device)
    tail=tail_data(torch,parent,spec,device)
    selection_tail=tail_data(torch,parent,spec,device,count=spec['selection_anchors'],selection=True)
    np.savez_compressed(output/'fit_tail.npz',**{k:v.cpu().numpy() for k,v in tail.items()})
    p.save_json(output/'fit_tail_support.json',dict(valid_crossings=int(tail['valid'].sum()),total=len(tail['valid']),FIT_only=True,
        label_horizons_ms=[1,8,40,160],crossing_horizon_max_ms=400))
    ev.projection_probe(torch,parent,spec,device,output)
    fb,support=f.feedback_data(torch,spec,cfg,calcium,base,device,output)
    sf,ss=f.feedback_data(torch,spec,cfg,calcium,base,device,output,selection=True)
    view=FeedbackView(torch,parent,model)
    ids=torch.arange(spec['batch'],device=device)[None].expand(36,-1).clone();pids=ids.clone()
    norm=f.atomic_terms(torch,model.f_correction,model.active,atomic,ids).detach().clamp_min(1e-12)
    opt=Adam(torch,model.f_correction,arms(),spec);graph=capture(torch,model,opt,atomic,ids,norm,keep,pids)
    dev=d.development(spec['new_seeds']['development']);np.savez_compressed(output/'development_inputs.npz',**dev)
    parent_rows=c.shadow_evaluate(torch,[parent],dev,spec['quality'],device)
    rng=np.random.default_rng(spec['new_seeds']['stream']);frng=np.random.default_rng(spec['new_seeds']['feedback_stream'])
    best=[None]*36;chosen=[None]*36;saved=[None]*36;curves=[];diag={};start=time.perf_counter()
    si=torch.arange(spec['selection_anchors'],device=device)[None].expand(36,-1)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            assert all(bool(torch.isfinite(v).all()) for v in model.parameters())
            torch.save(dict(step=step,correction=model.f_correction.detach().cpu(),optimizer=opt.state_dict(),
                rng=rng.bit_generator.state,feedback_rng=frng.bit_generator.state),output/f'checkpoint_{step}.pt')
            print(f'[GIADA Task41g] checkpoint_saved {step}',flush=True)
            models=[extract(torch,model,k,parent,device) for k in range(12)]
            with torch.no_grad():
                point=f.atomic_terms(torch,model.f_correction,model.active,selection,si)/norm
                response=feedback_terms(torch,model,view,sf,ss,torch.arange(spec['selection_feedback_anchors'],device=device),cfg,calcium,base)
                tail_selection=tail_loss(torch,model,selection_tail,torch.arange(spec['selection_anchors'],device=device))
                measured=c.shadow_evaluate(torch,models,dev,spec['quality'],device)
            rr=[]
            for j,rows in enumerate(measured):
                rank=f.ranks(rows,parent_rows[j%3],response[j]+tail_selection[j],point[j].sum(),spec['quality']);rr.append(rank)
                if best[j] is None or rank<best[j]:best[j],chosen[j],saved[j]=rank,step,model.f_correction[j].detach().clone()
            curves.append(dict(step=step,ranks=rr,point=point.tolist(),response=response.tolist(),tail_selection=tail_selection.tolist(),gradients=diag,
                elapsed_seconds=time.perf_counter()-start,atomic_examples=step*spec['batch'],preservation_examples=step*spec['batch'],
                feedback_windows=(step//spec['feedback_every'])*spec['feedback_batch'],tail_examples=(step//spec['feedback_every'])*spec['tail_batch']))
            p.save_json(output/'training_curves.json',curves);p.save_json(output/f'development_{step}.json',measured)
            print(f'[GIADA Task41g] checkpoint development {step} target-passing={sum(r[1]==0 for r in rr)}/36',flush=True)
        if step==spec['checkpoints'][-1]:break
        opt.rate(step)
        ids.copy_(torch.tensor(rng.integers(0,spec['anchors'],spec['batch']),device=device)[None].expand(36,-1))
        pids.copy_(torch.tensor(rng.integers(0,spec['anchors'],spec['batch']),device=device)[None].expand(36,-1))
        if (step+1)%spec['feedback_every']==0:
            fi=torch.tensor(frng.integers(0,len(support['broad']),spec['feedback_batch']),device=device)
            tids=torch.tensor(frng.integers(0,spec['tail_anchors'],spec['tail_batch']),device=device)
            diag=auxiliary_step(torch,model,opt,atomic,ids,norm,keep,pids,tail,tids,view,fb,support,fi,cfg,calcium,base)
        else:graph.replay()
        if (step+1)%1000==0:print(f'[GIADA Task41g] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    weights=torch.stack(saved);trained={a['name']:extract(torch,model,k,parent,device,weights) for k,a in enumerate(arms())}
    nominee=min(range(12),key=lambda k:tuple(max(best[3*k+s][i] for s in range(3)) for i in range(len(best[0])))+(k,))
    torch.save({k:v.state_dict() for k,v in trained.items()},output/'frozen_checkpoint.pt')
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen,ranks=best,nominee=arms()[nominee]['name'],
        checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'),opened_regressions_accessed=True,
        opened_regressions_used_for_selection=False,independent_confirmation=False,parent='E_Q0',
        selection='Canonical gate predicates unchanged; introduced-case guard relative to Q0; common independent held-tail+feedback selection score; new development only.')
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    print('[GIADA Task41g] checkpoint selection frozen',flush=True)
    with torch.inference_mode():result=ev.evaluate(torch,root,output,controls,trained,spec,cfg,calcium,base,device)
    result['training_and_evaluation_seconds']=time.perf_counter()-start;p.save_json(output/'final_report.json',result)
    return result
