"""Learned local equilibrium precision: paired factorials on a frozen P110.

Training labels may use canonical kinetics; deployed heads use V,Ca and the
same18 recurrent states. Opened historical panels are accessed after freeze.
"""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
from . import task41c_frozen_confirmation as prior
c,p,t,x=prior.local,prior.p,prior.t,prior.x
CONFIG='experiments/task41d_preregistration_v1.json'

def arms():
    ca=[dict(name=f'C{s}{r}',panel='Ca',gate=0,support='sensitive' if s else 'uniform',basis=64 if r else 32,shape='linear',paired=False,feedback=False) for s in (0,1) for r in (0,1)]
    nap=[dict(name=f'N{s}{o}{f}',panel='Nap',gate=8,support='uniform',basis=64,shape='smooth' if s else 'linear',paired=bool(o),feedback=bool(f)) for s in (0,1) for o in (0,1) for f in (0,1)]
    return ca+nap

def contract(root):
    spec=json.loads((root/CONFIG).read_text());old,cfg,calcium,base=prior.contract(root)
    assert spec['quality']==old['current_quality'] and spec['compartment_gates']==old['compartment_gates']
    assert spec['arms']==arms() and spec['seeds']==[17,29,43] and spec['dt_ms']==.1
    assert not set(spec['new_seeds'].values()) & set(old['excluded_input_seeds']+old['native_seeds']+old['stress_seeds'])
    for path,sha in spec['sources'].items():assert p.digest(root/path)==sha,path
    return spec,x.calibration_config(old,cfg),calcium,base

def feature(torch,v,n,shape='linear'):
    centers=torch.linspace(-100,60,n,device=v.device,dtype=v.dtype)
    d=(v[...,None]-centers)/(160/(n-1));a=d.abs()
    if shape=='linear':return (1-a).clamp_min(0)
    # Compact C2 cubic B-splines; normalize only at finite domain boundaries.
    f=torch.where(a<1,(4-6*a.square()+3*a.pow(3))/6,torch.where(a<2,(2-a).clamp_min(0).pow(3)/6,torch.zeros_like(a)))
    return f/f.sum(-1,keepdim=True).clamp_min(1e-12)

def model_class(torch):
    class Model(c.model_class(torch)):
        def __init__(self,definitions,seeds):
            super().__init__(dict(basis=64),seeds*len(definitions))
            for par in self.parameters():par.requires_grad_(False)
            self.definitions=definitions
            self.local_equilibrium=torch.nn.Parameter(torch.zeros(len(seeds)*len(definitions),64))
        def raw(self,v,ca):
            u,z=super().raw(v,ca);values=[]
            for j,a in enumerate(self.definitions):
                f=feature(torch,v[j*3:j*3+3],a['basis'],a['shape'])
                w=self.local_equilibrium[j*3:j*3+3,:a['basis']]
                extra=(f.double()*w[:,None].double()).sum(-1).to(v.dtype)
                extra=.1*torch.tanh(extra/.1)
                mask=torch.nn.functional.one_hot(torch.tensor(a['gate'],device=v.device),18).to(v.dtype)
                values.append(extra[...,None]*mask)
            return u+torch.cat(values),z
    return Model

def make_model(torch,parent,definitions,seeds,device):
    model=model_class(torch)(definitions,seeds).to(device);state=model.state_dict()
    for name,value in parent.state_dict().items():
        state[name]=value.repeat(len(definitions),*([1]*(value.ndim-1))) if value.ndim and value.shape[0]==3 and name!='gbar' else value
    model.load_state_dict(state);return model

def extract(torch,model,index,root,device):
    parent=prior.load_models(torch,root,device)['P110'];result=make_model(torch,parent,[model.definitions[index]],[17,29,43],device)
    with torch.no_grad():result.local_equilibrium.copy_(model.local_equilibrium[index*3:index*3+3])
    result.eval();return result

def anchors(seed,count,support='uniform',gate=0):
    # Separate RNGs: changing support selection cannot shift Ca/epsilon/history.
    rng=np.random.default_rng(seed);pilot=rng.uniform(-100,60,count*4)
    aux=np.random.default_rng(seed+100000);ca=10**aux.uniform(-7,-2,count*4)
    choice=np.random.default_rng(seed+200000)
    if support=='sensitive':
        left,_=p.coefficients(pilot-.1,ca,.1);right,_=p.coefficients(pilot+.1,ca,.1)
        score=abs(right[:,gate]-left[:,gate])/(.01+left[:,gate]*(1-left[:,gate]))
        score=np.minimum(score,np.quantile(score,.95));prob=.5/len(score)+.5*score/max(score.sum(),1e-30)
        indices=choice.choice(len(pilot),count,replace=False,p=prob/prob.sum())
    else:indices=np.arange(count)
    v=pilot[indices].astype('float32');ca=ca[indices].astype('float32')
    eps=np.random.default_rng(seed+300000).choice([.0001,.001,.01,.1],count).astype('float32')
    pair=np.stack((v-eps,v,v+eps),-1).clip(-100,60)
    u,z=p.coefficients(pair.astype('float64'),np.broadcast_to(ca[:,None],pair.shape).astype('float64'),.1)
    return dict(v=pair,ca=np.broadcast_to(ca[:,None],pair.shape).copy(),u=u,z=z,eps=eps,source_indices=indices)

def prepare_atomic(torch,parent,spec,device,seed=None,count=None,selection=False):
    seed=spec['new_seeds']['selection' if selection else 'fit'] if seed is None else seed
    n=count or spec['anchors'];datasets={};features=[];targets=[];base=[];secants=[]
    for a in arms():
        key=a['panel']+'_'+a['support']
        if key not in datasets:datasets[key]=anchors(seed,n,a['support'],a['gate'])
        d=datasets[key];v=torch.tensor(d['v'],device=device);ca=torch.tensor(d['ca'],device=device)
        with torch.no_grad():ru,_=parent.raw(v.reshape(1,-1).expand(3,-1),ca.reshape(1,-1).expand(3,-1))
        base.append(ru[...,a['gate']].reshape(3,n,3).double())
        targets.append(torch.tensor(d['u'][...,a['gate']],device=device)[None].expand(3,-1,-1))
        f=feature(torch,v,a['basis'],a['shape']);f=torch.nn.functional.pad(f,(0,64-a['basis']))
        features.append(f[None].expand(3,-1,-1,-1).double())
        # Pair differences, not dividing by a guessed derivative near saturation.
        secants.append(torch.tensor(d['eps'],device=device)[None].expand(3,-1).double())
    return dict(features=torch.cat(features),base=torch.cat(base),target=torch.cat(targets),eps=torch.cat(secants)),datasets

def atomic_terms(torch,w,data,idx):
    members=torch.arange(len(w),device=w.device)[:,None]
    f=data['features'][members,idx];raw=data['base'][members,idx]
    extra=(f*w[:,None,None].double()).sum(-1).float();extra=.1*torch.tanh(extra/.1)
    raw=(raw+extra.double()).float()
    target=data['target'][members,idx];expected=torch.logit(target.clamp(1e-7,1-1e-7))
    point=(raw.double()-expected).square().mean((1,2))
    u=torch.sigmoid(raw).double();scale=(2*data['eps'][members,idx]).clamp_min(1e-7)
    paired=(((u[...,2]-u[...,0])-(target[...,2]-target[...,0]))/scale).square().mean(1)
    return torch.stack((point,paired),-1)

def feedback_data(torch,spec,cfg,calcium,base,device,count):
    rng=np.random.default_rng(spec['new_seeds']['feedback']);v=rng.uniform(-85,-25,count);ca=10**rng.uniform(-4.8,-3.5,count)
    initial,_=p.coefficients(rng.uniform(-95,-25,count),ca,.1)
    mult=rng.uniform(.5,2,(count,11));inj=rng.uniform(-.003,.012,count);syn=rng.uniform(0,.0002,(count,4))
    actions=rng.integers(0,13,count);other=mult.copy();otherinj=inj.copy();othersyn=syn.copy()
    for k,a in enumerate(actions):
        if a<11:other[k,a]=0
        elif a==11:otherinj[k]*=1.01
        else:othersyn[k]=0
    values=dict(v=np.repeat(v[:,None],2,1),ca=np.repeat(ca[:,None],2,1),g=np.repeat(initial[:,None],2,1),mult=np.stack((mult,other),1),inj=np.stack((inj,otherinj),1),syn=np.stack((syn,othersyn),1))
    tensor={k:torch.tensor(a,device=device,dtype=torch.float64) for k,a in values.items()}
    core=t.make_core(torch,cfg,calcium,base,100*np.pi);vv,cc,gg=tensor['v'],tensor['ca'],tensor['g'];targets=[]
    with torch.no_grad():
        for _ in range(spec['feedback_ticks']):
            vv,cc,gg,ii=core(vv,cc,gg,tensor['mult'],tensor['inj'],tensor['syn']);targets.append(ii[:,:,4])
    target=torch.stack(targets);delta=target[:,:,1]-target[:,:,0]
    tensor={k:v.float() for k,v in tensor.items()};tensor.update(target=delta.double(),scale=delta.abs().amax(0).clamp_min(1e-7))
    return tensor,values

def feedback_terms(torch,model,data,idx,cfg,calcium,base):
    # All members use the same FIT physical pairs; each owns independent weights.
    members=len(model.local_equilibrium);batch=len(idx)
    expanded=lambda key:data[key][idx][None].expand(members,*data[key][idx].shape).reshape(members,batch*2,*data[key].shape[2:])
    vv,cc,gg=expanded('v'),expanded('ca'),expanded('g');mult,inj,syn=expanded('mult'),expanded('inj'),expanded('syn')
    core=t.make_core(torch,cfg,calcium,base,100*np.pi,model,t.FAMILIES['all']);loss=torch.zeros(members,device=vv.device,dtype=torch.float64)
    for tick in range(len(data['target'])):
        vv,cc,gg,ii=core(vv,cc,gg,mult,inj,syn)
        ii=ii[...,4].reshape(members,batch,2).double();delta=ii[...,1]-ii[...,0]
        loss+=((delta-data['target'][tick,idx])/data['scale'][idx]).square().mean(-1)
    return loss/len(data['target'])

def train_step(torch,model,opt,atomic,idx,normalizers,weights,fb,fi,cfg,calcium,base,feedback=False):
    opt.zero_grad(set_to_none=False)
    terms=atomic_terms(torch,model.local_equilibrium,atomic,idx)/normalizers
    loss=(terms*weights[:,:2]).sum()
    if feedback:loss=loss+(feedback_terms(torch,model,fb,fi,cfg,calcium,base)*weights[:,2]).sum()
    loss.backward();norm=model.local_equilibrium.grad.double().norm(dim=1)
    model.local_equilibrium.grad.mul_((1/norm.clamp_min(1))[:,None].float());opt.step()
    return terms.detach()

def calibrate(torch,model,atomic,idx,fb,fi,cfg,calcium,base):
    w=model.local_equilibrium;tt=atomic_terms(torch,w,atomic,idx)
    initial=tt.detach().clamp_min(1e-12)
    # Shared uniform FIT normalizers within each panel, independent of support.
    normalizers=torch.cat([initial[:3] if a['panel']=='Ca' else initial[12:15] for a in arms()])
    point=(tt[:,0]/normalizers[:,0]).sum()
    g0=torch.autograd.grad(point,w,retain_graph=True)[0].double().norm(dim=1)
    g1=torch.autograd.grad((tt[:,1]/normalizers[:,1]).sum(),w)[0].double().norm(dim=1)
    ff=feedback_terms(torch,model,fb,fi,cfg,calcium,base);g2=torch.autograd.grad(ff.sum(),w)[0].double().norm(dim=1)
    weights=torch.zeros(len(w),3,device=w.device,dtype=torch.float64);weights[:,0]=1
    for k,a in enumerate(arms()):
        if a['paired']:weights[k*3:k*3+3,1]=(.1*g0[k*3:k*3+3]/g1[k*3:k*3+3].clamp_min(1e-20)).clamp(1e-8,100)
        if a['feedback']:weights[k*3:k*3+3,2]=(.1*g0[k*3:k*3+3]/g2[k*3:k*3+3].clamp_min(1e-20)).clamp(1e-8,1e6)
    return normalizers,weights,dict(initial_gradient_norms=torch.stack((g0,g1,g2),-1).tolist())

def gradient_diagnostics(torch,model,atomic,idx,norm,fb,fi,cfg,calcium,base):
    terms=atomic_terms(torch,model.local_equilibrium,atomic,idx)/norm
    gg=[torch.autograd.grad(terms[:,i].sum(),model.local_equilibrium,retain_graph=i==0)[0].double().norm(dim=1) for i in (0,1)]
    ff=feedback_terms(torch,model,fb,fi,cfg,calcium,base)
    gg.append(torch.autograd.grad(ff.sum(),model.local_equilibrium)[0].double().norm(dim=1))
    return torch.stack(gg,-1).tolist()

def capture_atomic(torch,model,opt,atomic,idx,norm,weights,fb,fi,cfg,calcium,base):
    original=model.local_equilibrium.detach().clone()
    for _ in range(3):train_step(torch,model,opt,atomic,idx,norm,weights,fb,fi,cfg,calcium,base)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):train_step(torch,model,opt,atomic,idx,norm,weights,fb,fi,cfg,calcium,base)
    with torch.no_grad():
        model.local_equilibrium.copy_(original)
        for state in opt.state.values():
            for a in state.values():
                if torch.is_tensor(a):a.zero_()
    return graph

def development(seed,count=12):
    rng=np.random.default_rng(seed);steps=12000;v=np.empty((steps,count),dtype='float32');ca=np.empty_like(v);initial=[]
    for k in range(count):
        rest=rng.uniform(-65,-30);pulse=rng.uniform(-40,10);stop=round(rng.uniform(8,25)*10)
        v[:,k]=rest;v[:stop,k]=pulse
        if k%3==0:
            n=round(rng.uniform(10,40)*10);v[stop:stop+n,k]=np.linspace(pulse,rest,n)
        if k%3==1:v[800:950,k]=rng.uniform(-45,-20)
        ca[:,k]=10**rng.uniform(-4.8,-3.5)
        initial.append(p.coefficients(np.array(-95 if k%2==0 else -42),np.array(ca[0,k]),.1)[0])
    u,z=p.coefficients(v.astype('float64'),ca.astype('float64'),.1);g=np.array(initial);gg=[]
    for tick in range(steps):g=g+z[tick]*(u[tick]-g);gg.append(g.copy())
    return dict(voltage=v,calcium=ca,initial=np.array(initial),target_gates=np.stack(gg),multipliers=np.ones((count,11)))

def atomic_update_models(torch,model,root,device):return [extract(torch,model,k,root,device) for k in range(len(arms()))]

def preflight(torch,root,output,device,spec):
    _,cfg,calcium,base=contract(root);parent=prior.load_models(torch,root,device)['P110'];model=make_model(torch,parent,arms(),spec['seeds'],device)
    data,_=prepare_atomic(torch,parent,spec,device,count=32);fb,_=feedback_data(torch,spec,cfg,calcium,base,device,32)
    idx=torch.arange(8,device=device)[None].expand(36,-1);fi=torch.arange(4,device=device)
    n,w,gr=calibrate(torch,model,data,idx,fb,fi,cfg,calcium,base)
    v=torch.linspace(-85,35,19,device=device)[None].expand(36,-1);ca=torch.full_like(v,1e-4);g=torch.full((36,19,18),.4,device=device)
    with torch.no_grad():
        before=parent(v[:3],ca[:3],g[:3]);actual=model(v,ca,g)
        zero_error=max(float((before-actual[k*3:k*3+3]).abs().max()) for k in range(12))
        assert zero_error<1e-5
    opt=torch.optim.Adam([model.local_equilibrium],lr=spec['lr'],eps=1e-8)
    train_step(torch,model,opt,data,idx,n,w,fb,fi,cfg,calcium,base,True)
    assert torch.isfinite(model.local_equilibrium).all()
    # Independent atomic Adam moments and per-member clipping, exercised at zero.
    test=torch.nn.Parameter(torch.zeros_like(model.local_equilibrium));ot=torch.optim.Adam([test],lr=spec['lr'])
    ot.zero_grad();loss=atomic_terms(torch,test,data,idx)[:,0]/n[:,0];loss.sum().backward();norm=test.grad.double().norm(dim=1);test.grad.mul_((1/norm.clamp_min(1))[:,None].float());ot.step()
    isolation=0.
    for member in (0,7,16,35):
        ds={k:a[member:member+1] for k,a in data.items()};ww=torch.nn.Parameter(torch.zeros(1,64,device=device));oo=torch.optim.Adam([ww],lr=spec['lr'])
        oo.zero_grad();(atomic_terms(torch,ww,ds,idx[:1])[:,0]/n[member:member+1,0]).sum().backward();nn=ww.grad.double().norm(dim=1);ww.grad.mul_((1/nn.clamp_min(1))[:,None].float());oo.step();isolation=max(isolation,float((ww[0]-test[member]).abs().max()))
    assert isolation<1e-6
    with torch.no_grad():
        initial=parent.raw(v[:3],ca[:3]);after=model.raw(v,ca)
        for k,a in enumerate(arms()):
            other=[j for j in range(18) if j!=a['gate']]
            assert (initial[0][...,other]-after[0][k*3:k*3+3,...,other]).abs().max()<1e-5
            assert (initial[1]-after[1][k*3:k*3+3]).abs().max()<1e-5
    # No kinetic fallback and eager/graph parity for a real physical step.
    mm=extract(torch,model,0,root,device);core=t.make_core(torch,cfg,calcium,base,100*np.pi,mm,t.FAMILIES['all'])
    args=[v[:3,:2],ca[:3,:2],g[:3,:2],torch.ones(3,2,11,device=device),torch.zeros(3,2,device=device),torch.zeros(3,2,4,device=device)]
    old=t._rates
    try:
        t._rates=lambda *a,**k:(_ for _ in ()).throw(AssertionError('kinetic fallback'))
        with torch.no_grad():
            a=t.Stepper(torch,core,args);b=t.Stepper(torch,core,args,device=='cuda');error=0.
            for _ in range(3):error=max(error,max(float((aa-bb).abs().max()) for aa,bb in zip(a.tick(),b.tick())))
    finally:t._rates=old
    assert error<1e-5
    training_graph_error=None
    if device=='cuda':
        reference=copy.deepcopy(model);oo=torch.optim.Adam([reference.local_equilibrium],lr=spec['lr'],eps=1e-8,capturable=True)
        go=torch.optim.Adam([model.local_equilibrium],lr=spec['lr'],eps=1e-8,capturable=True)
        graph=capture_atomic(torch,model,go,data,idx,n,w,fb,fi,cfg,calcium,base)
        for feedback in (False,True,False):
            if feedback:train_step(torch,model,go,data,idx,n,w,fb,fi,cfg,calcium,base,True)
            else:graph.replay()
            train_step(torch,reference,oo,data,idx,n,w,fb,fi,cfg,calcium,base,feedback)
        torch.cuda.synchronize();training_graph_error=float((model.local_equilibrium-reference.local_equilibrium).abs().max());assert training_graph_error<1e-5
    clock=time.perf_counter()
    for _ in range(5):train_step(torch,model,opt,data,idx,n,w,fb,fi,cfg,calcium,base,False)
    if device=='cuda':torch.cuda.synchronize()
    report=dict(valid=True,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,torch=torch.__version__,members=36,all_parameters_finite=all(bool(torch.isfinite(a).all()) for a in model.parameters()),optimizer_isolation_max_error=isolation,graph_eager_max_error=error,training_graph_optimizer_max_error=training_graph_error,zero_init_max_error=zero_error,computational_parity_atol=1e-5,untargeted_weights_frozen=True,canonical_fallback=False,seconds_per_small_atomic_update=(time.perf_counter()-clock)/5,gradient_calibration=gr,source_sha256=p.digest(root/'src/giada_teacher/task41d_precision_response.py'),protocol_sha256=p.digest(root/CONFIG))
    p.save_json(output/'preflight.json',report);return report

def run(root,output,device='cuda',smoke=False):
    import torch
    spec,cfg,calcium,base=contract(root);assert device=='cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output.mkdir(parents=True,exist_ok=True);assert not (output/'checkpoint_0.pt').exists()
    p.save_json(output/'preregistration.json',spec);pf=preflight(torch,root,output,device,spec)
    print('[GIADA Task41d] preflights passed',flush=True)
    if smoke:
        result=dict(valid=True,decision='GPU_PREFLIGHT_ONLY',task='41d');p.save_json(output/'final_report.json',result);return result
    parent=prior.load_models(torch,root,device)['P110'];model=make_model(torch,parent,arms(),spec['seeds'],device)
    atomic,datasets=prepare_atomic(torch,parent,spec,device);selection,_=prepare_atomic(torch,parent,spec,device,count=spec['selection_anchors'],selection=True)
    fb,fbraw=feedback_data(torch,spec,cfg,calcium,base,device,spec['feedback_anchors']);fbsel,_=feedback_data(torch,{**spec,'new_seeds':{**spec['new_seeds'],'feedback':spec['new_seeds']['selection_feedback']}},cfg,calcium,base,device,spec['selection_feedback_anchors'])
    for key,data in datasets.items():np.savez_compressed(output/('fit_'+key+'.npz'),**data)
    np.savez_compressed(output/'fit_feedback.npz',**fbraw)
    idx=torch.arange(spec['batch'],device=device)[None].expand(36,-1).clone();fi=torch.arange(spec['feedback_batch'],device=device)
    norm,weights,gr=calibrate(torch,model,atomic,idx,fb,fi,cfg,calcium,base)
    p.save_json(output/'fit_calibration.json',dict(normalizers=norm.tolist(),weights=weights.tolist(),**gr))
    opt=torch.optim.Adam([model.local_equilibrium],lr=spec['lr'],eps=1e-8,capturable=True)
    # One shared static input buffer, independent member moments; CUDA graph atomic updates.
    graph=capture_atomic(torch,model,opt,atomic,idx,norm,weights,fb,fi,cfg,calcium,base)
    dev=development(spec['new_seeds']['development']);np.savez_compressed(output/'development_inputs.npz',**dev)
    stream=np.random.default_rng(spec['new_seeds']['stream']);fbrng=np.random.default_rng(spec['new_seeds']['feedback_stream'])
    curves=[];best=[None]*36;chosen=[None]*36;saved=[None]*36;clock=time.perf_counter()
    si=torch.arange(spec['selection_anchors'],device=device)[None].expand(36,-1);sfi=torch.arange(spec['selection_feedback_anchors'],device=device)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            torch.save(dict(step=step,correction=model.local_equilibrium.detach().cpu(),optimizer=opt.state_dict(),stream=stream.bit_generator.state,feedback_rng=fbrng.bit_generator.state),output/f'checkpoint_{step}.pt')
            assert all(torch.isfinite(a).all() for a in model.parameters());print(f'[GIADA Task41d] checkpoint_saved {step}',flush=True)
            diagnostics=gradient_diagnostics(torch,model,atomic,idx,norm,fb,fi,cfg,calcium,base)
            mm=atomic_update_models(torch,model,root,device)
            with torch.no_grad():
                atom=atomic_terms(torch,model.local_equilibrium,selection,si)/norm
                response=feedback_terms(torch,model,fbsel,sfi,cfg,calcium,base)
                measured=c.shadow_evaluate(torch,mm,dev,spec['quality'],device)
            ranks=[]
            for j,rows in enumerate(measured):
                rank=(sum(not r['passed'] for r in rows),max(c.severity(r,spec['quality']) for r in rows),float(response[j]),float(atom[j].sum()))
                ranks.append(rank)
                if best[j] is None or rank<best[j]:best[j]=rank;chosen[j]=step;saved[j]=model.local_equilibrium[j].detach().clone()
            curves.append(dict(step=step,ranks=ranks,atomic=atom.tolist(),feedback=response.tolist(),FIT_gradient_norms=diagnostics,elapsed_seconds=time.perf_counter()-clock))
            p.save_json(output/'training_curves.json',curves);p.save_json(output/f'development_{step}.json',measured)
            print(f'[GIADA Task41d] checkpoint development {step} passing_members={sum(r[0]==0 for r in ranks)}/36',flush=True)
        if step==spec['checkpoints'][-1]:break
        common=stream.integers(0,spec['anchors'],spec['batch']);idx.copy_(torch.tensor(common,device=device)[None].expand(36,-1))
        if (step+1)%spec['feedback_every']==0:
            fi.copy_(torch.tensor(fbrng.integers(0,spec['feedback_anchors'],spec['feedback_batch']),device=device))
            train_step(torch,model,opt,atomic,idx,norm,weights,fb,fi,cfg,calcium,base,True)
        else:graph.replay()
        if (step+1)%1000==0:print(f'[GIADA Task41d] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-clock)/60:.1f}min',flush=True)
    with torch.no_grad():model.local_equilibrium.copy_(torch.stack(saved))
    mm=atomic_update_models(torch,model,root,device);nominees={}
    for panel,indices in [('Ca',range(4)),('Nap',range(4,12))]:
        k=min(indices,key=lambda k:(max(best[k*3+s][0] for s in range(3)),max(best[k*3+s][1] for s in range(3)),max(best[k*3+s][2] for s in range(3)),sum(best[k*3+s][3] for s in range(3)),k))
        nominees[panel]=arms()[k]['name']
    torch.save({a['name']:m.state_dict() for a,m in zip(arms(),mm)},output/'frozen_checkpoint.pt')
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen,ranks=best,nominees=nominees,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'),opened_regressions_accessed=False,independent_confirmation=False,selection='New development all-case failure count,worst complete severity,paired short-response loss,point+secant score;panel worst seed nomination;earliest ties.')
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    print('[GIADA Task41d] checkpoint selection frozen',flush=True)
    from .task41d_evaluation import evaluate
    with torch.inference_mode():result=evaluate(torch,root,output,{a['name']:m for a,m in zip(arms(),mm)},spec,device)
    result['training_and_evaluation_seconds']=time.perf_counter()-clock;p.save_json(output/'final_report.json',result);return result
