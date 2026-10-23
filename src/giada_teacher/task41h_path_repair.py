"""41h: paired path support, objective exposure and local resolution.

All acceptance predicates are inherited unchanged. Prescribed-path affine
composition accelerates training only; deployed models retain18 states.
"""
import copy
import hashlib
import json
import time
import types
import numpy as np
from . import task41g_preservation as g

p,t,c,d,f,prior = g.p,g.t,g.c,g.d,g.f,g.prior
CONFIG='experiments/task41h_preregistration_v1.json'


def arms():
    rows=[dict(name=f'H{h}_D{dose}_N{n}',history=h,aux_every=dose,fit_count=n,
        basis=64,kind='U64',heads='M',optimizer='anneal',objective='sensitive',support='broad',feedback=True)
        for h in (0,1) for dose in (256,16) for n in (4096,16384)]
    for source in (rows[0],rows[-1]):
        rows.append(dict(source,name=source['name']+'_B128',basis=128,kind='U128'))
    return rows


def contrasts():
    aa=arms();result=[]
    for key in ('history','aux_every','fit_count'):
        levels=sorted({a[key] for a in aa[:8]})
        for a in aa[:8]:
            for b in aa[:8]:
                if a[key]==levels[0] and b[key]==levels[1] and all(a[k]==b[k] for k in ('history','aux_every','fit_count') if k!=key):
                    result.append((key,a['name'],b['name']))
    return result+[('basis',aa[0]['name'],aa[8]['name']),('basis',aa[7]['name'],aa[9]['name'])]


def contract(root):
    spec=json.loads((root/CONFIG).read_text());old,cfg,ca,base=g.contract(root)
    for k in ('quality','compartment_gates','cost'):assert spec[k]==old[k],k
    assert spec['arms']==arms() and spec['contrasts']==[list(x) for x in contrasts()]
    assert spec['dt_ms']==.1 and spec['seeds']==[17,29,43]
    assert not set(spec['new_seeds'].values()) & set(old['new_seeds'].values())
    for name,sha in spec['sources'].items():assert p.digest(root/name)==sha,name
    return spec,cfg,ca,base


def sparse_sum(torch,v,w,n):
    """Evaluate the same two nonzero hat features, with original centers."""
    centers=torch.linspace(-100,60,n,device=v.device,dtype=v.dtype)
    lo=torch.floor((v+100)/(160/(n-1))).long().clamp(0,n-2)
    # A neighbor on either side is included to cover floating center rounding.
    total=torch.zeros_like(v,dtype=torch.float64)
    for shift in (-1,0,1,2):
        ix=(lo+shift).clamp(0,n-1)
        valid=(lo+shift>=0)&(lo+shift<n)
        phi=(1-((v-centers[ix])/(160/(n-1))).abs()).clamp_min(0)*valid
        ww=torch.gather(w[:,:n],1,ix.reshape(len(w),-1)).reshape_as(v)
        total=total+phi.double()*ww.double()
    return total.to(v.dtype)


def model_class(torch):
    Base=prior.model_class(torch)
    class Model(Base):
        def __init__(self,definitions,seeds=(17,29,43)):
            super().__init__(definitions,list(seeds))
            for par in self.parameters():par.requires_grad_(False)
            self.f_correction=torch.nn.Parameter(torch.zeros(len(seeds)*len(definitions),6,128))
            mask=torch.zeros_like(self.f_correction)
            for j,a in enumerate(definitions):mask[3*j:3*j+3,0,:a['basis']]=1
            self.register_buffer('active',mask);self.compact=False
        def raw(self,v,ca):
            u,z=super().raw(v,ca)
            if self.compact and len(self.definitions)==1:
                n=self.definitions[0]['basis']
                delta=.1*torch.tanh(sparse_sum(torch,v,self.f_correction[:,0],n)/.1)
                return torch.cat((u[...,:1]+delta[...,None],u[...,1:]),-1),z
            return f.add(torch,u,z,f.extras(torch,v,self.f_correction*self.active,self.definitions,self.active))
    return Model


def make_model(torch,parent,definitions,device):
    model=model_class(torch)(definitions).to(device=device,dtype=next(parent.parameters()).dtype)
    state=model.state_dict()
    for name,value in parent.state_dict().items():
        if name in ('active','f_correction'):continue
        state[name]=value.repeat(len(definitions),*([1]*(value.ndim-1))) if value.ndim and value.shape[0]==3 and name!='gbar' else value
    model.load_state_dict(state);return model


def extract(torch,model,index,parent,device,weights=None):
    result=make_model(torch,parent,[model.definitions[index]],device)
    with torch.no_grad():result.f_correction.copy_((model.f_correction if weights is None else weights)[index*3:index*3+3])
    return result.eval()


def atomic_data(torch,parent,spec,device,selection=False,count=None):
    fn=types.FunctionType(f.atomic_data.__code__,dict(f.__dict__,arms=arms))
    return fn(torch,parent,spec,device,selection,count)


def frozen_controls(torch,root,spec,device):
    result=g.frozen_controls(torch,root,spec,device)
    saved=torch.load(root/spec['task41g_checkpoint'],map_location=device,weights_only=True)
    for key,label in [('task41g_nominee','G_nominee'),('task41g_diagnostic','G_tail_diagnostic')]:
        definition=next(a for a in g.arms() if a['name']==spec[key])
        model=g.make_model(torch,result['E_Q0'],[definition],device)
        model.load_state_dict(saved[spec[key]]);result[label]=model.eval()
    return result


def path_tail_data(torch,parent,spec,device,count=None,selection=False):
    """Matched endpoints/initial states, constant or depolarize/ramp histories.

    Store only m's affine prefix weights, other frozen endpoint gates and
    tail labels. No parent forward or full BPTT is needed in training slots.
    """
    n=count or spec['tail_anchors'];T=spec['path_prefix_ticks']
    rng=np.random.default_rng(spec['new_seeds']['selection' if selection else 'fit']+19001)
    rest=rng.uniform(-85,-30,n);boundary=rng.random(n)<.5;rest[boundary]=rng.uniform(-50,-35,int(boundary.sum()))
    ca=10**rng.uniform(-5,-3.5,n);peak=rng.uniform(-31,-14,n);stop=rng.integers(130,231,n)
    initial_v=rng.uniform(-95,35,n);initial,_=p.coefficients(initial_v,ca,.1)
    mult=rng.uniform(.5,2,(n,11));allmodes=[]
    for mode in (0,1):
        collected={k:[] for k in ('path_v','prefix_u','prefix_weight','prefix_initial','end_gates','rest_u','rest_z','target','ticks','scale','timing_scale','valid')}
        for first in range(0,n,64):
            ix=slice(first,min(first+64,n));nn=len(rest[ix]);clock=np.arange(T)
            v=np.broadcast_to(rest[ix,None],(nn,T)).copy()
            if mode:
                fraction=np.clip((clock[None]-stop[ix,None])/300,0,1)
                v=peak[ix,None]+(rest[ix,None]-peak[ix,None])*fraction
            vc=v.T.astype('float32');cc=np.broadcast_to(ca[ix],vc.shape).astype('float32')
            canonical_u,canonical_z=p.coefficients(vc.astype('float64'),cc.astype('float64'),.1)
            with torch.no_grad():
                vv=torch.tensor(vc.T.reshape(-1),device=device)[None].expand(3,-1)
                cv=torch.tensor(cc.T.reshape(-1),device=device)[None].expand(3,-1)
                bu,bz=parent.raw(vv,cv)
                bu=bu.reshape(3,nn,T,18).double().cpu().numpy();bz=bz.reshape(3,nn,T,18).double().cpu().numpy()
                ru,rz=parent.raw(torch.tensor(rest[ix],device=device,dtype=torch.float32)[None].expand(3,-1),
                    torch.tensor(ca[ix],device=device,dtype=torch.float32)[None].expand(3,-1))
                ru,rz=ru.double().cpu().numpy(),rz.double().cpu().numpy()
            inf=1/(1+np.exp(-np.clip(bu,-700,700)));rate=1/(1+np.exp(-np.clip(bz,-700,700)))
            target_end=initial[ix].copy();parent_end=np.broadcast_to(initial[ix],(3,nn,18)).copy()
            for tick in range(T):
                target_end+=canonical_z[tick]*(canonical_u[tick]-target_end)
                parent_end+=rate[:,:,tick]*(inf[:,:,tick]-parent_end)
            decay=1-rate[...,0]
            after=np.concatenate((np.cumprod(decay[...,::-1],axis=-1)[...,::-1][...,1:],np.ones((3,nn,1))),-1)
            weight=rate[...,0]*after;initial_factor=np.prod(decay,-1)
            cu,cz=p.coefficients(rest[ix],ca[ix],.1)
            def truth(ticks):
                gates=cu[:,None]+(target_end[:,None]-cu[:,None])*np.power(1-cz[:,None],ticks[...,None])
                return p.readout(rest[ix,None],gates,mult[ix,None],c.ionic.REVERSALS)[1][...,:1]
            maximum=12000-T;lo=np.zeros(nn);hi=np.full(nn,float(maximum))
            start=truth(np.zeros((nn,1)))[:,0,0];steady=p.readout(rest[ix],cu,mult[ix],c.ionic.REVERSALS)[1][:,0]
            amp=np.abs(start-steady);threshold=.1*amp
            for _ in range(16):
                mid=(lo+hi)/2;above=np.abs(truth(mid[:,None])[:,0,0]-steady)>=threshold
                lo=np.where(above,mid,lo);hi=np.where(above,hi,mid)
            crossing=np.rint((lo+hi)/2).clip(1,maximum-1)
            ticks=np.stack([np.full(nn,k) for k in (10,80,400,1600,4000,8000,maximum)]+[crossing],-1)
            slope=(truth((crossing+1)[:,None])[:,0,0]-truth((crossing-1)[:,None])[:,0,0])/.2
            valid=(amp>1e-6)&(np.abs(truth(np.full((nn,1),maximum))[:,0,0]-steady)<threshold)
            values=dict(path_v=v.astype('float64'),prefix_u=bu[...,0],prefix_weight=weight,prefix_initial=initial_factor*initial[ix,0],
                end_gates=parent_end,rest_u=ru,rest_z=rz,target=truth(ticks),ticks=ticks,
                scale=np.maximum(amp,1e-7),timing_scale=np.maximum(np.abs(slope)*spec['quality']['onset_offset_max_ms'],1e-10),valid=valid)
            for key,value in values.items():collected[key].append(value)
        mode_data={key:np.concatenate(values,axis=1 if key in ('prefix_u','prefix_weight','prefix_initial','end_gates','rest_u','rest_z') else 0) for key,values in collected.items()}
        allmodes.append(mode_data)
    out={key:torch.tensor(np.stack([a[key] for a in allmodes]),device=device) for key in allmodes[0]}
    out.update(rest=torch.tensor(rest,device=device),ca=torch.tensor(ca,device=device),mult=torch.tensor(mult,device=device),
        initial=torch.tensor(initial,device=device),prefix_ticks=T)
    return out


def path_loss(torch,model,data,ids,force_history=None,return_rows=False):
    members=len(model.f_correction);batch=len(ids);defs=model.definitions
    modes=torch.tensor([a['history'] for a in defs for _ in range(3)],device=ids.device)
    # Mixture preserves half constant trajectories in the history treatment.
    modes=modes[:,None]*((ids%2)==1)[None]
    if force_history is not None:modes=torch.full_like(modes,force_history)
    seeds=torch.arange(members,device=ids.device)%3
    def get(key):
        a=data[key]
        return a[modes,seeds[:,None],ids[None]] if key in ('prefix_u','prefix_weight','prefix_initial','end_gates','rest_u','rest_z') else a[modes,ids[None]]
    vv=get('path_v');rest=data['rest'][ids][None].expand(members,-1)
    # Four sparse gathers avoid materializing M*B*T*128 dense feature arrays.
    prefix=torch.zeros_like(vv)
    held=torch.zeros_like(rest)
    for basis in (64,128):
        enabled=torch.tensor([a['basis']==basis for a in defs for _ in range(3)],device=ids.device)
        prefix=prefix+torch.where(enabled[:,None,None],sparse_sum(torch,vv,model.f_correction[:,0],basis),0)
        held=held+torch.where(enabled[:,None],sparse_sum(torch,rest,model.f_correction[:,0],basis),0)
    mp=get('prefix_initial')+(get('prefix_weight')*torch.sigmoid(get('prefix_u')+.1*torch.tanh(prefix/.1))).sum(-1)
    u=get('rest_u');z=torch.sigmoid(get('rest_z'));ticks=get('ticks')
    inf=torch.sigmoid(torch.cat((u[...,:1]+(.1*torch.tanh(held/.1))[...,None],u[...,1:]),-1))
    ending=torch.cat((mp[...,None],get('end_gates')[...,1:]),-1)
    gates=inf[:,:,None]+(ending[:,:,None]-inf[:,:,None])*torch.pow(1-z[:,:,None],ticks[...,None])
    current=t.readout(torch,rest[:,:,None].expand(-1,-1,ticks.shape[-1]),gates,data['mult'][ids][None,:,None])[1][...,:1]
    error=current-get('target');normalized=error/get('scale')[...,None,None]
    ordinary=normalized.square().mean((1,2,3));cross=(error[:,:,-1,0]/get('timing_scale')).square().clamp_max(1e6)
    valid=get('valid');crossing=(cross*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return (ordinary+.01*crossing,normalized) if return_rows else ordinary+.01*crossing


class FeedbackView(g.FeedbackView):
    def __init__(self,torch,parent,model):
        self.torch,self.model=torch,model
        self.parent=make_model(torch,parent,model.definitions,next(model.parameters()).device).double()
        for par in self.parent.parameters():par.requires_grad_(False)


def auxiliary_step(torch,model,opt,data,ids,norm,keep,pids,tail,tids,view,fb,support,fi,cfg,calcium,base,step):
    point,heads=g.point_loss(torch,model,data,ids,norm,keep,pids)
    gp=torch.autograd.grad(point.sum(),model.f_correction)[0]
    aux=path_loss(torch,model,tail,tids)+g.feedback_terms(torch,model,view,fb,support,fi,cfg,calcium,base)
    ga=torch.autograd.grad(aux.sum(),model.f_correction)[0]
    pn,an=gp.double().norm(dim=(-1,-2)),ga.double().norm(dim=(-1,-2))
    ratio=(.1*pn/an.clamp_min(1e-20)).clamp_max(1e6)
    enabled=torch.tensor([step%a['aux_every']==0 for a in model.definitions for _ in range(3)],device=gp.device)
    gradient=(gp+ga*(ratio*enabled)[:,None,None]).to(gp.dtype);assert torch.isfinite(gradient).all()
    # CUDA graphs retain the gradient buffer address; do not replace it.
    if model.f_correction.grad is None:
        model.f_correction.grad=gradient.clone()
    else:
        model.f_correction.grad.copy_(gradient)
    f.clip(torch,model.f_correction.grad);opt.step()
    return dict(point=point.tolist(),aux=aux.tolist(),enabled=enabled.tolist(),point_norm=pn.tolist(),aux_norm=an.tolist(),
        scale=ratio.tolist(),cosine=((gp.double()*ga.double()).sum((-1,-2))/(pn*an).clamp_min(1e-30)).tolist())


def preflight(torch,root,out,device,spec):
    controls=frozen_controls(torch,root,spec,device);parent=controls['E_Q0'];model=make_model(torch,parent,arms(),device)
    v=torch.linspace(-95,55,21,device=device)[None].expand(30,-1);ca=torch.full_like(v,1e-4)
    with torch.no_grad():
        a=model.raw(v,ca);b=parent.raw(v[:3],ca[:3]);assert all(torch.equal(x[3*k:3*k+3],y) for x,y in zip(a,b) for k in range(10))
        model.f_correction.normal_(0,.001);model.f_correction.mul_(model.active)
    tail=path_tail_data(torch,parent,spec,device,count=8)
    loss=path_loss(torch,model,tail,torch.arange(4,device=device));grad=torch.autograd.grad(loss.sum(),model.f_correction)[0]
    assert torch.isfinite(grad).all() and (grad*(1-model.active)==0).all()
    # Independent sequential recurrence checks the affine weighted composition.
    u=tail['prefix_u'][1,:,1];ww=tail['prefix_weight'][1,:,1];initial=tail['initial'][1,0]
    # Derive per-step survival from known prefix weights via a direct scalar test.
    rng=np.random.default_rng(17);z=torch.tensor(rng.uniform(.001,.9,(3,53)),device=device)
    uu=torch.randn(3,53,device=device,dtype=torch.float64,requires_grad=True)
    state=torch.full((3,),.3,device=device,dtype=torch.float64)
    for tick in range(53):state=state+z[:,tick]*(torch.sigmoid(uu[:,tick])-state)
    seqgrad=torch.autograd.grad(state.sum(),uu,retain_graph=True)[0]
    survive=torch.cumprod((1-z).flip(-1),-1).flip(-1)
    weights=z*torch.cat((survive[:,1:],torch.ones_like(survive[:,:1])),-1)
    composed=.3*torch.prod(1-z,-1)+(weights*torch.sigmoid(uu)).sum(-1)
    compgrad=torch.autograd.grad(composed.sum(),uu)[0]
    err=float((state-composed).abs().max());ge=float((seqgrad-compgrad).abs().max())
    assert err<1e-12 and ge<1e-12
    compact_error=0.
    for k in (0,8):
        small=extract(torch,model,k,parent,device);v0=v[:3].double();ca0=ca[:3].double();small=small.double()
        with torch.no_grad():
            old=small.raw(v0,ca0);small.compact=True;new=small.raw(v0,ca0)
            compact_error=max(compact_error,max(float((x-y).abs().max()) for x,y in zip(old,new)))
    assert compact_error<1e-10
    report=dict(valid=True,device=device,members=30,all_parameters_finite=True,path_gradient_finite=True,
        composition_forward_error=err,composition_gradient_error=ge,compact_raw_error=compact_error,
        protocol_sha256=p.digest(root/CONFIG),thresholds_changed=False,scientific_training_performed=False)
    p.save_json(out/'preflight.json',report);return report


def run(root,out,device='cuda',smoke=False):
    import torch
    from . import task41h_evaluation as ev
    spec,cfg,calcium,base=contract(root)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert device=='cuda' and torch.cuda.is_available()
    out.mkdir(parents=True,exist_ok=True);assert not (out/'checkpoint_0.pt').exists()
    p.save_json(out/'preregistration.json',spec);preflight(torch,root,out,device,spec)
    if not smoke:
        initial_controls=frozen_controls(torch,root,spec,device)
        initial_model=make_model(torch,initial_controls['E_Q0'],arms(),device)
        torch.save(dict(step=0,correction=initial_model.f_correction.detach().cpu(),
            model=initial_model.state_dict(),phase='before_frozen_diagnostics',training_started=False),out/'checkpoint_0.pt')
        del initial_model,initial_controls
    print('[GIADA Task41h] preflights passed',flush=True)
    if smoke:return dict(valid=True,decision='GPU_PREFLIGHT_ONLY')
    controls=frozen_controls(torch,root,spec,device)
    with torch.inference_mode():ev.frozen_audit(torch,root,out,controls,spec,cfg,calcium,base,device)
    f.ee.differentiable_constants();parent=controls['E_Q0'];model=make_model(torch,parent,arms(),device)
    atomic,raw=atomic_data(torch,parent,spec,device);np.savez_compressed(out/'fit_atomic.npz',**raw)
    selection,_=atomic_data(torch,parent,spec,device,selection=True)
    keep,_=atomic_data(torch,parent,dict(spec,new_seeds={**spec['new_seeds'],'fit':spec['new_seeds']['fit']+200000}),device,count=4096)
    tail=path_tail_data(torch,parent,spec,device);selection_tail=path_tail_data(torch,parent,spec,device,count=spec['selection_anchors'],selection=True)
    np.savez_compressed(out/'fit_path_tail.npz',**{k:v.cpu().numpy() if torch.is_tensor(v) else v for k,v in tail.items()})
    fb,support=f.feedback_data(torch,spec,cfg,calcium,base,device,out)
    sf,ss=f.feedback_data(torch,spec,cfg,calcium,base,device,out,selection=True)
    view=FeedbackView(torch,parent,model);ids=torch.arange(spec['batch'],device=device)[None].expand(30,-1).clone();pids=ids.clone()
    norm=f.atomic_terms(torch,model.f_correction,model.active,atomic,ids).detach().clamp_min(1e-12)
    opt=g.Adam(torch,model.f_correction,arms(),spec);graph=g.capture(torch,model,opt,atomic,ids,norm,keep,pids)
    dev=d.development(spec['new_seeds']['development']);np.savez_compressed(out/'development_inputs.npz',**dev)
    parent_rows=c.shadow_evaluate(torch,[parent],dev,spec['quality'],device)
    rng=np.random.default_rng(spec['new_seeds']['stream']);frng=np.random.default_rng(spec['new_seeds']['feedback_stream'])
    best={rule:[None]*30 for rule in ('aggregate','stratified')};chosen={r:[None]*30 for r in best};saved={r:[None]*30 for r in best}
    curves=[];diag={};started=time.perf_counter();si=torch.arange(spec['selection_anchors'],device=device)[None].expand(30,-1)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            assert all(torch.isfinite(v).all() for v in model.parameters())
            torch.save(dict(step=step,correction=model.f_correction.detach().cpu(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,feedback_rng=frng.bit_generator.state),out/f'checkpoint_{step}.pt')
            print(f'[GIADA Task41h] checkpoint_saved {step}',flush=True)
            models=[extract(torch,model,k,parent,device) for k in range(10)]
            with torch.no_grad():
                point=f.atomic_terms(torch,model.f_correction,model.active,selection,si)/norm
                response=g.feedback_terms(torch,model,view,sf,ss,torch.arange(spec['selection_feedback_anchors'],device=device),cfg,calcium,base)
                ca0=path_loss(torch,model,selection_tail,torch.arange(spec['selection_anchors'],device=device),force_history=0)
                ca1=path_loss(torch,model,selection_tail,torch.arange(spec['selection_anchors'],device=device),force_history=1)
                measured=c.shadow_evaluate(torch,models,dev,spec['quality'],device)
            ranks={r:[] for r in best}
            for j,rows in enumerate(measured):
                for rule in best:
                    value=response[j]+((ca0[j]+ca1[j])/2 if rule=='aggregate' else torch.maximum(ca0[j],ca1[j]))
                    rank=f.ranks(rows,parent_rows[j%3],value,point[j].sum(),spec['quality']);ranks[rule].append(rank)
                    if best[rule][j] is None or rank<best[rule][j]:best[rule][j],chosen[rule][j],saved[rule][j]=rank,step,model.f_correction[j].detach().clone()
            curves.append(dict(step=step,ranks=ranks,point=point.tolist(),constant_tail=ca0.tolist(),ramp_tail=ca1.tolist(),gradients=diag,
                elapsed_seconds=time.perf_counter()-started,atomic_examples=step*spec['batch'],
                auxiliary_examples=[(step//a['aux_every'])*spec['tail_batch'] for a in arms()]))
            p.save_json(out/'training_curves.json',curves);p.save_json(out/f'development_{step}.json',measured)
        if step==spec['checkpoints'][-1]:break
        opt.rate(step)
        draw=rng.integers(0,2**30,spec['batch']);ids.copy_(torch.tensor(np.stack([draw%a['fit_count'] for a in arms() for _ in range(3)]),device=device))
        pids.copy_(torch.tensor(rng.integers(0,4096,spec['batch']),device=device)[None].expand(30,-1))
        if (step+1)%16==0:
            fi=torch.tensor(frng.integers(0,len(support['broad']),spec['feedback_batch']),device=device)
            tids=torch.tensor(frng.integers(0,spec['tail_anchors'],spec['tail_batch']),device=device)
            diag=auxiliary_step(torch,model,opt,atomic,ids,norm,keep,pids,tail,tids,view,fb,support,fi,cfg,calcium,base,step+1)
        else:graph.replay()
        if (step+1)%1000==0:print(f'[GIADA Task41h] update {step+1}/60000 elapsed={(time.perf_counter()-started)/60:.1f}min',flush=True)
    primary='stratified';weights=torch.stack(saved[primary])
    trained={a['name']:extract(torch,model,k,parent,device,weights) for k,a in enumerate(arms())}
    aggregate=torch.stack(saved['aggregate'])
    for k,a in enumerate(arms()):
        if chosen['aggregate'][3*k:3*k+3]!=chosen[primary][3*k:3*k+3]:trained['aggregate_'+a['name']]=extract(torch,model,k,parent,device,aggregate)
    nominee=min(range(10),key=lambda k:tuple(max(best[primary][3*k+s][i] for s in range(3)) for i in range(len(best[primary][0])))+(k,))
    torch.save({k:v.state_dict() for k,v in trained.items()},out/'frozen_checkpoint.pt')
    freeze=dict(arms=arms(),seeds=spec['seeds'],chosen_steps=chosen[primary],ranks=best[primary],selection_rules=dict(steps=chosen,ranks=best),
        primary_selection=primary,nominee=arms()[nominee]['name'],checkpoint_sha256=p.digest(out/'frozen_checkpoint.pt'),
        opened_regressions_used_for_selection=False,independent_confirmation=False,parent='E_Q0')
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(out/'selection_freeze.json',freeze)
    print('[GIADA Task41h] checkpoint selection frozen',flush=True)
    with torch.inference_mode():report=ev.evaluate(torch,root,out,controls,trained,spec,cfg,calcium,base,device)
    report['training_and_evaluation_seconds']=time.perf_counter()-started;p.save_json(out/'final_report.json',report);return report
