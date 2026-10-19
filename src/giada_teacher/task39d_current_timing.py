"""Task39d: paired current scaling, history, conditioning and learned tail bias."""
import hashlib
import itertools
import json
import time
import numpy as np
from . import task39c_dynamic_identification as c

b,q,old,p=c.b,c.q,c.old,c.p
CONFIG='experiments/task39d_preregistration_v1.json'


def arms():
    out=[dict(name=f'base_n{n}_h{h}_b{w}',architecture='base',local=n,history=h,balance=w,event=0)
         for n,h,w in itertools.product((0,1),repeat=3)]
    for architecture,event in (('base',1),('linear',0)):
        for value in (0,1):
            out.append(dict(name=f'{"event" if event else "linear"}_{value}{value}{value}',architecture=architecture,
                            local=value,history=value,balance=value,event=event))
    return out


def contrast_pairs(defs):
    result=[]
    for axis in ('local','history','balance'):
        for a in defs[:8]:
            if a[axis]:continue
            z=next(z for z in defs[:8] if z[axis] and all(z[k]==a[k] for k in ('local','history','balance') if k!=axis))
            result.append((axis,a['name'],z['name']))
    for z in defs[8:]:
        a=next(a for a in defs[:8] if all(a[k]==z[k] for k in ('local','history','balance')))
        result.append(('event_package' if z['event'] else 'linear_bias',a['name'],z['name']))
    return result


def model_class(torch):
    class Model(c.model_class(torch)):
        def __init__(self,architecture,seeds):
            super().__init__('base',seeds)
            self.architecture=architecture
            if architecture=='linear':
                # Remove36 nonlinear parameters, add36 learnable logit slopes:1220 total.
                self.a=torch.nn.Parameter(self.a[:,:,:31].detach().clone())
                self.b=torch.nn.Parameter(self.b[:,:,:31].detach().clone())
                self.c=torch.nn.Parameter(self.c[:,:31].detach().clone())
                self.slope=torch.nn.Parameter(torch.zeros(len(seeds),36))

        def raw(self,v,ca):
            u,z=super().raw(v,ca)
            if self.architecture=='linear':
                vv=(v+40)/80;cc=(ca.clamp_min(1e-9).log10()+4.5)/2.5
                inp=torch.cat((vv[...,None].expand(-1,-1,16),cc[...,None],vv[...,None]),-1)
                u=u+inp*self.slope[:,None,:18];z=z+inp*self.slope[:,None,18:]
            return u,z
    return Model


def cohorts(torch,defs,seeds,device):
    result=[]
    for arch in ('base','linear'):
        ds=[d for d in defs if d['architecture']==arch]
        if not ds:continue
        flags=torch.tensor([[d[k] for k in ('local','history','balance','event')] for d in ds for _ in seeds],device=device)
        result.append(dict(model=model_class(torch)(arch,seeds*len(ds)).to(device),definitions=ds,flags=flags))
    return result


def split_models(torch,groups,seeds,device):
    out=[]
    for g in groups:
        for i,d in enumerate(g['definitions']):
            m=model_class(torch)(d['architecture'],seeds).to(device)
            m.load_state_dict({k:v.clone() if k=='gbar' else v[i*len(seeds):(i+1)*len(seeds)].clone() for k,v in g['model'].state_dict().items()})
            out.append(m)
    return out


def pool(root,role,families,count,length,seed,targeted=False):
    if not targeted:return q.windows(root,role,families,count,length,seed)
    assert role in ('fit','fit_extra')
    rng=np.random.default_rng(seed);data=[];support=[]
    for family in families:
        with np.load(root/f'{role}__{family}.npz') as z:a={k:z[k] for k in z.files}
        ticks,cases=a['voltage'].shape
        cur=a['current_targets'];scale=np.maximum(np.abs(cur).max(0),1e-7)
        delta=np.max(np.abs(np.diff(cur,axis=0,prepend=cur[:1]))/scale,axis=-1)
        activity=np.max(np.abs(cur)/scale,axis=-1)
        # Four fit-only strata; empty strata fall back to the full fit population.
        grid=np.indices((ticks,cases)).reshape(2,-1).T
        masks=[delta>=max(float(np.quantile(delta,.9)),1e-8),
               (activity>.01)&(activity<.2),np.arange(ticks)[:,None]>=ticks//2,
               np.ones((ticks,cases),bool)]
        starts=[];ci=[];counts=[]
        for k in range(count):
            mask=np.broadcast_to(masks[k%4],(ticks,cases));choices=grid[mask.ravel()]
            if not len(choices):choices=grid
            t,case=choices[rng.integers(len(choices))]
            starts.append(int(np.clip(t-rng.integers(length),0,ticks-length)));ci.append(int(case));counts.append(k%4)
        starts=np.array(starts);ci=np.array(ci);ti=starts[:,None]+np.arange(length)
        ix=np.broadcast_to(ci[:,None],ti.shape);pre=a['gate_targets'][np.maximum(ti-1,0),ix].copy()
        for j in np.flatnonzero(starts==0):pre[j,0]=a['initial_gates'][ci[j]]
        data.append(np.concatenate((a['voltage'][ti,ix,None],a['bridge_ca'][ti,ix,None],pre,a['gate_targets'][ti,ix],
            np.broadcast_to(a['multipliers'][ci,None],(count,length,11)),np.broadcast_to(a['reversals'][ci,None],(count,length,11)),cur[ti,ix]),-1))
        support.append(dict(role=role,family=family,windows=count,unique_windows=len(set(zip(starts.tolist(),ci.tolist()))),
            strata=np.bincount(counts,minlength=4).tolist(),assignment_sha256=hashlib.sha256(np.stack((starts,ci)).tobytes()).hexdigest()))
    return np.concatenate(data).astype('float32'),support


def rollout_training(torch,model,batch,history,short):
    n,bs,ticks,_=batch.shape
    ru,rz=model.raw(batch[...,0].reshape(n,-1),batch[...,1].reshape(n,-1))
    u,z=torch.sigmoid(ru).reshape(n,bs,ticks,18),torch.sigmoid(rz).reshape(n,bs,ticks,18)
    x=batch[:,:,0,2:20];states=[]
    for t in range(ticks):
        if t and t%short==0:x=torch.where(history[:,None,None],x,batch[:,:,t,2:20])
        x=x+z[:,:,t]*(u[:,:,t]-x);states.append(x)
    return torch.stack(states,2)


def event_terms(torch,pred,ref,floor=1e-7):
    # Only offline targets set scales and event eligibility; no candidate baseline correction.
    scale=ref.abs().amax(2).clamp_min(floor)
    charge=((pred-ref).mean(2)/scale).square().mean((1,2))
    base=ref[:,:,:1];rr=(ref-base).abs();pp=(pred-base).abs()
    amp=rr.amax(2);peak=rr.argmax(2)
    eligible=(amp>=10*floor)&(amp>=.25*scale)&(peak>0)&(peak<ref.shape[2]-1)
    width=(.02*amp).clamp_min(floor)[:,:,None]
    a=torch.sigmoid((pp-.1*amp[:,:,None])/width)
    target=torch.sigmoid((rr-.1*amp[:,:,None])/width)
    event=((a-target).square().mean(2)*eligible).sum((1,2))/eligible.sum((1,2)).clamp_min(1)
    derivative=((torch.diff(pred,dim=2)-torch.diff(ref,dim=2))/scale[:,:,None]).square().mean((1,2,3))
    return charge,event+derivative


def terms(torch,model,s,scales,flags,short=16):
    batch,anchor,ticks=s['batch'],s['anchor'],s['ticks'];gs,cs,us,zs=scales
    state=rollout_training(torch,model,batch,flags[:,1].bool(),short)
    current=b.captured_readout(torch,batch[...,0],state,batch[...,38:49],batch[...,49:60],model.gbar)
    ref=batch[...,60:71]
    local=ref.abs().amax(2,keepdim=True).clamp_min(1e-7)
    scale=torch.where(flags[:,0,None,None,None].bool(),local,cs[None,None,None])
    variable=((state-batch[...,20:38])/gs).square().mean((1,2,3))+((current-ref)/scale).square().mean((1,2,3))
    u,z,ld=model.coefficients(anchor[...,0],anchor[...,1])
    predicted=c.compose(torch,u,ld,anchor[...,2:20],ticks);target=c.held_targets(torch,anchor,ticks)
    cur=b.captured_readout(torch,anchor[...,0],predicted,anchor[...,38:49],anchor[...,49:60],model.gbar)
    true=b.captured_readout(torch,anchor[...,0],target,anchor[...,38:49],anchor[...,49:60],model.gbar)
    initial=b.captured_readout(torch,anchor[...,0],anchor[...,2:20],anchor[...,38:49],anchor[...,49:60],model.gbar)
    local_held=torch.maximum(true.abs(),initial.abs()).clamp_min(1e-7)
    held_scale=torch.where(flags[:,0,None,None].bool(),local_held,cs[None,None])
    held=((predicted-target)/gs).square().mean((1,2))+((cur-true)/held_scale).square().mean((1,2))
    inf=((u-anchor[...,71:89])/us).square().mean((1,2))
    _,rz=model.raw(anchor[...,0],anchor[...,1])
    rate=((torch.nn.functional.logsigmoid(rz)-anchor[...,89:107].clamp_min(1e-30).log())/zs).square().mean((1,2))
    charge,event=event_terms(torch,current,ref)
    return torch.stack((variable,held,inf,rate,charge,event),-1)


def objective(torch,g,s,scales,short=16):
    return (terms(torch,g['model'],s,scales,g['flags'],short)*s['weights']).sum(-1)


def gradient_diagnostics(torch,g,s,scales,step,short=16):
    model=g['model'];values=terms(torch,model,s,scales,g['flags'],short)
    grads=[]
    for i in range(6):
        gs=torch.autograd.grad(values[:,i].sum(),tuple(model.parameters()),retain_graph=i<5)
        grads.append(torch.cat([x.flatten(1) for x in gs],-1).detach())
    stack=torch.stack(grads,1);norm=stack.norm(dim=-1)
    cosine=torch.bmm(stack,stack.transpose(1,2))/(norm[:,:,None]*norm[:,None,:]).clamp_min(1e-20)
    return dict(step=step,architecture=model.architecture,losses=values.detach().cpu().tolist(),
        term_norms=norm.cpu().tolist(),cosines=cosine.cpu().tolist(),weights=s['weights'].cpu().tolist()),norm


def calibrate(torch,groups,static,scales,step,short=16,initial=False):
    records=[];common=None
    for g,s in zip(groups,static):
        rec,norm=gradient_diagnostics(torch,g,s,scales,step,short)
        if initial:
            # Common seed-paired control calibration, so factorial axes do not change initial weights.
            s['weights'][:,:2]=.5
        base=.5*(norm[:,0]+norm[:,1])
        proposed=(base[:,None]/norm[:,2:].clamp_min(1e-12)).clamp(.001,1000)
        if initial:
            if common is None:common=proposed[:3].clone()
            proposed=common.repeat(len(proposed)//3,1)
        update=torch.ones_like(g['flags'][:,2]).bool() if initial else g['flags'][:,2].bool()
        s['weights'][:,2:]=torch.where(update[:,None],proposed,s['weights'][:,2:])
        s['weights'][:,4:]*=g['flags'][:,3,None]
        rec['new_weights']=s['weights'].cpu().tolist();records.append(rec)
    return records


def optimizer_preflight(torch,g,s,scales,seeds,short):
    model=g['model'];initial={k:x.detach().clone() for k,x in model.named_parameters()}
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    objective(torch,g,s,scales,short).sum().backward();old.clip(torch,model);opt.step();error=0.
    for j in range(model.a.shape[0]):
        m=model_class(torch)(model.architecture,[seeds[j%len(seeds)]]).to(s['batch'].device)
        with torch.no_grad():
            for k,x in m.named_parameters():x.copy_(initial[k][j:j+1])
        iso=dict(model=m,flags=g['flags'][j:j+1]);ss={k:v[j:j+1] for k,v in s.items()}
        oo=torch.optim.Adam(m.parameters(),lr=.001)
        objective(torch,iso,ss,scales,short).sum().backward();old.clip(torch,m);oo.step()
        error=max(error,max(float((x[0]-dict(model.named_parameters())[k][j]).abs().max()) for k,x in m.named_parameters()))
    with torch.no_grad():
        for k,x in model.named_parameters():x.copy_(initial[k]);x.grad=None
    assert error<=1e-5,error
    return dict(valid=True,members=model.a.shape[0],max_abs_error=error)


def capture_training(torch,groups,static,scales,spec,objective_fn=None):
    loss_fn=objective_fn or objective
    originals=[{k:x.detach().clone() for k,x in g['model'].named_parameters()} for g in groups]
    opts=[torch.optim.Adam(g['model'].parameters(),lr=spec['learning_rate'],capturable=True) for g in groups]
    clips=[torch.zeros(g['model'].a.shape[0],device='cuda') for g in groups]
    def update():
        for g,s,opt,counter in zip(groups,static,opts,clips):
            opt.zero_grad(set_to_none=False);loss_fn(torch,g,s,scales,spec['short_ticks']).sum().backward()
            norm=torch.stack([x.grad.flatten(1).square().sum(1) for x in g['model'].parameters()]).sum(0).sqrt()
            counter.add_((norm>1).float())
            for x in g['model'].parameters():x.grad.div_(norm.clamp_min(1).reshape(-1,*([1]*(x.ndim-1))))
            opt.step()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):update()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):update()
    def reset():
        with torch.no_grad():
            for g,init,opt in zip(groups,originals,opts):
                for k,x in g['model'].named_parameters():x.copy_(init[k]);x.grad.zero_()
                for st in opt.state.values():
                    for x in st.values():
                        if torch.is_tensor(x):x.zero_()
            for x in clips:x.zero_()
    reset();update();update();expected=[{k:x.detach().clone() for k,x in g['model'].named_parameters()} for g in groups]
    reset();graph.replay();graph.replay();torch.cuda.synchronize()
    err=max(float((x-e[k]).abs().max()) for g,e in zip(groups,expected) for k,x in g['model'].named_parameters())
    assert all(bool(torch.isfinite(x).all()) for g in groups for x in g['model'].parameters())
    assert err<=spec['numeric_atol'],err
    reset();start=time.perf_counter()
    for _ in range(20):graph.replay()
    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/20;reset()
    assert seconds*spec['checkpoints'][-1]<spec['max_projected_training_seconds'],'Fit-only performance preflight exceeded registered runtime cap'
    # Adam state is allocated during eager warmup, OUTSIDE the private graph
    # pool. CUDA Graph records addresses, not ownership of those tensors.
    # Keep optimizers alive for exactly the lifetime of the replayable graph.
    graph._giada_optimizers=opts
    return graph,clips,dict(valid=True,two_update_max_error=err,seconds_per_update=seconds,
        projected_training_seconds=seconds*spec['checkpoints'][-1],optimizer_reset=True,optimizer_state_lifetime_retained=True,backend='two member-vectorized cohorts in one CUDA Graph')


def load_parents(torch,root,spec,device,dtype):
    folder=root/spec['parent'];freeze=json.loads((folder/'selection_freeze.json').read_text());sha=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==sha
    assert p.digest(folder/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']==spec['parent_sha256']
    weights=torch.load(folder/'frozen_checkpoint.pt',map_location=device,weights_only=True);out=[]
    for name in spec['oracle_parent_arms']:
        definition=next(a for a in freeze['arms'] if a['name']==name);arch=definition['architecture']
        idx=[a['name'] for a in freeze['arms'] if a['architecture']==arch].index(name)*3
        m=c.model_class(torch)(arch,spec['seeds']).to(device)
        m.load_state_dict({k:v.clone() if k=='gbar' else v[idx:idx+3].clone() for k,v in weights[arch].items()})
        out.append(m.to(dtype=dtype))
    return out


def timing_audit(reference,prediction,dt=.1):
    out=[]
    for j,channel in enumerate(p.ionic.CHANNELS):
        r=reference[:,j];a=prediction[:,j];amp=float(np.abs(r-r[0]).max());thr=.1*amp
        active=np.flatnonzero(np.abs(r-r[0])>=thr);ap=np.flatnonzero(np.abs(a-r[0])>=thr)
        if not len(active) or amp<1e-6:continue
        indices=sorted(set([int(active[0]),int(active[-1])]+([int(ap[0]),int(ap[-1])] if len(ap) else [])))
        eps=max(1e-12,float(np.max(np.abs(r)))*float(np.finfo(np.float32).eps)*4)
        perturb=[p.event_metrics(r,a+sign*eps,dt,1e-7) for sign in (-1,1)]
        out.append(dict(channel=channel,event=p.event_metrics(r,a,dt,1e-7),threshold=thr,
            crossing_samples=[dict(index=k,reference=float(r[k]),prediction=float(a[k]),
                reference_slope=float((r[min(k+1,len(r)-1)]-r[max(k-1,0)])/(dt*(min(k+1,len(r)-1)-max(k-1,0)) or dt))) for k in indices],
            perturbation_absolute=eps,perturbed_events=perturb,perturbation_is_diagnostic_only=True))
    return out


def oracle_audit(torch,root,output,spec,quality,device,parent_loader=None,timing_fn=None):
    rows=[];floors=[];timings=[]
    for dtype,precision in ((torch.float32,'float32'),(torch.float64,'float64')):
        models=(parent_loader or load_parents)(torch,root,spec,device,dtype)
        seed_count=len(spec['seeds']);members=len(models)*seed_count
        assert models and len(models)==len(spec['oracle_parent_arms'])
        assert all(m.a.shape[0]==seed_count for m in models)
        modes=['learned','teacher_u','teacher_z','teacher_both']+([f'gate_{g}' for g in range(18)] if precision=='float32' and spec.get('oracle_single_gates',True) else [])
        if spec.get('oracle_state_reset',False):modes.append('teacher_state_reset')
        for family in quality['families']:
            with np.load(root/spec['dataset']/f'development__{family}.npz') as zz:a={k:zz[k] for k in zz.files}
            shape=a['voltage'].shape;lus=[];lzs=[]
            with torch.no_grad():
                for m in models:
                    uu=[];zz=[]
                    for start in range(0,shape[0],2048):
                        v=torch.as_tensor(a['voltage'][start:start+2048].reshape(-1),device=device,dtype=dtype)[None].expand(seed_count,-1)
                        ca=torch.as_tensor(a['bridge_ca'][start:start+2048].reshape(-1),device=device,dtype=dtype)[None].expand(seed_count,-1)
                        u,z,_=m.coefficients(v,ca);uu.append(u.reshape(seed_count,-1,shape[1],18).cpu().numpy());zz.append(z.reshape(seed_count,-1,shape[1],18).cpu().numpy())
                    lus.append(np.concatenate(uu,1));lzs.append(np.concatenate(zz,1))
            lu,lz=np.concatenate(lus),np.concatenate(lzs);npdtype=np.float32 if precision=='float32' else np.float64
            eu,ez=p.coefficients(a['voltage'].astype(float),a['bridge_ca'].astype(float),.1);eu=eu.astype(npdtype);ez=ez.astype(npdtype)
            for mode in modes:
                u,z=lu,lz
                if mode=='teacher_u':u=np.broadcast_to(eu,lu.shape)
                elif mode=='teacher_z':z=np.broadcast_to(ez,lz.shape)
                elif mode=='teacher_both':u,z=np.broadcast_to(eu,lu.shape),np.broadcast_to(ez,lz.shape)
                elif mode.startswith('gate_'):
                    gate=int(mode.split('_')[1]);u,z=lu.copy(),lz.copy();u[...,gate]=eu[...,gate];z[...,gate]=ez[...,gate]
                state=np.broadcast_to(a['initial_gates'],(members,*a['initial_gates'].shape)).astype(npdtype).copy()
                pred=np.empty((shape[0],members,shape[1],18),dtype=npdtype)
                for tick in range(shape[0]):
                    if mode=='teacher_state_reset' and tick:state=np.broadcast_to(a['gate_targets'][tick-1],state.shape).astype(npdtype).copy()
                    state=state+z[:,tick]*(u[:,tick]-state);pred[tick]=state
                for j in range(members):
                    _,cur=p.readout(a['voltage'],pred[:,j],a['multipliers'],a['reversals'])
                    for case in range(shape[1]):
                        metric=p.evaluate_currents(a['current_targets'][:,case],cur[:,case],.1,quality['quality'],a['gate_targets'][:,case],pred[:,j,case],a['multipliers'][case]==0)
                        row=dict(parent_arm=spec['oracle_parent_arms'][j//seed_count],seed=spec['seeds'][j%seed_count],family=family,case=case,precision=precision,intervention=mode,metrics=metric,selection_eligible=False);rows.append(row)
                        if mode=='teacher_both' and precision=='float64':
                            floors.append(dict(passed=metric['passed'],gate_max=float(np.abs(pred[:,j,case]-a['gate_targets'][:,case]).max()),current_max=float(np.abs(cur[:,case]-a['current_targets'][:,case]).max())))
                        if mode=='learned':
                            events=(timing_fn or timing_audit)(a['current_targets'][:,case],cur[:,case]);timings.append(dict(parent=row['parent_arm'],seed=row['seed'],family=family,case=case,precision=precision,channels=events))
                            failed=any(e['event']['status']=='missing_event' or (e['event']['status']=='evaluable' and (e['event']['peak_error_ms']>.2+1e-9 or max(e['event']['onset_error_ms'],e['event']['offset_error_ms'])>.3+1e-9)) for e in events)
                            if failed:np.savez_compressed(output/f'timing__{row["parent_arm"]}__{precision}__{family}__{case}__{row["seed"]}.npz',reference=a['current_targets'][:,case],prediction=cur[:,case],time_ms=np.arange(shape[0])*.1)
            p.save_json(output/'frozen_oracle_attribution.json',rows);p.save_json(output/'timing_audit.json',timings)
            print(f'[GIADA Task{spec.get("task","39d")}] frozen attribution {precision}/{family}',flush=True)
    valid=all(r['passed'] and r['gate_max']<=spec['floor_gate_atol'] and r['current_max']<=spec['floor_current_atol'] for r in floors)
    p.save_json(output/'oracle_preflight.json',dict(valid=valid,teacher_floor=floors,rows=len(rows),selection_eligible=False))
    if not valid:raise RuntimeError('Teacher oracle failed; stop before training')


def run(root,output,device='cuda'):
    import torch
    assert device=='cuda' and torch.cuda.is_available(),'Task39d requires CUDA'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());quality=json.loads((root/p.CONFIG).read_text())
    output.mkdir(parents=True,exist_ok=True)
    if (output/'final_report.json').exists():raise FileExistsError(output)
    p.save_json(output/'preregistration.json',spec)
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()):
        assert p.digest(root/spec['dataset']/row['file'])==row['sha256']
    oracle_audit(torch,root,output,spec,quality,device)
    defs=arms();seeds=spec['seeds'];families=list(quality['families']);support=[];arrays=[]
    for targeted in (False,True):
        for role,offset in (('fit',0),('fit_extra',1009)):
            a,info=pool(root/spec['dataset'],role,families,spec['windows_per_family'],spec['sequence_length'],spec['data_seed']+offset,targeted)
            arrays.append(a);support+=info
    anchors=c.anchor_pool(np.concatenate(arrays[:2]),spec['data_seed']+2)
    dev=[]
    for k,family in enumerate(families):
        a,info=q.windows(root/spec['dataset'],'development',[family],spec['development_windows_per_family'],512,spec['data_seed']+4000+k)
        dev.append(a);support+=info
    for family in ('same_v_history','slow_recovery'):
        with np.load(root/spec['dataset']/f'development__{family}.npz') as z:a={k:z[k] for k in z.files}
        length=spec['development_long_ticks'];idx=np.arange(len(a['voltage'])-length,len(a['voltage']))
        pre=a['gate_targets'][np.maximum(idx-1,0)]
        block=np.concatenate((a['voltage'][idx,...,None],a['bridge_ca'][idx,...,None],pre,a['gate_targets'][idx],
             np.broadcast_to(a['multipliers'],(length,2,11)),np.broadcast_to(a['reversals'],(length,2,11)),a['current_targets'][idx]),-1)
        dev.append(block.transpose(1,0,2).astype('float32'))
    held=c.anchor_pool(np.tile(dev[0],(spec['development_anchor_nodes']//len(dev[0]),1,1)),spec['data_seed']+5000,True)
    p.save_json(output/'window_support.json',dict(windows=support,anchor_nodes=len(anchors),held_development_nodes=len(held),
        history_intervention='targeted fit coverage plus continuous64 versus reset every16; a package, not separately identifiable'))
    normal=json.loads((root/spec['normalizers']).read_text());us=np.std(anchors[:,71:89],0).clip(.05)
    zs=np.std(np.log(anchors[:,89:107].clip(1e-30)),0).clip(.5)
    scales=[torch.tensor(a,device=device,dtype=torch.float32) for a in (normal['gate'],normal['current'],us,zs)]
    pools=[torch.tensor(a,device=device) for a in arrays];at=torch.tensor(anchors,device=device)
    groups=cohorts(torch,defs,seeds,device);static=[]
    for g in groups:
        n=g['model'].a.shape[0]
        batch=torch.where(g['flags'][:,1,None,None,None].bool(),pools[2][:spec['batch_sequences']][None],pools[0][:spec['batch_sequences']][None])
        static.append(dict(batch=batch.clone(),anchor=at[:spec['anchor_batch']][None].expand(n,-1,-1).clone(),
            ticks=torch.tensor(np.resize(spec['held_ticks'],spec['anchor_batch']),device=device,dtype=torch.float32)[None,:,None].expand(n,-1,-1).clone(),
            weights=torch.ones(n,6,device=device)))
    # Fixed FIT probe, identical across phases; development never adapts weights.
    probe=[{k:v.clone() for k,v in s.items()} for s in static]
    calibration=calibrate(torch,groups,static,scales,0,spec['short_ticks'],True)
    p.save_json(output/'fit_loss_calibration.json',dict(records=calibration,u_scales=us.tolist(),log_z_scales=zs.tolist(),normalizers=normal))
    p.save_json(output/'composition_preflight.json',c.composition_preflight(torch,groups,device))
    p.save_json(output/'optimizer_preflight.json',[optimizer_preflight(torch,g,s,scales,seeds,spec['short_ticks']) for g,s in zip(groups,static)])
    old.recurrence_preflight(output,spec,split_models(torch,groups,seeds,device),torch,device)
    graph,clips,execution=capture_training(torch,groups,static,scales,spec)
    execution.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,members=36,
        parameters_per_member={g['model'].architecture:sum(x.numel() for x in g['model'].parameters())//g['model'].a.shape[0] for g in groups},thresholds_changed=False)
    p.save_json(output/'execution.json',execution)
    print(f'[GIADA Task39d] preflights passed; projected fit {execution["projected_training_seconds"]/60:.1f}min',flush=True)
    rng=np.random.default_rng(spec['data_seed']+99);curves=[];probes=[];best=np.full(36,np.inf);chosen=[None]*36;saved=[None]*36
    exposure=np.zeros((len(families),2),dtype=np.int64);durations=np.zeros(6,dtype=np.int64);start=time.perf_counter()
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            scores,parts=c.selection(torch,groups,dev,held,normal,spec,device);assert np.isfinite(scores).all()
            j=0
            for g in groups:
                for k in range(g['model'].a.shape[0]):
                    if scores[j]<best[j]:
                        best[j]=scores[j];chosen[j]=step;saved[j]={name:x[k].detach().cpu().clone() for name,x in g['model'].named_parameters()}
                    j+=1
            curves.append(dict(step=step,common_development_score=scores.tolist(),component_scores=parts,clipping_counts=[x.cpu().tolist() for x in clips],elapsed_seconds=time.perf_counter()-start))
            p.save_json(output/'training_curves.json',curves)
            torch.save({g['model'].architecture:g['model'].state_dict() for g in groups},output/f'checkpoint_{step}.pt')
            for g,s,pr in zip(groups,static,probe):
                pr['weights'].copy_(s['weights'])
                rec,_=gradient_diagnostics(torch,g,pr,scales,step,spec['short_ticks']);probes.append(rec)
            p.save_json(output/'gradient_probes.json',probes)
            print(f'[GIADA Task39d] checkpoint {step} median={np.median(scores):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
        if step==spec['checkpoints'][-1]:break
        if step in spec['balance_steps']:
            calibration+=calibrate(torch,groups,probe,scales,step,spec['short_ticks'])
            for s,pr in zip(static,probe):s['weights'].copy_(pr['weights'])
            p.save_json(output/'phase_calibration.json',calibration)
        idx=rng.integers(len(arrays[0]),size=spec['batch_sequences']);ex=rng.random(len(idx))<.5
        ai=rng.integers(len(anchors),size=spec['anchor_batch']);di=rng.integers(6,size=len(ai))
        ti=torch.tensor(idx,device=device);ee=torch.tensor(ex,device=device)[:,None,None]
        ordinary=torch.where(ee,pools[1][ti],pools[0][ti]);targeted=torch.where(ee,pools[3][ti],pools[2][ti])
        ticks=torch.tensor(np.asarray(spec['held_ticks'])[di],device=device,dtype=torch.float32)[None,:,None]
        for g,s in zip(groups,static):
            s['batch'].copy_(torch.where(g['flags'][:,1,None,None,None].bool(),targeted[None],ordinary[None]))
            s['anchor'].copy_(at[ai][None]);s['ticks'].copy_(ticks)
        graph.replay();np.add.at(exposure,(idx//spec['windows_per_family'],ex.astype(int)),spec['sequence_length']);np.add.at(durations,di,1)
        if (step+1)%2000==0:print(f'[GIADA Task39d] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    torch.cuda.synchronize();training_seconds=time.perf_counter()-start;j=0
    with torch.no_grad():
        for g in groups:
            for k in range(g['model'].a.shape[0]):
                for name,x in g['model'].named_parameters():x[k].copy_(saved[j][name])
                j+=1
    torch.save({g['model'].architecture:g['model'].state_dict() for g in groups},output/'frozen_checkpoint.pt')
    freeze=dict(arms=defs,seeds=seeds,chosen_steps=chosen,scores=best.tolist(),development_only=True,independent_confirmation_accessed=False,
        stress_used_for_selection=False,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'))
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    p.save_json(output/'exposure.json',dict(variable_ticks_family_role=exposure.tolist(),anchor_duration_counts=durations.tolist(),
        variable_targets_per_member=spec['checkpoints'][-1]*spec['sequence_length']*spec['batch_sequences'],anchor_targets_per_member=spec['checkpoints'][-1]*spec['anchor_batch']))
    base=json.loads((root/old.CONFIG).read_text());evaluation=dict(spec,architectures=[d['name'] for d in defs])
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=base[key]
    assert p.digest(root/base['baseline_parent'])==base['baseline_parent_sha256']
    assert p.digest(root/base['baseline_checkpoint'])==base['baseline_sha256']
    summary=old.evaluate(root,output,evaluation,split_models(torch,groups,seeds,device),torch,device,quality)
    rows=json.loads((output/'recurrence_evaluation.json').read_text());scores={}
    for d in defs:
        scores[d['name']]={}
        for seed in seeds:
            rr=[r for r in rows if r['model']=='joint_w'+d['name'] and r['role']=='development' and r['seed']==seed]
            vals=[q.severity(r['metrics']) for r in rr]
            scores[d['name']][str(seed)]=dict(median=float(np.median(vals)),worst=max(vals),passed=sum(r['metrics']['passed'] for r in rr))
    contrasts=[]
    for axis,a,z in contrast_pairs(defs):
        ratios=[scores[z][str(s)]['median']/max(scores[a][str(s)]['median'],1e-12) for s in seeds]
        worst=[scores[z][str(s)]['worst']/max(scores[a][str(s)]['worst'],1e-12) for s in seeds]
        contrasts.append(dict(axis=axis,control=a,treatment=z,paired_ratios=ratios,worst_ratios=worst,supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=contrasts))
    budget=[]
    for low,high in ((30000,60000),(60000,120000)):
        aa=np.asarray(next(r['common_development_score'] for r in curves if r['step']==low))
        zz=np.asarray(next(r['common_development_score'] for r in curves if r['step']==high))
        for i,d in enumerate(defs):
            ratios=(zz[i*3:(i+1)*3]/aa[i*3:(i+1)*3].clip(1e-12)).tolist()
            budget.append(dict(arm=d['name'],from_step=low,to_step=high,paired_ratios=ratios,improvement_supported=bool(np.median(ratios)<=.8 and max(ratios)<1)))
    p.save_json(output/'budget_contrasts.json',budget)
    c.cost_benchmark(torch,groups,seeds,device,output,model_factory=model_class,splitter=split_models)
    passing=[d['name'] for d in defs if summary['joint_w'+d['name']]['all_development_passed']]
    execution.update(training_wall_seconds=training_seconds,total_wall_seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated());p.save_json(output/'execution.json',execution)
    final=dict(valid=True,task='39d',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',passing_arms=passing,
        summaries=summary,thresholds_changed=False,independent_confirmation_accessed=False,speedup_claimed=False,limits=spec['limits'])
    p.save_json(output/'final_report.json',final);return final
