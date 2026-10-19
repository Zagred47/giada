"""Task39i: paired equilibrium/relaxation, information and learned SK shape.

Canonical coefficients appear only in offline labels and diagnostic oracles.
All deployed kinetics remain learned; current readout and quality are Task37.
"""
import hashlib
import json
import time
import numpy as np
from . import task39h_relaxation_identification as h
g,f,e,d,c,b,q,old,p=h.g,h.f,h.e,h.d,h.c,h.b,h.q,h.old,h.p
CONFIG='experiments/task39i_preregistration_v1.json'
TERMS=('variable','held','u','log_z','relative_u','composition','sk_relative_u')


def arms():
    settings=[(0,0,'plain',0,0),(1,0,'plain',0,0),(0,1,'plain',0,0),
              (1,1,'plain',0,0),(1,1,'plain',1,0),(1,1,'channel',0,0),
              (1,1,'generic',0,0),(1,1,'plain',0,1),(1,1,'monotone',0,0),
              (1,1,'monotone',0,1),(1,1,'transfer',0,0),(1,1,'transfer',0,1)]
    return [dict(name=f'I{i}',architecture=a,relative_u=u,composition=z,history=hist,sk_relative=sk,
                 horizon=64,learning_rate=.0001) for i,(u,z,a,hist,sk) in enumerate(settings)]


def contrast_pairs(definitions=None):
    return [('u','I0','I1'),('u','I2','I3'),('relaxation','I0','I2'),('relaxation','I1','I3'),
            ('history','I3','I4'),('channel_adapter','I3','I5'),('generic_capacity','I3','I6'),
            ('channel_bias_matched_capacity','I6','I5'),('sk_objective','I3','I7'),
            ('sk_objective','I8','I9'),('sk_shape_matched_transfer','I10','I8'),
            ('sk_shape_matched_transfer','I11','I9'),('transfer','I3','I10'),('transfer','I7','I11')]


def model_class(torch,training=False):
    parent=g.model_class(torch) if training else f.model_class(torch)
    class Model(parent):
        def __init__(self,architecture,seeds):
            super().__init__('wide',seeds);self.architecture=architecture;n=len(seeds)
            if architecture in ('channel','generic'):
                width=40 if architecture=='channel' else 6
                rows=[torch.randn(1,width,generator=torch.Generator().manual_seed(s+3909)) for s in seeds]
                self.va=torch.nn.Parameter(torch.stack(rows));self.vb=torch.nn.Parameter(torch.zeros(n,1,width))
                self.vc=torch.nn.Parameter(torch.zeros(n,4 if architecture=='channel' else 6,34))
                self.vd=torch.nn.Parameter(torch.zeros(n,34))
                if architecture=='channel':
                    # Ten voltage-driven channel groups,17gates; SK remains calcium-only.
                    labels=[j for j in range(11) for _ in range(p.OFFSETS[j+1]-p.OFFSETS[j]) if j!=9]
                    unique=sorted(set(labels));mapping=[unique.index(j) for j in labels]*2
                    self.register_buffer('channel_index',torch.tensor(mapping),persistent=False)
            if architecture=='monotone':
                self.ms=torch.nn.Parameter(torch.zeros(n,16));self.mb=torch.nn.Parameter(torch.zeros(n,1))
                self.register_buffer('knots',torch.linspace(-1,1,17)[1:-1],persistent=False)

        def mono_raw(self,ca):
            cc=(ca.clamp_min(1e-9).log10()+4.5)/2.5
            slopes=torch.nn.functional.softplus(self.ms)
            # Positive slopes on all intervals, including outside the FIT domain.
            return (self.mb+(cc+1)*slopes[:,:1]+
                    (torch.relu(cc[...,None]-self.knots)*(slopes[:,None,1:]-slopes[:,None,:-1])).sum(-1))

        def raw(self,v,ca):
            u,z=super().raw(v,ca)
            if self.architecture in ('channel','generic'):
                vv=(v+40)/80
                dtype=torch.float64 if training else v.dtype
                hidden=torch.nn.functional.silu(torch.bmm(vv.to(dtype)[...,None],self.va.to(dtype))+self.vb.to(dtype))
                if self.architecture=='channel':
                    selected=hidden.reshape(hidden.shape[0],hidden.shape[1],10,4)[:,:,self.channel_index]
                    extra=(selected*self.vc.to(dtype).transpose(1,2)[:,None]).sum(-1)+self.vd.to(dtype)[:,None]
                else:extra=torch.bmm(hidden,self.vc.to(dtype))+self.vd.to(dtype)[:,None]
                extra=extra.to(v.dtype)
                u=u+torch.cat((extra[...,:16],torch.zeros_like(extra[...,0:1]),extra[...,16:17]),-1)
                z=z+torch.cat((extra[...,17:33],torch.zeros_like(extra[...,0:1]),extra[...,33:34]),-1)
            if self.architecture=='monotone':u=torch.cat((u[...,:16],self.mono_raw(ca)[...,None],u[...,17:]),-1)
            return u,z
    return Model


def load_parents(torch,root,spec,device,dtype):
    folder=root/spec['parent'];freeze=json.loads((folder/'selection_freeze.json').read_text());sha=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==sha
    assert p.digest(folder/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']==spec['parent_sha256']
    weights=torch.load(folder/'frozen_checkpoint.pt',map_location=device,weights_only=True)
    aa=[a for a in freeze['arms'] if a['learning_rate']==.0001];index=[a['name'] for a in aa].index('H8')*3
    model=f.model_class(torch)('wide',spec['seeds']).to(device)
    model.load_state_dict({k:v.clone() if k=='gbar' else v[index:index+3].clone() for k,v in weights['lr0001'].items()})
    return [model.to(dtype=dtype)]


def masks(torch,model):
    out={k:torch.ones_like(v) for k,v in model.named_parameters()}
    if model.architecture=='monotone':out['sc'][...,0]=0;out['sd'][...,0]=0
    return out


def groups(torch,root,spec,device):
    parent=load_parents(torch,root,spec,device,torch.float32)[0];out=[]
    for arch in ('plain','channel','generic','monotone','transfer'):
        aa=[a for a in arms() if a['architecture']==arch];model=model_class(torch,True)(arch,spec['seeds']*len(aa)).to(device)
        initial=model.state_dict()
        for k,v in parent.state_dict().items():initial[k]=v.clone() if k=='gbar' else v.repeat(len(aa),*([1]*(v.ndim-1)))
        model.load_state_dict(initial)
        out.append(dict(key=arch,definitions=aa,model=model,flags=torch.ones(len(aa)*3,4,device=device),
                        masks=masks(torch,model),learning_rate=.0001))
    return out


def split_models(torch,cohorts,seeds,device):
    result=[]
    for group in cohorts:
        for i,a in enumerate(group['definitions']):
            model=model_class(torch)(a['architecture'],seeds).to(device)
            model.load_state_dict({k:v.clone() if k=='gbar' else v[i*len(seeds):(i+1)*len(seeds)].clone() for k,v in group['model'].state_dict().items()})
            result.append(model)
    return result


def evaluation_groups(torch,cohorts,seeds,device):
    out=[]
    for group in cohorts:
        model=model_class(torch)(group['model'].architecture,seeds*len(group['definitions'])).to(device)
        model.load_state_dict(group['model'].state_dict())
        out.append(dict(group,model=model))
    return out


def transfer_sk(torch,cohorts,parent,spec,device):
    """FIT-only function transfer with explicit original-head matched controls."""
    mono=next(x['model'] for x in cohorts if x['key']=='monotone')
    control=next(x['model'] for x in cohorts if x['key']=='transfer')
    ca=torch.logspace(-7,-2,513,device=device)[None].expand(3,-1);vv=torch.zeros_like(ca)
    with torch.no_grad():
        target=parent.raw(vv,ca)[0][...,16]
        knots_ca=torch.logspace(-7,-2,17,device=device)[None].expand(3,-1)
        knots=parent.raw(torch.zeros_like(knots_ca),knots_ca)[0][...,16]
        slopes=(torch.diff(knots,dim=1)/.125).clamp_min(1e-4)
        mono.ms.copy_(torch.log(torch.expm1(slopes)).repeat(2,1));mono.mb.copy_(knots[:,:1].repeat(2,1))
    original={k:v.detach().clone() for k,v in control.named_parameters()}
    records=[]
    # Equal budgets and17trainable SK-u parameters per member in both heads.
    opt=torch.optim.Adam((mono.ms,mono.mb),lr=.001)
    ca6=ca.repeat(2,1);v6=vv.repeat(2,1);target6=target.repeat(2,1)
    for _ in range(spec['transfer_steps']):
        opt.zero_grad(set_to_none=False);ru=mono.mono_raw(ca6)
        (torch.tanh((ru-target6)/6).square().mean(1).sum()).backward();opt.step()
    with torch.no_grad():mapped=mono.mono_raw(ca6).detach()
    opt2=torch.optim.Adam((control.sc,control.sd),lr=.001)
    for _ in range(spec['transfer_steps']):
        opt2.zero_grad(set_to_none=False);ru=control.raw(v6,ca6)[0][...,16]
        torch.tanh((ru-mapped)/6).square().mean(1).sum().backward()
        control.sc.grad[...,1]=0;control.sd.grad[...,1]=0;opt2.step()
    with torch.no_grad():
        grid=torch.logspace(-7,-2,4097,device=device)[None].expand(6,-1);v=torch.zeros_like(grid)
        target_grid=parent.coefficients(v[:3],grid[:3])[0][...,16].repeat(2,1)
        m=mono.coefficients(v,grid)[0][...,16];ctrl=control.coefficients(v,grid)[0][...,16]
        before=target_grid;err=(m-ctrl).abs().amax(1)
        for j in range(6):records.append(dict(seed=spec['seeds'][j%3],arm=cohorts[-2]['definitions'][j//3]['name'],
            original_to_monotone_max_u=float((m[j]-before[j]).abs().max()),
            matched_transfer_max_u=float(err[j]),monotone=bool((torch.diff(m[j])>=-1e-7).all())))
        assert all(x['monotone'] for x in records)
        assert float(err.max())<=spec['transfer_atol'],('SK function matching failed',records)
        # The unused original u head must not affect monotone inference; z is preserved.
        for k in original:
            if k not in ('sc','sd'):assert torch.equal(original[k],dict(control.named_parameters())[k])
    return dict(valid=True,FIT_only=True,teacher_labels_used=False,steps_each=spec['transfer_steps'],
        trainable_sk_u_parameters_each=17,matching_atol=spec['transfer_atol'],rows=records,
        limits='Residual transfer mismatch retained explicitly. I3/I10 andI7/I11 quantify transfer effect; I10/I8 andI11/I9 compare learned shapes.')


def relative_u_loss(torch,raw,target):
    # Censor saturated labels; bounded relative occupancy error emphasizes small gates.
    clipped=target.clamp(1e-6,1-1e-6)
    relative=(torch.sigmoid(raw)-target)/torch.minimum(clipped,1-clipped)
    return .5*e.censored_logit_error(torch,raw,target)+.5*(5*torch.tanh(relative/5)).square()


def terms(torch,group,s,scales):
    main=e.terms(torch,group,s,scales,64)
    model=group['model'];ru,rz=model.raw(s['anchor'][...,0],s['anchor'][...,1])
    err=relative_u_loss(torch,ru,s['anchor'][...,71:89])
    # Teacher log-decay is computed OFFLINE for variable FIT inputs,never in inference.
    batch=s['batch'];shape=batch.shape
    _,raw=model.raw(batch[...,0].reshape(shape[0],-1),batch[...,1].reshape(shape[0],-1))
    predicted=torch.nn.functional.logsigmoid(-raw).reshape(*shape[:3],18)
    actual=batch[...,71:89]
    pred_sum=predicted.cumsum(2).clamp_min(-20);true_sum=actual.cumsum(2).clamp_min(-20)
    accumulated=((pred_sum-true_sum)/true_sum.abs().clamp_min(.01)).square()
    induced=(torch.expm1(pred_sum)-torch.expm1(true_sum)).square()
    composition=(accumulated+induced).mean((1,2,3))
    return torch.cat((main,err.mean((1,2))[:,None],composition[:,None],err[...,16].mean(1)[:,None]),-1)


def objective(torch,group,s,scales):return (terms(torch,group,s,scales)*s['weights']).sum(-1)


def gradient_probe(torch,group,s,scales,step):
    values=terms(torch,group,s,scales);grads=[]
    for j in range(len(TERMS)):
        grad=torch.autograd.grad(values[:,j].sum(),tuple(group['model'].parameters()),retain_graph=j+1<len(TERMS),allow_unused=True)
        grads.append(torch.cat([(torch.zeros_like(v) if x is None else x*group['masks'][k]).flatten(1)
            for (k,v),x in zip(group['model'].named_parameters(),grad)],-1).detach().double())
    stack=torch.stack(grads,1);norm=stack.norm(dim=-1);cos=torch.bmm(stack,stack.transpose(1,2))/(norm[:,:,None]*norm[:,None,:]).clamp_min(1e-20)
    with torch.no_grad():
        u,z,_=group['model'].coefficients(s['anchor'][...,0],s['anchor'][...,1]);ref=s['anchor'][...,71:89]
        bias=(u-ref);rate=z.clamp_min(1e-30).log()-s['anchor'][...,89:107].clamp_min(1e-30).log()
    return dict(step=step,cohort=group['key'],terms=TERMS,losses=values.detach().cpu().tolist(),
        term_norms=norm.cpu().tolist(),cosines=cos.cpu().tolist(),weights=s['weights'].cpu().tolist(),
        u_signed_bias=bias.mean(1).cpu().tolist(),u_rmse=bias.square().mean(1).sqrt().cpu().tolist(),
        log_z_signed_bias=rate.mean(1).cpu().tolist())


def update(torch,group,s,scales,opt,counter=None):
    opt.zero_grad(set_to_none=False);objective(torch,group,s,scales).sum().backward()
    for k,v in group['model'].named_parameters():
        if v.grad is None:v.grad=torch.zeros_like(v)
        v.grad.mul_(group['masks'][k])
    norm=torch.stack([v.grad.flatten(1).square().sum(1) for v in group['model'].parameters()]).sum(0).sqrt()
    if counter is not None:counter.add_((norm>1).float())
    for v in group['model'].parameters():v.grad.div_(norm.clamp_min(1).reshape(-1,*([1]*(v.ndim-1))))
    opt.step()


def optimizer_preflight(torch,group,s,scales,seeds):
    model=group['model'];initial={k:v.detach().clone() for k,v in model.named_parameters()}
    opt=torch.optim.Adam(model.parameters(),lr=.0001);update(torch,group,s,scales,opt);err=0.
    for j in range(model.a.shape[0]):
        iso=model_class(torch,True)(model.architecture,[seeds[j%3]]).to(s['batch'].device)
        with torch.no_grad():
            for k,v in iso.named_parameters():v.copy_(initial[k][j:j+1])
        gg=dict(model=iso,flags=group['flags'][j:j+1],masks=masks(torch,iso));ss={k:v[j:j+1] for k,v in s.items()}
        oo=torch.optim.Adam(iso.parameters(),lr=.0001);update(torch,gg,ss,scales,oo)
        err=max(err,max(float((v[0]-dict(model.named_parameters())[k][j]).abs().max()) for k,v in iso.named_parameters()))
    frozen=max(float(((v-initial[k])*(1-group['masks'][k])).abs().max()) for k,v in model.named_parameters())
    with torch.no_grad():
        for k,v in model.named_parameters():v.copy_(initial[k]);v.grad=None
    assert err<=1e-5 and frozen==0,(err,frozen)
    return dict(valid=True,members=model.a.shape[0],max_abs_error=err,frozen_parameter_error=frozen,independent_moments_and_clipping=True)


def capture(torch,cohorts,static,scales,spec):
    initial=[{k:v.detach().clone() for k,v in x['model'].named_parameters()} for x in cohorts]
    opts=[torch.optim.Adam(x['model'].parameters(),lr=.0001,capturable=True) for x in cohorts]
    clips=[torch.zeros(x['model'].a.shape[0],device='cuda') for x in cohorts]
    def work():
        for group,s,opt,clip in zip(cohorts,static,opts,clips):update(torch,group,s,scales,opt,clip)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):work()
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):work()
    def reset():
        with torch.no_grad():
            for group,state,opt in zip(cohorts,initial,opts):
                for k,v in group['model'].named_parameters():v.copy_(state[k]);v.grad.zero_()
                for st in opt.state.values():
                    for value in st.values():
                        if torch.is_tensor(value):value.zero_()
            for clip in clips:clip.zero_()
    reset();work();work();expected=[{k:v.detach().clone() for k,v in x['model'].named_parameters()} for x in cohorts]
    reset();graph.replay();graph.replay();torch.cuda.synchronize()
    err=max(float((v-state[k]).abs().max()) for group,state in zip(cohorts,expected) for k,v in group['model'].named_parameters())
    assert err<=spec['numeric_atol'] and all(bool(torch.isfinite(v).all()) for x in cohorts for v in x['model'].parameters()),err
    reset();start=time.perf_counter()
    for _ in range(20):graph.replay()
    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/20;reset()
    assert seconds*spec['checkpoints'][-1]<spec['max_projected_training_seconds']
    graph._giada_optimizers=opts
    return graph,clips,dict(valid=True,two_update_max_error=err,seconds_per_update=seconds,
        projected_training_seconds=seconds*spec['checkpoints'][-1],optimizer_state_lifetime_retained=True,optimizer_reset=True)


def labelled(data):
    _,z=p.coefficients(data[...,0].astype(float),data[...,1].astype(float),.1)
    decay=np.log1p(-np.minimum(z,1-1e-12)).astype('float32')
    return np.concatenate((data[...,:71],decay),-1)


def independent_history(root,output,spec,quality):
    folder=output.parent/'variable_history_fit';folder.mkdir(exist_ok=False);arrays=[];anchors=[];manifest=[]
    for role in ('fit','fit_extra'):
        rng=np.random.default_rng(spec['history_seeds'][role]);ticks=spec['recovery_ticks'];count=36
        v=np.empty((ticks,count));ca=np.empty_like(v)
        # Independent preconditioning, ramps, plateaux and release; no development path copied.
        for j in range(count):
            boundaries=sorted(set([0,ticks]+rng.integers(1,ticks,6).tolist()))
            previous_v=rng.uniform(-100,60);previous_ca=rng.uniform(-7,-2)
            for left,right in zip(boundaries[:-1],boundaries[1:]):
                end_v=rng.uniform(-100,60);end_ca=-7+5*((j+.5)/count)
                if j%3==0:
                    v[left:right,j]=np.linspace(previous_v,end_v,right-left);ca[left:right,j]=10**np.linspace(previous_ca,end_ca,right-left)
                else:v[left:right,j]=end_v;ca[left:right,j]=10**end_ca
                previous_v,previous_ca=end_v,end_ca
        initial=rng.uniform(0,1,(count,18));mult=rng.uniform(.25,4,(count,11));rev=p.ionic.REVERSALS+rng.uniform(-5,5,(count,11))
        gates,current,source=p.recurrent_reference(v,ca,initial,mult,rev,.1)
        path=folder/f'{role}__history_grid.npz'
        np.savez_compressed(path,voltage=v.astype('float32'),bridge_ca=ca.astype('float32'),initial_gates=initial.astype('float32'),
            multipliers=mult.astype('float32'),reversals=rev.astype('float32'),gate_targets=gates.astype('float32'),current_targets=current.astype('float32'),source_ca=source.astype('float32'))
        data,info=q.windows(folder,role,['history_grid'],spec['windows_per_family']*len(quality['families']),256,spec['history_seeds'][role]+9)
        arrays.append(data);anchors.append(c.anchor_pool(data,spec['history_seeds'][role]+2,True))
        signatures=[]
        for j in range(count):
            hh=hashlib.sha256()
            for value in (v[:,j],ca[:,j],initial[j],mult[j],rev[j]):hh.update(np.asarray(value,dtype='<f4').tobytes())
            signatures.append(hh.hexdigest())
        manifest.append(dict(role=role,seed=spec['history_seeds'][role],sha256=p.digest(path),file=path.name,
            input_signatures=signatures,windows=info,trajectories=count,ticks=ticks,window_sha256=hashlib.sha256(data.tobytes()).hexdigest()))
        print(f'[GIADA Task39i] independent history FIT {role} generated',flush=True)
    existing=set()
    for role in quality['roles']:
        for family in quality['families']:
            with np.load(root/spec['dataset']/f'{role}__{family}.npz') as zz:
                for j in range(zz['voltage'].shape[1]):
                    hh=hashlib.sha256()
                    for key in ('voltage','bridge_ca','initial_gates','multipliers','reversals'):
                        value=zz[key][:,j] if key in ('voltage','bridge_ca') else zz[key][j]
                        hh.update(np.asarray(value,dtype='<f4').tobytes())
                    existing.add(hh.hexdigest())
    for row in json.loads((output/'recovery_fit_manifest.json').read_text())['files']:existing.update(row['input_signatures'])
    new=[s for row in manifest for s in row['input_signatures']]
    assert len(set(new))==len(new) and not existing&set(new)
    p.save_json(output/'history_fit_manifest.json',dict(valid=True,files=manifest,FIT_only=True,disjoint_from_all_original_roles=True))
    return arrays,np.concatenate(anchors)


def synthetic_preflight(torch,device):
    u=torch.tensor([0.,1e-9,.01,.5,.99,1.],device=device)[None,:,None].expand(2,-1,18)
    raw=torch.logit(u.clamp(1e-6,1-1e-6)).requires_grad_();loss=relative_u_loss(torch,raw,u).mean();loss.backward()
    assert bool(torch.isfinite(raw.grad).all()) and bool(torch.isfinite(loss))
    target=torch.logspace(-9,-.001,18,device=device)[None,None].expand(2,64,18)
    ld=torch.log1p(-target);damage=ld*1.01
    exact=(ld.cumsum(1).clamp_min(-20)-ld.cumsum(1).clamp_min(-20)).square().mean()
    harmed=(damage.cumsum(1).clamp_min(-20)-ld.cumsum(1).clamp_min(-20)).square().mean()
    assert float(exact)==0 and float(harmed)>0
    return dict(valid=True,saturated_u_loss_finite=True,exact_composition_loss=float(exact),perturbed_composition_loss=float(harmed))


def run(root,output,device='cuda'):
    import torch
    assert device=='cuda' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());quality=json.loads((root/p.CONFIG).read_text());assert p.digest(root/p.CONFIG)==spec['quality_config_sha256']
    output.mkdir(parents=True,exist_ok=True);assert not (output/'final_report.json').exists();p.save_json(output/'preregistration.json',spec)
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()):assert p.digest(root/spec['dataset']/row['file'])==row['sha256']
    p.save_json(output/'synthetic_loss_preflight.json',synthetic_preflight(torch,device))
    d.oracle_audit(torch,root,output,spec,quality,device,parent_loader=load_parents,timing_fn=f.timing_audit)
    constant,constant_at=h.recovery_fit(root,output,spec,quality)
    history,history_at=independent_history(root,output,spec,quality)
    originals=[];support=[]
    for role,offset in [('fit',0),('fit_extra',1009)]:
        data,info=q.windows(root/spec['dataset'],role,list(quality['families']),spec['windows_per_family'],256,spec['data_seed']+offset)
        originals.append(data);support+=info
    at=c.anchor_pool(np.concatenate(originals),spec['data_seed']+2,True)
    parent_cal=json.loads((root/spec['parent']/'fit_loss_calibration.json').read_text());normal=json.loads((root/spec['normalizers']).read_text())
    scales=[torch.tensor(value,device=device,dtype=torch.float32) for value in (normal['gate'],normal['current'],parent_cal['u_scales'],parent_cal['log_z_scales'])]
    pools=[[torch.tensor(labelled(x),device=device) for x in arrays] for arrays in (originals,constant,history)]
    # Only the first71 columns enter anchors; decay labels are an offline loss sidecar.
    anchors=[torch.tensor(x,device=device) for x in (at,constant_at,history_at)];del originals,constant,history
    cohorts=groups(torch,root,spec,device);parent=load_parents(torch,root,spec,device,torch.float32)[0]
    p.save_json(output/'sk_transfer_preflight.json',transfer_sk(torch,cohorts,parent,spec,device))
    static=[];map_error=0.;calib=np.asarray(parent_cal['parent_weights'])
    for group,deployed in zip(cohorts,evaluation_groups(torch,cohorts,spec['seeds'],device)):
        n=group['model'].a.shape[0]
        with torch.no_grad():
            v=torch.linspace(-95,55,7,device=device)[None].expand(n,-1);ca=torch.logspace(-7,-2,7,device=device)[None].expand(n,-1);state=torch.full((n,7,18),.4,device=device)
            map_error=max(map_error,float((group['model'](v,ca,state)-deployed['model'](v,ca,state)).abs().max()))
        weights=[[.5,.5,*calib[j],0.,0.,0.] for a in group['definitions'] for j in range(3)]
        static.append(dict(batch=pools[0][0][:2].reshape(8,64,89)[None].expand(n,-1,-1,-1).clone(),
            anchor=anchors[0][:64][None].expand(n,-1,-1).clone(),ticks=torch.full((n,64,1),1000.,device=device),weights=torch.tensor(weights,device=device,dtype=torch.float32)))
    assert map_error<=spec['numeric_atol'];p.save_json(output/'training_deployment_preflight.json',dict(valid=True,one_step_max_error=map_error,atol=spec['numeric_atol']))
    probe=gradient_probe(torch,cohorts[0],static[0],scales,0);norm=np.asarray(probe['term_norms'])[:3];cos=np.asarray(probe['cosines'])[:3]
    main=np.asarray(static[0]['weights'].cpu())[:3,:4]*norm[:,:4]
    combined=np.sqrt(np.maximum(np.einsum('bi,bij,bj->b',main,cos[:,:4,:4],main),1e-30))
    lam=np.clip(.1*combined[:,None]/np.maximum(norm[:,4:],1e-12),1e-4,1e4)
    for group,s in zip(cohorts,static):
        for i,a in enumerate(group['definitions']):
            flags=np.asarray([a['relative_u'],a['composition'],a['sk_relative']])
            s['weights'][i*3:(i+1)*3,4:]=torch.tensor(lam*flags,device=device,dtype=torch.float32)
    p.save_json(output/'fit_loss_calibration.json',dict(parent_weights=calib.tolist(),u_scales=parent_cal['u_scales'],log_z_scales=parent_cal['log_z_scales'],
        additional_weights=lam.tolist(),source='Initial originalFIT I0 gradients,0.1main norm per new active term;fixed across treatments',probe=probe))
    p.save_json(output/'window_support.json',dict(original=support,constant_fraction=.5,history_replaces_constant_only_for_I4=True,target_ticks_per_update=512))
    p.save_json(output/'optimizer_preflight.json',[optimizer_preflight(torch,x,s,scales,spec['seeds']) for x,s in zip(cohorts,static)])
    p.save_json(output/'composition_preflight.json',c.composition_preflight(torch,cohorts,device))
    old.recurrence_preflight(output,spec,split_models(torch,cohorts,spec['seeds'],device),torch,device)
    graph,clips,execution=capture(torch,cohorts,static,scales,spec)
    definitions=[a for group in cohorts for a in group['definitions']];count=len(definitions)*3
    execution.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,members=count,backend='five vectorized cohorts in one CUDA Graph',
        parameters={x['key']:dict(stored=sum(v.numel() for v in x['model'].parameters())//x['model'].a.shape[0],active=int(sum(v[0].sum() for v in x['masks'].values()))) for x in cohorts},
        training_affine_accumulation='float64;FP32parameters/states/Adam',evaluation_inference_dtype='float32',thresholds_changed=False)
    p.save_json(output/'execution.json',execution);print(f'[GIADA Task39i] preflights passed;projected fit {execution["projected_training_seconds"]/60:.1f}min',flush=True)
    dev,held=h.development(root,spec,quality);fixed=[{k:v.clone() for k,v in s.items()} for s in static]
    rng=np.random.default_rng(spec['data_seed']+399);best=[None]*count;saved=[None]*count;chosen=[None]*count;curves=[];probes=[];invalid=set();start=time.perf_counter();exposure=np.zeros((3,2),dtype=np.int64)
    baseline=json.loads((root/old.CONFIG).read_text());compare_spec=json.loads((root/h.CONFIG).read_text())
    def evaluation_spec():
        out=dict(spec,architectures=[a['name'] for a in definitions]+['frozen39h_H8','frozen39g_A0','frozen39g_A6'])
        for k in ('baseline_parent','baseline_checkpoint'):out[k]=baseline[k]
        return out
    assert p.digest(root/baseline['baseline_parent'])==baseline['baseline_parent_sha256'] and p.digest(root/baseline['baseline_checkpoint'])==baseline['baseline_sha256']
    def evaluate(folder):
        return old.evaluate(root,folder,evaluation_spec(),split_models(torch,cohorts,spec['seeds'],device)+[parent]+h.load_parents(torch,root,compare_spec,device,torch.float32),torch,device,quality)
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            tmp=output/f'checkpoint_{step}.tmp';torch.save({x['key']:x['model'].state_dict() for x in cohorts},tmp);tmp.replace(output/f'checkpoint_{step}.pt')
            tmp=output/f'optimizer_{step}.tmp';torch.save(dict(optimizers=[o.state_dict() for o in graph._giada_optimizers],numpy_rng=rng.bit_generator.state,step=step,clipping_counts=[x.cpu() for x in clips]),tmp);tmp.replace(output/f'optimizer_{step}.pt')
            p.save_json(output/'checkpoint_phase.json',dict(step=step,phase='before_selection',parameters_finite={x['key']:all(bool(torch.isfinite(v).all()) for v in x['model'].parameters()) for x in cohorts}))
            print(f'[GIADA Task39i] checkpoint_saved {step} before_selection',flush=True)
            common,parts,ranks,panel=e.checkpoint_scores(torch,evaluation_groups(torch,cohorts,spec['seeds'],device),dev,held,normal,spec,device,quality);j=0
            for group in cohorts:
                for k in range(group['model'].a.shape[0]):
                    if ranks[j] is None:invalid.add(j)
                    elif best[j] is None or ranks[j]<best[j]:best[j]=ranks[j];chosen[j]=step;saved[j]={name:v[k].detach().cpu().clone() for name,v in group['model'].named_parameters()}
                    j+=1
            assert all(x is not None for x in saved)
            curves.append(dict(step=step,common_development_score=common.tolist(),panel=panel,elapsed_seconds=time.perf_counter()-start,clipping_counts=[x.cpu().tolist() for x in clips]))
            p.save_json(output/'training_curves.json',e.diagnostic_json(curves));p.save_json(output/'numerical_health.json',dict(invalid_members=sorted(invalid),invalid_checkpoints_are_never_selected=True))
            for group,s in zip(cohorts,fixed):probes.append(gradient_probe(torch,group,s,scales,step))
            p.save_json(output/'fixed_fit_gradient_probes.json',e.diagnostic_json(probes))
            if step in spec['full_trace_checkpoints']:
                folder=output/f'diagnostic_checkpoint_{step}';folder.mkdir();summary=evaluate(folder)
                p.save_json(folder/'diagnostic_only.json',dict(step=step,selection_eligible=False,summaries=summary))
            print(f'[GIADA Task39i] checkpoint {step} median_common={np.median(common):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
        if step==spec['checkpoints'][-1]:break
        ix=rng.integers(pools[0][0].shape[0],size=2);role=rng.random(2)<.5;replacement=rng.random(2)<.5
        ti=torch.tensor(ix,device=device);role_t=torch.tensor(role,device=device)[:,None,None];replace_t=torch.tensor(replacement,device=device)[:,None,None]
        batches=[torch.where(role_t,pool[1][ti],pool[0][ti]) for pool in pools]
        ai=rng.integers(len(at),size=64);which=rng.random(64)<.5;di=rng.integers(len(spec['held_ticks']),size=64)
        tt=torch.tensor(np.asarray(spec['held_ticks'])[di],device=device,dtype=torch.float32)[None,:,None]
        for group,s in zip(cohorts,static):
            hist=torch.tensor([a['history'] for a in group['definitions'] for _ in spec['seeds']],device=device).bool()
            recovery=torch.where(hist[:,None,None,None],batches[2][None],batches[1][None])
            s['batch'].copy_(torch.where(replace_t[None],recovery,batches[0][None]).reshape(s['batch'].shape))
            old_a=anchors[0][ai];const_a=anchors[1][ai % len(constant_at)];hist_a=anchors[2][ai % len(history_at)]
            extra_a=torch.where(hist[:,None,None],hist_a[None],const_a[None])
            s['anchor'].copy_(torch.where(torch.tensor(which,device=device)[None,:,None],extra_a,old_a[None]));s['ticks'].copy_(tt)
        graph.replay();np.add.at(exposure[0],role[~replacement].astype(int),256);np.add.at(exposure[1],role[replacement].astype(int),256);np.add.at(exposure[2],role[replacement].astype(int),256)
        if (step+1)%2000==0:print(f'[GIADA Task39i] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    training=time.perf_counter()-start;j=0
    with torch.no_grad():
        for group in cohorts:
            for k in range(group['model'].a.shape[0]):
                for name,v in group['model'].named_parameters():v[k].copy_(saved[j][name])
                j+=1
    torch.save({x['key']:x['model'].state_dict() for x in cohorts},output/'frozen_checkpoint.pt')
    freeze=dict(arms=definitions,seeds=spec['seeds'],chosen_steps=chosen,ranks=best,development_only=True,stress_used_for_selection=False,
        independent_confirmation_accessed=False,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'))
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    p.save_json(output/'exposure.json',dict(role_target_ticks=exposure.tolist(),variable_targets_per_member=512*spec['checkpoints'][-1],anchor_targets_per_member=64*spec['checkpoints'][-1]))
    summary=evaluate(output);rows=json.loads((output/'recurrence_evaluation.json').read_text());scores={}
    for a in definitions:
        scores[a['name']]={}
        for seed in spec['seeds']:
            rr=[r for r in rows if r['model']=='joint_w'+a['name'] and r['role']=='development' and r['seed']==seed];values=[q.severity(r['metrics']) for r in rr]
            scores[a['name']][str(seed)]=dict(median=float(np.median(values)),worst=max(values),passed=sum(r['metrics']['passed'] for r in rr))
    contrasts=[]
    for axis,control,treatment in contrast_pairs():
        ratios=[scores[treatment][str(s)]['median']/max(scores[control][str(s)]['median'],1e-12) for s in spec['seeds']]
        worst=[scores[treatment][str(s)]['worst']/max(scores[control][str(s)]['worst'],1e-12) for s in spec['seeds']]
        contrasts.append(dict(axis=axis,control=control,treatment=treatment,paired_ratios=ratios,worst_ratios=worst,supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=contrasts))
    c.cost_benchmark(torch,cohorts,spec['seeds'],device,output,model_factory=model_class,splitter=split_models,task_label='39i')
    unstable={definitions[j//3]['name'] for j in invalid};passing=[a['name'] for a in definitions if summary['joint_w'+a['name']]['all_development_passed'] and a['name'] not in unstable]
    execution.update(training_wall_seconds=training,total_wall_seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated());p.save_json(output/'execution.json',execution)
    final=dict(valid=True,task='39i',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',passing_arms=passing,
        numerically_unstable_arms=sorted(unstable),summaries=summary,thresholds_changed=False,independent_confirmation_accessed=False,speedup_claimed=False,limits=spec['limits'])
    p.save_json(output/'final_report.json',final);return final
