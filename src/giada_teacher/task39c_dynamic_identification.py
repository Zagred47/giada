"""Task39c: identify equilibrium, relaxation, support and optimization jointly.

Teacher coefficients are offline labels/oracles only. Candidate inference has
18 explicit bounded states, a learned affine update and the existing readout.
"""
import copy
import hashlib
import itertools
import json
import time
import numpy as np
from . import task39b_architecture_diagnosis as b

q, old, p = b.q, b.old, b.p
CONFIG = 'experiments/task39c_preregistration_v1.json'


def arms():
    rows = [dict(name=f'base_i{i}_s{s}_d{d}', architecture='base', information=i,
                 support=s, duration=d) for i,s,d in itertools.product((0,1), repeat=3)]
    for architecture in ('wide', 'separate'):
        for i,s,d in ((0,0,0), (1,1,1)):
            rows.append(dict(name=f'{architecture}_i{i}_s{s}_d{d}', architecture=architecture,
                             information=i, support=s, duration=d))
    return rows


def model_class(torch):
    class Model(torch.nn.Module):
        def __init__(self, architecture, seeds):
            super().__init__()
            self.architecture = architecture
            rows = []
            for seed in seeds:
                gen = torch.Generator().manual_seed(seed)
                if architecture == 'separate':
                    rows.append((torch.randn(18,16,generator=gen), torch.zeros(18,16),
                                 torch.randn(18,16,2,generator=gen)*.01,
                                 torch.tensor([0.,-3.]).repeat(18,1)))
                else:
                    width = 32 if architecture == 'base' else 64
                    a = torch.randn(1,64,generator=gen)
                    c = torch.randn(64,34,generator=gen)*.01
                    sa = torch.randn(1,8,generator=gen)
                    sc = torch.randn(8,2,generator=gen)*.01
                    rows.append((a[:,:width],torch.zeros(1,width),c[:width],
                        torch.cat((torch.zeros(17),torch.full((17,),-3.))),
                        sa,torch.zeros(1,8),sc,torch.tensor([0.,-3.])))
            names = ('a','b','c','d') if architecture == 'separate' else ('a','b','c','d','sa','sb','sc','sd')
            for k,name in enumerate(names):
                self.register_parameter(name,torch.nn.Parameter(torch.stack([r[k] for r in rows])))
            self.register_buffer('gbar',torch.tensor(p.ionic.GBAR,dtype=torch.float32))

        def raw(self,v,ca):
            vv = (v+40)/80
            cc = (ca.clamp_min(1e-9).log10()+4.5)/2.5
            if self.architecture == 'separate':
                inp = torch.cat((vv[...,None].expand(-1,-1,16),cc[...,None],vv[...,None]),-1)
                hidden = torch.nn.functional.silu(inp[...,None]*self.a[:,None]+self.b[:,None])
                raw = (hidden[...,None]*self.c[:,None]).sum(-2)+self.d[:,None]
                return raw[...,0],raw[...,1]
            raw = torch.bmm(torch.nn.functional.silu(torch.bmm(vv[...,None],self.a)+self.b),self.c)+self.d[:,None]
            sk = torch.bmm(torch.nn.functional.silu(torch.bmm(cc[...,None],self.sa)+self.sb),self.sc)+self.sd[:,None]
            u = torch.cat((raw[...,:16],sk[...,0:1],raw[...,16:17]),-1)
            z = torch.cat((raw[...,17:33],sk[...,1:2],raw[...,33:34]),-1)
            return u,z

        def coefficients(self,v,ca):
            ru,rz = self.raw(v,ca)
            return torch.sigmoid(ru),torch.sigmoid(rz),torch.nn.functional.logsigmoid(-rz)

        def forward(self,v,ca,state):
            ru,rz = self.raw(v,ca)
            return state+torch.sigmoid(rz)*(torch.sigmoid(ru)-state)
    return Model


def compose(torch,u,log_decay,state,ticks):
    """Stable exact composition of a learned held-input map, not variable V."""
    return state-torch.expm1(ticks*log_decay)*(u-state)


def anchor_pool(data,seed,stratified=False):
    rng=np.random.default_rng(seed)
    a=data[:,0].astype('float64').copy()
    if stratified:
        n=len(a); order=rng.permutation(n)
        a[:,0]=-100+160*(order+rng.random(n))/n
        a[:,1]=10**(-7+5*(rng.permutation(n)+rng.random(n))/n)
        a[:,2:20]=rng.uniform(0,1,(n,18))
        mask=rng.random((n,18))<.5
        a[:,2:20]=np.where(mask,rng.choice([0.,1.],(n,18)),a[:,2:20])
    u,z=p.coefficients(a[:,0],a[:,1],.1)
    return np.concatenate((a,u,z),-1).astype('float32')


def held_targets(torch,anchor,ticks):
    u,z=anchor[...,71:89],anchor[...,89:107]
    # Labels only; no gradient enters teacher values. z==1 is a valid fast gate.
    relaxation=-torch.expm1(ticks*torch.log1p(-z))
    return anchor[...,2:20]+relaxation*(u-anchor[...,2:20])


def terms(torch,model,batch,anchor,ticks,gs,cs,us,zs):
    state=batch[:,:,0,2:20]
    variable=torch.zeros(batch.shape[0],device=batch.device)
    for k in range(batch.shape[2]):
        row=batch[:,:,k]
        state=model(row[...,0],row[...,1],state)
        current=b.captured_readout(torch,row[...,0],state,row[...,38:49],row[...,49:60],model.gbar)
        variable=variable+((state-row[...,20:38])/gs).square().mean((1,2))
        variable=variable+((current-row[...,60:71])/cs).square().mean((1,2))
    variable=variable/batch.shape[2]
    u,z,logd=model.coefficients(anchor[...,0],anchor[...,1])
    predicted=compose(torch,u,logd,anchor[...,2:20],ticks)
    target=held_targets(torch,anchor,ticks)
    current=b.captured_readout(torch,anchor[...,0],predicted,anchor[...,38:49],anchor[...,49:60],model.gbar)
    reference=b.captured_readout(torch,anchor[...,0],target,anchor[...,38:49],anchor[...,49:60],model.gbar)
    held=((predicted-target)/gs).square().mean((1,2))+((current-reference)/cs).square().mean((1,2))
    inf=((u-anchor[...,71:89])/us).square().mean((1,2))
    # log(z) exposes small fractions. The same transform is used for all gates.
    _,rz=model.raw(anchor[...,0],anchor[...,1])
    logz=torch.nn.functional.logsigmoid(rz)
    rate=((logz-anchor[...,89:107].clamp_min(1e-30).log())/zs).square().mean((1,2))
    return torch.stack((variable,held,inf,rate),-1)


def objective(torch,model,batch,anchor,ticks,gs,cs,us,zs,information,weights):
    values=terms(torch,model,batch,anchor,ticks,gs,cs,us,zs)
    return .5*(values[:,0]+values[:,1])+information*(weights[0]*values[:,2]+weights[1]*values[:,3])


def groups(torch,definitions,seeds,device):
    result=[]
    for architecture in ('base','wide','separate'):
        defs=[d for d in definitions if d['architecture']==architecture]
        model=model_class(torch)(architecture,seeds*len(defs)).to(device)
        info=torch.tensor([d['information'] for d in defs for _ in seeds],device=device,dtype=torch.float32)
        sup=torch.tensor([d['support'] for d in defs for _ in seeds],device=device,dtype=torch.bool)[:,None,None]
        dur=torch.tensor([d['duration'] for d in defs for _ in seeds],device=device,dtype=torch.bool)[:,None,None]
        result.append(dict(model=model,definitions=defs,information=info,support=sup,duration=dur))
    return result


def split_models(torch,cohorts,seeds,device):
    result=[]
    for group in cohorts:
        for k,d in enumerate(group['definitions']):
            m=model_class(torch)(d['architecture'],seeds).to(device)
            m.load_state_dict({name:(x.clone() if name=='gbar' else x[k*len(seeds):(k+1)*len(seeds)].clone())
                               for name,x in group['model'].state_dict().items()})
            result.append(m)
    return result


def optimizer_preflight(torch,group,batch,anchor,ticks,scales,weights,seeds):
    model=group['model'];initial={k:x.detach().clone() for k,x in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    objective(torch,model,batch,anchor,ticks,*scales,group['information'],weights).sum().backward()
    old.clip(torch,model);optimizer.step();maximum=0.
    for j in range(len(batch)):
        iso=model_class(torch)(model.architecture,[seeds[j%len(seeds)]]).to(batch.device)
        with torch.no_grad():
            for name,x in iso.named_parameters():x.copy_(initial[name][j:j+1])
        opt=torch.optim.Adam(iso.parameters(),lr=.001)
        objective(torch,iso,batch[j:j+1],anchor[j:j+1],ticks[j:j+1],*scales,
                  group['information'][j:j+1],weights).sum().backward()
        old.clip(torch,iso);opt.step()
        maximum=max(maximum,max(float((x[0]-dict(model.named_parameters())[name][j]).abs().max())
                                for name,x in iso.named_parameters()))
    with torch.no_grad():
        for name,x in model.named_parameters():x.copy_(initial[name])
    for x in model.parameters():x.grad=None
    assert maximum<=1e-5,maximum
    return dict(valid=True,members=len(batch),max_abs_error=maximum)


def composition_preflight(torch,cohorts,device):
    rows=[]
    for g in cohorts:
        m=g['model'];n=m.a.shape[0]
        v=torch.tensor([-95.,-55.,0.,45.],device=device)[None].expand(n,-1)
        ca=torch.full_like(v,.0001);x=torch.full((n,4,18),.3,device=device)
        with torch.no_grad():
            u,z,ld=m.coefficients(v,ca)
            reference=x.double().cpu().numpy().copy();uu=u.double().cpu().numpy();zz=z.double().cpu().numpy()
            for tick in range(100000):
                reference+=zz*(uu-reference)
                if tick+1 in (1,1000,100000):
                    actual=compose(torch,u,ld,x,torch.full((n,4,1),float(tick+1),device=device)).cpu().numpy()
                    error=float(np.abs(actual-reference).max())
                    assert error<=2e-5,error
                    rows.append(dict(architecture=m.architecture,ticks=tick+1,max_abs_error=error))
    return dict(valid=True,rows=rows,reference='Repeated affine map in float64, same learned u/z; GPU stable composition in float32',
                inference_step_changed=False)


def frozen39b(torch,root,spec,device,dtype):
    parent=root/spec['parent'];freeze=json.loads((parent/'selection_freeze.json').read_text())
    claimed=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==claimed
    assert p.digest(parent/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']==spec['parent_sha256']
    execution=json.loads((parent/'execution.json').read_text())
    m=b.model_class(torch)(freeze['arms'],freeze['seeds'],execution['rapid_output_positions']).to(device)
    m.load_state_dict(torch.load(parent/'frozen_checkpoint.pt',map_location=device,weights_only=True))
    return m.to(dtype=dtype),freeze


def parent_coefficients(torch,model,v,ca):
    n=model.a.shape[0]
    inp=torch.cat(((v[...,None]+40)/80,torch.zeros(n,v.shape[1],17,device=v.device,dtype=v.dtype)),-1)
    raw=torch.bmm(torch.nn.functional.silu(torch.bmm(inp,model.a)+model.b),model.c*model.head_mask)+model.d[:,None]
    sk_in=torch.stack((torch.zeros_like(v),(ca.clamp_min(1e-9).log10()+4.5)/2.5),-1)
    sk=torch.bmm(torch.nn.functional.silu(torch.bmm(sk_in,model.sa)+model.sb),model.sc)+model.sd[:,None]
    uu=torch.sigmoid(raw[...,:17]);zz=model.fraction(raw[...,17:])
    return (torch.cat((uu[...,:16],torch.sigmoid(sk[...,0:1]),uu[...,16:]),-1),
            torch.cat((zz[...,:16],model.fraction(sk[...,1:2]),zz[...,16:]),-1))


def oracle_audit(torch,root,output,spec,quality,device):
    """Frozen same-weight FP32/64 and coefficient swaps; no model selection."""
    modes=['learned','teacher_u','teacher_z','teacher_both','both_calcium','both_potassium','both_sodium','both_Ih']
    channels={name:k for k,name in enumerate(p.ionic.CHANNELS)}
    groups_={'both_calcium':['Ca_HVA','Ca_LVAst','SK_E2'],
             'both_potassium':['Im','K_Pst','K_Tst','SKv3_1'],
             'both_sodium':['NaTa_t','NaTs2_t','Nap_Et2'],'both_Ih':['Ih']}
    rows=[];floor=[]
    for dtype,name in ((torch.float32,'float32'),(torch.float64,'float64')):
        model,freeze=frozen39b(torch,root,spec,device,dtype)
        indices=[next(i for i,d in enumerate(freeze['arms']) if d['name']==a)*3+j
                 for a in spec['oracle_parent_arms'] for j in range(3)]
        for family in quality['families']:
            with np.load(root/spec['dataset']/f'development__{family}.npz') as z:a={k:z[k] for k in z.files}
            shape=a['voltage'].shape;learned_u=[];learned_z=[]
            with torch.no_grad():
                for start in range(0,shape[0],1024):
                    v=torch.as_tensor(a['voltage'][start:start+1024].reshape(-1),device=device,dtype=dtype)[None].expand(36,-1)
                    ca=torch.as_tensor(a['bridge_ca'][start:start+1024].reshape(-1),device=device,dtype=dtype)[None].expand(36,-1)
                    u,z=parent_coefficients(torch,model,v,ca)
                    learned_u.append(u[indices].reshape(6,-1,shape[1],18).cpu().numpy())
                    learned_z.append(z[indices].reshape(6,-1,shape[1],18).cpu().numpy())
            lu,lz=np.concatenate(learned_u,1),np.concatenate(learned_z,1)
            eu,ez=p.coefficients(a['voltage'].astype('float64'),a['bridge_ca'].astype('float64'),.1)
            npdtype=np.float32 if name=='float32' else np.float64;eu=eu.astype(npdtype);ez=ez.astype(npdtype)
            for mode in modes:
                u,z=lu,lz
                if mode=='teacher_u':u=np.broadcast_to(eu,lu.shape)
                elif mode=='teacher_z':z=np.broadcast_to(ez,lz.shape)
                elif mode=='teacher_both':u,z=np.broadcast_to(eu,lu.shape),np.broadcast_to(ez,lz.shape)
                elif mode in groups_:
                    ids=[g for channel in groups_[mode] for g in range(p.OFFSETS[channels[channel]],p.OFFSETS[channels[channel]+1])]
                    u,z=lu.copy(),lz.copy();u[...,ids]=eu[...,ids];z[...,ids]=ez[...,ids]
                state=np.broadcast_to(a['initial_gates'],(6,*a['initial_gates'].shape)).astype(npdtype).copy()
                prediction=np.empty((shape[0],6,shape[1],18),dtype=npdtype)
                for tick in range(shape[0]):
                    state=state+z[:,tick]*(u[:,tick]-state);prediction[tick]=state
                for j in range(6):
                    _,current=p.readout(a['voltage'],prediction[:,j],a['multipliers'],a['reversals'])
                    for case in range(shape[1]):
                        metrics=p.evaluate_currents(a['current_targets'][:,case],current[:,case],.1,quality['quality'],
                                                   a['gate_targets'][:,case],prediction[:,j,case],a['multipliers'][case]==0)
                        rows.append(dict(parent_arm=spec['oracle_parent_arms'][j//3],seed=spec['seeds'][j%3],
                            family=family,case=case,precision=name,intervention=mode,metrics=metrics,selection_eligible=False))
                        if mode=='teacher_both' and name=='float64':
                            ge=float(np.abs(prediction[:,j,case]-a['gate_targets'][:,case]).max())
                            ce=float(np.abs(current[:,case]-a['current_targets'][:,case]).max())
                            floor.append(dict(family=family,case=case,gate_max=ge,current_max=ce,passed=metrics['passed']))
                        if mode=='learned' and name=='float32' and j<3:
                            # Preserve full traces for every isolated event-timing failure.
                            cs=metrics['channels']
                            amplitude_ok=all(c['nrmse']<=.01 and c['nmax']<=.05 and
                                c['signed_charge_normalized']<=.02 and c['absolute_charge_normalized']<=.02 for c in cs)
                            if amplitude_ok and metrics['gate']['passed'] and not metrics['passed']:
                                np.savez_compressed(output/f'timing__{family}__{case}__{spec["seeds"][j]}.npz',
                                    time_ms=np.arange(shape[0])*.1,reference=a['current_targets'][:,case],prediction=current[:,case])
                p.save_json(output/'frozen_oracle_attribution.json',rows)
            print(f'[GIADA Task39c] frozen attribution {name}/{family}',flush=True)
        del model
    valid=all(r['passed'] and r['gate_max']<=spec['floor_gate_atol'] and r['current_max']<=spec['floor_current_atol'] for r in floor)
    report=dict(valid=valid,teacher_both_float64_floor=floor,rows=len(rows),selection_eligible=False,
                limits='Same learned weights cast to two precisions; offline coefficient composition, not whole GPU runtime precision equivalence.')
    p.save_json(output/'oracle_preflight.json',report)
    if not valid:raise RuntimeError('Both coefficient swaps failed registered floor/quality; stop before training')
    return report


def selection(torch,cohorts,dev,held,normal,spec,device):
    result=[];parts=[]
    gs,cs=np.asarray(normal['gate']),np.asarray(normal['current'])
    for group in cohorts:
        m=group['model'];n=m.a.shape[0];scores=[]
        # Every family and both short/full variable sequences contribute equally to max selection.
        scanner=old.Scanner(torch,[m],spec['development_windows_per_family'],spec['recurrence_chunk'],device)
        for block in dev:
            gates=scanner.run(block[...,0].T,block[...,1].T,block[:,0,2:20])[0].transpose(1,2,0,3)
            for ticks in (8,block.shape[1]):
                ref=block[None,:,:ticks];pred=gates[:,:,:ticks]
                _,current=p.readout(ref[...,0],pred,ref[...,38:49],ref[...,49:60])
                scores.append(np.mean(((pred-ref[...,20:38])/gs)**2,axis=(1,2,3))+
                              np.mean(((current-ref[...,60:71])/cs)**2,axis=(1,2,3)))
        with torch.no_grad():
            a=torch.tensor(held,device=device)[None].expand(n,-1,-1)
            u,z,ld=m.coefficients(a[...,0],a[...,1])
            for ticks in spec['held_ticks']:
                t=torch.full((n,len(held),1),float(ticks),device=device)
                pred=compose(torch,u,ld,a[...,2:20],t);target=held_targets(torch,a,t)
                current=b.captured_readout(torch,a[...,0],pred,a[...,38:49],a[...,49:60],m.gbar)
                reference=b.captured_readout(torch,a[...,0],target,a[...,38:49],a[...,49:60],m.gbar)
                scores.append((((pred-target)/torch.tensor(gs,device=device)).square().mean((1,2))+
                               ((current-reference)/torch.tensor(cs,device=device)).square().mean((1,2))).cpu().numpy())
        array=np.asarray(scores);result.extend(array.max(0).tolist());parts.append(array.tolist())
    return np.asarray(result),parts


def calibrate(torch,group,batch,anchor,ticks,scales):
    model=group['model'];values=terms(torch,model,batch,anchor,ticks,*scales)
    norms=[]
    for i in range(4):
        grad=torch.autograd.grad(values[:,i].sum(),tuple(model.parameters()),retain_graph=i<3)
        norms.append(torch.stack([x.flatten(1).square().sum(1) for x in grad]).sum(0).sqrt().detach())
    # Shared scalar calibration from base members/fit only; no treatment-specific tuning.
    base=(norms[0]+norms[1])/2
    weights=torch.stack([(base/norms[i].clamp_min(1e-12)).median().clamp(.001,1000) for i in (2,3)])
    return weights,dict(source='initial base cohort, fit only',per_term_per_member_norms=[x.cpu().tolist() for x in norms],
                        weights=weights.cpu().tolist(),clamp=[.001,1000],frozen_before_training=True)


def training_graph(torch,cohorts,static,scales,weights,spec):
    originals=[{k:x.detach().clone() for k,x in g['model'].named_parameters()} for g in cohorts]
    optimizers=[torch.optim.Adam(g['model'].parameters(),lr=spec['learning_rate'],capturable=True) for g in cohorts]
    clipping=[torch.zeros(g['model'].a.shape[0],device=static[0]['batch'].device) for g in cohorts]
    def update():
        for g,s,opt,counter in zip(cohorts,static,optimizers,clipping):
            opt.zero_grad(set_to_none=False)
            objective(torch,g['model'],s['batch'],s['anchor'],s['ticks'],*scales,g['information'],weights).sum().backward()
            norm=torch.stack([x.grad.flatten(1).square().sum(1) for x in g['model'].parameters()]).sum(0).sqrt()
            counter.add_((norm>1).to(counter.dtype))
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
            for g,initial,opt in zip(cohorts,originals,optimizers):
                for name,x in g['model'].named_parameters():x.copy_(initial[name]);x.grad.zero_()
                for state in opt.state.values():
                    for x in state.values():
                        if torch.is_tensor(x):x.zero_()
            for counter in clipping:counter.zero_()
    reset();update();update()
    expected=[{k:x.detach().clone() for k,x in g['model'].named_parameters()} for g in cohorts]
    reset();graph.replay();graph.replay();torch.cuda.synchronize()
    error=max(float((x-e[name]).abs().max()) for g,e in zip(cohorts,expected) for name,x in g['model'].named_parameters())
    assert error<=spec['numeric_atol'],error
    reset();start=time.perf_counter()
    for _ in range(20):graph.replay()
    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/20;reset()
    return graph,clipping,dict(valid=True,two_update_max_error=error,seconds_per_update=seconds,
        projected_training_seconds=seconds*spec['checkpoints'][-1],state_reset=True,
        backend='single CUDA graph, three member-vectorized architecture cohorts, independent Adam/clipping')


def cost_benchmark(torch,cohorts,seeds,device,output):
    """Matched resident ionic work, including every current on every tick."""
    from .gate_d_compute_audit import _rates
    n,ticks,repetitions=642,80,30
    rng=np.random.default_rng(99591)
    v=torch.tensor(rng.uniform(-95,45,(ticks,n)),device=device,dtype=torch.float32)
    ca=torch.tensor(10**rng.uniform(-7,-2,(ticks,n)),device=device,dtype=torch.float32)
    initial=torch.full((1,n,18),.4,device=device)
    mult=torch.ones(1,n,11,device=device)
    rev=torch.tensor(p.ionic.REVERSALS,device=device,dtype=torch.float32) if hasattr(p.ionic,'REVERSALS') else None
    # Dataset's canonical reversals are supplied by the same teacher module.
    if rev is None:
        from .gate_d_compute_audit import REVERSAL_VALUES
        rev=torch.tensor(REVERSAL_VALUES,device=device,dtype=torch.float32)
    gbar=torch.tensor(p.ionic.GBAR,device=device,dtype=torch.float32)
    def exact(vv,cc,x):
        parts=[]
        for k,name in enumerate(p.ionic.CHANNELS):
            u,tau=_rates(name,vv,cc,torch);a,c=p.OFFSETS[k:k+2]
            if c-a==1:u,tau=u[...,None],tau[...,None]
            z=-torch.expm1(-.1/tau);parts.append(x[...,a:c]+z*(u-x[...,a:c]))
        return torch.cat(parts,-1)
    def capture(fn):
        state=initial.clone();out=torch.empty(ticks,1,n,11,device=device)
        def work():
            x=initial
            for tick in range(ticks):
                x=fn(v[tick][None],ca[tick][None],x)
                out[tick].copy_(b.captured_readout(torch,v[tick][None],x,mult,rev,gbar))
            state.copy_(x)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):work()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(graph):work()
        return graph,state,out
    def elapsed(graph):
        a,c=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();c.record();c.synchronize();return float(a.elapsed_time(c))
    baseline,_,_=capture(exact);rows=[]
    models=split_models(torch,cohorts,seeds,device)
    definitions=[d for g in cohorts for d in g['definitions']]
    # Every arm/seed is frozen before timing; architecture timing differences are descriptive.
    for definition,model in zip(definitions,models):
        for j,seed in enumerate(seeds):
            iso=model_class(torch)(model.architecture,[seed]).to(device)
            iso.load_state_dict({name:(x.clone() if name=='gbar' else x[j:j+1].clone()) for name,x in model.state_dict().items()})
            graph,state,cur=capture(iso);pairs=[]
            for k in range(repetitions):
                if k%2:a=elapsed(baseline);c=elapsed(graph)
                else:c=elapsed(graph);a=elapsed(baseline)
                pairs.append([a,c])
            array=np.asarray(pairs);saving=1-array[:,1]/array[:,0]
            boot=rng.choice(saving,(2000,repetitions),replace=True).mean(1)
            rows.append(dict(arm=definition['name'],seed=seed,paired_formula_candidate_ms=pairs,
                median_saving_fraction=float(np.median(saving)),paired_mean_saving_ci95=np.quantile(boot,[.025,.975]).tolist(),
                finite=bool(torch.isfinite(state).all() and torch.isfinite(cur).all())))
            del graph,iso
        print(f'[GIADA Task39c] resident cost {definition["name"]}',flush=True)
    report=dict(valid=all(r['finite'] for r in rows),rows=rows,batch=n,ticks=ticks,repetitions=repetitions,
        backend='CUDA Graph for both formula and network; exact current readout, same80ticks and external V/Ca',
        training_throughput_used=False,operational_transfer_cost_measured=False,
        limits='Only resident ionic core; timing alone never authorizes promotion; no optimized LUT comparison in39c.')
    p.save_json(output/'resident_cost_benchmark.json',report)
    return report


def gradient_probe(torch,cohorts,static,scales,weights,step):
    rows=[]
    for group,s in zip(cohorts,static):
        m=group['model'];values=terms(torch,m,s['batch'],s['anchor'],s['ticks'],*scales)
        norms=[]
        for j in range(4):
            grad=torch.autograd.grad(values[:,j].sum(),tuple(m.parameters()),retain_graph=j<3)
            norms.append(torch.stack([x.flatten(1).square().sum(1) for x in grad]).sum(0).sqrt().detach().cpu().tolist())
        weighted=objective(torch,m,s['batch'],s['anchor'],s['ticks'],*scales,group['information'],weights)
        grad=torch.autograd.grad(weighted.sum(),tuple(m.parameters()))
        unclipped=torch.stack([x.flatten(1).square().sum(1) for x in grad]).sum(0).sqrt()
        rows.append(dict(architecture=m.architecture,step=step,losses=values.detach().cpu().tolist(),
                         gradient_term_norms=norms,gradient_norm=unclipped.detach().cpu().tolist(),
                         clipping_count=int((unclipped>1).sum()),members=len(unclipped)))
    return rows


def contrast_pairs(definitions):
    pairs=[]
    for axis in ('information','support','duration'):
        for a in definitions[:8]:
            if a[axis]:continue
            c=next(c for c in definitions[:8] if c[axis]==1 and all(c[k]==a[k] for k in ('information','support','duration') if k!=axis))
            pairs.append((axis,a['name'],c['name']))
    for c in definitions[8:]:
        a=next(a for a in definitions[:8] if all(a[k]==c[k] for k in ('information','support','duration')))
        pairs.append(('architecture',a['name'],c['name']))
    return pairs


def run(root,output,device='cuda'):
    import torch
    assert device=='cuda' and torch.cuda.is_available(),'Task39c requires CUDA'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());quality=json.loads((root/p.CONFIG).read_text())
    output.mkdir(parents=True,exist_ok=True)
    if (output/'final_report.json').exists():raise FileExistsError(output)
    p.save_json(output/'preregistration.json',spec)
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()):
        assert p.digest(root/spec['dataset']/row['file'])==row['sha256']
    print('[GIADA Task39c] verified data; frozen coefficient/precision attribution begins',flush=True)
    oracle_audit(torch,root,output,spec,quality,device)
    defs=arms();seeds=spec['seeds'];families=list(quality['families'])
    fit,support=q.windows(root/spec['dataset'],'fit',families,spec['windows_per_family'],spec['sequence_length'],spec['data_seed'])
    extra,info=q.windows(root/spec['dataset'],'fit_extra',families,spec['windows_per_family'],spec['sequence_length'],spec['data_seed']+1009);support+=info
    anchors=anchor_pool(np.concatenate((fit,extra)),spec['data_seed']+2)
    targeted=anchor_pool(np.concatenate((fit,extra)),spec['data_seed']+2,True)
    # Development nodes and long windows are prospectively specified, never sampled from stress.
    dev=[]
    for k,family in enumerate(families):
        data,info=q.windows(root/spec['dataset'],'development',[family],spec['development_windows_per_family'],512,spec['data_seed']+4000+k)
        dev.append(data);support+=info
    for family in ('same_v_history','slow_recovery'):
        with np.load(root/spec['dataset']/f'development__{family}.npz') as z:
            a={k:z[k] for k in z.files}
        length=spec['development_long_ticks'];idx=np.arange(len(a['voltage'])-length,len(a['voltage']))
        pre=a['gate_targets'][np.maximum(idx-1,0)]
        data=np.concatenate((a['voltage'][idx,...,None],a['bridge_ca'][idx,...,None],pre,a['gate_targets'][idx],
             np.broadcast_to(a['multipliers'],(length,2,11)),np.broadcast_to(a['reversals'],(length,2,11)),a['current_targets'][idx]),-1)
        dev.append(data.transpose(1,0,2).astype('float32'))
    held=anchor_pool(dev[0],spec['data_seed']+5000,True)
    # More than two held nodes: use the same declared development voltage grid for every seed.
    held=anchor_pool(np.tile(dev[0],(spec['development_anchor_nodes']//len(dev[0]),1,1)),spec['data_seed']+5000,True)
    p.save_json(output/'window_support.json',dict(windows=support,anchor_pool_nodes=len(anchors),
        targeted_nodes=len(targeted),targeted_voltage_range=[float(targeted[:,0].min()),float(targeted[:,0].max())],
        targeted_ca_range=[float(targeted[:,1].min()),float(targeted[:,1].max())],
        targeted_extreme_fraction=float(np.mean((targeted[:,2:20]==0)|(targeted[:,2:20]==1))),
        held_development_nodes=len(held),long_development_ticks=spec['development_long_ticks']))
    normal=json.loads((root/spec['normalizers']).read_text())
    coefficient=np.concatenate((anchors,targeted));us=np.std(coefficient[:,71:89],0).clip(.05)
    zs=np.std(np.log(coefficient[:,89:107].clip(1e-30)),0).clip(.5)
    scales=[torch.tensor(a,device=device,dtype=torch.float32) for a in (normal['gate'],normal['current'],us,zs)]
    pools=[torch.tensor(a,device=device) for a in (fit,extra,anchors,targeted)]
    cohorts=groups(torch,defs,seeds,device);static=[]
    for g in cohorts:
        n=g['model'].a.shape[0]
        static.append(dict(batch=pools[0][:spec['batch_sequences']][None].expand(n,-1,-1,-1).clone(),
            anchor=pools[2][:spec['anchor_batch']][None].expand(n,-1,-1).clone(),
            ticks=torch.ones(n,spec['anchor_batch'],1,device=device)))
    weights,calibration=calibrate(torch,cohorts[0],static[0]['batch'],static[0]['anchor'],static[0]['ticks'],scales)
    p.save_json(output/'fit_loss_calibration.json',dict(**calibration,u_scales=us.tolist(),log_z_scales=zs.tolist(),normalizers=normal))
    preflights=[]
    p.save_json(output/'composition_preflight.json',composition_preflight(torch,cohorts,device))
    for g,s in zip(cohorts,static):
        # Include both long anchors and variable inputs in isolated optimizer parity.
        s['ticks'].fill_(10000)
        preflights.append(optimizer_preflight(torch,g,s['batch'][:,:,:8],s['anchor'],s['ticks'],scales,weights,seeds))
        s['ticks'].fill_(1)
    old.recurrence_preflight(output,spec,split_models(torch,cohorts,seeds,device),torch,device)
    p.save_json(output/'optimizer_preflight.json',preflights)
    graph,clipping,execution=training_graph(torch,cohorts,static,scales,weights,spec)
    execution.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,members=36,
        parameters_per_member={g['model'].architecture:sum(x.numel() for x in g['model'].parameters())//g['model'].a.shape[0] for g in cohorts},
        thresholds_changed=False,budget_locked_before_quality=True)
    p.save_json(output/'execution.json',execution)
    print(f'[GIADA Task39c] all preflights passed; projected training {execution["projected_training_seconds"]/60:.1f} min',flush=True)
    rng=np.random.default_rng(spec['data_seed']+99);curves=[];probes=[];best=np.full(36,np.inf);chosen=[None]*36;saved=[None]*36
    exposure=np.zeros((len(families),2),dtype=np.int64);duration_counts=np.zeros(6,dtype=np.int64);start=time.perf_counter()
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            scores,parts=selection(torch,cohorts,dev,held,normal,spec,device);assert np.isfinite(scores).all()
            j=0
            for g in cohorts:
                for k in range(g['model'].a.shape[0]):
                    if scores[j]<best[j]:
                        best[j]=scores[j];chosen[j]=step
                        saved[j]={name:x[k].detach().cpu().clone() for name,x in g['model'].named_parameters()}
                    j+=1
            curves.append(dict(step=step,common_development_score=scores.tolist(),component_scores=parts,
                clipping_counts=[x.detach().cpu().tolist() for x in clipping],elapsed_seconds=time.perf_counter()-start))
            p.save_json(output/'training_curves.json',curves)
            torch.save({g['model'].architecture:g['model'].state_dict() for g in cohorts},output/f'checkpoint_{step}.pt')
            probes+=gradient_probe(torch,cohorts,static,scales,weights,step)
            p.save_json(output/'gradient_probes.json',probes)
            print(f'[GIADA Task39c] checkpoint {step}/{spec["checkpoints"][-1]} median={np.median(scores):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
        if step==spec['checkpoints'][-1]:break
        idx=rng.integers(len(fit),size=spec['batch_sequences']);ex=rng.random(len(idx))<.5
        ai=rng.integers(len(anchors),size=spec['anchor_batch']);di=rng.integers(6,size=len(ai))
        ti=torch.tensor(idx,device=device);ee=torch.tensor(ex,device=device)[:,None,None]
        variable=torch.where(ee,pools[1][ti],pools[0][ti]);an=pools[2][ai];tar=pools[3][ai]
        ticks=torch.tensor(np.asarray(spec['held_ticks'])[di],device=device,dtype=torch.float32)[None,:,None]
        for g,s in zip(cohorts,static):
            s['batch'].copy_(variable[None]);s['anchor'].copy_(torch.where(g['support'],tar[None],an[None]))
            s['ticks'].copy_(torch.where(g['duration'],ticks,torch.ones_like(ticks)))
        graph.replay()
        np.add.at(exposure,(idx//spec['windows_per_family'],ex.astype(int)),spec['sequence_length'])
        np.add.at(duration_counts,di,1)
        if (step+1)%2000==0 and step+1 not in spec['checkpoints']:
            print(f'[GIADA Task39c] update {step+1}/{spec["checkpoints"][-1]} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    training_seconds=time.perf_counter()-start;j=0
    with torch.no_grad():
        for g in cohorts:
            for k in range(g['model'].a.shape[0]):
                for name,x in g['model'].named_parameters():x[k].copy_(saved[j][name])
                j+=1
    torch.save({g['model'].architecture:g['model'].state_dict() for g in cohorts},output/'frozen_checkpoint.pt')
    freeze=dict(arms=defs,seeds=seeds,chosen_steps=chosen,scores=best.tolist(),development_only=True,
                independent_confirmation_accessed=False,stress_used_for_selection=False,checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'))
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    p.save_json(output/'exposure.json',dict(variable_ticks_family_role=exposure.tolist(),long_anchor_duration_counts=duration_counts.tolist(),
        variable_targets_per_member=spec['checkpoints'][-1]*spec['sequence_length']*spec['batch_sequences'],
        anchor_targets_per_member=spec['checkpoints'][-1]*spec['anchor_batch'],different_duration_not_different_inference_dt=True))
    base=json.loads((root/old.CONFIG).read_text());evaluation=dict(spec,architectures=[d['name'] for d in defs])
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=base[key]
    assert p.digest(root/base['baseline_parent'])==base['baseline_parent_sha256']
    assert p.digest(root/base['baseline_checkpoint'])==base['baseline_sha256']
    summary=old.evaluate(root,output,evaluation,split_models(torch,cohorts,seeds,device),torch,device,quality)
    rows=json.loads((output/'recurrence_evaluation.json').read_text())
    scores={d['name']:{str(seed):dict(median=float(np.median([q.severity(r['metrics']) for r in rows if r['model']=='joint_w'+d['name'] and r['role']=='development' and r['seed']==seed])),
        worst=max(q.severity(r['metrics']) for r in rows if r['model']=='joint_w'+d['name'] and r['role']=='development' and r['seed']==seed)) for seed in seeds} for d in defs}
    contrasts=[]
    for axis,a,c in contrast_pairs(defs):
        ratios=[scores[c][str(s)]['median']/max(scores[a][str(s)]['median'],1e-12) for s in seeds]
        worst=[scores[c][str(s)]['worst']/max(scores[a][str(s)]['worst'],1e-12) for s in seeds]
        contrasts.append(dict(axis=axis,control=a,treatment=c,paired_ratios=ratios,worst_ratios=worst,
            supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=contrasts))
    budget=[]
    for low,high in ((30000,60000),(60000,120000)):
        before=np.asarray(next(c['common_development_score'] for c in curves if c['step']==low))
        after=np.asarray(next(c['common_development_score'] for c in curves if c['step']==high))
        for k,d in enumerate(defs):
            ratios=(after[k*3:(k+1)*3]/before[k*3:(k+1)*3].clip(1e-12)).tolist()
            budget.append(dict(arm=d['name'],from_step=low,to_step=high,paired_ratios=ratios,
                               improvement_supported=bool(np.median(ratios)<=.8 and max(ratios)<1)))
    p.save_json(output/'budget_contrasts.json',budget)
    probes=[]
    for g in cohorts:
        m=g['model'];n=m.a.shape[0]
        a=torch.tensor(held,device=device)[None].expand(n,-1,-1)
        with torch.no_grad():
            u,z,_=m.coefficients(a[...,0],a[...,1]);_,rawz=m.raw(a[...,0],a[...,1])
            ue=(u-a[...,71:89]).square().mean(1).sqrt()
            ze=(torch.nn.functional.logsigmoid(rawz)-a[...,89:107].clamp_min(1e-30).log()).square().mean(1).sqrt()
        for j in range(n):
            probes.append(dict(arm=g['definitions'][j//3]['name'],seed=seeds[j%3],
                u_rmse_by_gate=ue[j].cpu().tolist(),log_z_rmse_by_gate=ze[j].cpu().tolist(),
                source='common development held nodes after freeze; not independent confirmation'))
    p.save_json(output/'internal_probes.json',probes)
    cost_benchmark(torch,cohorts,seeds,device,output)
    passing=[d['name'] for d in defs if summary['joint_w'+d['name']]['all_development_passed']]
    execution.update(training_wall_seconds=training_seconds,total_wall_seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    p.save_json(output/'execution.json',execution)
    final=dict(valid=True,task='39c',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',
        passing_arms=passing,summaries=summary,thresholds_changed=False,independent_confirmation_accessed=False,
        speedup_claimed=False,limits=spec['limits'])
    p.save_json(output/'final_report.json',final);return final
