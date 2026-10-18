"""Task39b: paired training interventions and architecture anchors, no new physics."""
import hashlib
import itertools
import json
import time
import numpy as np
from . import task39_learnability as q

old, p = q.old, q.p
CONFIG = 'experiments/task39b_preregistration_v1.json'


def arms():
    result = [dict(name=f'base_h{h}_c{c}_r{r}', architecture='base', horizon=h, collocation=c, relative=r)
              for h,c,r in itertools.product((8,64),(0,1),(0,1))]
    for architecture in ('grouped','multiscale'):
        for h,c,r in ((8,0,0),(64,1,1)):
            result.append(dict(name=f'{architecture}_h{h}_c{c}_r{r}',architecture=architecture,horizon=h,collocation=c,relative=r))
    return result


def multiscale_shift():
    lo,hi=0.,12.; target=1/(1+np.exp(3))
    for _ in range(80):
        mid=(lo+hi)/2
        if np.mean(1/(1+np.exp(-(-3+mid+np.array([0,-3,-6,-9])))))<target:lo=mid
        else:hi=mid
    return (lo+hi)/2


def model_class(torch):
    class Model(old.model_class(torch)):
        def __init__(self,definitions,seeds,rapid):
            super().__init__(32,seeds*len(definitions))
            self.definitions=definitions; self.seeds=seeds; self.rapid=list(rapid)
            mask=torch.ones(len(definitions)*len(seeds),32,34)
            for k,d in enumerate(definitions):
                if d['architecture']=='grouped':
                    for gate in range(17):
                        inactive=slice(24,32) if gate in rapid else slice(16,24)
                        mask[k*len(seeds):(k+1)*len(seeds),inactive,gate]=0
                        mask[k*len(seeds):(k+1)*len(seeds),inactive,gate+17]=0
            self.register_buffer('head_mask',mask)
            self.register_buffer('multi',torch.tensor([d['architecture']=='multiscale' for d in definitions for _ in seeds])[:,None,None])
            self.register_buffer('offsets',torch.tensor([0.,-3.,-6.,-9.])+multiscale_shift())

        def fraction(self,raw):
            return torch.where(self.multi,torch.sigmoid(raw[...,None]+self.offsets).mean(-1),torch.sigmoid(raw))

        def forward(self,v,ca,state):
            previous=torch.cat((state[...,:16],state[...,17:]),-1)
            inp=torch.cat(((v[...,None]+40)/80,torch.zeros_like(previous)),-1)
            raw=torch.bmm(torch.nn.functional.silu(torch.bmm(inp,self.a)+self.b),self.c*self.head_mask)+self.d[:,None]
            candidate=torch.sigmoid(raw[...,:17]); fraction=self.fraction(raw[...,17:])
            new=previous+fraction*(candidate-previous)
            sk_in=torch.stack((torch.zeros_like(v),(ca.clamp_min(1e-9).log10()+4.5)/2.5),-1)
            sk=torch.bmm(torch.nn.functional.silu(torch.bmm(sk_in,self.sa)+self.sb),self.sc)+self.sd[:,None]
            z=state[...,16]+self.fraction(sk[...,1:2])[...,0]*(torch.sigmoid(sk[...,0])-state[...,16])
            return torch.cat((new[...,:16],z[...,None],new[...,16:]),-1)
    return Model


def collocation(data,seed):
    """Offline oracle only; pre-state/target columns remain causally aligned."""
    rng=np.random.default_rng(seed); result=data.copy()
    state=rng.uniform(0,1,(len(data),18))
    inf,z=p.coefficients(data[...,0],data[...,1],.1)
    for tick in range(data.shape[1]):
        result[:,tick,2:20]=state
        state=state+z[:,tick]*(inf[:,tick]-state)
        result[:,tick,20:38]=state
    _,current=p.readout(data[...,0],result[...,20:38],data[...,38:49],data[...,49:60])
    result[...,60:71]=current
    return result


def flags(torch,definitions,seeds,device):
    ds=[d for d in definitions for _ in seeds]
    short=torch.tensor([d['horizon']==8 for d in ds],device=device)[:,None,None]
    colloc=torch.tensor([d['collocation'] for d in ds],device=device,dtype=torch.bool)[:,None,None,None]
    relative=torch.tensor([d['relative'] for d in ds],device=device,dtype=torch.float32)
    return short,colloc,relative


def objective(torch,model,batch,gs,cs,short,relative):
    state=batch[:,:,0,2:20]; total=torch.zeros(batch.shape[0],device=batch.device)
    scale=batch[...,60:71].abs().amax(2).clamp_min(1e-7)
    for tick in range(batch.shape[2]):
        b=batch[:,:,tick]
        if tick and tick%8==0:state=torch.where(short,b[...,2:20],state)
        state=model(b[...,0],b[...,1],state)
        current=old.readout(torch,b[...,0],state,b[...,38:49],b[...,49:60])
        total=total+((state-b[...,20:38])/gs).square().mean((1,2))+((current-b[...,60:71])/cs).square().mean((1,2))
        total=total+relative*((current-b[...,60:71])/scale).square().mean((1,2))
    return total/batch.shape[2]


def split_models(torch,model,definitions,seeds,device):
    result=[]
    for k,d in enumerate(definitions):
        m=model_class(torch)([d],seeds,model.rapid).to(device)
        state={n:(x.clone() if n=='offsets' else x[k*len(seeds):(k+1)*len(seeds)].clone()) for n,x in model.state_dict().items()}
        m.load_state_dict(state); result.append(m)
    return result


def optimizer_preflight(torch,model,batch,gs,cs,definitions,seeds):
    device=batch.device; short,_,relative=flags(torch,definitions,seeds,device)
    init={n:x.detach().clone() for n,x in model.named_parameters()}
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    objective(torch,model,batch,gs,cs,short,relative).sum().backward();old.clip(torch,model);opt.step()
    maximum=0.
    for k,d in enumerate(definitions):
        for j,seed in enumerate(seeds):
            i=k*len(seeds)+j; iso=model_class(torch)([d],[seed],model.rapid).to(device)
            with torch.no_grad():
                for n,x in iso.named_parameters():x.copy_(init[n][i:i+1])
            sh,_,rel=flags(torch,[d],[seed],device);o=torch.optim.Adam(iso.parameters(),lr=.001)
            objective(torch,iso,batch[i:i+1],gs,cs,sh,rel).sum().backward();old.clip(torch,iso);o.step()
            maximum=max(maximum,max(float((x[0]-dict(model.named_parameters())[n][i]).abs().max()) for n,x in iso.named_parameters()))
    with torch.no_grad():
        for n,x in model.named_parameters():x.copy_(init[n])
    for x in model.parameters():x.grad=None
    assert maximum<=1e-5,maximum
    return dict(valid=True,max_abs_error=maximum,members_tested=len(definitions)*len(seeds),independent_clipping=True)


def captured_training(torch,model,batch,gs,cs,short,relative,lr,atol):
    """One native CUDA graph across all36 members, including independent Adam."""
    initial={n:x.detach().clone() for n,x in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,capturable=True)
    def update():
        optimizer.zero_grad(set_to_none=False)
        objective(torch,model,batch,gs,cs,short,relative).sum().backward()
        old.clip(torch,model);optimizer.step()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):update()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):update()
    def reset():
        with torch.no_grad():
            for n,x in model.named_parameters():x.copy_(initial[n])
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):value.zero_()
            for x in model.parameters():x.grad.zero_()
    # Compare two captured updates with the same optimizer in ordinary execution.
    reset();update();update();expected={n:x.detach().clone() for n,x in model.named_parameters()}
    reset();graph.replay();graph.replay();torch.cuda.synchronize()
    error=max(float((x-expected[n]).abs().max()) for n,x in model.named_parameters())
    assert error<=atol,error
    reset();torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(20):graph.replay()
    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/20;reset()
    return graph,dict(valid=True,backend='native CUDA graph forward/backward/Adam',two_step_error=error,
        benchmark_seconds_per_update=seconds,projected_60k_training_seconds=seconds*60000,state_reset_after_benchmark=True)


def selection(torch,model,dev,gs,cs,spec,device):
    count=spec['development_windows_per_family']; length=spec['development_length']
    scanner=old.Scanner(torch,[model],count,spec['recurrence_chunk'],device)
    scores=[]
    for start in range(0,len(dev),count):
        b=dev[start:start+count]; pred=scanner.run(b[...,0].T,b[...,1].T,b[:,0,2:20])[0]
        # T,M,N,G -> M,N,T,G
        pred=pred.transpose(1,2,0,3)
        for ticks in (8,length):
            state=pred[:,:,:ticks]; ref=b[None,:,:ticks]
            _,current=p.readout(ref[...,0],state,ref[...,38:49],ref[...,49:60])
            scores.append(np.mean(((state-ref[...,20:38])/gs)**2,axis=(1,2,3))+
                np.mean(((current-ref[...,60:71])/cs)**2,axis=(1,2,3)))
    return np.max(scores,axis=0)


def planned_contrasts(rows,definitions):
    scores={}
    for d in definitions:
        rs=[r for r in rows if r['model']=='joint_w'+d['name'] and r['role']=='development']
        scores[d['name']]={str(s):dict(median=float(np.median([q.severity(r['metrics']) for r in rs if r['seed']==s])),
            worst=max(q.severity(r['metrics']) for r in rs if r['seed']==s)) for s in (17,29,43)}
    pairs=[]
    for axis,low,high in [('horizon',8,64),('collocation',0,1),('relative',0,1)]:
        for a in definitions[:8]:
            if a[axis]!=low:continue
            b=next(b for b in definitions[:8] if b[axis]==high and all(a[k]==b[k] for k in ('horizon','collocation','relative') if k!=axis))
            pairs.append((axis,a['name'],b['name']))
    for a in definitions[8:]:
        b=next(b for b in definitions[:8] if all(a[k]==b[k] for k in ('horizon','collocation','relative')))
        pairs.append(('architecture',b['name'],a['name']))
    result=[]
    for axis,a,b in pairs:
        ratios=[scores[b][str(s)]['median']/max(scores[a][str(s)]['median'],1e-12) for s in (17,29,43)]
        worst=[scores[b][str(s)]['worst']/max(scores[a][str(s)]['worst'],1e-12) for s in (17,29,43)]
        result.append(dict(axis=axis,control=a,treatment=b,paired_ratios=ratios,worst_ratios=worst,
            supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    return dict(scores=scores,contrasts=result)


def run(root,output,device='cuda'):
    import torch
    assert device=='cuda' and torch.cuda.is_available(),'Task39b requires CUDA'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((root/CONFIG).read_text());quality=json.loads((root/p.CONFIG).read_text())
    output.mkdir(parents=True,exist_ok=True)
    if (output/'final_report.json').exists():raise FileExistsError(output)
    p.save_json(output/'preregistration.json',spec)
    assert p.digest(root/spec['parent_checkpoint'])==spec['parent_sha256']
    for row in json.loads((root/spec['dataset']/'dataset_manifest.json').read_text()):assert p.digest(root/spec['dataset']/row['file'])==row['sha256']
    definitions=arms();seeds=spec['seeds'];n=len(definitions)*len(seeds)
    families=list(quality['families']);pools={};support=[]
    for role in ('fit','fit_extra'):
        data,info=q.windows(root/spec['dataset'],role,families,spec['windows_per_family'],64,spec['data_seed']+len(pools)*1009)
        synthetic=collocation(data,spec['data_seed']+len(pools)*1009+7)
        pools[role]=torch.tensor(data,device=device);pools[role+'_collocation']=torch.tensor(synthetic,device=device)
        support+=info
        if role=='fit':
            sample=data.reshape(len(families),spec['windows_per_family'],64,71)[:,:32].reshape(-1,64,71)
            _,fraction=p.coefficients(sample[...,0],sample[...,1],.1)
            medians=np.median(fraction.reshape(-1,18),axis=0)[list(old.IDS)]
            rapid=np.argsort(medians)[-8:].tolist()
    dev,info=q.windows(root/spec['dataset'],'development',families,spec['development_windows_per_family'],spec['development_length'],spec['data_seed']+4040)
    support+=info;p.save_json(output/'window_support.json',support)
    normal=json.loads((root/spec['normalizers']).read_text());gs,cs=[torch.tensor(normal[k],device=device) for k in ('gate','current')]
    model=model_class(torch)(definitions,seeds,rapid).to(device)
    short,colloc_flag,relative=flags(torch,definitions,seeds,device)
    static=pools['fit'][:spec['batch_sequences']][None].expand(n,-1,-1,-1).clone()
    print('[GIADA Task39b] dataset and parent hashes verified; testing36 independent members',flush=True)
    smoke=static[:,:,:16]
    preflight=optimizer_preflight(torch,model,smoke,gs,cs,definitions,seeds)
    with torch.no_grad():
        sample=pools['fit_collocation'][:128,0]
        from .gate_d_compute_audit import exact_gpu
        oracle,_=exact_gpu(sample[:,0],sample[:,1],sample[:,2:20],torch.full_like(sample[:,0],.1),torch)
        err=float((oracle-sample[:,20:38]).abs().max());assert err<=1e-5,err
        x=smoke[:,:,0,2:20];v=smoke[:,:,0,0];ca=smoke[:,:,0,1]
        a=model(v,ca,x);b=model(v,ca*10,x);xx=x.clone();xx[...,1:]=1-xx[...,1:]
        assert torch.equal(a[...,:16],b[...,:16]) and torch.equal(a[...,17:],b[...,17:])
        assert torch.equal(a[...,0],model(v,ca,xx)[...,0]);assert a.min()>=0 and a.max()<=1
        current=old.readout(torch,sample[:,0],oracle,sample[:,38:49],sample[:,49:60])
        ce=float((current-sample[:,60:71]).abs().max());assert ce<=1e-7,ce
    preflight.update(collocation_teacher_error=err,current_error=ce)
    p.save_json(output/'preflight.json',preflight)
    old.recurrence_preflight(output,spec,split_models(torch,model,definitions,seeds,device),torch,device)
    graph,execution=captured_training(torch,model,static,gs,cs,short,relative,spec['learning_rate'],spec['numeric_atol'])
    execution.update(members=n,gpu=torch.cuda.get_device_name(),torch=torch.__version__,rapid_output_positions=rapid,
        parameters_per_member=sum(x.numel() for x in model.parameters())//n,
        nominal_parameter_count_equal=True,effective_connectivity_differs=True,targets_per_update=1024,
        structurally_active_parameters=dict(base=1220,grouped=948,multiscale=1220),
        grouping_source='32 windows per family, fit only, median teacher fraction; eight faster outputs')
    p.save_json(output/'execution.json',execution)
    print('[GIADA Task39b] all preflights passed; CUDA graph training ready',flush=True)
    rng=np.random.default_rng(spec['data_seed']+99);best=np.full(n,np.inf);chosen=[None]*n;saved=[None]*n;curves=[]
    exposure=np.zeros((len(families),2,2),dtype=np.int64);start=time.perf_counter()
    for step in range(spec['checkpoints'][-1]+1):
        if step in spec['checkpoints']:
            score=selection(torch,model,dev,np.asarray(normal['gate']),np.asarray(normal['current']),spec,device)
            assert np.isfinite(score).all()
            for j,value in enumerate(score):
                if value<best[j]:
                    best[j]=float(value);chosen[j]=step
                    saved[j]={k:x[j].detach().cpu().clone() for k,x in model.named_parameters()}
            curves.append(dict(step=step,common_development_score=score.tolist(),elapsed_seconds=time.perf_counter()-start))
            p.save_json(output/'training_curves.json',curves);torch.save(model.state_dict(),output/f'checkpoint_{step}.pt')
            print(f'[GIADA Task39b] checkpoint{step}/60000 median={np.median(score):.6g} elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
        if step==spec['checkpoints'][-1]:break
        idx=rng.integers(len(pools['fit']),size=spec['batch_sequences'])
        extra=rng.random(spec['batch_sequences'])<.5;replace=rng.random(spec['batch_sequences'])<spec['collocation_fraction']
        ti=torch.tensor(idx,device=device);ex=torch.tensor(extra,device=device)[:,None,None];rr=torch.tensor(replace,device=device)[:,None,None]
        real=torch.where(ex,pools['fit_extra'][ti],pools['fit'][ti])
        synthetic=torch.where(ex,pools['fit_extra_collocation'][ti],pools['fit_collocation'][ti])
        replaced=torch.where(rr,synthetic,real)
        static.copy_(torch.where(colloc_flag,replaced[None],real[None]));graph.replay()
        np.add.at(exposure,(idx//spec['windows_per_family'],extra.astype(int),replace.astype(int)),64)
        if (step+1)%1000==0 and step+1 not in spec['checkpoints']:print(f'[GIADA Task39b] update{step+1}/60000 elapsed={(time.perf_counter()-start)/60:.1f}min',flush=True)
    training_seconds=time.perf_counter()-start
    with torch.no_grad():
        for name,value in model.named_parameters():
            for j in range(n):value[j].copy_(saved[j][name])
    torch.save(model.state_dict(),output/'frozen_checkpoint.pt')
    freeze=dict(arms=definitions,seeds=seeds,chosen_steps=chosen,scores=best.tolist(),checkpoint_sha256=p.digest(output/'frozen_checkpoint.pt'),
        development_only=True,independent_confirmation_accessed=False,stress_used_for_selection=False)
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();p.save_json(output/'selection_freeze.json',freeze)
    p.save_json(output/'exposure.json',dict(families=families,target_ticks_family_extra_collocation_slot=exposure.tolist(),
        targets_per_member=spec['checkpoints'][-1]*1024,collocation_replaces_not_adds=True))
    # Full Jacobian is diagonal by construction; record all18 slopes, equilibria and gradient diagnostics.
    probe_indices=torch.tensor(np.linspace(0,len(pools['fit'])-1,256,dtype=np.int64),device=device)
    b=pools['fit'][probe_indices,0];v=b[:,0][None].expand(n,-1);ca=b[:,1][None].expand(n,-1)
    zero=torch.zeros(n,len(b),18,device=device);one=torch.ones_like(zero)
    probe_state=torch.tensor(np.random.default_rng(spec['data_seed']+303).uniform(0,1,(len(b),18)),dtype=torch.float32,device=device)[None].expand(n,-1,-1)
    with torch.no_grad():
        intercept=model(v,ca,zero);upper=model(v,ca,one);slope=upper-intercept
        fraction=1-slope;equilibrium=intercept/fraction.clamp_min(1e-9)
        reference,_=exact_gpu(v,ca,probe_state,torch.full_like(v,.1),torch)
        random_error=(model(v,ca,probe_state)-reference).square().mean(1).sqrt()
    for x in model.parameters():x.grad=None
    objective(torch,model,static,gs,cs,short,relative).sum().backward()
    probes=dict(fraction_min=fraction.amin(1).cpu().tolist(),fraction_max=fraction.amax(1).cpu().tolist(),
        equilibrium_min=equilibrium.amin(1).cpu().tolist(),equilibrium_max=equilibrium.amax(1).cpu().tolist(),
        physical_state_gate_rmse=random_error.cpu().tolist(),output_head_gradient_norm=model.c.grad.square().sum(1).sqrt().cpu().tolist(),
        jacobian='Diagonal by architecture; diagonal slopes=1-fraction',source='fit-only post-freeze diagnostics')
    prediction=model(v,ca,probe_state);error=((prediction-reference)/gs).square()
    fast=[old.IDS[k] for k in rapid];slow=[k for k in old.IDS if k not in fast]
    gf=torch.autograd.grad(error[...,fast].mean((1,2)).sum(),model.a,retain_graph=True)[0].flatten(1)
    gl=torch.autograd.grad(error[...,slow].mean((1,2)).sum(),model.a)[0].flatten(1)
    probes['fast_slow_trunk_gradient_cosine']=((gf*gl).sum(1)/(gf.square().sum(1)*gl.square().sum(1)).sqrt().clamp_min(1e-20)).cpu().tolist()
    probes['gradient_cosine_limit']='Local off-trajectory diagnostic, not proof of training interference.'
    p.save_json(output/'internal_probes.json',probes)
    models=split_models(torch,model,definitions,seeds,device)
    oldspec=json.loads((root/q.CONFIG).read_text());parent=torch.load(root/spec['parent_checkpoint'],map_location=device,weights_only=True)
    historical=q.model_class(torch)(q.arms(),seeds).to(device);historical.load_state_dict(parent)
    index=next(i for i,d in enumerate(q.arms()) if d['name']=='factorized_expanded_fr')
    models.append(q.split_models(torch,historical,q.arms(),seeds,device)[index])
    base_spec=json.loads((root/old.CONFIG).read_text())
    eval_spec=dict(spec,architectures=[d['name'] for d in definitions]+['historical39_best'])
    for key in ('baseline_parent','baseline_checkpoint'):eval_spec[key]=base_spec[key]
    for key in ('baseline_checkpoint','baseline_parent'):
        expected=base_spec['baseline_sha256' if key=='baseline_checkpoint' else 'baseline_parent_sha256']
        assert p.digest(root/base_spec[key])==expected
    summary=old.evaluate(root,output,eval_spec,models,torch,device,quality)
    rows=json.loads((output/'recurrence_evaluation.json').read_text());contrasts=planned_contrasts(rows,definitions)
    p.save_json(output/'paired_contrasts.json',contrasts)
    passing=[d['name'] for d in definitions if summary['joint_w'+d['name']]['all_development_passed']]
    execution.update(training_wall_seconds=training_seconds,total_wall_seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    p.save_json(output/'execution.json',execution)
    report=dict(valid=True,task='39b',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',
        passing_arms=passing,summaries=summary,contrasts=contrasts,thresholds_changed=False,
        independent_confirmation_accessed=False,speedup_claimed=False,limits=spec['limits'])
    p.save_json(output/'final_report.json',report);return report
