"""45d: support/learnability factorial and separate exact-execution diagnostics.

Canonical rates are offline labels only. All inference uses the same18 states,
affine bounded recurrence, original physical readout and membrane solver.
"""
import copy,hashlib,io,json,os,time,types,zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from . import task45c_preserving_kinetics as c
n=c.n
ORIGINAL_BENCHMARK=n.benchmark
ARMS=('none','C0','C1','A00','A01','A10','A11','Ncap','Nline','T0','T1','J','R41','R111')
COLS=(3,9,10,20,26,27)

def prepare_training_readout(torch,device):
    """Normalize resident cache lifetime, not the immutable historical solver.

    Inference-warmed constants are cloned OUTSIDE inference mode; values,
    dtypes and equations are unchanged. Old CUDA graphs are not reused here.
    """
    rows=[]
    with torch.inference_mode(False),torch.no_grad():
        for dtype in (torch.float32,torch.float64):
            v=torch.empty(0,device=device,dtype=dtype);key=(str(v.device),v.dtype);before=c.t.constants(torch,v)
            after=tuple(x.detach().clone() for x in before)
            assert all(not torch.is_inference(x) and torch.equal(x,y) for x,y in zip(after,before))
            c.t._CONSTANTS[key]=after;rows.append(dict(dtype=str(dtype),ordinary_tensors=True,values_bitwise_unchanged=True))
    return rows

def spec(root):
    p=json.loads((root/'experiments/task45d_preregistration_v1.json').read_text());old,cfg,ca,base=n.contract(root)
    assert p['quality']==old['quality'] and p['compartment_gates']==cfg['gates']
    assert tuple(p['coupled_arms'])==ARMS and not p['thresholds_changed'] and not p['abstraction_changed']
    return p,cfg,ca,base

def matched(data):
    data=copy.deepcopy(data);v=data['voltage'];clock=np.arange(len(v))*.1
    for k,meta in enumerate(data['meta']):
        if meta['family'] in ('doublepulse','staircase'):
            # Weak, long variable-voltage returns near slow-current thresholds.
            rest=-48.+(meta['input_seed']%9)*.7
            use=clock>=230;v[use,k]=rest+3*np.exp(-(clock[use]-230)/250.)
        elif meta['family']=='tail_hold':
            use=(clock>=40)&(clock<200);v[use,k]=-130.+(meta['input_seed']%3)*5.
    u,z=n.p.coefficients(v.astype('float64'),data['calcium'].astype('float64'),.1);state=data['initial'].copy();truth=[]
    for tick in range(len(v)):state+=z[tick]*(u[tick]-state);truth.append(state.copy())
    data.update(target_u=u,target_z=z,truth=np.stack(truth));data['target_current']=n.p.readout(v,data['truth'],data['mult'],c.t.ionic.REVERSALS)[1]
    return data

def role(torch,baseline,p,which,adaptive,out,device):
    d=c.histories(p,which);d=matched(d) if adaptive else d
    d['features'],d['raw']=c.features_and_raw(torch,baseline,d['voltage'],d['calcium'],device)
    with torch.inference_mode():d['baseline_gates']=n.scan(torch,baseline,d['voltage'],d['calcium'],d['initial'],device)
    np.savez_compressed(out/(which+str(int(adaptive))+'_inputs.npz'),voltage=d['voltage'],calcium=d['calcium'],initial=d['initial'],multipliers=d['mult'])
    n.p.save_json(out/(which+str(int(adaptive))+'_manifest.json'),dict(meta=d['meta'],sha256=hashlib.sha256(d['voltage'].tobytes()+d['calcium'].tobytes()).hexdigest(),support_matched=adaptive))
    return d

def union(a,b):
    result={}
    for k in ('voltage','calcium','truth','target_u','target_z','target_current'):result[k]=np.concatenate((a[k],b[k]),1)
    for k in ('initial','mult'):result[k]=np.concatenate((a[k],b[k]),0)
    result['meta']=[dict(r,support=s) for s,d in enumerate((a,b)) for r in d['meta']]
    return result

def pool(torch,baseline,p,data,support,device):
    f=p['fitting'];q=(np.arange(f['points'])+.5)/f['points'];v=(-100+160*q).astype('float32');v[::2]=(-135+35*q[::2]).astype('float32');ca=np.full_like(v,1e-4)
    if support:
        rng=np.random.default_rng(p['matrix']['support_seed']);ix=rng.integers(data['voltage'].size,size=len(v)//2)
        selected=np.arange(len(v)).reshape(-1,4)[:,:2].reshape(-1)
        v[selected]=data['voltage'].reshape(-1)[ix];ca[selected]=data['calcium'].reshape(-1)[ix]
    ff,raw=c.features_and_raw(torch,baseline,v[None],ca[None],device);u,z=n.p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    lab=np.stack([np.log(np.clip(a,1e-12,1-1e-12)/(1-np.clip(a,1e-12,1-1e-12))) for a in (u,z)],-1)
    return dict(f=ff[0],raw=raw[0],label=lab,censored=int(np.count_nonzero((u<=1e-12)|(u>=1-1e-12)|(z<=1e-12)|(z>=1-1e-12))),v=v,ca=ca,u=u,z=z)

def whitening(pool,p):
    rows=[];probes=[]
    for seed in range(3):
        f=pool['f'][seed];_,s,vh=np.linalg.svd(f,full_matrices=False);rows.append(vh.T/np.maximum(s/np.sqrt(len(f)),p['fitting']['whitening_floor'])[None])
        for gate in (3,9,10):
            for h in (0,1):
                residual=pool['label'][:,gate,h]-pool['raw'][seed,:,gate,h];w=np.linalg.lstsq(f,residual,rcond=1e-12)[0]
                probes.append(dict(seed=(17,29,43)[seed],gate=gate,head=h,ideal_linear_rmse=float(np.sqrt(np.mean((f@w-residual)**2))),singular_max=float(s.max()),singular_min=float(s.min()),selectable=False))
    return np.stack(rows),probes

def training_view(torch,baseline,pool,p,adapt,device):
    class View(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=copy.deepcopy(baseline)
            for par in self.model.parameters():par.requires_grad_(False)
            self.model.parent.a.requires_grad_(adapt);self.model.parent.b.requires_grad_(adapt)
            white,probes=whitening(pool,p);self.register_buffer('white',torch.tensor(white,device=device,dtype=torch.float64))
            self.theta=torch.nn.Parameter(torch.zeros(3,65,6,device=device,dtype=torch.float64));self.probes=probes
        def raw(self,v,ca):
            u,z=self.model.raw(v,ca);f=c.b.features(torch,self.model,v).double();delta=torch.bmm(torch.bmm(f,self.white),self.theta)
            u=u.double();z=z.double();uu=[];zz=[]
            for gate in range(18):
                uu.append(u[...,gate]+(delta[...,COLS.index(gate)] if gate in COLS[:3] else 0.))
                zz.append(z[...,gate]+(delta[...,COLS.index(gate+17)] if gate+17 in COLS[3:] else 0.))
            return torch.stack(uu,-1),torch.stack(zz,-1)
        def materialize(self):
            model=copy.deepcopy(self.model)
            with torch.no_grad():
                delta=torch.bmm(self.white,self.theta).float()
                for j,col in enumerate(COLS):model.parent.c[:,:,col].add_(delta[:,:-1,j]);model.parent.d[:,col].add_(delta[:,-1,j])
                for par in model.parameters():par.requires_grad_(False)
            return model.eval()
    return View()

def train_loss(torch,view,data,ids,p):
    v=data['v'][ids][None].expand(3,-1);ca=data['ca'][ids][None].expand(3,-1);u,z=view.raw(v,ca)
    teacher=data['label'][ids][None];keep=torch.tensor([g not in (3,9,10) for g in range(18)],device=v.device)
    coeff=((torch.stack((u,z),-1)[...,[3,9,10],:]-teacher[...,[3,9,10],:])**2).mean((1,2,3))
    frozen=data['raw'][:,ids];preserve=((torch.stack((u,z),-1)[...,keep,:]-frozen[...,keep,:])**2).mean((1,2,3))
    wanted_u=data['u'][ids][None];wanted_z=data['z'][ids][None];state=data['initial'][ids][None]
    result=coeff+preserve
    for tick in p['matrix']['held_ticks']:
        pred=state-torch.expm1(tick*torch.nn.functional.logsigmoid(-z))*(torch.sigmoid(u)-state)
        truth=state-torch.expm1(tick*torch.log1p(-wanted_z))*(wanted_u-state)
        result=result+((pred[...,[3,9,10]]-truth[...,[3,9,10]])/.005).square().mean((1,2))/len(p['matrix']['held_ticks'])
        mult=torch.ones((*v.shape,11),device=v.device,dtype=pred.dtype)
        pc=c.t.readout(torch,v,pred,mult)[1][...,[1,4,5]];tc=c.t.readout(torch,v,truth,mult)[1][...,[1,4,5]]
        result=result+((pc-tc)/(tc.abs().clamp_min(p['quality']['current_scale_floor_ma_cm2']))/.01).square().mean((1,2))/len(p['matrix']['held_ticks'])
    return result

def clip_group(torch,params,limit):
    grads=[q.grad for q in params if q.grad is not None]
    if not grads:return
    norm=sum(g.reshape(3,-1).double().square().sum(1) for g in grads).sqrt();scale=(limit/norm.clamp_min(1e-30)).clamp_max(1)
    for g in grads:g.mul_(scale.reshape(3,*([1]*(g.ndim-1))))

def score_rows(rows):
    return sum(sum(q['passed'] for q in r['targets'].values()) for r in rows),max(q['severity'] for r in rows for q in r['targets'].values())

def safe(rows,base):
    indexed={(r['seed'],r['case'],r['horizon_ms']):r for r in base}
    return not any(indexed[(r['seed'],r['case'],r['horizon_ms'])]['passed'] and not r['passed'] for r in rows)

def choose(records,base):
    eligible=[r for r in records if safe(r[2],base)]
    return min(eligible or records,key=lambda r:(-score_rows(r[2])[0],score_rows(r[2])[1],r[1],r[0]))

def fit(torch,baseline,p,out,device):
    if device=='cuda':
        with np.load(out.parent/'verified_source45c/native_shadow_traces.npz') as z:preflight_data=dict(z)
        # Use the already checksummed source/config in evaluate's root.
        root=Path(__file__).resolve().parents[2];prior,cfg,ca,base=n.contract(root)
        physics=(n.old_native.x.calibration_config(prior,cfg),ca,base);small=copy.deepcopy(p);small['cost'].update(batches=[1],ticks=2,repetitions=2)
        dest=out/'GPU_cost_preflight';dest.mkdir()
        with torch.inference_mode():checks=ORIGINAL_BENCHMARK(torch,dict(dual=baseline),baseline.parent,preflight_data,physics,small,dest,device)
        assert len(checks)==2 and all(r['all_outputs_finite'] for r in checks)
        n.p.save_json(out/'GPU_cost_preflight.json',dict(valid=True,autograd_disabled=True,scopes=2,ticks=2))
    n.p.save_json(out/'training_readout_cache.json',dict(valid=True,rows=prepare_training_readout(torch,device),historical_source_unchanged=True))
    with zipfile.ZipFile(os.environ['GIADA_TASK45C_CHECKPOINTS']) as z:
        previous=copy.deepcopy(baseline);previous.load_state_dict(torch.load(io.BytesIO(z.read('J.pt')),map_location=device,weights_only=True))
    fits=[role(torch,baseline,p,'fit',bool(s),out,device) for s in (0,1)]
    devs=[role(torch,baseline,p,'dev',bool(s),out,device) for s in (0,1)];dev=union(*devs)
    base_rows=c.evaluate_role(torch,previous,dev,p,device);n.p.save_json(out/'development_C1.json',base_rows)
    pools=[pool(torch,baseline,p,fits[s],s,device) for s in (0,1)]
    records=[];models={'C1':previous};snapshots={};diagnostics=[];views={};optimizers={};tensors={};rng=np.random.default_rng(p['matrix']['support_seed'])
    common_initial=rng.uniform(0,1,(p['fitting']['points'],18))
    for label,a in p['matrix']['arms'].items():
        s=a['support'];view=training_view(torch,baseline,pools[s],p,a['adapt_features'],device);views[label]=view
        groups=[dict(params=[view.theta],lr=p['matrix']['theta_lr'])]
        if a['adapt_features']:groups.append(dict(params=[view.model.parent.a,view.model.parent.b],lr=p['matrix']['trunk_lr']))
        optimizers[label]=torch.optim.Adam(groups,foreach=False)
        data=pools[s];tensors[label]={k:torch.tensor(data[k],device=device,dtype=torch.float64 if k not in ('v','ca') else torch.float32) for k in ('v','ca','u','z','raw','label')}
        tensors[label]['initial']=torch.tensor(common_initial,device=device,dtype=torch.float64)
        n.p.save_json(out/(label+'_projection.json'),view.probes)
    if device=='cuda':
        # Exercise the actual inference-warmed cache -> backward -> Adam path
        # before spending the registered training budget. Scratch copies only.
        for label in ('A00','A01'):
            scratch=copy.deepcopy(views[label]);groups=[dict(params=[scratch.theta],lr=p['matrix']['theta_lr'])]
            if p['matrix']['arms'][label]['adapt_features']:groups.append(dict(params=[scratch.model.parent.a,scratch.model.parent.b],lr=p['matrix']['trunk_lr']))
            opt=torch.optim.Adam(groups,foreach=False)
            for _ in range(2):
                opt.zero_grad(set_to_none=True);loss=train_loss(torch,scratch,tensors[label],torch.arange(16,device=device),p).sum();assert bool(torch.isfinite(loss));loss.backward()
                clip_group(torch,[scratch.theta],p['matrix']['gradient_norm_per_seed_group']);clip_group(torch,[scratch.model.parent.a,scratch.model.parent.b],p['matrix']['gradient_norm_per_seed_group']);opt.step()
            assert all(bool(torch.isfinite(q).all()) for q in scratch.parameters())
        del scratch,opt,groups
        n.p.save_json(out/'GPU_training_transition_preflight.json',dict(valid=True,inference_then_backward_then_Adam=True,scratch_steps_per_style=2,scratch_updates_discarded=True,registered_updates_consumed=0))
        print('[GIADA45d] inference-to-training preflight passed; scratch updates discarded',flush=True)
    # All arms share minibatch indices and update times. Seeds are tensorized;
    # Adam moments and clipping are independent per seed and parameter group.
    for step in range(1,max(p['matrix']['atomic_checkpoints'])+1):
        ids=torch.tensor(rng.integers(p['fitting']['points'],size=p['matrix']['batch']),device=device)
        losses=[]
        for label,view in views.items():optimizers[label].zero_grad(set_to_none=True);losses.append(train_loss(torch,view,tensors[label],ids,p).sum())
        sum(losses).backward()
        for label,view in views.items():
            clip_group(torch,[view.theta],p['matrix']['gradient_norm_per_seed_group']);clip_group(torch,[view.model.parent.a,view.model.parent.b],p['matrix']['gradient_norm_per_seed_group']);optimizers[label].step()
        if step%25==0:
            assert all(bool(torch.isfinite(par).all()) for v in views.values() for par in v.parameters()),'Nonfinite45d training parameters'
        if step not in p['matrix']['atomic_checkpoints']:continue
        for label,view in views.items():
            model=view.materialize();torch.save(model.state_dict(),out/(label+'_'+str(step)+'.pt'));snapshots[(label,step)]=model
            rows=c.evaluate_role(torch,model,dev,p,device);records.append((label,step,rows));n.p.save_json(out/('development_'+label+'_'+str(step)+'.json'),rows)
            with torch.no_grad():
                v=tensors[label]['v'][:128][None].expand(3,-1);ca=tensors[label]['ca'][:128][None].expand(3,-1)
                want=view.raw(v,ca);got=model.raw(v,ca);err=max(float((a-b.double()).abs().max()) for a,b in zip(want,got))
            diagnostics.append(dict(arm=label,step=step,training_loss=float(losses[list(views).index(label)].detach()),materialized_logit_max_error=err))
        torch.save(dict(step=step,views={k:v.state_dict() for k,v in views.items()},optimizers={k:v.state_dict() for k,v in optimizers.items()},numpy_rng=rng.bit_generator.state),out/('atomic_training_state_'+str(step)+'.pt'))
        n.p.save_json(out/'atomic_diagnostics.json',diagnostics);print('[GIADA45d] atomic checkpoint '+str(step)+' all4 arms finite and DEV saved',flush=True)
    del views,optimizers,tensors
    # Nap coefficient-only quadratic: exact same objective and seed solves,
    # cap2 versus actual FIT line search, with increasing-budget snapshots.
    pp=copy.deepcopy(p);empty=fits[0];po=pools[0];w,_=whitening(po,p);width=w.shape[-1]
    for label in ('Ncap','Nline','T0','T1'):
        gate=9 if label.startswith('N') else 10;joint=gate==10;white=np.zeros((3,width*(2 if joint else 1),width*(2 if joint else 1)))
        for h in range(2 if joint else 1):white[:,h*width:(h+1)*width,h*width:(h+1)*width]=w
        theta=np.zeros((3,white.shape[-1]));w0=c.readout_weights(torch,baseline,gate,joint);stopped=None
        usepo=po if gate==9 else pools[1];usedata=empty if gate==9 else fits[1]
        if gate==10:
            w,_=whitening(usepo,p)
            for h in range(2):white[:,h*width:(h+1)*width,h*width:(h+1)*width]=w
        for step in range(1,max(p['fitting']['checkpoints'])+1):
            if stopped is None:
                pp['fitting']['max_whitened_step_norm']=2. if label=='Ncap' else 1e20
                proposed,info=c.solve_refinement(usedata,usepo,gate,joint,gate==10,label=='T1',theta,white,pp)
                if label=='Nline':
                    def loss(th):
                        delta=np.einsum('sij,sj->si',white,th);raw=usepo['raw'][...,gate,1]+np.einsum('snk,sk->sn',usepo['f'],delta)
                        return ((raw-usepo['label'][None,:,gate,1])**2).mean((1,))+p['fitting']['ridge']*(th**2).sum(1)
                    candidates=[theta+alpha*(proposed-theta) for alpha in (0.,.25,.5,1.)];ll=np.stack([loss(q) for q in candidates]);indices=ll.argmin(0);theta=np.stack([candidates[int(indices[s])][s] for s in range(3)])
                else:theta=proposed
                diagnostics.append(dict(arm=label,step=step,**info))
                if max(info['unclipped_step_norm'])<1e-8:stopped=step
            if step not in p['fitting']['checkpoints']:continue
            delta=np.einsum('sij,sj->si',white,theta).reshape(3,2 if joint else 1,width).transpose(0,2,1)
            model=c.materialize(torch,baseline,gate,joint,(w0+delta).astype('float32'));snapshots[(label,step)]=model;torch.save(model.state_dict(),out/(label+'_'+str(step)+'.pt'))
            rows=c.evaluate_role(torch,model,dev,p,device);records.append((label,step,rows));n.p.save_json(out/('development_'+label+'_'+str(step)+'.json'),dict(rows=rows,early_stopped_at=stopped))
            print('[GIADA45d] '+label+' budget'+str(step)+' saved; early stop='+str(stopped),flush=True)
    choices={}
    for label in ('A00','A01','A10','A11','Ncap','Nline','T0','T1'):
        record=choose([r for r in records if r[0]==label],base_rows);models[label]=snapshots[(record[0],record[1])];choices[label]=dict(checkpoint=record[1],DEV_nonregressing=safe(record[2],base_rows))
    selected=choose([('C1',0,base_rows)]+[r for r in records if safe(r[2],base_rows)],base_rows)
    models['J']=copy.deepcopy(previous if selected[0]=='C1' else snapshots[(selected[0],selected[1])])
    freeze=dict(valid=True,chosen=dict(arm=selected[0],checkpoint=selected[1]),per_arm_choices=choices,selection_used_coupled=False,selection_used_old_stress=False,weights_sha256={k:n.weights_hash(v) for k,v in models.items()},refinement_diagnostics=diagnostics)
    n.p.save_json(out/'nomination_freeze.json',freeze);return models,freeze

def adapter(torch,baseline,models):
    class Packed(torch.nn.Module):
        def __init__(self):
            super().__init__();self.reference=baseline;self.ensemble=copy.deepcopy(baseline.parent);self.parallel=True
            sources=[baseline if name=='none' else models[name] for name in ARMS]
            self.nets=torch.nn.ModuleList(sources)
            for net in sources:
                assert all(torch.equal(v,net.donor.state_dict()[k]) for k,v in baseline.donor.state_dict().items()),'45d donor must remain frozen across arms'
            for key,param in self.ensemble.named_parameters():
                if param.ndim and param.shape[0]==3:param.data=torch.stack([dict(m.parent.named_parameters())[key].detach() for m in sources],1).reshape(3*len(ARMS),*param.shape[1:]).clone()
                param.requires_grad_(False)
            for key,value in list(self.ensemble.named_buffers()):
                if value.ndim and value.shape[0]==3:
                    owner,_,field=key.rpartition('.');module=self.ensemble.get_submodule(owner) if owner else self.ensemble
                    setattr(module,field,torch.stack([dict(m.parent.named_buffers())[key] for m in sources],1).reshape(3*len(ARMS),*value.shape[1:]).clone())
        def __getattr__(self,k):
            if k in ('a','sa','sb','sc','sd','gbar'):return getattr(self.reference,k)
            return super().__getattr__(k)
        def raw(self,v,ca):
            count=v.shape[1]//len(ARMS)
            if not self.parallel:
                parts=[net.raw(v[:,i*count:(i+1)*count],ca[:,i*count:(i+1)*count]) for i,net in enumerate(self.nets)]
                return tuple(torch.cat([r[k] for r in parts],1) for k in (0,1))
            parent=self.ensemble.raw(v.reshape(3*len(ARMS),count),ca.reshape(3*len(ARMS),count))
            donor=self.reference.donor.raw(v,ca)
            return tuple(torch.where(self.reference.mask,b,a.reshape(3,-1,18)) for a,b in zip(parent,donor))
        def forward(self,v,ca,state):
            u,z=self.raw(v,ca);return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)
    net=Packed().eval();checks=[]
    with torch.inference_mode():
        for count in (1,2,64,176,642,4096):
            v=torch.linspace(-135,60,count,device=baseline.a.device)[None].expand(3,-1).repeat(1,len(ARMS));ca=torch.full_like(v,1e-4);state=torch.full((*v.shape,18),.37,device=v.device)
            got=net(v,ca,state);worst=0.
            for i,source in enumerate(net.nets):
                sl=slice(i*count,(i+1)*count);want=source(v[:,sl],ca[:,sl],state[:,sl]);worst=max(worst,float((want-got[:,sl]).abs().max()))
            checks.append(dict(batch=count,max_gate_error=worst))
            if worst>2e-7:net.parallel=False;break
    net.execution_preflight=dict(vectorized=net.parallel,checks=checks,original_occupancy_atol=2e-7,fallback='Actual standalone model per arm, no tolerance relaxation' if not net.parallel else None)
    return net

@contextmanager
def configured(p):
    before=c.y.spec,c.y.ARMS,c.y.previous.design,c.y.channel_masks
    c.y.spec=lambda root:(p,*n.contract(root)[1:]);c.y.ARMS=ARMS;c.y.previous.design=c.b.design
    c.y.channel_masks=lambda count,arms:np.array([np.full((count,11),arm!='none',dtype=bool) for arm in arms])
    try:yield
    finally:c.y.spec,c.y.ARMS,c.y.previous.design,c.y.channel_masks=before

def remap(fn,**kw):return types.FunctionType(fn.__code__,dict(fn.__globals__,**kw),argdefs=fn.__defaults__)

def crossing_and_gate_probes(pred,ref,geometry,rows,mode):
    crossings=remap(c.crossing_audit,ARMS=ARMS)(pred,ref,geometry,rows,mode);probes=[]
    # Store compact18-state diagnostics, not a second copy of every trajectory.
    # Their thresholds are diagnostic; original evaluator predicates unchanged.
    for ai,arm in enumerate(ARMS[1:],1):
        error=abs(pred['gates'][:,:,ai]-ref['gates'][:,:,0]);maximum=error.max(axis=(0,3));rmse=np.sqrt((error**2).mean(0)).max(2)
        crossings_at=(error>.03).any(3);first=crossings_at.argmax(0);exceeded=crossings_at.any(0)
        for si,seed in enumerate((17,29,43)):
            for case in range(len(rows)):
                for gate in range(18):
                    if maximum[si,case,gate]>.005:
                        probes.append(dict(arm=arm,seed=seed,case=case,gate_index=gate,max_abs=float(maximum[si,case,gate]),worst_node_rmse=float(rmse[si,case,gate]),first_abs_003_exceedance_ms=float(first[si,case,gate]*.1) if exceeded[si,case,gate] else None,diagnostic_only=True))
    return dict(crossings=crossings,all18_gate_probes=probes,reference='Same canonical coupled/imposed reference, not native phase relabeling')

def control_check(report,checkpoints):
    with zipfile.ZipFile(checkpoints) as z:old=json.loads(z.read('accuracy_summary.json'))['panels']
    for panel,by_h in report['panels'].items():
        for horizon,by_arm in by_h.items():
            for arm,previous in (('C0','C0'),('C1','J'),('R41','R41'),('R111','R111')):
                for mode in ('imposed','autonomous'):
                    got=by_arm[arm][mode];want=old[panel][horizon][previous][mode]
                    assert (got['eligible'],got['passed'])==(want['eligible'],want['passed']),(panel,horizon,arm,mode,'Frozen45c control counts failed reproduction')
    return dict(valid=True,controls=['C0','C1','R41','R111'],same_original_counts=True,original_identity_details_retained_in_raw_results=True)

def evaluate(torch,root,source,fixtures,checkpoints,out,device='cuda'):
    plan=json.loads((root/'experiments/task45c_cost_transport_v2.json').read_text());assert c.sha(checkpoints)==plan['sha256']
    os.environ['GIADA_TASK45C_CHECKPOINTS']=str(checkpoints)
    benchmark=n.benchmark
    def cost(*args):
        nets=args[1];chosen=nets['J'];variants=n.execution.variants(torch,chosen)
        parity=[];eligible={}
        with torch.inference_mode():
            for label,net in variants.items():
                okay=True;maximum=0.
                for batch in (1,2,64,176,642,4096):
                    v=torch.linspace(-135,60,batch,device=device)[None].expand(3,-1);ca=torch.logspace(-8,-1,batch,device=device)[None].expand(3,-1)
                    want=chosen.raw(v,ca);got=net.raw(v,ca);exact=all(n.tensor_bits_equal(torch,a,b) for a,b in zip(want,got));okay &= exact;maximum=max(maximum,max(float((a-b).abs().max()) for a,b in zip(want,got)))
                parity.append(dict(variant=label,bitwise_equal=okay,max_raw_error=maximum,classification='equivalent_on_preflight' if okay else 'diagnostic_not_eligible'))
                if label!='dual':eligible[('J_' if okay else 'diagnostic_J_')+label]=net
            n.p.save_json(out/'execution_variant_parity.json',parity)
            mutable=list(args);mutable[1]=dict(dual=nets['dual'],C1=nets['C1'],J=chosen,**eligible)
            result=benchmark(*mutable)
            v=torch.linspace(-130,50,642,device=device)[None].expand(3,-1);ca=torch.full_like(v,1e-4);profiles=[]
            for label,fn in ([('parent_raw',lambda:chosen.parent.raw(v,ca)),('donor_raw',lambda:chosen.donor.raw(v,ca)),('full_raw',lambda:chosen.raw(v,ca)),('parent_feature',lambda:c.b.features(torch,chosen,v))] if device=='cuda' else []):
                for _ in range(3):fn()
                samples=[]
                for _ in range(30):
                    aa,bb=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);aa.record();fn();bb.record();bb.synchronize();samples.append(aa.elapsed_time(bb))
                profiles.append(dict(component=label,batch_per_seed=642,cuda_ms=samples,additive_attribution=False))
            n.p.save_json(out/'component_profile.json',profiles)
        return result
    from unittest.mock import patch
    def make_adapter(*args):
        model=adapter(*args);n.p.save_json(out/'packed_execution_preflight.json',model.execution_preflight);return model
    globals_=dict(spec=spec,fit=fit,batched_model=make_adapter,configured=configured,ARMS=ARMS,
        packed_parity=remap(c.packed_parity,ARMS=ARMS),crossing_audit=crossing_and_gate_probes)
    with patch.object(n,'benchmark',side_effect=cost):report=remap(c.evaluate,**globals_)(torch,root,source,fixtures,out,device)
    report['frozen_control_reproduction']=control_check(report,checkpoints)
    report.update(task='45d',decision='LEARNABILITY_AND_EXECUTION_MATRIX_COMPLETE_REQUIRES_AUDIT',release_authorized=False)
    n.p.save_json(out/'final_report.json',report);return report
