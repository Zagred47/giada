"""Original quality conjunction plus head-visible contrasts and frozen sidecars."""
import copy
import json
import types
from contextlib import contextmanager
import numpy as np
from . import task41i_selective_repair as i
from . import task41h_evaluation as he
from . import task41f_evaluation as fe
p,t,c,d=i.p,i.t,i.c,i.d


class EvaluationPause(Exception):pass


def install_resume_cache(out):
    """Reuse verified completed panels; pause only outside numerical bodies."""
    import sys,time
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    from task41h_stage_cache import install
    install(i.h,he,out)
    started=time.monotonic()
    def guard(original):
        def wrapped(*args,**kwargs):
            if time.monotonic()-started>6*3600:raise EvaluationPause()
            return original(*args,**kwargs)
        return wrapped
    # Same formulas/metric bodies. Completed-cache checks remain inside wrappers.
    for module,name in ((he.ee.old,'suite'),(he.ee.old,'stress'),(c.b,'causal_evaluate'),(c.j.i.old,'evaluate'),(he.ee.e,'benchmark')):
        setattr(module,name,guard(getattr(module,name)))


def projection_probe(torch,model,fit,held,out):
    rows=[]
    for ai,a in enumerate(model.definitions):
        for si,seed in enumerate((17,29,43)):
            member=3*ai+si
            for head,gate in enumerate(i.GATES):
                for count in (1024,4096):
                    n=min(count,len(fit['v']))
                    v=fit['v'][:n,1][None];knots=model.knots[member:member+1,head].double()
                    xx=i.feature(torch,v.double(),knots)[0].cpu()
                    target=fit['target'][:n,1,head].double();base=fit['base'][si,:n,1,head].double()
                    delta=torch.logit(target.clamp(1e-7,1-1e-7))-base
                    outside=int((delta.abs()>=.1).sum())
                    transformed=.1*torch.atanh((delta/.1).clamp(-.999999,.999999))
                    w=torch.linalg.lstsq(xx,transformed.cpu()[:,None],driver='gelsy').solution
                    test=i.feature(torch,held['v'][:,1][None].double(),knots)[0].cpu()
                    predicted=.1*torch.tanh((test@w).squeeze(-1)/.1)
                    wanted=torch.logit(held['target'][:,1,head].double().clamp(1e-7,1-1e-7))-held['base'][si,:,1,head].double()
                    error=predicted-wanted.cpu()
                    rows.append(dict(arm=a['name'],seed=seed,gate=gate,FIT_count=n,outside_bound_fit=outside,
                                     heldout_mse=float(error.square().mean()),heldout_max=float(error.abs().max()),selection_eligible=False))
    p.save_json(out/'projection_probe.json',dict(rows=rows,scope='Bounded FIT-only coefficient approximation diagnostic; no candidate installation or SGD claim. Saturated targets explicitly counted.'))


def increment_membrane(torch,v,conductance,injection,syn,area,cfg,base):
    """FP32 Newton increments with a compensated high/low voltage carrier.

    Same membrane equation and12 iterations. All exponential/current arithmetic
    here is FP32; FP64 stores only the sum of the high/low voltage components.
    This is a numerical representation, not an additional biological state.
    """
    hi=v.float();lo=(v-hi.double()).float();gg=conductance.float();inj=injection.float()
    _,rev,synrev=t.constants(torch,hi);density=syn.float()*(100./area)
    fixed=torch.cat((density[...,:1],torch.zeros_like(density[...,1:2]),density[...,2:]),-1)
    cap=1e-3*base['cm_uf_cm2']/cfg['dt_ms'];gp=base['g_pas_s_cm2']
    denom=cap+gp+gg.sum(-1)+fixed.sum(-1)
    delta=torch.zeros_like(hi)
    for _ in range(12):
        nv=(hi+delta)+lo
        ex=torch.exp((-.08*nv).clamp(-80,80))/3.57;nmda=density[...,1]/(1+ex)
        derivative=.08*density[...,1]*ex/(1+ex).square()
        residual=cap*delta+gp*((hi-base['e_pas_mv'])+lo+delta)
        residual=residual+(gg*((hi[...,None]-rev)+(lo+delta)[...,None])).sum(-1)
        residual=residual+(fixed*((hi[...,None]-synrev)+(lo+delta)[...,None])).sum(-1)+nmda*nv-inj
        delta=delta-residual/(denom+nmda+derivative*nv)
    return hi.double()+lo.double()+delta.double()


def compensated_core(torch,model,cfg,calcium,base,area):
    proxy=types.SimpleNamespace(**{**vars(t),'membrane':increment_membrane})
    fn=types.FunctionType(he.ge.precision_core.__code__,dict(he.ge.__dict__,t=proxy))
    return fn(torch,model,cfg,calcium,base,area,True,False,False)


class RuntimeModel:
    """Frozen numerical/execution diagnostic, never used in checkpoint selection."""
    def __init__(self,torch,parent,policy):
        self.torch,self.parent,self.runtime_policy=torch,parent,policy
        self._raw=None;self._forward=None
    def __getattr__(self,name):
        if name in ('parent','_raw','_forward'):raise AttributeError(name)
        return getattr(self.parent,name)
    def __deepcopy__(self,memo):return RuntimeModel(self.torch,copy.deepcopy(self.parent,memo),self.runtime_policy)
    def to(self,*args,**kwargs):self.parent.to(*args,**kwargs);self._raw=None;self._forward=None;return self
    def raw(self,v,ca):
        if self.runtime_policy!='compiled':return self.parent.raw(v,ca)
        if self._raw is None:self._raw=self.torch.compile(self.parent.raw,fullgraph=True,options={'triton.cudagraphs':False})
        return self._raw(v,ca)
    def __call__(self,v,ca,state):
        u,z=self.raw(v,ca);return state+self.torch.sigmoid(z)*(self.torch.sigmoid(u)-state)


@contextmanager
def runtime_adapters():
    """Dispatch only explicitly labelled sidecars; canonical reference untouched.

    Scoped process-local adapters preserve every historical source file/hash.
    Imposed paths and V64 recurrent buffers are represented at their real dtype.
    """
    original_core,original_rollout=t.make_core,t.rollout
    def make_core(torch,cfg,calcium,base,area,model=None,channels=(),ionic_only=False):
        policy=getattr(model,'runtime_policy',None) if model is not None else None
        if policy not in ('joint64','increment32'):
            return original_core(torch,cfg,calcium,base,area,model,channels,ionic_only)
        assert not ionic_only,'Precision sidecars require the coupled membrane'
        core=he.precision_core(torch,model,cfg,calcium,base,area,True,True,False) if policy=='joint64' else compensated_core(torch,model,cfg,calcium,base,area)
        core._giada_runtime=(model,policy)
        return core
    def rollout(torch,core,data,device,imposed=None,graph=False):
        if not hasattr(core,'_giada_runtime'):return original_rollout(torch,core,data,device,imposed,graph)
        def ex(value,dtype):return torch.as_tensor(value,device=device,dtype=dtype)[None].expand(3,*value.shape).contiguous()
        args=[ex(data['voltage'][:,0],torch.float64)]+[ex(data[key][:,0],torch.float32) for key in ('calcium','gates')]
        args += [ex(data['multipliers'],torch.float32),ex(data['injection'][:,0],torch.float64),ex(data['base_g_us'][:,0],torch.float64)]
        runner=t.Stepper(torch,core,args,graph);values=[[a.clone()] for a in args[:3]]+[[]]
        for tick in range(data['injection'].shape[1]):
            forced=None if imposed is None else (ex(imposed['voltage'][tick],torch.float64),ex(imposed['calcium'][tick],torch.float32))
            result=runner.tick(ex(data['injection'][:,tick],torch.float64),ex(data['base_g_us'][:,tick],torch.float64),forced)
            for target,val in zip(values,result):target.append(val.clone())
        return {key:torch.stack(vals).cpu().numpy() for key,vals in zip(('voltage','calcium','gates','current'),values)}
    t.make_core,t.rollout=make_core,rollout
    try:yield
    finally:t.make_core,t.rollout=original_core,original_rollout


def visible_contrasts(out,spec):
    rows=json.loads((out/'development_gpu.json').read_text())
    result=[]
    for axis,a,b in i.contrasts():
        channels=(4,) if axis=='nap' else i.CHANNELS
        per_head={}
        introduced=sum(not treatment['metrics']['passed'] and control['metrics']['passed'] for control,treatment in zip(rows[a],rows[b]))
        for channel in channels:
            severity=lambda name,seed:max(i.f.channel_severity(r['metrics'],spec['quality'],(channel,)) for r in rows[name] if r['seed']==seed)
            ratios=[max(severity(b,s),1e-12)/max(severity(a,s),1e-12) for s in spec['seeds']]
            per_head[str(channel)]=dict(ratios=ratios,median_ratio=float(np.median(ratios)),improvement_in_all_seeds=all(r<1 for r in ratios),
                                        registered_contrast_passed=bool(introduced==0 and all(r<1 for r in ratios) and np.median(ratios)<=.8))
        result.append(dict(axis=axis,control=a,treatment=b,per_head=per_head,
                           introduced_case_failures=introduced,registered_contrast_passed=all(row['registered_contrast_passed'] for row in per_head.values()),
                           global_gate_waived=False,interpretation='Per-head severity distinguishes effects; original full conjunction remains mandatory.'))
    p.save_json(out/'paired_contrasts.json',result)


def execution_cost(torch,models,data,spec,cfg,calcium,base,device,out,qualified):
    import time
    from .ie_ionic_cost import paired_summary
    rows=[]
    for batch in spec['cost']['batches']:
        ix=np.arange(batch)%len(data['voltage']);samples={};cuda_samples={};runners={}
        for name,net in {'formula':None,**models}.items():
            dtype=torch.float64 if getattr(net,'runtime_policy',None) in ('joint64','increment32') else torch.float32
            ex=lambda value,dt:torch.tensor(value,device=device,dtype=dt)[None].expand(3,*value.shape).contiguous()
            args=[ex(data['voltage'][ix,100],dtype),ex(data['calcium'][ix,100],torch.float32),ex(data['gates'][ix,100],torch.float32),
                  ex(data['multipliers'][ix],torch.float32),ex(data['injection'][ix,100],dtype),ex(data['base_g_us'][ix,100],dtype)]
            core=t.make_core(torch,cfg,calcium,base,float(data['area_um2']),net,t.FAMILIES['all'] if net is not None else ())
            runner=t.Stepper(torch,core,args,True)
            inj=[ex(data['injection'][ix,100+k],dtype) for k in range(spec['cost']['ticks'])]
            syn=[ex(data['base_g_us'][ix,100+k],dtype) for k in range(spec['cost']['ticks'])]
            runners[name]=(runner,args,inj,syn);samples[name]=[];cuda_samples[name]=[]
        names=list(runners)
        for repeat in range(spec['cost']['repetitions']):
            for name in names if repeat%2==0 else names[::-1]:
                runner,args,inj,syn=runners[name];runner.reset(args);torch.cuda.synchronize()
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record()
                for k in range(len(inj)):runner.tick(inj[k],syn[k])
                end.record();end.synchronize();samples[name].append(1000*(time.perf_counter()-start));cuda_samples[name].append(begin.elapsed_time(end))
        rows.append(dict(batch_per_seed=batch,actual_sites=3*batch,samples=samples,cuda_samples=cuda_samples,
                         vs_formula={name:paired_summary(samples['formula'],values) for name,values in samples.items() if name!='formula'},
                         cuda_vs_formula={name:paired_summary(cuda_samples['formula'],values) for name,values in cuda_samples.items() if name!='formula'},
                         quality_qualified={name:name in qualified for name in samples},reference='Original FP32 formula; all candidates measured at their actual runtime state dtype'))
        p.save_json(out/'whole_chain_execution_cost.json',rows)
        if batch==642:
            activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:
                for name,(runner,args,inj,syn) in runners.items():
                    with torch.profiler.record_function(name):
                        runner.reset(args)
                        for k in range(min(4,len(inj))):runner.tick(inj[k],syn[k])
                torch.cuda.synchronize()
            records=[dict(name=row.key,calls=row.count,self_cpu_us=row.self_cpu_time_total,
                          device_us=getattr(row,'device_time_total',0),self_device_us=getattr(row,'self_device_time_total',0)) for row in prof.key_averages()]
            p.save_json(out/'whole_chain_profile.json',dict(rows=records,scope='Four coupled ticks per variant, including CUDA graph kernels; not a speedup claim'))
        del runners;__import__('gc').collect();torch.cuda.empty_cache()
    return rows


def evaluate(torch,root,out,controls,trained,spec,cfg,calcium,base,device):
    freeze=json.loads((out/'selection_freeze.json').read_text());nominee=trained[freeze['nominee']]
    # All three numerical/execution sidecars face the SAME full quality suite.
    variants={f'{freeze["nominee"]}__{policy}':RuntimeModel(torch,copy.deepcopy(nominee),policy)
              for policy in ('joint64','increment32','compiled')}
    proxy=types.SimpleNamespace(**{**vars(i),'contrasts':lambda:[]})
    fn=types.FunctionType(fe.evaluate.__code__,dict(fe.__dict__,f=proxy))
    with runtime_adapters():
        report=fn(torch,root,out,controls,{**trained,**variants},spec,cfg,calcium,base,device)
        if device=='cuda':
            data,_,_=he.ee.load41c(root,spec)
            execution_cost(torch,{'legacy':nominee,**variants},data,spec,cfg,calcium,base,device,out,
                           set(report['passing_arms'])|({'legacy'} if freeze['nominee'] in report['passing_arms'] else set()))
    visible_contrasts(out,spec)
    # Compare the frozen H and the new finalist with already audited precision
    # policies as well; no numerical policy is used to choose the learned arm.
    he.numerical_panels(torch,root,out/'selected_numerical_audit',{'H_nominee':controls['H_nominee'],'nominee':nominee},spec,cfg,calcium,base,device)
    primary={a['name'] for a in i.arms()}
    report['diagnostic_passing_variants']=[name for name in report['passing_arms'] if name not in primary]
    report['passing_arms']=[name for name in report['passing_arms'] if name in primary]
    report.update(task='41i',decision='REPAIR_REQUIRES_INDEPENDENT_CONFIRMATION' if report['passing_arms'] else 'SELECTIVE_REPAIR_NO_GO',
                  task42_authorized=False,thresholds_changed=False,speedup_claim=False,numerical_selection_performed=False)
    return report
