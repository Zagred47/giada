"""Frozen numerical seams, paired execution and unchanged scientific panels."""
import copy
import itertools
import json
import types
import numpy as np
from . import task41h_path_repair as h
from . import task41g_evaluation as ge
p,t,c,d,f,ee=h.p,h.t,h.c,h.d,h.f,h.f.ee


def stable_membrane(torch,v,conductance,injection,syn,area,cfg,base):
    _,rev,synrev=t.constants(torch,v);density=syn*(100./area)
    fixed=torch.cat((density[...,:1],torch.zeros_like(density[...,1:2]),density[...,2:]),-1)
    cap=1e-3*base['cm_uf_cm2']/cfg['dt_ms']
    denom=cap+base['g_pas_s_cm2']+conductance.sum(-1)+fixed.sum(-1)
    numer=cap*v+base['g_pas_s_cm2']*base['e_pas_mv']+(conductance*rev).sum(-1)+(fixed*synrev).sum(-1)+injection
    nv=numer/denom
    for _ in range(12):
        ex=torch.exp((-.08*nv).clamp(-80,80))/3.57;nmda=density[...,1]/(1+ex)
        derivative=.08*density[...,1]*ex/(1+ex).square()
        residual=cap*(nv-v)+base['g_pas_s_cm2']*(nv-base['e_pas_mv'])
        residual=residual+(conductance*(nv[...,None]-rev)).sum(-1)+(fixed*(nv[...,None]-synrev)).sum(-1)+nmda*nv-injection
        nv=nv-residual/(denom+nmda+derivative*nv)
    return nv


def precision_core(torch,model,cfg,calcium,base,area,state64=False,solve64=False,stable=False):
    state_dtype=torch.float64 if state64 else torch.float32
    solve_dtype=torch.float64 if solve64 else torch.float32
    def membrane(torch,v,g,i,s,a,c,b):
        fn=stable_membrane if stable else t.membrane
        return fn(torch,v.to(solve_dtype),g.to(solve_dtype),i.to(solve_dtype),s.to(solve_dtype),a,c,b).to(state_dtype)
    proxy=types.SimpleNamespace(**{**vars(t),'membrane':membrane})
    fn=types.FunctionType(ge.precision_core.__code__,dict(ge.__dict__,t=proxy))
    return fn(torch,model,cfg,calcium,base,area,state64,False,False)


def rollout(torch,model,data,cfg,calcium,base,device,policy):
    state64,solve64,stable=policy;dtype=torch.float64 if state64 else torch.float32
    ex=lambda x,dt:torch.tensor(x,device=device,dtype=dt)[None].expand(3,*x.shape).contiguous()
    args=[ex(data['voltage'][:,0],dtype)]+[ex(data[k][:,0],torch.float32) for k in ('calcium','gates')]
    args += [ex(data['multipliers'],torch.float32),ex(data['injection'][:,0],dtype),ex(data['base_g_us'][:,0],dtype)]
    runner=t.Stepper(torch,precision_core(torch,model,cfg,calcium,base,float(data['area_um2']),*policy),args,device=='cuda')
    values=[[x.clone()] for x in args[:3]]+[[]]
    for tick in range(data['injection'].shape[1]):
        result=runner.tick(ex(data['injection'][:,tick],dtype),ex(data['base_g_us'][:,tick],dtype))
        for target,value in zip(values,result):target.append(value.clone())
    return {k:torch.stack(v).cpu().numpy() for k,v in zip(('voltage','calcium','gates','current'),values)}


def numerical_panels(torch,root,out,models,spec,cfg,calcium,base,device):
    out.mkdir(exist_ok=True);oldspec=json.loads((root/c.CONFIG).read_text())
    data,meta,_,_=ee.e.load_opened(root,oldspec);dd,mm,_=ee.load41c(root,spec);rows=[]
    for label,inputs,metadata in [('41',data,meta),('41c',dd,mm)]:
        ref={k:v[:,0] for k,v in t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(inputs['area_um2'])),inputs,device,graph=device=='cuda').items()}
        for name,model in models.items():
            for state,solve,stable in [(a,b,False) for a,b in itertools.product((False,True),repeat=2)]+[(False,False,True)]:
                pred=rollout(torch,model,inputs,cfg,calcium,base,device,(state,solve,stable))
                metrics=t.metrics(pred,{**ref,**{k:inputs[k].T for k in ('voltage','calcium')}},inputs,metadata,spec['quality'],cfg['gates'],'autonomous')
                response=[dict(case=k,seed=seed,metrics=p.evaluate_intervention(ref['current'][:,int(r['source_index'])],ref['current'][:,k],
                    pred['current'][:,si,int(r['source_index'])],pred['current'][:,si,k],.1,spec['quality']))
                    for si,seed in enumerate(spec['seeds']) for k,r in enumerate(metadata) if r['intervention']!='none']
                rows.append(dict(panel=label,model=name,state64=state,solve64=solve,recentered=stable,metrics=metrics,responses=response,selection_eligible=False))
                p.save_json(out/'voltage_precision.json',rows)
            print(f'[GIADA Task41h] checkpoint numerical {label}/{name}',flush=True)
    return rows


def oracle_stress(torch,root,out,models,spec,device):
    _,_,stress=ee.load41c(root,spec);design=d.prior.stress_design(json.loads((root/d.prior.CONFIG).read_text()))
    combined={}
    for name,model in models.items():
        for label,us,zs in [('learned',(),()),('u_m',(0,),()),('z_m',(),(0,)),('h',(1,),(1,))]:
            combined[name+'/'+label]=model if label=='learned' else h.g.fe.CoefficientOracle(torch,model,us,zs)
    report=ee.old.stress(torch,combined,stress,spec,device,design)
    p.save_json(out/'coefficient_oracle_stress.json',report);return report


def frozen_audit(torch,root,out,controls,spec,cfg,calcium,base,device):
    dest=out/'frozen_audit';dest.mkdir()
    selected={name:controls[name] for name in ('E_Q0','G_nominee','G_tail_diagnostic')}
    numerical_panels(torch,root,dest,selected,spec,cfg,calcium,base,device)
    oracle_stress(torch,root,dest,{'G_tail_diagnostic':controls['G_tail_diagnostic']},spec,device)
    p.save_json(dest/'scope.json',dict(valid=True,selection_eligible=False,thresholds_changed=False,reference_changed=False))


def execution_cost(torch,root,out,model,data,spec,cfg,calcium,base,device):
    import time
    from .ie_ionic_cost import paired_summary
    rows=[]
    for batch in spec['cost']['batches']:
        ix=np.arange(batch)%len(data['voltage']);vv=torch.tensor(data['voltage'][ix,100],device=device)[None].expand(3,-1)
        cc=torch.tensor(data['calcium'][ix,100],device=device)[None].expand(3,-1)
        gg=torch.tensor(data['gates'][ix,100],device=device)[None].expand(3,-1,-1)
        mult=torch.tensor(data['multipliers'][ix],device=device)[None].expand(3,-1,-1)
        inj=torch.tensor(data['injection'][ix,100:100+spec['cost']['ticks']],device=device).T
        syn=torch.tensor(data['base_g_us'][ix,100:100+spec['cost']['ticks']],device=device).transpose(0,1)
        compact=copy.deepcopy(model);compact.compact=True
        for stable in (False,True):
            runners={};samples={};args=[vv.float().contiguous(),cc.float().contiguous(),gg.float().contiguous(),mult.float().contiguous(),inj[0][None].expand(3,-1).float().contiguous(),syn[0][None].expand(3,-1,-1).float().contiguous()]
            for name,net in [('formula',None),('legacy',model),('compact',compact)]:
                if net is None:
                    if stable:
                        proxy=types.SimpleNamespace(**{**vars(t),'membrane':stable_membrane})
                        fn=types.FunctionType(t.make_core.__code__,dict(t.__dict__,membrane=stable_membrane))
                        core=fn(torch,cfg,calcium,base,float(data['area_um2']))
                    else:core=t.make_core(torch,cfg,calcium,base,float(data['area_um2']))
                else:core=precision_core(torch,net,cfg,calcium,base,float(data['area_um2']),False,False,stable)
                runners[name]=t.Stepper(torch,core,[a.clone() for a in args],True);samples[name]=[]
            for repeat in range(spec['cost']['repetitions']):
                for name in (('formula','legacy','compact') if repeat%2==0 else ('compact','legacy','formula')):
                    run=runners[name];run.reset(args);torch.cuda.synchronize();start=time.perf_counter()
                    for k in range(len(inj)):run.tick(inj[k][None].expand(3,-1).float(),syn[k][None].expand(3,-1,-1).float())
                    torch.cuda.synchronize();samples[name].append(1000*(time.perf_counter()-start))
            rows.append(dict(batch=batch,recentered_membrane=stable,samples=samples,
                compact_vs_legacy=paired_summary(samples['legacy'],samples['compact']),compact_vs_formula=paired_summary(samples['formula'],samples['compact']),quality_qualified=False))
            p.save_json(out/'compact_execution_cost.json',rows)
            del runners;__import__('gc').collect();torch.cuda.empty_cache()
    return rows


def evaluate(torch,root,out,controls,trained,spec,cfg,calcium,base,device):
    fn=types.FunctionType(h.g.fe.evaluate.__code__,dict(h.g.fe.__dict__,f=h))
    report=fn(torch,root,out,controls,trained,spec,cfg,calcium,base,device)
    primary={a['name'] for a in h.arms()}
    report['diagnostic_passing_arms']=[k for k in report['passing_arms'] if k not in primary]
    report['passing_arms']=[k for k in report['passing_arms'] if k in primary]
    nominee=trained[report['nominee']]
    numerical_panels(torch,root,out/'selected_numerical_audit',{'nominee':nominee},spec,cfg,calcium,base,device)
    oracle_stress(torch,root,out,{'nominee':nominee},spec,device)
    data,_,_=ee.load41c(root,spec)
    if device=='cuda':execution_cost(torch,root,out,nominee,data,spec,cfg,calcium,base,device)
    report.update(task='41h',decision='REPAIR_REQUIRES_INDEPENDENT_CONFIRMATION' if report['passing_arms'] else 'PATH_EXPOSURE_REPAIR_NO_GO',
        primary_selection='stratified',task42_authorized=False,thresholds_changed=False)
    return report
