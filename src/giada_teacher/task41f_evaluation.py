"""41f frozen precision attribution, unchanged regressions and execution matrix."""
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from . import task41f_precision_relaxation as f
p,t,c,d,x,ee=f.p,f.t,f.c,f.d,f.x,f.ee


def cast_model(torch,model,dtype):
    if isinstance(model,CoefficientOracle):
        return CoefficientOracle(torch,cast_model(torch,model.parent,dtype),model.us,model.zs)
    result=ee.clone_model(torch,model)
    if hasattr(result,'ca_head'):
        for child in (result.parent,result.ca_head,result.nap_head):child.to(dtype=dtype)
    else:result.to(dtype=dtype)
    return result


class MixedModel:
    """64-bit recurrent arithmetic, FP32 voltage-trunk coefficients.

    The small calcium-only SK head uses FP64 arithmetic with frozen FP32
    weights cast exactly, as its input is the new FP64 calcium.
    """
    def __init__(self,torch,model):self.torch,self.parent=torch,cast_model(torch,model,torch.float32)
    def __getattr__(self,name):
        if name=='parent':raise AttributeError(name)
        value=getattr(self.parent,name)
        return value.double() if name in ('sa','sb','sc','sd') else value
    def raw(self,v,ca):return tuple(r.to(v.dtype) for r in self.parent.raw(v.float(),ca.float()))
    def __call__(self,v,ca,g):
        u,z=self.raw(v,ca)
        return g+self.torch.sigmoid(z)*(self.torch.sigmoid(u)-g)


class CoefficientOracle:
    def __init__(self,torch,parent,us=(),zs=()):self.torch,self.parent,self.us,self.zs=torch,parent,tuple(us),tuple(zs)
    def __getattr__(self,name):
        if name=='parent':raise AttributeError(name)
        return getattr(self.parent,name)
    def raw(self,v,ca):
        u,z=self.parent.raw(v,ca);truth_u=[];truth_z=[]
        for channel,name in enumerate(c.ionic.CHANNELS):
            inf,tau=t._rates(name,v,ca,self.torch)
            if len(c.ionic.STATE_NAMES[channel])==1:inf,tau=inf[...,None],tau[...,None]
            truth_u.append(self.torch.logit(inf));truth_z.append(self.torch.logit(-self.torch.expm1(-.1/tau)))
        tu,tz=self.torch.cat(truth_u,-1),self.torch.cat(truth_z,-1)
        return tuple(self.torch.cat([truth[...,g:g+1] if g in gates else learned[...,g:g+1] for g in range(18)],-1)
            for truth,learned,gates in ((tu,u,self.us),(tz,z,self.zs)))
    def __call__(self,v,ca,g):
        u,z=self.raw(v,ca)
        return g+self.torch.sigmoid(z)*(self.torch.sigmoid(u)-g)


def typed_rollout(torch,model,data,cfg,calcium,base,device,policy='FP32',clamp_v=False,clamp_ca=False):
    dtype=torch.float32 if policy=='FP32' else torch.float64
    if model is not None:
        model=MixedModel(torch,model) if policy=='mixed_state64' else cast_model(torch,model,dtype)
    convert=lambda a:torch.tensor(a,device=device,dtype=dtype)
    expand=lambda a:convert(a)[None].expand(3,*a.shape).contiguous()
    args=[expand(data[k][:,0]) for k in ('voltage','calcium','gates')]
    args += [expand(data['multipliers']),expand(data['injection'][:,0]),expand(data['base_g_us'][:,0])]
    core=t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,t.FAMILIES['all'] if model is not None else ())
    runner=t.Stepper(torch,core,args,device=='cuda')
    inputs={k:convert(data[k]).transpose(0,1) for k in ('voltage','calcium','injection','base_g_us')}
    values=[[a.clone()] for a in args[:3]]+[[]]
    for tick in range(data['injection'].shape[1]):
        if clamp_v:runner.args[0].copy_(inputs['voltage'][tick][None].expand(3,-1))
        if clamp_ca:runner.args[1].copy_(inputs['calcium'][tick][None].expand(3,-1))
        out=runner.tick(inputs['injection'][tick][None].expand(3,-1),inputs['base_g_us'][tick][None].expand(3,-1,-1))
        for dest,value in zip(values,out):dest.append(value.clone())
    return {k:torch.stack(v).cpu().numpy() for k,v in zip(('voltage','calcium','gates','current'),values)}


def paired(pred,reference,meta,q,cases=(108,122)):
    return [dict(case=k,seed=seed,metrics=p.evaluate_intervention(reference[:,int(meta[k]['source_index'])],reference[:,k],
        pred[:,si,int(meta[k]['source_index'])],pred[:,si,k],.1,q)) for si,seed in enumerate((17,29,43)) for k in cases]


def signed_crossings(reference,prediction,dt=.1):
    base=reference[0];rr=np.abs(reference-base);pp=np.abs(prediction-base)
    threshold=.1*float(rr.max());ri=np.flatnonzero(rr>=threshold);pi=np.flatnonzero(pp>=threshold)
    if not len(ri) or not len(pi):return dict(evaluable=False)
    end=int(ri[-1]);lo=max(0,end-1);hi=min(len(rr)-1,end+1)
    return dict(evaluable=True,reference_offset_ms=end*dt,prediction_offset_ms=int(pi[-1])*dt,
        signed_offset_ms=(int(pi[-1])-end)*dt,threshold_ma_cm2=threshold,
        reference_crossing_slope=float((rr[hi]-rr[lo])/max((hi-lo)*dt,dt)),
        current_error_at_crossing=float(prediction[end]-reference[end]))


def frozen_audit(torch,root,output,controls,spec,cfg,calcium,base,device):
    folder=output/'frozen_audit';folder.mkdir()
    data,meta,stress=ee.load41c(root,spec)
    q=spec['quality'];expected=json.loads((root/spec['task41e_reproduction']).read_text())
    ref32=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,device,graph=device=='cuda')['current'][:,0]
    ref64=typed_rollout(torch,None,data,cfg,calcium,base,device,'FP64')['current'][:,0]
    numpy_ref=t.numpy_reference(data,cfg,calcium,base)['current']
    np.savez_compressed(folder/'references.npz',FP32=ref32,FP64=ref64,numpy=numpy_ref)
    names=('P110','D_C01','D_combined','D_fused','E_Q0','E_H10')
    rows=[];reproduction={};original=list(dict.fromkeys([int(meta[k]['source_index']) for k in (108,122)]+[108,122]))
    for name in names:
        for policy in ('FP32','mixed_state64','FP64'):
            pred=typed_rollout(torch,controls[name],data,cfg,calcium,base,device,policy)
            measurements={label:paired(pred['current'],ref,meta,q) for label,ref in [('official_FP32',ref32),('diagnostic_FP64',ref64),('diagnostic_numpy',numpy_ref)]}
            if policy=='FP32':
                actual=[r['metrics']['passed'] for r in measurements['official_FP32']]
                wanted=expected[name]
                assert actual==wanted,('Original failed-response reproduction',name,actual,wanted)
                reproduction[name]=dict(valid=True,passed=sum(actual),total=len(actual),full_original_batch=True)
            rows.append(dict(model=name,policy=policy,reference_comparisons=measurements))
            np.savez_compressed(folder/(name+'_'+policy+'.npz'),**{k:a[:,:,original] for k,a in pred.items()})
            p.save_json(folder/'precision.json',rows)
            print(f'[GIADA Task41f] checkpoint precision {name}/{policy}',flush=True)
    # Batch-size sensitivity: use the exact official reference, retaining the
    # original indices explicitly rather than silently changing the benchmark.
    subset={k:(v[original] if k!='area_um2' else float(v)) for k,v in data.items()}
    small_meta=[dict(meta[k],source_index=original.index(int(meta[k]['source_index']))) for k in original]
    small_ref=ref32[:,original];batch=[]
    for name in names:
        pred=typed_rollout(torch,controls[name],subset,cfg,calcium,base,device)
        batch.append(dict(model=name,original_ids=original,responses=paired(pred['current'],small_ref,small_meta,q,[original.index(k) for k in (108,122)])))
    p.save_json(folder/'batch_sensitivity.json',batch)
    clamps=[]
    for cv,cc in ((True,False),(False,True),(True,True)):
        ref=typed_rollout(torch,None,data,cfg,calcium,base,device,'FP32',cv,cc)['current'][:,0]
        for name in ('P110','D_combined','E_H10'):
            pred=typed_rollout(torch,controls[name],data,cfg,calcium,base,device,'FP32',cv,cc)
            clamps.append(dict(model=name,clamp_voltage=cv,clamp_calcium=cc,reference='FP32 canonical with identical clamps',responses=paired(pred['current'],ref,meta,q)))
    p.save_json(folder/'clamps.json',clamps)
    oracle_cases=[('u_m',(0,),()),('z_m',(),(0,)),('u_h',(1,),()),('z_h',(),(1,)),
                  ('u_Nap',(8,),()),('z_Nap',(),(8,)),('u_all',f.GATES,()),('z_all',(),f.GATES),('u_z_all',f.GATES,f.GATES)]
    oracles=[]
    # Long imposed slow return: isolate each coefficient at the same clock.
    design=d.prior.stress_design(json.loads((root/d.prior.CONFIG).read_text()))
    ids=[k for k,r in enumerate(design) if r['family']=='slow_boundary_return']
    st={k:(v[:,ids] if k in ('voltage','calcium','target_gates') else v[ids]) for k,v in stress.items()}
    truth=p.readout(st['voltage'],st['target_gates'],st['multipliers'],c.ionic.REVERSALS)[1]
    for name in ('P110','E_Q0','E_H10'):
        for label,us,zs in [('learned',(),())]+oracle_cases:
            model=controls[name] if label=='learned' else CoefficientOracle(torch,controls[name],us,zs)
            gates=c.j.i.old.Scanner(torch,[model],st['voltage'].shape[1],32,device).run(st['voltage'],st['calcium'],st['initial'].astype('float32'))[0]
            items=[]
            for si,seed in enumerate((17,29,43)):
                current=p.readout(st['voltage'],gates[:,si],st['multipliers'],c.ionic.REVERSALS)[1]
                for k,original_id in enumerate(ids):
                    items.append(dict(seed=seed,original_case=design[original_id],metrics=p.evaluate_currents(truth[:,k],current[:,k],.1,q,st['target_gates'][:,k],gates[:,si,k]),
                        Ca_crossing=signed_crossings(truth[:,k,0],current[:,k,0])))
            oracles.append(dict(model=name,oracle=label,selection_eligible=False,rows=items))
            p.save_json(folder/'coefficient_oracles.json',oracles)
        print(f'[GIADA Task41f] checkpoint coefficient oracles {name}',flush=True)
    p.save_json(folder/'reproduction.json',dict(valid=True,controls=reproduction,official_reference='FP32 canonical; unchanged',
        alternate_references_selection_eligible=False,opened_diagnostics_before_training=True,thresholds_changed=False))


def execution_matrix(torch,root,output,models,data,spec,cfg,calcium,base,device):
    from .ie_ionic_cost import paired_summary
    if device!='cuda':return []
    assert len(models)==1,'Pair one candidate with formula to bound whole-loop CUDA graph memory'
    group=next(iter(models));results=[]
    for batch in spec['cost']['batches']:
        ids=np.arange(batch)%len(data['voltage'])
        expand=lambda a:torch.tensor(a,device=device,dtype=torch.float32)[None].expand(3,*a.shape).contiguous()
        args=[expand(data[k][ids,100]) for k in ('voltage','calcium','gates')]
        inj=[expand(data['injection'][ids,100+k]) for k in range(spec['cost']['ticks'])]
        syn=[expand(data['base_g_us'][ids,100+k]) for k in range(spec['cost']['ticks'])]
        forced=[(expand(data['voltage'][ids,100+k]),expand(data['calcium'][ids,100+k])) for k in range(spec['cost']['ticks'])]
        args += [expand(data['multipliers'][ids]),inj[0],syn[0]]
        for physics in ('imposed','coupled'):
            samples={};parity={};runners={}
            for name,model in dict(formula=None,**models).items():
                core=t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,t.FAMILIES['all'] if model is not None else ())
                step=t.Stepper(torch,core,args,True)
                def run_step(step=step):
                    for k in range(len(inj)):step.tick(inj[k],syn[k],forced[k] if physics=='imposed' else None)
                run_step();expected=[a.clone() for a in step.out]
                whole=t.Stepper(torch,core,args,False)
                def run_whole(whole=whole):
                    for k in range(len(inj)):whole.tick(inj[k],syn[k],forced[k] if physics=='imposed' else None)
                run_whole();torch.cuda.synchronize();whole.reset(args)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):run_whole()
                whole.reset(args);graph.replay();torch.cuda.synchronize()
                error=max(float((a-b).abs().max()) for a,b in zip(expected,whole.out))
                assert error<=spec['computational_parity_atol'],('Graph granularity changed outputs',name,error)
                parity[name]=error
                runners[name+'/step']=(step,run_step)
                runners[name+'/whole']=(whole,graph.replay)
            for name in runners:samples[name]=dict(wall_ms=[],cuda_ms=[])
            for rep in range(spec['cost']['repetitions']):
                order=list(runners);order=order if rep%2==0 else order[::-1]
                for name in order:
                    runner,call=runners[name];runner.reset(args);torch.cuda.synchronize()
                    start=time.perf_counter();aa,bb=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    aa.record();call();bb.record();bb.synchronize()
                    samples[name]['wall_ms'].append(1000*(time.perf_counter()-start));samples[name]['cuda_ms'].append(aa.elapsed_time(bb))
            results.append(dict(candidate=group,batch_per_seed=batch,actual_sites=3*batch,physics=physics,samples=samples,parity=parity,
                comparisons={name:paired_summary(samples['formula/'+style]['wall_ms'],samples[name]['wall_ms']) for name in samples for style in [name.split('/')[-1]]},
                quality_qualified=False,thresholds_changed=False))
            p.save_json(output/('execution_matrix_'+group+'.json'),results)
            print(f'[GIADA Task41f] checkpoint execution {physics}/{batch}',flush=True)
    return results


def evaluate(torch,root,output,controls,trained,spec,cfg,calcium,base,device):
    freeze=json.loads((output/'selection_freeze.json').read_text())
    payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    assert not freeze['opened_regressions_used_for_selection'] and p.digest(output/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    models={**controls,**trained};allmodels={**models,'LUT_control':x.lut_model(torch,device,.1)}
    oldspec=json.loads((root/c.CONFIG).read_text())
    data,meta,st,_=ee.e.load_opened(root,oldspec)
    dd,mm,ss=ee.load41c(root,spec)
    reports={};stresses={}
    for label,inputs,metadata,stress,design in [('41',data,meta,st,x.stress_design(json.loads((root/x.CONFIG).read_text()))),
        ('41c',dd,mm,ss,d.prior.stress_design(json.loads((root/d.prior.CONFIG).read_text())))]:
        reports[label]=ee.old.suite(torch,output/('opened'+label),allmodels,inputs,metadata,spec,cfg,calcium,base,device)
        stresses[label]=ee.old.stress(torch,models,stress,spec,device,design)
        p.save_json(output/('stress'+label+'.json'),stresses[label])
    dev=d.development(spec['new_seeds']['development']);devrows={}
    target=p.readout(dev['voltage'],dev['target_gates'],dev['multipliers'],c.ionic.REVERSALS)[1]
    for name,model in models.items():
        gates=c.j.i.old.Scanner(torch,[model],dev['voltage'].shape[1],32,device).run(dev['voltage'],dev['calcium'],dev['initial'].astype('float32'))[0]
        devrows[name]=[dict(seed=seed,case=k,metrics=p.evaluate_currents(target[:,k],
            p.readout(dev['voltage'],gates[:,s],dev['multipliers'],c.ionic.REVERSALS)[1][:,k],.1,spec['quality'],dev['target_gates'][:,k],gates[:,s,k]))
            for s,seed in enumerate(spec['seeds']) for k in range(dev['voltage'].shape[1])]
    p.save_json(output/'development_gpu.json',devrows)
    hist=output/'historical_regression';hist.mkdir()
    historical=c.b.causal_evaluate(torch,root,hist,dict(B0=t.load_models(torch,root,device)['J1'],**models),spec,device)
    oldcfg=json.loads((root/c.j.i.old.CONFIG).read_text());evaluation={**json.loads((root/c.j.CONFIG).read_text()),'architectures':list(models)}
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=oldcfg[key]
    recurrence=[];items=list(models.items())
    for k in range(0,len(items),3):
        subset=items[k:k+3];dest=hist/f'recurrence_group_{k//3}';dest.mkdir()
        c.j.i.old.evaluate(root,dest,{**evaluation,'architectures':[n for n,v in subset]},[v for n,v in subset],torch,device,json.loads((root/p.CONFIG).read_text()))
        recurrence+=json.loads((dest/'recurrence_evaluation.json').read_text())
        p.save_json(hist/'recurrence_evaluation.json',recurrence)
        __import__('gc').collect();torch.cuda.empty_cache()
        print(f'[GIADA Task41f] checkpoint historical recurrence {k//3}',flush=True)
    summaries={};passing=[]
    for name in models:
        flat=[r['metrics'] for r in devrows[name]]
        summary=dict(development=dict(passed=sum(r['passed'] for r in flat),total=len(flat)),panels={},stress={},historical={},recurrence={})
        okay=all(r['passed'] for r in flat)
        for label in ('41','41c'):
            summary['panels'][label]={}
            for mode in ('imposed','autonomous'):
                rr=reports[label][name+'/'+mode];n=sum(r['metrics']['passed'] for r in rr['interventions'])
                summary['panels'][label][mode]=dict(trajectory=rr['passed_cases'],total=rr['total'],responses=n,response_total=len(rr['interventions']))
                okay &= rr['passed'] and n==len(rr['interventions'])
            sr=[r for r in stresses[label]['rows'] if r['model']==name]
            summary['stress'][label]=dict(passed=sum(r['metrics']['passed'] for r in sr),total=len(sr));okay &= all(r['metrics']['passed'] for r in sr)
        for mode in ('imposed','autonomous'):
            rr=historical[name+'/'+mode];n=sum(r['metrics']['passed'] for r in rr['interventions'])
            summary['historical'][mode]=dict(trajectory=rr['passed_cases'],total=rr['total'],responses=n)
            okay &= rr['passed'] and n==len(rr['interventions'])
        for role in ('development','stress_diagnostic'):
            rows=[r for r in recurrence if r['model']=='joint_w'+name and r['role']==role];assert len(rows)==72
            summary['recurrence'][role]=dict(passed=sum(r['metrics']['passed'] for r in rows),total=72);okay &= all(r['metrics']['passed'] for r in rows)
        summaries[name]=summary
        if okay and name in trained:passing.append(name)
    comparison=[]
    for axis,aa,bb in f.contrasts():
        a=next(k for k,r in enumerate(f.arms()) if r['name']==aa);b=next(k for k,r in enumerate(f.arms()) if r['name']==bb)
        rr=freeze['ranks'];ratios=[rr[b*3+s][2]/max(rr[a*3+s][2],1e-12) for s in range(3)]
        okay=all(rr[b*3+s][0]<=rr[a*3+s][0] and rr[b*3+s][1]<=rr[a*3+s][1] for s in range(3))
        comparison.append(dict(axis=axis,control=aa,treatment=bb,ratios=ratios,registered_contrast_passed=bool(okay and all(r<1 for r in ratios) and np.median(ratios)<=.8)))
    p.save_json(output/'paired_contrasts.json',comparison)
    ee.e.benchmark(torch,root,output,models,dd,spec,device)
    selected={k:models[k] for k in ('P110','E_Q0','E_H10',freeze['nominee'])};execution=[]
    for name,model in selected.items():
        execution+=execution_matrix(torch,root,output,{name:model},dd,spec,cfg,calcium,base,device)
        p.save_json(output/'execution_matrix.json',execution)
        __import__('gc').collect();torch.cuda.empty_cache()
    return dict(valid=True,task='41f',decision='REPAIR_REQUIRES_INDEPENDENT_CONFIRMATION' if passing else 'PRECISION_RELAXATION_NO_GO',
        passing_arms=passing,nominee=freeze['nominee'],summaries=summaries,thresholds_changed=False,
        independent_confirmation=False,task42_authorized=False,speedup_claim=False,
        original_reference_retained=True,opened_diagnostic_before_training=True,
        selection_uses_new_development_only=True,limits=spec['limits'])
