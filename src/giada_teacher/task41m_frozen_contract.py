"""Bounded full-panel verification of frozen selective repair and numerics.

No optimizer, selection, additional kinetic formula or altered scientific gate.
Numerical interventions are diagnostics, not a silent release-policy change.
"""
import gc,hashlib,json,os,time,types,zipfile
from pathlib import Path
import numpy as np
from . import task41l_frozen_attribution as a
l,he=a.l,a.he
p,t,c=l.p,l.t,l.c
CONFIG='experiments/task41m_preregistration_v1.json'
POLICIES={'primary':(False,False,False),'voltage_solve64':(True,True,False),'recentered32':(False,False,True)}

class BudgetReached(RuntimeError):pass

def contract(root):
    spec=json.loads((root/CONFIG).read_text());old,cfg,calcium,base=l.contract(root)
    for name in ('quality','compartment_gates','cost'):assert spec[name]==old[name],name
    assert spec['training_updates']==0 and spec['selection_performed'] is False
    assert spec['seeds']==old['seeds'] and spec['policies']==list(POLICIES)
    for name,sha in spec['sources'].items():assert p.digest(root/name)==sha,name
    return spec,old,cfg,calcium,base

def core(torch,model,physics,area,policy):
    state,solve,stable=POLICIES[policy]
    if model is not None:return he.precision_core(torch,model,*physics,area,state,solve,stable)
    if not stable:return a.canonical_core(torch,*physics,area,state,solve)
    fn=types.FunctionType(t.make_core.__code__,dict(t.make_core.__globals__,membrane=he.stable_membrane),argdefs=t.make_core.__defaults__)
    return fn(torch,*physics,area)

def rollout(torch,model,data,physics,device,policy,imposed=None):
    dtype=torch.float64 if POLICIES[policy][0] else torch.float32
    ex=lambda v,dt:torch.tensor(v,device=device,dtype=dt)[None].expand(3,*v.shape).contiguous()
    args=[ex(data['voltage'][:,0],dtype),ex(data['calcium'][:,0],torch.float32),ex(data['gates'][:,0],torch.float32),
          ex(data['multipliers'],torch.float32),ex(data['injection'][:,0],dtype),ex(data['base_g_us'][:,0],dtype)]
    runner=t.Stepper(torch,core(torch,model,physics,float(data['area_um2']),policy),args,device=='cuda')
    values=[[x.clone()] for x in args[:3]]+[[]]
    for tick in range(data['injection'].shape[1]):
        if imposed is not None:
            for index,key in enumerate(('voltage','calcium')):runner.args[index].copy_(ex(imposed[key][tick],args[index].dtype))
        result=runner.tick(ex(data['injection'][:,tick],dtype),ex(data['base_g_us'][:,tick],dtype))
        for dest,value in zip(values,result):dest.append(value.clone())
    return {k:torch.stack(v).cpu().numpy() for k,v in zip(('voltage','calcium','gates','current'),values)}

def measure(pred,ref,data,meta,spec,cfg,mode,panel):
    metric_ref={k:v[:,0].copy() for k,v in ref.items()}
    for key in ('voltage','calcium'):
        if panel=='historical':metric_ref[key][:,:32]=data[key][:32].T
        else:metric_ref[key]=data[key].T
    measured=t.metrics(pred,metric_ref,data,meta,spec['quality'],cfg['gates'],mode)
    measured['interventions']=[dict(case=k,seed=seed,metrics=p.evaluate_intervention(
        ref['current'][:,s,int(r['source_index'])],ref['current'][:,s,k],
        pred['current'][:,s,int(r['source_index'])],pred['current'][:,s,k],.1,spec['quality']))
        for s,seed in enumerate(spec['seeds']) for k,r in enumerate(meta) if r['intervention']!='none']
    return measured

def summary(metric):
    responses=metric['interventions']
    return dict(trajectory_passed=metric['passed_cases'],trajectory_total=metric['total'],
        responses_passed=sum(r['metrics']['passed'] for r in responses),responses_total=len(responses),
        failed_responses=[dict(case=r['case'],seed=r['seed']) for r in responses if not r['metrics']['passed']],
        passed=metric['passed'] and all(r['metrics']['passed'] for r in responses))

def primary_benchmark(torch,root,out,models,data,spec,device):
    # Preserve the original timer body; only rename its historical B6 comparator
    # to the actual frozen H control (swap000) present in this experiment.
    import ast,sys
    sys.path.insert(0,str(root/'scripts'))
    from task41h_stage_cache import compile_function
    def rename(fn):
        count=0
        for node in ast.walk(fn):
            if isinstance(node,ast.Constant) and node.value=='B6':node.value='swap000';count+=1
        assert count==2
    return compile_function(he.ee.e.benchmark,rename)(torch,root,out,models,data,spec,device)

def cost(torch,models,data,physics,spec,device,save,check):
    from .ie_ionic_cost import paired_summary
    rows=[]
    for batch in spec['cost']['batches']:
        check();ix=np.arange(batch)%64;samples={};runners={};inputs={}
        ex=lambda v,dt:torch.tensor(v,device=device,dtype=dt)[None].expand(3,*v.shape).contiguous()
        for policy in POLICIES:
            dt=torch.float64 if POLICIES[policy][0] else torch.float32
            args=[ex(data['voltage'][ix,100],dt),ex(data['calcium'][ix,100],torch.float32),ex(data['gates'][ix,100],torch.float32),
                  ex(data['multipliers'][ix],torch.float32),ex(data['injection'][ix,100],dt),ex(data['base_g_us'][ix,100],dt)]
            ins=[(ex(data['injection'][ix,100+k],dt),ex(data['base_g_us'][ix,100+k],dt)) for k in range(spec['cost']['ticks'])]
            for name,model in {'formula':None,**models}.items():
                key=policy+'/'+name;runners[key]=t.Stepper(torch,core(torch,model,physics,float(data['area_um2']),policy),args,True)
                inputs[key]=(args,ins);samples[key]=[]
        keys=list(runners)
        for repeat in range(spec['cost']['repetitions']):
            check()
            for key in keys if repeat%2==0 else reversed(keys):
                args,ins=inputs[key];run=runners[key];run.reset(args);torch.cuda.synchronize();start=time.perf_counter()
                for inj,syn in ins:run.tick(inj,syn)
                torch.cuda.synchronize();samples[key].append(1000*(time.perf_counter()-start))
        row=dict(batch_per_seed=batch,actual_sites=3*batch,physics='integrated',samples=samples,
            versus_original_formula={key:paired_summary(samples['primary/formula'],value) for key,value in samples.items()},
            versus_same_policy_formula={key:paired_summary(samples[key.split('/')[0]+'/formula'],value) for key,value in samples.items()},
            quality_qualified=False,diagnostic_two_network_transplant=True)
        rows.append(row);save('numerical_cost.json',rows)
        del runners;gc.collect();torch.cuda.empty_cache()
    return rows

def run(root,out,device='cuda',preflight=False):
    import torch
    start=time.monotonic();spec,old,cfg,calcium,base=contract(root);physics=(cfg,calcium,base)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert device!='cuda' or (torch.cuda.is_available() and torch.cuda.get_device_name()==spec['gpu'] and torch.__version__==spec['torch'])
    out.mkdir(parents=True,exist_ok=True)
    save=lambda name,value:p.save_json(out/name,a.fe.json_native(value))
    def check():
        if time.monotonic()-start>spec['max_gpu_seconds']:raise BudgetReached('Registered60-minute frozen-evaluation cap reached; partial evidence preserved, not scientific NO-GO.')
    controls=l.j.controls(torch,root,device);swaps,_=a.source(torch,root,controls,old,device)
    models={name:swaps[name] for name in spec['models']}
    for model in models.values():
        model.eval()
        for param in model.parameters():param.requires_grad_(False);assert torch.isfinite(param).all()
    if preflight:
        fake=dict(voltage=np.full((4,6),-65.,dtype='float32'),calcium=np.full((4,6),1e-4,dtype='float32'),
            gates=np.full((4,6,18),.01,dtype='float32'),multipliers=np.ones((4,11),dtype='float32'),
            injection=np.full((4,5),.001,dtype='float32'),base_g_us=np.full((4,5,4),1e-5,dtype='float32'),area_um2=np.array(314.1592653589793))
        with torch.inference_mode():
            canonical=rollout(torch,None,fake,physics,device,'primary')
            oldpred=a.rollout(torch,None,fake,physics,device,(False,False))
            assert all(np.array_equal(canonical[k],oldpred[k]) for k in canonical)
            for policy in POLICIES:
                for model in [None,*models.values()]:
                    pred=rollout(torch,model,fake,physics,device,policy)
                    assert all(np.isfinite(x).all() for x in pred.values())
            meta=[dict(id=k,source_index=0,intervention='none' if k==0 else 'test') for k in range(4)]
            measure(pred,canonical,fake,meta,spec,cfg,'autonomous','41')
            assert all(torch.equal(x,y) for x,y in zip(models['swap000'].raw(torch.full((3,4),-65.,device=device),torch.full((3,4),1e-4,device=device)),controls['H_nominee'].raw(torch.full((3,4),-65.,device=device),torch.full((3,4),1e-4,device=device))))
            if device=='cuda':
                fake_cost=dict(fake)
                for key in ('voltage','calcium','gates','injection','base_g_us'):
                    fake_cost[key]=np.repeat(fake[key][:1,:1],64,axis=0).repeat(103,axis=1)
                fake_cost['multipliers']=np.ones((64,11),dtype='float32')
                small=dict(spec,cost=dict(spec['cost'],batches=[1],ticks=2,repetitions=2))
                primary_benchmark(torch,root,out,models,fake_cost,small,device)
                cost(torch,models,fake_cost,physics,small,device,save,lambda:None)
        result=dict(valid=True,gpu=torch.cuda.get_device_name() if device=='cuda' else 'cpu',torch=torch.__version__,all_parameters_finite=True,
            protocol_sha256=p.digest(root/CONFIG),training_updates=0,canonical_primary_parity=True,all_policy_graphs_exercised=True)
        save('gpu_preflight.json',result);return result
    package=Path(os.environ['GIADA_TASK41M_REUSE']);assert p.digest(package)==spec['reuse']['sha256']
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for name,sha in spec['reuse']['members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
        reused={name:json.loads(z.read(name)) for name in spec['reuse']['members']}
    save('reused_reports.json',dict(provenance=spec['reuse'],reports=reused))
    save('preregistration.json',spec)
    torch.save({name:model.state_dict() for name,model in models.items()},out/'frozen_models.pt')
    aa,am,_,_=he.ee.e.load_opened(root,json.loads((root/c.CONFIG).read_text()))
    bb,bm,_=he.ee.load41c(root,old)
    historical,hm=t.design(root,t.contract(root)[0]);panels=[('41',aa,am),('41c',bb,bm),('historical',historical,hm)]
    results=[];floors=[]
    with torch.inference_mode():
        for panel,data,meta in panels:
            for mode in ('autonomous','imposed'):
                check();imposed=None
                golden=t.numpy_reference(data,cfg,calcium,base) if mode=='autonomous' or panel=='historical' else a.fe.x.numpy_imposed(data,cfg,calcium,base)
                if mode=='imposed':
                    if panel=='historical':
                        imposed={k:golden[k].copy() for k in ('voltage','calcium')}
                        for k in imposed:imposed[k][:,:32]=data[k][:32].T
                    else:imposed={k:data[k].T for k in ('voltage','calcium')}
                policies=list(POLICIES) if mode=='autonomous' else ['primary']
                refs={}
                for policy in policies:
                    check();ref=rollout(torch,None,data,physics,device,policy,imposed);refs[policy]=ref
                    np.savez_compressed(out/f'reference_{panel}_{mode}_{policy}.npz',**ref)
                    # Original full numerical-floor test for autonomous; imposed
                    # historical reference is handled by the original suite semantics.
                    floor=[]
                    if mode=='autonomous' or panel!='historical':
                        for case in range(len(meta)):
                            floor.append(p.evaluate_currents(golden['current'][:,case],ref['current'][:,0,case],.1,spec['quality'],golden['gates'][1:,case],ref['gates'][1:,0,case]))
                        for case,row in enumerate(meta):
                            if row['intervention']!='none':
                                src=int(row['source_index']);floor.append(p.evaluate_intervention(golden['current'][:,src],golden['current'][:,case],ref['current'][:,0,src],ref['current'][:,0,case],.1,spec['quality']))
                    floors.append(dict(panel=panel,mode=mode,policy=policy,metrics=floor,passed=bool(floor) and all(r['passed'] for r in floor),historical_imposed_floor_not_recomputed=not floor))
                    save('canonical_floors.json',floors)
                    for name,model in models.items():
                        check();pred=rollout(torch,model,data,physics,device,policy,imposed)
                        assert all(np.isfinite(v).all() for v in pred.values()),(panel,policy,name)
                        np.savez_compressed(out/f'traces_{panel}_{mode}_{policy}_{name}.npz',**pred)
                        for reference in ['primary']+([] if policy=='primary' else [policy]):
                            measured=measure(pred,refs[reference],data,meta,spec,cfg,mode,panel)
                            if name=='swap000' and policy==reference=='primary' and mode=='autonomous':
                                expected=spec['control_failures'][panel]
                                assert summary(measured)['failed_responses']==expected,(panel,'H control differs',summary(measured))
                                assert measured['passed_cases']==len(meta)*3,'H trajectory control differs'
                            row=dict(panel=panel,mode=mode,model=name,policy=policy,reference=reference,summary=summary(measured),
                                continuous=a.continuous(pred,refs[reference]),metrics=measured,selection_eligible=False)
                            save(f'{panel}_{mode}_{policy}_{name}_vs_{reference}.json',row)
                            results.append({k:v for k,v in row.items() if k!='metrics'})
                        save('progress_summary.json',results)
                        print(f'[GIADA Task41m] {panel}/{mode}/{policy}/{name}: '+str(results[-1]['summary']),flush=True)
                        del pred;gc.collect();torch.cuda.empty_cache()
            del refs
        # Same primary ionic/integrated cost protocol; variants are diagnostics.
        check();primary_cost=primary_benchmark(torch,root,out,models,bb,spec,device)
        numeric_cost=cost(torch,models,bb,physics,spec,device,save,check) if device=='cuda' else []
    for model in models.values():assert all(torch.isfinite(v).all() for v in model.parameters())
    # Report the evidence, never silently qualify a diagnostic against new policy.
    result=dict(valid=True,complete=True,task='41m',decision='FROZEN_DIAGNOSTIC_COMPLETE_REQUIRES_AUDIT',
        summaries=results,all_parameters_finite=True,training_updates_performed=0,selection_performed=False,
        thresholds_changed=False,abstraction_changed=False,task42_authorized=False,speedup_claim=False,
        independent_confirmation=False,elapsed_seconds=time.monotonic()-start,compression_tested=False,
        original_reference_retained=True,canonical_floor_passed=all(r['passed'] for r in floors if not r['historical_imposed_floor_not_recomputed']),
        full_qualification_claimed=False,limits=spec['limits'])
    save('final_report.json',result);return result
