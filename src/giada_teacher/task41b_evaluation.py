"""Frozen diagnostics for Task41b, including all opened regression panels."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from . import task41b_local_precision as c
p,t,x=c.p,c.t,c.prior

def load_opened(root,spec):
    from pathlib import Path
    import os
    path=Path(os.environ.get('GIADA_TASK41_INPUTS','/content/task41b_opened_inputs.zip'))
    assert p.digest(path)==spec['opened_inputs_sha256'],'Opened input bundle changed'
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,sha in spec['opened_members'].items():
            import hashlib
            assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
        arrays=lambda n:dict(np.load(io.BytesIO(z.read(n))))
        return arrays('native_shadow_traces.npz'),json.loads(z.read('cases.json')),arrays('fresh_stress_traces.npz'),json.loads(z.read('causal_results.json'))

def stress(torch,models,data,spec,device):
    q=spec['quality'];v,ca=data['voltage'],data['calcium'];target=p.readout(v,data['target_gates'],data['multipliers'],c.ionic.REVERSALS)[1]
    meta=x.stress_design(json.loads((Path(__file__).resolve().parents[2]/x.CONFIG).read_text()))
    result=[]
    # Canonical FP32 stress floor, required separately from the short coupled floor.
    u,z=p.coefficients(v.astype('float64'),ca.astype('float64'),.1);g=data['initial'].astype('float32');formula=[]
    for tick in range(len(v)):g=g+z[tick].astype('float32')*(u[tick].astype('float32')-g);formula.append(g.copy())
    formula=np.stack(formula);curr=p.readout(v,formula,data['multipliers'],c.ionic.REVERSALS)[1]
    floor=[p.evaluate_currents(target[:,k],curr[:,k],.1,q,data['target_gates'][:,k],formula[:,k]) for k in range(v.shape[1])]
    assert all(r['passed'] for r in floor),'Task41b FP32 stress formula floor failed'
    for name,model in models.items():
        gates=c.j.i.old.Scanner(torch,[model],v.shape[1],32,device).run(v,ca,data['initial'].astype('float32'))[0]
        for s,seed in enumerate(spec['seeds']):
            current=p.readout(v,gates[:,s],data['multipliers'],c.ionic.REVERSALS)[1]
            for k,row in enumerate(meta):
                result.append(dict(model=name,seed=seed,input_seed=row['seed'],family=row['family'],case=row['case'],
                    metrics=p.evaluate_currents(target[:,k],current[:,k],.1,q,data['target_gates'][:,k],gates[:,s,k])))
    return dict(formula_floor=floor,rows=result)

def benchmark(torch,root,output,models,data,spec,device):
    import time
    from .ie_ionic_cost import paired_summary
    _,cfg,calcium,base=x.contract(root);cfg=x.calibration_config(json.loads((root/x.CONFIG).read_text()),cfg)
    lut=x.lut_model(torch,device,.1);allmodels=dict(formula=None,LUT_control=lut,**models);results=[]
    for batch in spec['cost']['batches']:
        ids=np.arange(batch)%64
        expand=lambda a:torch.tensor(a,device=device,dtype=torch.float32)[None].expand(3,*a.shape).contiguous()
        args=[expand(data[k][ids,100]) for k in ('voltage','calcium','gates')]
        inj=[expand(data['injection'][ids,100+k]) for k in range(spec['cost']['ticks'])]
        syn=[expand(data['base_g_us'][ids,100+k]) for k in range(spec['cost']['ticks'])]
        args += [expand(data['multipliers'][ids]),inj[0],syn[0]]
        for scope in ('ionic','integrated'):
            runners={}
            for name,m in allmodels.items():
                channels=() if name=='formula' else (tuple(k for k in range(11) if k!=9) if name=='LUT_control' else t.FAMILIES['all'])
                runners[name]=t.Stepper(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),m,channels,scope=='ionic'),args,True)
            samples={name:dict(wall_ms=[],cuda_ms=[]) for name in runners}
            for rep in range(spec['cost']['repetitions']):
                for name in list(runners) if rep%2==0 else list(runners)[::-1]:
                    runner=runners[name];runner.reset(args);torch.cuda.synchronize();start=time.perf_counter()
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                    for tick in range(len(inj)):runner.tick(inj[tick],syn[tick])
                    b.record();b.synchronize();samples[name]['wall_ms'].append((time.perf_counter()-start)*1000);samples[name]['cuda_ms'].append(a.elapsed_time(b))
            results.append(dict(batch_per_seed=batch,actual_sites=3*batch,scope=scope,samples=samples,
                comparisons={name:{base:paired_summary(samples[base]['wall_ms'],samples[name]['wall_ms']) for base in ('formula','LUT_control','B6')} for name in models if name!='B6'},
                finite={name:all(bool(torch.isfinite(a).all()) for a in r.out) for name,r in runners.items()},quality_qualified=False))
            p.save_json(output/'cost.json',results)
            print(f'[GIADA Task41b] checkpoint cost {scope} batch={batch}',flush=True)
    return results

def evaluate(torch,root,output,models,spec,device):
    data,meta,st,expected=load_opened(root,spec);_,cfg,calcium,base=x.contract(root);cfg=x.calibration_config(json.loads((root/x.CONFIG).read_text()),cfg)
    q=spec['quality'];reports={};refs={};imposed={k:data[k].T for k in ('voltage','calcium')}
    dev,_=c.development(spec['development_seed']);devtarget=p.readout(dev['voltage'],dev['target_gates'],dev['multipliers'],c.ionic.REVERSALS)[1]
    devrows={}
    for name,model in models.items():
        gates=c.j.i.old.Scanner(torch,[model],dev['voltage'].shape[1],32,device).run(dev['voltage'],dev['calcium'],dev['initial'].astype('float32'))[0]
        rows=[]
        for s,seed in enumerate(spec['seeds']):
            curr=p.readout(dev['voltage'],gates[:,s],dev['multipliers'],c.ionic.REVERSALS)[1]
            for k in range(dev['voltage'].shape[1]):rows.append(dict(seed=seed,case=k,metrics=p.evaluate_currents(devtarget[:,k],curr[:,k],.1,q,dev['target_gates'][:,k],gates[:,s,k])))
        devrows[name]=rows;p.save_json(output/'development_gpu.json',devrows)
    print('[GIADA Task41b] checkpoint frozen development CUDA complete',flush=True)
    for mode in ('imposed','autonomous'):
        rr=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,device,imposed if mode=='imposed' else None,True)
        ref={k:v[:,0] for k,v in rr.items()};refs[mode]=ref
        target=x.numpy_imposed(data,cfg,calcium,base) if mode=='imposed' else t.numpy_reference(data,cfg,calcium,base)
        floor=[p.evaluate_currents(target['current'][:,k],ref['current'][:,k],.1,q,target['gates'][1:,k],ref['gates'][1:,k],data['multipliers'][k]==0) for k in range(len(meta))]
        assert all(r['passed'] for r in floor),'Opened41 numerical formula floor failed'
        p.save_json(output/f'formula_floor_{mode}.json',floor)
        if mode=='imposed':
            ng,nc=x.native_phase_targets(data)
            phase=[p.evaluate_currents(nc[:,k],ref['current'][:,k],.1,q,ng[:,k],ref['gates'][1:,k],data['multipliers'][k]==0) for k in range(len(meta))]
            assert all(r['passed'] for r in phase);p.save_json(output/'native_phase_floor.json',phase)
        for name,model in models.items():
            candidate=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,t.FAMILIES['all']),data,device,imposed if mode=='imposed' else None,True)
            metricref={**ref,**{k:data[k].T for k in ('voltage','calcium')}}
            report=t.metrics(candidate,metricref,data,meta,q,cfg['gates'],mode);inter=[]
            for s,seed in enumerate(spec['seeds']):
                for k,row in enumerate(meta):
                    if row['intervention']=='none':continue
                    o=row['source_index'];inter.append(dict(case=k,seed=seed,metrics=p.evaluate_intervention(ref['current'][:,o],ref['current'][:,k],candidate['current'][:,s,o],candidate['current'][:,s,k],.1,q)))
            report['interventions']=inter;reports[name+'/'+mode]=report
            if name=='B6':
                old=expected[name+'/'+mode]
                assert [r['passed'] for r in old['cases']]==[r['passed'] for r in report['cases']],'B6 Task41 reproduction changed'
                assert [r['metrics']['passed'] for r in old['interventions']]==[r['metrics']['passed'] for r in inter]
            # Selected current/gate traces remain auditable, one compact file per arm/mode.
            np.savez_compressed(output/f'traces_{name}_{mode}.npz',**candidate)
            p.save_json(output/'causal_results.json',reports)
            print(f'[GIADA Task41b] checkpoint opened41 {name}/{mode} {report["passed_cases"]}/528 responses={sum(r["metrics"]["passed"] for r in inter)}/336',flush=True)
    np.savez_compressed(output/'reference_traces.npz',**{a+'__'+k:v for a,r in refs.items() for k,v in r.items()})
    sr=stress(torch,models,st,spec,device);p.save_json(output/'stress_results.json',sr)
    assert sum(r['metrics']['passed'] for r in sr['rows'] if r['model']=='B6')==33
    print('[GIADA Task41b] checkpoint opened41 stress complete',flush=True)
    # Preserve every historical gate, not only the three newly implicated ones.
    hist=output/'historical_regression';hist.mkdir()
    baseline=t.load_models(torch,root,device)['J1']
    oldreports=c.b.causal_evaluate(torch,root,hist,dict(B0=baseline,**models),spec,device)
    oldcfg=json.loads((root/c.j.i.old.CONFIG).read_text());evaluation=dict(json.loads((root/c.j.CONFIG).read_text()),architectures=list(models))
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=oldcfg[key]
    oldsummary=c.j.i.old.evaluate(root,hist,evaluation,list(models.values()),torch,device,json.loads((root/p.CONFIG).read_text()))
    print('[GIADA Task41b] checkpoint historical regression complete',flush=True)
    cost=benchmark(torch,root,output,models,data,spec,device)
    summary={};passing=[]
    for name in models:
        summary[name]=dict(new_development_passed=sum(r['metrics']['passed'] for r in devrows[name]),new_development_total=len(devrows[name]),opened41={mode:dict(trajectory=reports[name+'/'+mode]['passed_cases'],total=528,
            responses=sum(r['metrics']['passed'] for r in reports[name+'/'+mode]['interventions']),response_total=336) for mode in ('imposed','autonomous')},
            stress_passed=sum(r['metrics']['passed'] for r in sr['rows'] if r['model']==name),stress_total=36,
            historical={mode:dict(trajectory=oldreports[name+'/'+mode]['passed_cases'],responses=sum(r['metrics']['passed'] for r in oldreports[name+'/'+mode]['interventions'])) for mode in ('imposed','autonomous')},
            old_recurrence=oldsummary['joint_w'+name])
        if name!='B6' and all(r['metrics']['passed'] for r in devrows[name]) and all(reports[name+'/'+m]['passed'] and all(r['metrics']['passed'] for r in reports[name+'/'+m]['interventions']) and oldreports[name+'/'+m]['passed'] and all(r['metrics']['passed'] for r in oldreports[name+'/'+m]['interventions']) for m in ('imposed','autonomous')) and summary[name]['stress_passed']==36:
            rr=json.loads((hist/'recurrence_evaluation.json').read_text());rr=[r for r in rr if r['model']=='joint_w'+name]
            if all(r['metrics']['passed'] for r in rr):passing.append(name)
    contrasts=[];freeze=json.loads((output/'selection_freeze.json').read_text());selected=freeze['ranks']
    for axis in ('support','basis','conditioned'):
        for a in c.arms():
            for b in c.arms():
                others=set(('support','basis','conditioned'))-{axis}
                if all(a[k]==b[k] for k in others) and ((axis=='support' and a[axis]=='uniform' and b[axis]=='sensitive') or (axis=='basis' and a[axis]==32 and b[axis]==64) or (axis=='conditioned' and not a[axis] and b[axis])):
                    ai=c.arms().index(a)*3;bi=c.arms().index(b)*3
                    ratios=[selected[bi+s][1]/max(selected[ai+s][1],1e-12) for s in range(3)]
                    contrasts.append(dict(axis=axis,control=a['name'],treatment=b['name'],worst_complete_development_ratios=ratios,
                        improved_all_seeds=all(r<1 for r in ratios),interpretation='Descriptive paired effect; inspect failed-case counts and cost. No stress-driven checkpoint selection.'))
    p.save_json(output/'paired_contrasts.json',contrasts)
    return dict(valid=True,task='41b',decision='LOCAL_PRECISION_CANDIDATES_REQUIRE_AUDIT' if passing else 'LOCAL_PRECISION_DIAGNOSTIC_NO_GO',
        passing_arms=passing,summaries=summary,baseline_reproduced=True,thresholds_changed=False,
        independent_confirmation=False,task42_authorized=False,speedup_claim=False,
        limits=spec['limits'])
