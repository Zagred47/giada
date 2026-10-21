"""Independent confirmation of frozen P110, not another model selection.

Native and CUDA phases run in separate processes. The scoped adapter reuses
the audited Task41 native interface without editing its immutable source.
"""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from . import task41b_local_precision as local
from . import task41b_evaluation as evaluation

x,p,t=local.prior,local.p,local.t
CONFIG='experiments/task41c_preregistration_v1.json'

def contract(root):
    spec=json.loads((root/CONFIG).read_text())
    old,cfg,calcium,base=local.contract(root)
    assert spec['current_quality']==old['quality']
    assert spec['compartment_gates']==old['compartment_gates']
    assert spec['candidate']=='P110' and spec['model_seeds']==[17,29,43]
    assert spec['dt_ms']==.1 and spec['training_performed'] is False
    for name,sha in spec['sources'].items(): assert p.digest(root/name)==sha,name
    assert not set(spec['native_seeds']+spec['stress_seeds']+[spec['calibration_seed']]) & set(spec['excluded_input_seeds'])
    return spec,cfg,calcium,base

def injection(protocol,clock):
    pulses={
        'calibration':[(2.,5.,.003)],
        'asymmetric_doublet':[(5.,7.2,.0095),(13.,16.4,.0105)],
        'rebound_triplet':[(1.5,6.5,-.0032),(8.5,10.2,.010),(13.2,15.,.0105),(18.,19.,.0095)],
        'reversed_staircase':[(3.5,6.5,.009),(6.5,9.5,.006),(9.5,12.5,.003),(16.,18.,.010)],
        'irregular_train':[(4.,4.7,.011),(6.3,7.2,.0105),(9.7,10.3,.0115),(14.5,15.3,.0105)]}
    out=np.zeros_like(clock)
    for a,b,amp in pulses[protocol]:out[(clock>=a)&(clock<b)]=amp
    return out

def design(spec):
    return x.design(spec)

def case_inputs(row,spec):
    clock=np.arange(round(spec['duration_ms']/spec['dt_ms']))*spec['dt_ms']
    inj=injection(row['protocol'],clock);mult=np.array(spec['panels'][row['panel']],dtype=float)
    action=row['intervention']
    if action.startswith('block_'):mult[local.ionic.CHANNELS.index(action[6:])]=0
    if action.startswith('injection_'):inj*=float(action.split('_')[1])
    return inj,mult

@contextmanager
def native_adapter(root):
    prepared=contract(root)
    old_contract,old_inputs=x.contract,x.case_inputs
    x.contract=lambda unused:prepared
    x.case_inputs=case_inputs
    try:yield
    finally:x.contract,x.case_inputs=old_contract,old_inputs

def native(root,teacher,output):
    with native_adapter(root):x.native(root,teacher,output)

def load_models(torch,root,device):
    spec,_,_,_=contract(root)
    parent=x.load_b6(torch,root,device)
    definition=next(a for a in local.arms() if a['name']=='P110')
    model=local.make_model(torch,parent,definition,spec['model_seeds'],device)
    state=torch.load(root/spec['checkpoint'],map_location=device,weights_only=True)
    model.load_state_dict(state,strict=True);model.eval()
    for par in model.parameters():par.requires_grad_(False);assert torch.isfinite(par).all()
    return dict(B6=parent,P110=model)

def stress_design(spec):
    clock=np.arange(round(spec['stress_duration_ms']/.1))*.1;rows=[]
    for family in ('doublepulse_recovery','slow_boundary_return'):
        for seed in spec['stress_seeds']:
            rng=np.random.default_rng(seed+(0 if family=='doublepulse_recovery' else 2000))
            rest=rng.uniform(-65,-55) if family=='doublepulse_recovery' else rng.uniform(-48,-37)
            v=np.full(len(clock),rest);ca=np.full(len(clock),10**rng.uniform(-4.4,-3.6))
            stop=rng.uniform(13,23);v[clock<stop]=rng.uniform(-31,-14)
            if family=='doublepulse_recovery':v[(clock>=90)&(clock<90+rng.uniform(8,18))]=rng.uniform(-35,-18)
            else:
                ix=(clock>=stop)&(clock<stop+30)
                v[ix]=np.linspace(v[max(0,np.flatnonzero(ix)[0]-1)],rest,ix.sum())
            for k,initial_v in enumerate((-95.,-42.)):
                initial,_=p.coefficients(np.array(initial_v),np.array(ca[0]),.1)
                panel=list(spec['panels'])[(k+spec['stress_seeds'].index(seed))%4]
                rows.append(dict(family=family,seed=seed,case=k,panel=panel,v=v.copy(),ca=ca.copy(),initial=initial))
    return rows

def stress(torch,models,spec,device,output):
    rows=stress_design(spec);v=np.stack([r['v'] for r in rows],1).astype('float32');ca=np.stack([r['ca'] for r in rows],1).astype('float32')
    initial=np.stack([r['initial'] for r in rows]);u,z=p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    g=initial.copy();truth=[]
    for tick in range(len(v)):g=g+z[tick]*(u[tick]-g);truth.append(g.copy())
    data=dict(voltage=v,calcium=ca,initial=initial,target_gates=np.stack(truth),multipliers=np.array([spec['panels'][r['panel']] for r in rows]))
    old=x.stress_design;x.stress_design=lambda unused:rows
    try:report=evaluation.stress(torch,models,data,{**spec,'quality':spec['current_quality'],'seeds':spec['model_seeds']},device)
    finally:x.stress_design=old
    np.savez_compressed(output/'fresh_stress_traces.npz',**data)
    p.save_json(output/'fresh_stress_results.json',report)
    return report

def smoke(torch,root,output,device):
    models=load_models(torch,root,device);_,cfg,calcium,base=contract(root)
    v=torch.linspace(-85,35,32,device=device)[None].expand(3,-1).contiguous();ca=torch.full_like(v,1e-4);g=torch.full((3,32,18),.4,device=device)
    out=models['P110'](v,ca,g);assert torch.isfinite(out).all()
    graph_error=0.
    if device=='cuda':
        for _ in range(3):models['P110'](v,ca,g)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):actual=models['P110'](v,ca,g)
        graph.replay();torch.cuda.synchronize();graph_error=float((actual-out).abs().max())
        assert graph_error<=1e-5
    report=dict(valid=True,all_parameters_finite=True,fresh_confirmation_accessed=False,training_performed=False,
                graph_eager_max_error=graph_error,device=device,checkpoint_sha256=p.digest(root/contract(root)[0]['checkpoint']))
    p.save_json(output/'gpu_preflight.json',report);return report

def evaluate(torch,root,output,device='cuda'):
    spec,cfg,calcium,base=contract(root);cfg=x.calibration_config(spec,cfg)
    assert json.loads((output/'native_interface_report.json').read_text())['valid']
    models=load_models(torch,root,device);models['LUT_control']=x.lut_model(torch,device,.1)
    with np.load(output/'native_shadow_traces.npz') as z:data=dict(z)
    meta=design(spec);reports={};refs={};floors={}
    for mode in ('imposed','autonomous'):
        imposed={k:data[k].T for k in ('voltage','calcium')} if mode=='imposed' else None
        rr=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,device,imposed,True)
        ref={k:v[:,0] for k,v in rr.items()};refs[mode]=ref
        target=x.numpy_imposed(data,cfg,calcium,base) if mode=='imposed' else t.numpy_reference(data,cfg,calcium,base)
        floor=[p.evaluate_currents(target['current'][:,k],ref['current'][:,k],.1,spec['current_quality'],target['gates'][1:,k],ref['gates'][1:,k],data['multipliers'][k]==0) for k in range(len(meta))]
        floor_inter=[]
        for k,row in enumerate(meta):
            if row['intervention']!='none':
                o=row['source_index'];floor_inter.append(p.evaluate_intervention(target['current'][:,o],target['current'][:,k],ref['current'][:,o],ref['current'][:,k],.1,spec['current_quality']))
        floors[mode]=dict(cases=floor,interventions=floor_inter)
        p.save_json(output/'formula_floor.json',floors)
        assert all(r['passed'] for r in floor+floor_inter),'Formula32/64 floor failed; candidate not judged'
        if mode=='imposed':
            ng,nc=x.native_phase_targets(data)
            phase=[p.evaluate_currents(nc[:,k],ref['current'][:,k],.1,spec['current_quality'],ng[:,k],ref['gates'][1:,k],data['multipliers'][k]==0) for k in range(len(meta))]
            p.save_json(output/'native_current_phase_floor.json',phase)
            assert all(r['passed'] for r in phase),'Native phase floor failed; candidate not judged'
        for name,model in models.items():
            channels=tuple(k for k in range(11) if k!=9) if name=='LUT_control' else t.FAMILIES['all']
            prediction=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,channels),data,device,imposed,True)
            metricref={**ref,**{k:data[k].T for k in ('voltage','calcium')}}
            report=t.metrics(prediction,metricref,data,meta,spec['current_quality'],cfg['gates'],mode);inter=[]
            for s,seed in enumerate(spec['model_seeds']):
                for k,row in enumerate(meta):
                    if row['intervention']=='none':continue
                    o=row['source_index'];inter.append(dict(case=k,seed=seed,metrics=p.evaluate_intervention(ref['current'][:,o],ref['current'][:,k],prediction['current'][:,s,o],prediction['current'][:,s,k],.1,spec['current_quality'])))
            report['interventions']=inter;reports[name+'/'+mode]=report
            p.save_json(output/'causal_results.json',reports)
            np.savez_compressed(output/f'traces_{name}_{mode}.npz',**prediction)
            print(f'[GIADA Task41c] checkpoint {name}/{mode} trajectory={report["passed_cases"]}/528 responses={sum(r["metrics"]["passed"] for r in inter)}/336',flush=True)
    np.savez_compressed(output/'reference_traces.npz',**{a+'__'+k:v for a,r in refs.items() for k,v in r.items()})
    sr=stress(torch,{k:v for k,v in models.items() if k!='LUT_control'},spec,device,output)
    print('[GIADA Task41c] checkpoint fresh stress complete',flush=True)
    cost=evaluation.benchmark(torch,root,output,{k:v for k,v in models.items() if k!='LUT_control'},data,spec,device)
    summary={k:dict(trajectory=v['passed_cases'],total=v['total'],responses=sum(r['metrics']['passed'] for r in v['interventions']),response_total=len(v['interventions'])) for k,v in reports.items()}
    stress_summary={name:dict(passed=sum(r['metrics']['passed'] for r in sr['rows'] if r['model']==name),total=sum(r['model']==name for r in sr['rows'])) for name in ('B6','P110')}
    passed=all(reports['P110/'+m]['passed'] and all(r['metrics']['passed'] for r in reports['P110/'+m]['interventions']) for m in ('imposed','autonomous')) and stress_summary['P110']==dict(passed=36,total=36)
    report=dict(valid=True,task='41c',quality_passed=passed,model_judged=True,
        decision='FROZEN_P110_INDEPENDENT_QUALITY_CONFIRMED_REQUIRE_AUDIT' if passed else 'FROZEN_P110_INDEPENDENT_SCIENTIFIC_NO_GO',
        independent_confirmation=True,summary=summary,stress=stress_summary,training_performed=False,model_selection_performed=False,
        thresholds_changed=False,task42_authorized=False,speedup_claim=False,cost_quality_qualified=False,limits=spec['limits'])
    p.save_json(output/'final_report.json',report);return report
