"""Recompute Task39d quality and paired diagnostics without selecting on stress."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from analyze_task39c_result import severity

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/task39d_colab_v1'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def eventfail(e):
    return e['status']=='missing_event' or (e['status']=='evaluable' and any(e[k]>v+1e-9 for k,v in [('peak_error_ms',.2),('onset_error_ms',.3),('offset_error_ms',.3)]))
def summarize(cases):
    fail=Counter();channels=Counter();families={};byseed={};channel_metrics=defaultdict(list)
    for r in cases:
        m=r['metrics'];reasons=set()
        if not m['gate']['passed']:reasons.add('gate')
        for c in m['channels']:
            channel_metrics[c['channel']].append(c)
            if not c['passed']:channels[c['channel']]+=1
            if c['nrmse']>.01:reasons.add('current_rmse')
            if c['nmax']>.05:reasons.add('current_max')
            if max(abs(c['signed_charge_normalized']),c['absolute_charge_normalized'])>.02:reasons.add('charge')
            if eventfail(c['event']):reasons.add('timing')
            if not c['blocked_current_passed']:reasons.add('blocked')
        assert bool(reasons)==(not m['passed']), (r['model'] if 'model' in r else r['parent_arm'],r['family'],reasons)
        fail.update(reasons)
        if reasons=={'timing'}:fail['timing_only']+=1
        for mapping,key in [(families,r['family']),(byseed,str(r['seed']))]:
            v=mapping.setdefault(key,dict(passed=0,total=0,worst_severity=0.))
            v['passed']+=int(m['passed']);v['total']+=1;v['worst_severity']=max(v['worst_severity'],severity(m))
    return dict(passed=sum(r['metrics']['passed'] for r in cases),total=len(cases),gate_passed=sum(r['metrics']['gate']['passed'] for r in cases),
        failures=dict(fail),failed_channels=dict(channels),families=families,seeds=byseed,
        median_severity=float(np.median([severity(r['metrics']) for r in cases])),worst_severity=max(severity(r['metrics']) for r in cases),
        channel_worst={k:{field:max(abs(c[field]) for c in cc) for field in ['nrmse','nmax','signed_charge_normalized','absolute_charge_normalized']} for k,cc in channel_metrics.items()})
def main():
    final=read('final_report.json');rows=read('recurrence_evaluation.json');freeze=read('selection_freeze.json')
    payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert digest(OUT/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    assert len(rows)==1872==len({(r['model'],r['role'],r['family'],r['case'],r['seed']) for r in rows})
    assert all(r['metrics']['finite'] and r['metrics']['gate']['occupancy_violations']==0 for r in rows)
    assert final['valid'] and not final['thresholds_changed'] and not final['independent_confirmation_accessed']
    assert not freeze['stress_used_for_selection'] and freeze['development_only']
    models={}
    for model in sorted({r['model'] for r in rows}):
        models[model]={role:summarize([r for r in rows if r['model']==model and r['role']==role]) for role in ('development','stress_diagnostic')}
        assert models[model]['development']['passed']==final['summaries'][model]['passed_cases']
    prior=json.loads((ROOT/'experiments/results/task39c_kaggle_v1/result_audit.json').read_text())
    priorrows=json.loads((ROOT/'experiments/results/task39c_kaggle_v1/recurrence_evaluation.json').read_text())
    assert {(r['role'],r['family'],r['case'],r['seed']) for r in rows}=={(r['role'],r['family'],r['case'],r['seed']) for r in priorrows}
    keys=lambda r:(r['family'],r['case'],r['seed'])
    prev={keys(r):r for r in priorrows if r['model']=='joint_wwide_i1_s1_d1' and r['role']=='development'}
    changes={}
    for model in models:
        now={keys(r):r for r in rows if r['model']==model and r['role']=='development'}
        changes[model]=dict(gained=[list(k) for k in now if now[k]['metrics']['passed'] and not prev[k]['metrics']['passed']],
            lost=[list(k) for k in now if not now[k]['metrics']['passed'] and prev[k]['metrics']['passed']])
    oracle=read('frozen_oracle_attribution.json');group=defaultdict(list)
    for r in oracle:group[(r['parent_arm'],r['precision'],r['intervention'])].append(r)
    oracles={'/'.join(k):summarize(v) for k,v in group.items()}
    timing=read('timing_audit.json');tg=defaultdict(Counter)
    for r in timing:
        key=r['parent']+'/'+r['precision']
        for c in r['channels']:
            e=c['event'];pert=c['perturbed_events']
            tg[key]['channels']+=1;tg[key]['timing_failures']+=int(eventfail(e))
            tg[key]['perturbation_changes_pass_status']+=int(any(eventfail(x)!=eventfail(e) for x in pert))
            tg[key]['perturbation_changes_event']+=int(any(x!=e for x in pert))
    curves=read('training_curves.json');steps={r['step']:np.array(r['common_development_score']) for r in curves}
    budgets={str(lo)+'_'+str(hi):dict(improved_members=int((steps[hi]<steps[lo]).sum()),median_reduction=float(np.median(1-steps[hi]/steps[lo]))) for lo,hi in [(30000,60000),(60000,120000),(90000,120000)]}
    costs={}
    for arm in [d['name'] for d in freeze['arms']]:
        cc=[r for r in read('resident_cost_benchmark.json')['rows'] if r['arm']==arm]
        assert len(cc)==3 and all(r['finite'] for r in cc)
        costs[arm]=dict(saving_per_seed=[r['median_saving_fraction'] for r in cc],
            speedup_per_seed=[float(np.median(np.array(r['paired_formula_candidate_ms'])[:,0]/np.array(r['paired_formula_candidate_ms'])[:,1])) for r in cc],
            ci95_per_seed=[r['paired_mean_saving_ci95'] for r in cc])
    baseline_diff=[]
    before={keys(r):r for r in priorrows if r['model']=='frozen_local161_SKcanonical' and r['role']=='development'}
    for r in rows:
        if r['model']=='frozen_local161_SKcanonical' and r['role']=='development' and r['metrics']['passed']!=before[keys(r)]['metrics']['passed']:baseline_diff.append(keys(r))
    audit=dict(valid=True,rows=len(rows),channel_metrics=len(rows)*11,checkpoint_hash_valid=True,freeze_hash_valid=True,
        finite=True,occupancy_violations=0,models=models,oracles=oracles,timing_audit={k:dict(v) for k,v in tg.items()},
        prior_best=prior['models']['joint_wwide_i1_s1_d1'],case_changes_vs39c_wide=changes,baseline_pass_disagreements=baseline_diff,
        budgets=budgets,selected_steps=dict(Counter(freeze['chosen_steps'])),
        supported_contrasts=[r for r in read('paired_contrasts.json')['contrasts'] if r['supported']],resident_cost=costs,
        sources={p.name:digest(p) for p in OUT.iterdir() if p.suffix in ('.json','.pt') and p.name not in ('result_audit.json','memory_receipt.json','memory_readback.json')},
        limits='Post-freeze descriptive analysis. 39c Kaggle vs39d Colab software differs; between-run comparisons are not matched causal contrasts. Same 72 dev cases per arm; overlapping cases/seeds not independent experiments. Oracles concern frozen39c parents, not new39d models. No independent confirmation; no autonomous voltage/multicompartment claim; resident-core cost only.')
    (OUT/'result_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('sources','oracles','models','prior_best','case_changes_vs39c_wide')},indent=2))
    print('MODELS',json.dumps({k:{role:{f:v for f,v in z.items() if f in ('passed','gate_passed','failures','failed_channels','median_severity','worst_severity','families')} for role,z in rr.items()} for k,rr in models.items()},indent=1))
    print('ORACLES',json.dumps({k:{f:v for f,v in z.items() if f in ('passed','failures','median_severity','worst_severity')} for k,z in oracles.items()},indent=1))
if __name__=='__main__':main()
