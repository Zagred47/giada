"""Read downloaded Task39c artifacts; recompute integrity and failure strata."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/task39c_kaggle_v1'


def severity(m):
    return max([m['gate']['rmse']/.005,m['gate']['max_abs']/.03]+
        [v for c in m['channels'] for v in (c['nrmse']/.01,c['nmax']/.05,
         abs(c['signed_charge_normalized'])/.02,c['absolute_charge_normalized']/.02)])


def main():
    read=lambda n:json.loads((OUT/n).read_text(encoding='utf-8'))
    final=read('final_report.json');rows=read('recurrence_evaluation.json');freeze=read('selection_freeze.json')
    claimed=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==claimed
    assert hashlib.sha256((OUT/'frozen_checkpoint.pt').read_bytes()).hexdigest()==freeze['checkpoint_sha256']
    assert len(rows)==len({(r['model'],r['role'],r['family'],r['case'],r['seed']) for r in rows})==1872
    assert all(r['metrics']['finite'] and r['metrics']['gate']['occupancy_violations']==0 for r in rows)
    models={}
    for model in sorted({r['model'] for r in rows}):
        models[model]={}
        for role in ('development','stress_diagnostic'):
            selected=[r for r in rows if r['model']==model and r['role']==role]
            causes=Counter();channels=Counter();families={};seeds={}
            for row in selected:
                m=row['metrics'];reasons=set()
                if not m['gate']['passed']:reasons.add('gate')
                if m['gate']['rmse']>.005:reasons.add('gate_rmse')
                if m['gate']['max_abs']>.03:reasons.add('gate_max')
                for c in m['channels']:
                    if not c['passed']:channels[c['channel']]+=1
                    if c['nrmse']>.01:reasons.add('current_rmse')
                    if c['nmax']>.05:reasons.add('current_max')
                    if max(abs(c['signed_charge_normalized']),c['absolute_charge_normalized'])>.02:reasons.add('charge')
                    e=c['event']
                    if e['status']=='evaluable' and any(e[k]>lim+1e-9 for k,lim in [('peak_error_ms',.2),('onset_error_ms',.3),('offset_error_ms',.3)]):reasons.add('timing')
                causes.update(reasons)
                if reasons=={'timing'}:causes['timing_only']+=1
                f=families.setdefault(row['family'],dict(passed=0,total=0,worst_severity=0.))
                f['passed']+=int(m['passed']);f['total']+=1;f['worst_severity']=max(f['worst_severity'],severity(m))
                s=seeds.setdefault(str(row['seed']),dict(passed=0,total=0,worst_severity=0.))
                s['passed']+=int(m['passed']);s['total']+=1;s['worst_severity']=max(s['worst_severity'],severity(m))
            models[model][role]=dict(passed=sum(r['metrics']['passed'] for r in selected),total=len(selected),
                failures=dict(causes),failed_channels=dict(channels),families=families,seeds=seeds,
                median_severity=float(np.median([severity(r['metrics']) for r in selected])),
                worst_severity=max(severity(r['metrics']) for r in selected))
        assert models[model]['development']['passed']==final['summaries'][model]['passed_cases']
    contrasts=read('paired_contrasts.json')['contrasts']
    assert len(contrasts)==16
    curves=read('training_curves.json');by_step={r['step']:np.asarray(r['common_development_score']) for r in curves}
    budgets={}
    for lo,hi in ((30000,60000),(60000,120000),(90000,120000)):
        ratios=by_step[hi]/by_step[lo]
        budgets[f'{lo}_{hi}']=dict(improved_members=int((ratios<1).sum()),median_reduction=float(np.median(1-ratios)),
            ratios=ratios.tolist())
    cost=read('resident_cost_benchmark.json');costs={}
    for arm in sorted({r['arm'] for r in cost['rows']}):
        cases=[r for r in cost['rows'] if r['arm']==arm]
        assert all(r['finite'] for r in cases)
        speeds=[];savings=[]
        for r in cases:
            pair=np.asarray(r['paired_formula_candidate_ms']);speeds.append(float(np.median(pair[:,0]/pair[:,1])))
            savings.append(r['median_saving_fraction'])
        costs[arm]=dict(speedup_per_seed=speeds,saving_per_seed=savings,
            ci95_per_seed=[r['paired_mean_saving_ci95'] for r in cases],
            formula_median_ms=[float(np.median(np.asarray(r['paired_formula_candidate_ms'])[:,0])) for r in cases],
            candidate_median_ms=[float(np.median(np.asarray(r['paired_formula_candidate_ms'])[:,1])) for r in cases])
    oracle=read('frozen_oracle_attribution.json');oracles={}
    for arm in sorted({r['parent_arm'] for r in oracle}):
        oracles[arm]={}
        for precision in ('float32','float64'):
            oracles[arm][precision]={}
            for mode in sorted({r['intervention'] for r in oracle}):
                selected=[r for r in oracle if r['parent_arm']==arm and r['precision']==precision and r['intervention']==mode]
                oracles[arm][precision][mode]=dict(passed=sum(r['metrics']['passed'] for r in selected),total=len(selected),
                    median_severity=float(np.median([severity(r['metrics']) for r in selected])),
                    worst_severity=max(severity(r['metrics']) for r in selected),
                    worst_gate_rmse=max(r['metrics']['gate']['rmse'] for r in selected))
    calibration=read('fit_loss_calibration.json');gradients=read('gradient_probes.json')
    clips=curves[-1]['clipping_counts']
    result=dict(valid=True,rows=len(rows),channel_metrics=len(rows)*11,checkpoint_hash_valid=True,freeze_hash_valid=True,
        summaries_recomputed=True,finite=True,occupancy_violations=0,models=models,budgets=budgets,
        selected_steps=dict(Counter(freeze['chosen_steps'])),supported_contrasts=[c for c in contrasts if c['supported']],
        all_contrasts=contrasts,resident_cost=costs,oracle_attribution=oracles,
        coefficient_loss_weights=calibration['weights'],clipping_fractions=[(np.asarray(x)/120000).tolist() for x in clips],
        last_gradient_probe=gradients[-3:],sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
            if p.name in ('final_report.json','recurrence_evaluation.json','selection_freeze.json','frozen_checkpoint.pt','training_curves.json','resident_cost_benchmark.json')},
        limits='Post-freeze descriptive audit; no independent CUDA replay. Failure categories overlap. Seed-case counts are not independent studies. Cost is one642-site resident core workload; no LUT or operational transfer comparison.')
    (OUT/'result_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(valid=True,rows=len(rows),selected_steps=result['selected_steps'],budgets=budgets,
        supported_contrasts=result['supported_contrasts'],resident_cost=costs,
        best_wide=models['joint_wwide_i1_s1_d1'],best_base=models['joint_wbase_i1_s0_d1'],
        oracle_attribution=oracles),indent=2))


if __name__=='__main__':main()
