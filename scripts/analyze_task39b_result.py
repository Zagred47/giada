"""Post-freeze integrity and failure analysis, unchanged acceptance thresholds."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task39b_kaggle_v2'


def main():
    # Reuse the identical failure categorization, with this run's expected cardinality.
    rows = json.loads((OUT/'recurrence_evaluation.json').read_text())
    final = json.loads((OUT/'final_report.json').read_text())
    freeze = json.loads((OUT/'selection_freeze.json').read_text())
    claimed = freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest() == claimed
    assert hashlib.sha256((OUT/'frozen_checkpoint.pt').read_bytes()).hexdigest() == freeze['checkpoint_sha256']
    assert len(rows) == len({(r['model'],r['role'],r['family'],r['seed'],r['case']) for r in rows}) == 2016
    assert all(r['metrics']['finite'] and r['metrics']['gate']['occupancy_violations']==0 for r in rows)
    # Existing analyzer's source is kept unchanged. This adapted computation retains all cases.
    models = {}
    for model in sorted({r['model'] for r in rows}):
        models[model] = {}
        for role in ('development','stress_diagnostic'):
            sub = [r for r in rows if r['model']==model and r['role']==role]
            counts=Counter(); channels=Counter(); families={}
            for r in sub:
                m=r['metrics']; causes=set()
                if not m['gate']['passed']: causes.add('gate')
                for c in m['channels']:
                    if not c['passed']: channels[c['channel']]+=1
                    if c['nrmse']>.01: causes.add('current_rmse')
                    if c['nmax']>.05: causes.add('current_max')
                    if abs(c['signed_charge_normalized'])>.02 or abs(c['absolute_charge_normalized'])>.02: causes.add('charge')
                    e=c['event']
                    if e['status']=='evaluable' and any(e[k]>lim+1e-9 for k,lim in [('peak_error_ms',.2),('onset_error_ms',.3),('offset_error_ms',.3)]): causes.add('timing')
                counts.update(causes)
                if causes=={'timing'}: counts['timing_only']+=1
                if not m['passed'] and not causes: counts['other_or_boundary']+=1
                f=families.setdefault(r['family'],dict(passed=0,total=0,worst_gate_rmse=0))
                f['total']+=1; f['passed']+=int(m['passed']); f['worst_gate_rmse']=max(f['worst_gate_rmse'],m['gate']['rmse'])
            models[model][role]=dict(passed=sum(r['metrics']['passed'] for r in sub),total=len(sub),failures=dict(counts),channels=dict(channels),families=families)
        assert models[model]['development']['passed']==final['summaries'][model]['passed_cases']
    curves=json.loads((OUT/'training_curves.json').read_text()); a=np.array(curves[-2]['common_development_score']); b=np.array(curves[-1]['common_development_score'])
    probes=json.loads((OUT/'internal_probes.json').read_text()); budgets={}
    for i,arm in enumerate(freeze['arms']):
        sl=slice(3*i,3*i+3)
        budgets[arm['name']]=dict(chosen_steps=freeze['chosen_steps'][sl],score_30k=a[sl].tolist(),score_60k=b[sl].tolist(),
            ratio_60k_over_30k=(b[sl]/a[sl]).tolist(),gradient_cosine=probes['fast_slow_trunk_gradient_cosine'][sl],
            random_state_gate_rmse_mean=np.mean(np.array(probes['physical_state_gate_rmse'])[sl],axis=1).tolist())
    result=dict(valid=True,rows=2016,channel_rows=22176,unique_cases=True,finite=True,occupancy_violations=0,
        freeze_hash_valid=True,checkpoint_hash_valid=True,summaries_recomputed=True,models=models,budgets=budgets,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.name in ('final_report.json','recurrence_evaluation.json','selection_freeze.json','frozen_checkpoint.pt','training_curves.json')},
        limits='Post-freeze descriptive audit, no independent CUDA replay. Overlapping failure counts. Same fixed thresholds; no selection from stress.')
    (OUT/'result_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(valid=True,rows=len(rows),all36_improve_30k_to_60k=bool((b<a).all()),
        median_score_reduction=float(np.median(1-b/a)),selected=Counter(freeze['chosen_steps']))))


if __name__=='__main__': main()
