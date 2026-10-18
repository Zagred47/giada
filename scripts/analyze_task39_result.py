"""Read-only post-freeze failure stratification; no changed acceptance gates."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'experiments/results/task39_kaggle_v1'


def analyze():
    rows = json.loads((FOLDER/'recurrence_evaluation.json').read_text())
    final = json.loads((FOLDER/'final_report.json').read_text())
    freeze = json.loads((FOLDER/'selection_freeze.json').read_text())
    claimed = freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest() == claimed
    keys = {(r['model'], r['role'], r['family'], r['seed'], r['case']) for r in rows}
    assert len(keys) == len(rows) == 1872
    assert all(r['metrics']['finite'] for r in rows)
    assert all(r['metrics']['gate']['occupancy_violations'] == 0 for r in rows)
    results = {}
    for model in sorted({r['model'] for r in rows}):
        results[model] = {}
        for role in ('development', 'stress_diagnostic'):
            subset = [r for r in rows if r['model'] == model and r['role'] == role]
            counts, channels, families = Counter(), Counter(), {}
            worst = []
            for row in subset:
                m = row['metrics']; g = m['gate']; causes = set()
                if not g['passed']: causes.add('gate')
                for c in m['channels']:
                    if not c['passed']: channels[c['channel']] += 1
                    if c['nrmse'] > .01: causes.add('current_rmse')
                    if c['nmax'] > .05: causes.add('current_max')
                    if abs(c['signed_charge_normalized']) > .02 or abs(c['absolute_charge_normalized']) > .02: causes.add('charge')
                    e = c['event']
                    if e['status'] == 'evaluable' and (e['peak_error_ms'] > .2+1e-9 or e['onset_error_ms'] > .3+1e-9 or e['offset_error_ms'] > .3+1e-9): causes.add('timing')
                counts.update(causes)
                if causes == {'timing'}: counts['timing_only'] += 1
                if not m['passed'] and not causes: counts['other_or_boundary'] += 1
                family = families.setdefault(row['family'], dict(passed=0, total=0, worst_gate_rmse=0))
                family['total'] += 1; family['passed'] += int(m['passed'])
                family['worst_gate_rmse'] = max(family['worst_gate_rmse'], g['rmse'])
                worst.append(dict(family=row['family'], seed=row['seed'], case=row['case'], gate_rmse=g['rmse'],
                                  worst_channel=max(m['channels'], key=lambda c:c['nrmse'])))
            results[model][role] = dict(passed=sum(r['metrics']['passed'] for r in subset), total=len(subset),
                overlapping_failure_counts=dict(counts), failed_channel_counts=dict(channels), families=families,
                worst_gate_cases=sorted(worst,key=lambda r:r['gate_rmse'],reverse=True)[:3])
        assert results[model]['development']['passed'] == final['summaries'][model]['passed_cases']
    report = dict(valid=True, rows=len(rows), channel_rows=len(rows)*11, unique_cases=True, finite=True,
        occupancy_violations=0, freeze_hash_valid=True, summaries_recomputed=True,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FOLDER.iterdir() if p.suffix=='.json' and p.name not in ('result_audit.json','memory_receipt.json')},
        models=results, limits='Post-freeze diagnostics; counts overlap. No retuning on stress, no independent confirmation, no autonomous V/Ca or speedup evidence.')
    (FOLDER/'result_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__ == '__main__':
    r = analyze()
    print(json.dumps({k:{role:{'passed':v['passed'],'failures':v['overlapping_failure_counts'],'channels':v['failed_channel_counts']} for role,v in roles.items()} for k,roles in r['models'].items()},indent=2))
