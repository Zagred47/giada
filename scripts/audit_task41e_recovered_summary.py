"""Audit recovered final summaries without claiming full-archive recovery."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'artifacts/task41e_reserve_external'
report = json.loads((folder/'final_report_recovered.json').read_text())
prior = json.loads((ROOT/'artifacts/task41d_primary_external/detailed_audit.json').read_text())
cost = json.loads((folder/'cost_recovered.json').read_text())
assert report['valid'] and report['baseline_reproduced'] and not report['thresholds_changed']
mapping = {'B6':'B6', 'P110':'P110', 'D_C01':'C01', 'D_N000':'N000',
           'D_combined':'combined', 'D_fused':'combined'}
reproduction = {}
for new, old in mapping.items():
    reproduction[new] = {key: report['summaries'][new][key] == prior['summaries'][old][key]
                         for key in ('panels','stress','historical','recurrence')}
assert all(all(v.values()) for v in reproduction.values())
table = []
for name, s in report['summaries'].items():
    table.append(dict(model=name, development=s['development'],
        autonomous_41=s['panels']['41']['autonomous'],
        autonomous_41c=s['panels']['41c']['autonomous'], stress=s['stress'],
        historical=s['historical'], recurrence=s['recurrence']))
times = []
for row in cost:
    medians = {n: statistics.median(v['wall_ms']) for n,v in row['samples'].items()}
    times.append(dict(batch_per_seed=row['batch_per_seed'], actual_sites=row['actual_sites'],
        scope=row['scope'], median_wall_ms=medians, fusion=row['fusion_comparison'],
        q0_slower_than_p110_fraction=medians['Q0']/medians['P110']-1,
        q0_saved_vs_formula_fraction=1-medians['Q0']/medians['formula'],
        quality_qualified=row['quality_qualified'], finite=row['finite']))
out = dict(technical_valid=report['valid'], decision=report['decision'],
    passing_arms=report['passing_arms'], nominee=report['nominee'],
    controls_reproduce_41d=reproduction, summaries=table, timing=times,
    fusion_equivalence=report['fusion_equivalence'],
    task42_authorized=report['task42_authorized'], thresholds_changed=False,
    caveats=['Development protocols are new: do not compare development counts directly to41d.',
             'Summary audit only; failure identities and factorial mechanisms need final detailed files.',
             'No final complete local backup is claimed until SHA and ZIP CRC are verified.',
             'Speed measurements alone do not qualify a model failing quality gates.'])
path=folder/'recovered_summary_audit.json'
path.write_text(json.dumps(out,indent=2))
print(json.dumps({'audit':str(path),'control_reproduction':True,'models':len(table),'timing_rows':len(times)}))
