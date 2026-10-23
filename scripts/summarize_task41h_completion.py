"""Concise, complete audit including the newly generated development failures."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'experiments/results/task41h_recovery_v1'
r=json.loads((D/'comparison_with_41g.json').read_text());n=r['nominee']
models={}
for name in (n,'G_nominee','G_tail_diagnostic'):
    s=r['report']['summaries'][name];fail=r['failures'][name]
    models[name]=dict(summary=s,total_failed_evaluations=sum(map(len,fail.values())),failure_counts={k:len(v) for k,v in fail.items() if v})
assert models[n]['total_failed_evaluations']==6
assert all(v is True for flags in r['baseline_reproduction'].values() for key,v in flags.items() if not isinstance(v,dict))
assert r['all_floors_passed'] and r['all_tensors_finite'] and r['complete_final_backup_verified']
out=dict(valid=True,complete=True,scientific_decision=r['report']['decision'],nominee=n,models=models,
    training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,task42_authorized=False,
    all_parameters_finite=True,original_training_checkpoints_unchanged=True,selected_checkpoint_unchanged=True,
    final_backup=r['backup']['remote'],raw_trace_restoration=r['report']['trace_restoration'],
    baseline_reproduction=r['baseline_reproduction'],cost_row_provenance=r['report'].get('cost_row_provenance'),
    interpretation=[
        'Fixed opened-panel failures: official41g5 ->41h3. Best posthoc41g also has3, but with a different response/stress tradeoff.',
        'New41h development adds3 Nap_Et2 offset failures shared by H and both G controls. On the full current evaluation set the totals are H6, G official8, G diagnostic6. These are not a comparison to old development seeds.',
        'Exact canonical u_m fixes the remaining Ca_HVA stress offsets (36/36); z_m and h oracles do not. This is a diagnostic intervention, not a deployable candidate.',
        'Joint state64+solve64 closes336/336 opened41c responses; state64 alone remains335 and solve64 alone334. Numerical policies remain ineligible for selection.',
        'The official dense candidate is slower at642 sites/seed in the standard paired benchmark; whole-loop coupling is approximately at formula parity. Compact execution is diagnostic and not an established speedup over formula.',
    ])
(D/'audit_completion.json').write_text(json.dumps(out,indent=2))
latest=json.loads((D/'LATEST.json').read_text());latest['audit']='audit_completion.json';(D/'LATEST.json').write_text(json.dumps(latest,indent=2))
print(json.dumps(dict(valid=True,complete=True,nominee=n,decision=out['scientific_decision'],failed_evaluations={k:v['total_failed_evaluations'] for k,v in models.items()},backup_sha256=out['final_backup']['sha256'])),flush=True)
