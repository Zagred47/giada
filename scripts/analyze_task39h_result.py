"""Reproducible complete result audit; immutable primary selection is preserved."""
import hashlib,io,json,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from analyze_task39d_result import summarize,eventfail
from analyze_task39g_result import tensors,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/task39h_colab_v1'
ARCHIVE=ROOT/'artifacts/task39h_primary_external/giada_task39h_result.zip'
EXPECTED='b90fbfeb633abf194304e22a0ef603339b7322e7fe5a732c2fe356baeae08a15'

def main():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==EXPECTED
    OUT.mkdir(exist_ok=True);manifest={};health={}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for entry in z.infolist():
            if entry.is_dir():continue
            n=entry.filename;assert not Path(n).is_absolute() and '..' not in Path(n).parts
            data=z.read(n);manifest[n]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            if n.endswith('.json') or n=='frozen_checkpoint.pt':
                path=OUT/n;path.parent.mkdir(exist_ok=True)
                if path.exists():assert path.read_bytes()==data,n
                else:path.write_bytes(data)
            if (n.startswith(('checkpoint_','optimizer_')) and n.endswith('.pt')) or n=='frozen_checkpoint.pt':
                tt=tensors(torch.load(io.BytesIO(data),map_location='cpu',weights_only=False))
                health[n]=dict(tensors=len(tt),all_finite=bool(tt) and all(bool(torch.isfinite(t).all()) for t in tt));assert health[n]['all_finite'],n
    read=lambda n:json.loads((OUT/n).read_text())
    final=read('final_report.json');freeze=read('selection_freeze.json');rows=read('recurrence_evaluation.json')
    assert final['valid'] and not final['thresholds_changed'] and not final['independent_confirmation_accessed']
    assert not freeze['stress_used_for_selection'] and freeze['development_only']
    assert hashlib.sha256((OUT/'frozen_checkpoint.pt').read_bytes()).hexdigest()==freeze['checkpoint_sha256']
    payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    key=lambda r:(r['family'],r['case'],r['seed'])
    assert len(rows)==len({(r['model'],r['role'],*key(r)) for r in rows})==1872
    assert all(r['metrics']['finite'] and r['metrics']['gate']['occupancy_violations']==0 for r in rows)
    models={m:{role:summarize([r for r in rows if r['model']==m and r['role']==role]) for role in ('development','stress_diagnostic')} for m in sorted({r['model'] for r in rows})}
    for m,v in models.items():assert v['development']['passed']==final['summaries'][m]['passed_cases']
    prior=json.loads((ROOT/'experiments/results/task39g_colab_v1/recurrence_evaluation.json').read_text())
    prior={(r['model'],r['role'],*key(r)):r['metrics'] for r in prior}
    for r in rows:
        if r['model'].startswith('joint_wfrozen39g_'):
            assert r['metrics']==prior[('joint_w'+r['model'].split('joint_wfrozen39g_')[1],r['role'],*key(r))]
    best='joint_wH8';base='joint_wfrozen39g_A0'
    now={key(r):r for r in rows if r['role']=='development' and r['model']==best}
    before={key(r):r for r in rows if r['role']=='development' and r['model']==base}
    residual=[dict(family=r['family'],case=r['case'],seed=r['seed'],gate=r['metrics']['gate'],channels=[c for c in r['metrics']['channels'] if not c['passed']]) for r in now.values() if not r['metrics']['passed']]
    groups=defaultdict(list)
    for r in read('frozen_oracle_attribution.json'):groups[(r['parent_arm'],r['precision'],r['intervention'])].append(r)
    oracle={'/'.join(k):summarize(v) for k,v in groups.items()}
    residual_probe=[]
    for r in read('residual_oracle_attribution.json'):
        residual_probe.append({k:r[k] for k in ('parent','seed','family','case','intervention','corrected_gates')}|dict(passed=r['metrics']['passed'],failed_channels=[c for c in r['metrics']['channels'] if not c['passed']]))
    timing=defaultdict(Counter)
    for r in read('timing_audit.json'):
        for c in r['channels']:
            if not eventfail(c['event']):continue
            k=r['parent']+'/'+r['precision'];timing[k]['failed_channel_events']+=1
            timing[k]['perturbation_changes_pass']+=int(any(eventfail(v)!=eventfail(c['event']) for v in c['perturbed_events']))
            timing[k]['phase_changes_pass']+=int(any(eventfail(v['event'])!=eventfail(c['event']) for v in c['grid_phase_diagnostics']))
    common=read('common_selection_diagnostic/recurrence_evaluation.json')
    gradients=[]
    for probe in read('fixed_fit_gradient_probes.json'):
        x=np.array(probe['term_norms'])*np.array(probe['weights']);cos=np.array(probe['cosines'])
        main_norm=np.sqrt(np.maximum(np.einsum('bi,bij,bj->b',x[:,:4],cos[:,:4,:4],x[:,:4]),1e-30))
        definitions=[a for a in freeze['arms'] if a['learning_rate']==(.001 if probe['cohort']=='lr001' else .0001)]
        for j in range(len(x)):
            gradients.append(dict(step=probe['step'],arm=definitions[j//3]['name'],seed=freeze['seeds'][j%3],auxiliary_to_main=float(x[j,4]/main_norm[j]),weighted_term_norms=x[j].tolist()))
    costs={}
    for a in freeze['arms']:
        cc=[r for r in read('resident_cost_benchmark.json')['rows'] if r['arm']==a['name']]
        costs[a['name']]=[float(np.median(np.array(r['paired_formula_candidate_ms'])[:,0]/np.array(r['paired_formula_candidate_ms'])[:,1])) for r in cc]
    audit=dict(valid=True,rows=len(rows),channel_metrics=len(rows)*11,archive_sha256=EXPECTED,numerical_verification=health,
        freeze_hash_valid=True,checkpoint_hash_valid=True,frozen39g_metrics_exactly_reproduced=True,models=models,best_development_arm=best,
        residual_cases=residual,gained=[list(k) for k in now if now[k]['metrics']['passed'] and not before[k]['metrics']['passed']],
        lost=[list(k) for k in now if not now[k]['metrics']['passed'] and before[k]['metrics']['passed']],oracles=oracle,
        parent_residual_probes=residual_probe,parent_timing_diagnostics={k:dict(v) for k,v in timing.items()},
        supported_contrasts=[c for c in read('paired_contrasts.json')['contrasts'] if c['supported']],registered_contrasts=14,
        common_pass_counts={m:sum(r['metrics']['passed'] for r in common if r['model']==m and r['role']=='development') for m in models},
        selected_steps={a['name']:freeze['chosen_steps'][3*i:3*i+3] for i,a in enumerate(freeze['arms'])},gradient_balance=gradients,resident_speedup=costs,
        limits='Post-hoc development analysis.72 cases=24protocols x3seeds; no independent confirmation. H8 identified by highest development pass count,not promoted. All existing oracle/timing interventions concern frozen39g A0/A6,not H8. Restricted-head failure conditional on fixed learned features. Common-selection counterfactual nonselectable. No quality-qualified end-to-end speed claim.',
        sources={n:v['sha256'] for n,v in manifest.items()})
    write(OUT/'archive_manifest.json',dict(archive=str(ARCHIVE.relative_to(ROOT)),sha256=EXPECTED,members=manifest));write(OUT/'result_audit.json',audit)
    for name in ('external_backup_receipt.json','supervisor_receipt.json','job_status.json'):write(OUT/name,json.loads((ARCHIVE.parent/name).read_text()))
    print(json.dumps(dict(valid=True,cases=len(rows),finite_files=len(health),best=models[best]['development']['passed'],residual=len(residual),gained=len(audit['gained']),lost=len(audit['lost']),supported=len(audit['supported_contrasts']),selected_steps=audit['selected_steps'],common=audit['common_pass_counts'])),flush=True)

if __name__=='__main__':main()
