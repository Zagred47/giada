"""Reproducible post-hoc audit; preserves every result without changing selection."""
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from analyze_task39d_result import summarize, eventfail

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/task39g_colab_v1'
ARCHIVE = ROOT/'artifacts/task39g_reserve_retry2_external/recovery_final.zip'
EXPECTED = 'b45bf5e01beb0102b18d9dce078396f2900282a9cb16d43ce4162669bf017cea'

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def tensors(x):
    if torch.is_tensor(x): return [x]
    if isinstance(x, dict): return [t for v in x.values() for t in tensors(v)]
    if isinstance(x, (list, tuple)): return [t for v in x for t in tensors(v)]
    return []

def main():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == EXPECTED
    OUT.mkdir(exist_ok=True)
    manifest, health = {}, {}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for info in z.infolist():
            if info.is_dir(): continue
            n=info.filename
            assert not Path(n).is_absolute() and '..' not in Path(n).parts
            data=z.read(n)
            manifest[n]=dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            if n.endswith('.json') or n=='frozen_checkpoint.pt':
                path=OUT/n; path.parent.mkdir(exist_ok=True)
                if path.exists(): assert path.read_bytes()==data, n
                else: path.write_bytes(data)
            if n.startswith(('checkpoint_', 'optimizer_')) and n.endswith('.pt') or n=='frozen_checkpoint.pt':
                tt=tensors(torch.load(io.BytesIO(data),map_location='cpu',weights_only=False))
                health[n]=dict(tensors=len(tt),all_finite=bool(tt) and all(bool(torch.isfinite(t).all()) for t in tt))
                assert health[n]['all_finite'],n
    read=lambda n:json.loads((OUT/n).read_text(encoding='utf-8'))
    final=read('final_report.json');freeze=read('selection_freeze.json');rows=read('recurrence_evaluation.json')
    assert final['valid'] and not final['thresholds_changed'] and not final['independent_confirmation_accessed']
    assert hashlib.sha256((OUT/'frozen_checkpoint.pt').read_bytes()).hexdigest()==freeze['checkpoint_sha256']
    payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    assert len(rows)==len({(r['model'],r['role'],r['family'],r['case'],r['seed']) for r in rows})==1872
    assert all(r['metrics']['finite'] and r['metrics']['gate']['occupancy_violations']==0 for r in rows)
    models={m:{role:summarize([r for r in rows if r['model']==m and r['role']==role]) for role in ('development','stress_diagnostic')} for m in sorted({r['model'] for r in rows})}
    for m,v in models.items(): assert v['development']['passed']==final['summaries'][m]['passed_cases']
    key=lambda r:(r['family'],r['case'],r['seed'])
    best='joint_wA0';prior='joint_wfrozen39f_wide_t0_s0'
    now={key(r):r for r in rows if r['model']==best and r['role']=='development'}
    before={key(r):r for r in rows if r['model']==prior and r['role']=='development'}
    residual=[dict(family=r['family'],case=r['case'],seed=r['seed'],gate=r['metrics']['gate'],channels=[c for c in r['metrics']['channels'] if not c['passed']]) for r in now.values() if not r['metrics']['passed']]
    original=json.loads((ROOT/'experiments/results/task39f_colab_v1/recurrence_evaluation.json').read_text())
    original={(r['model'],r['role'],*key(r)):r['metrics'] for r in original}
    for r in rows:
        if r['model'].startswith('joint_wfrozen39f_'):
            assert original[('joint_w'+r['model'].split('joint_wfrozen39f_')[1],r['role'],*key(r))]==r['metrics']
    groups=defaultdict(list)
    for r in read('frozen_oracle_attribution.json'): groups[(r['parent_arm'],r['precision'],r['intervention'])].append(r)
    oracle={'/'.join(k):summarize(v) for k,v in groups.items()}
    gradients=[]
    for probe in read('fixed_fit_gradient_probes.json'):
        x=np.array(probe['term_norms'])*np.array(probe['weights']);cos=np.array(probe['cosines'])
        base_norm=np.sqrt(np.maximum(np.einsum('bi,bij,bj->b',x[:,:4],cos[:,:4,:4],x[:,:4]),1e-30))
        aux_norm=np.sqrt(np.maximum(np.einsum('bi,bij,bj->b',x[:,4:],cos[:,4:,4:],x[:,4:]),0))
        offset=0 if probe['cohort']=='h64' else 6
        for j in range(len(x)):
            gradients.append(dict(step=probe['step'],arm=f'A{offset+j//3}',seed=freeze['seeds'][j%3],
                auxiliary_to_combined_main_gradient=float(aux_norm[j]/base_norm[j]),weighted_term_norms=x[j].tolist()))
    costs={}
    for a in freeze['arms']:
        cc=[r for r in read('resident_cost_benchmark.json')['rows'] if r['arm']==a['name']]
        costs[a['name']]=[float(np.median(np.array(r['paired_formula_candidate_ms'])[:,0]/np.array(r['paired_formula_candidate_ms'])[:,1])) for r in cc]
    timing=defaultdict(Counter)
    for r in read('timing_audit.json'):
        for c in r['channels']:
            if not eventfail(c['event']): continue
            k=r['parent']+'/'+r['precision'];timing[k]['failed_channel_events']+=1
            timing[k]['perturbation_changes_pass']+=int(any(eventfail(v)!=eventfail(c['event']) for v in c['perturbed_events']))
            timing[k]['phase_changes_pass']+=int(any(eventfail(v['event'])!=eventfail(c['event']) for v in c['grid_phase_diagnostics']))
    common=read('common_selection_diagnostic/recurrence_evaluation.json')
    audit=dict(common_pass_counts={m:sum(r['metrics']['passed'] for r in common if r['model']==m and r['role']=='development') for m in models},selected_step_zero_count=freeze['chosen_steps'].count(0),
valid=True,rows=len(rows),channel_metrics=len(rows)*11,archive_sha256=EXPECTED,numerical_verification=health,
        freeze_hash_valid=True,checkpoint_hash_valid=True,frozen39f_metrics_exactly_reproduced=True,models=models,best_development_arm=best,
        residual_cases=residual,gained=[list(k) for k in now if now[k]['metrics']['passed'] and not before[k]['metrics']['passed']],
        lost=[list(k) for k in now if not now[k]['metrics']['passed'] and before[k]['metrics']['passed']],
        supported_contrasts=[c for c in read('paired_contrasts.json')['contrasts'] if c['supported']],registered_contrasts=11,
        oracles=oracle,gradient_balance=gradients,resident_speedup=costs,parent_timing_diagnostics={k:dict(v) for k,v in timing.items()},
        limits='Post-hoc development analysis:72 cases=24 protocols x3 seeds. Thresholds unchanged; no independent confirmation. Oracles/timing concern frozen39f, NOT new A0/A6. Fixed-FIT gradient probes are descriptive. FP64 training affines apply to all arms; inference FP32. Warm continuation adds120000 updates. Common selection is diagnostic and nonselectable. No end-to-end speed claim.',
        sources={n:v['sha256'] for n,v in manifest.items()})
    write(OUT/'archive_manifest.json',dict(archive=str(ARCHIVE.relative_to(ROOT)),sha256=EXPECTED,members=manifest))
    write(OUT/'result_audit.json',audit)
    for name in ('external_backup_receipt.json','supervisor_receipt.json','job_status.json'):
        write(OUT/name,json.loads((ARCHIVE.parent/name).read_text()))
    print(json.dumps(dict(valid=True,cases=len(rows),finite_files=len(health),best=models[best]['development']['passed'],residual=len(residual),gained=len(audit['gained']),lost=len(audit['lost']),supported=len(audit['supported_contrasts']))))

if __name__=='__main__': main()


