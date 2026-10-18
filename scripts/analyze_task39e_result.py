"""Immutable Task39e result ingestion and post-hoc residual audit."""
import hashlib
import json
import zipfile
import io
from collections import Counter
from pathlib import Path
import numpy as np
from analyze_task39d_result import summarize, eventfail

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task39e_colab_v1'
ARCHIVE = ROOT / 'artifacts/task39e_retry2_external/recovery_final.zip'
EXPECTED = '10b515c6d2193378a9f85771f2333ad8cc93b4b81b950e3c61349f71e6ad7f95'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value): p.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
def main():
    assert sha(ARCHIVE) == EXPECTED
    OUT.mkdir(exist_ok=True)
    manifest = {}
    import torch
    health = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for info in z.infolist():
            if info.is_dir(): continue
            n = info.filename
            assert not Path(n).is_absolute() and '..' not in Path(n).parts
            content = z.read(n)
            manifest[n] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
            # Raw timing traces refer to inherited oracle probes, not new39e arms.
            # Full raw data and optimizer states remain in the verified external archive.
            if n.endswith('.json') or n == 'frozen_checkpoint.pt':
                p = OUT/n; p.parent.mkdir(exist_ok=True)
                if p.exists(): assert p.read_bytes() == content, n
                else: p.write_bytes(content)
        for n in ('checkpoint_180000.pt','frozen_checkpoint.pt','optimizer_180000.pt'):
            data=torch.load(io.BytesIO(z.read(n)),map_location='cpu',weights_only=False)
            tensors=[]
            def visit(x):
                if torch.is_tensor(x): tensors.append(x)
                elif isinstance(x,dict):
                    for v in x.values():visit(v)
                elif isinstance(x,(tuple,list)):
                    for v in x:visit(v)
            visit(data)
            health[n]=dict(tensors=len(tensors),all_finite=all(bool(torch.isfinite(x).all()) for x in tensors))
            assert health[n]['all_finite'] and tensors
    read = lambda n: json.loads((OUT/n).read_text(encoding='utf-8'))
    f = read('final_report.json'); freeze = read('selection_freeze.json')
    assert f['valid'] and not f['thresholds_changed'] and not f['independent_confirmation_accessed']
    assert sha(OUT/'frozen_checkpoint.pt') == freeze['checkpoint_sha256']
    payload = {k:v for k,v in freeze.items() if k != 'freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == freeze['freeze_sha256']
    rows = read('recurrence_evaluation.json')
    assert len(rows) == len({(r['model'],r['role'],r['family'],r['case'],r['seed']) for r in rows}) == 1584
    assert all(r['metrics']['finite'] and r['metrics']['gate']['occupancy_violations']==0 for r in rows)
    models = {m:{role:summarize([r for r in rows if r['model']==m and r['role']==role])
                  for role in ('development','stress_diagnostic')} for m in sorted({r['model'] for r in rows})}
    for m in models: assert models[m]['development']['passed'] == f['summaries'][m]['passed_cases']
    best = 'joint_wshared_o1_s1_h0'
    failures = []
    for r in rows:
        if r['model'] != best or r['role'] != 'development' or r['metrics']['passed']: continue
        cc = []
        for c in r['metrics']['channels']:
            if c['passed']: continue
            cc.append(dict(channel=c['channel'],nrmse=c['nrmse'],nmax=c['nmax'],
                signed_charge=c['signed_charge_normalized'],absolute_charge=c['absolute_charge_normalized'],
                scale=c['scale_ma_cm2'],event=c['event'],timing_failed=eventfail(c['event'])))
        failures.append(dict(family=r['family'],case=r['case'],seed=r['seed'],channels=cc))
    prior = json.loads((ROOT/'experiments/results/task39d_colab_v1/recurrence_evaluation.json').read_text())
    key = lambda r:(r['family'],r['case'],r['seed'])
    before = {key(r):r for r in prior if r['model']=='joint_wbase_n0_h0_b0' and r['role']=='development'}
    now = {key(r):r for r in rows if r['model']==best and r['role']=='development'}
    assert before.keys() == now.keys()
    contrasts = read('paired_contrasts.json')['contrasts']
    events = Counter()
    for r in failures:
        for c in r['channels']:
            ev = c['event']
            if ev['status']=='missing_event': events['missing_event'] += 1
            elif ev['status']=='evaluable':
                for k,limit in [('peak_error_ms',.2),('onset_error_ms',.3),('offset_error_ms',.3)]:
                    if ev[k]>limit+1e-9: events[k] += 1
    curves=read('training_curves.json');budgets={}
    for row in curves:
        budgets[str(row['step'])] = dict(common=row['common_development_score'],panel=row['panel'])
    audit = dict(valid=True,rows=len(rows),channel_metrics=len(rows)*11,archive_sha256=EXPECTED,numerical_verification=health,
        freeze_hash_valid=True,checkpoint_hash_valid=True,models=models,best_development_arm=best,
        residual_cases=failures,timing_failure_events=dict(events),
        prior_best=summarize(list(before.values())),
        gained=[list(k) for k in now if now[k]['metrics']['passed'] and not before[k]['metrics']['passed']],
        lost=[list(k) for k in now if not now[k]['metrics']['passed'] and before[k]['metrics']['passed']],
        supported_contrasts=[r for r in contrasts if r['supported']],registered_contrasts=len(contrasts),
        sources={n:v['sha256'] for n,v in manifest.items()},
        limits='Post-hoc development analysis;72 cases are24 protocols x3 seeds, not72 independent protocols. Task40 blocked. Historical39d comparison confounds budget, selection and repaired CUDA optimizer ownership. Inherited frozen oracle/timing files concern older parents, not39e models. No independent confirmation, no end-to-end speed claim.')
    write(OUT/'archive_manifest.json',dict(archive=str(ARCHIVE.relative_to(ROOT)),sha256=EXPECTED,members=manifest))
    write(OUT/'result_audit.json',audit)
    print(json.dumps(dict(residual_cases=failures,timing_failure_events=dict(events),supported=audit['supported_contrasts'],gained=len(audit['gained']),lost=len(audit['lost'])),indent=2))

if __name__=='__main__': main()
