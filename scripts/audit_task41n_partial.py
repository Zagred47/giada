"""Read-only stopped confirmation: never score a model behind a failed floor."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'artifacts/task41n_primary_external'
receipt=json.loads((folder/'external_backup_receipt.json').read_text())
assert receipt['valid'] and receipt['runtime_independent'] and receipt['remote']['label']=='final'
archive=Path(receipt['local'])
with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    reports={name:json.loads(z.read(name)) for name in z.namelist() if name.endswith('.json')}
    floors=reports['canonical_floors.json'];freeze=reports['execution_freeze.json']
    assert freeze['eligible']==['dual','donor_pruned','shared_features']
    detail=[]
    for row in floors:
        failed=[]
        for index,metric in enumerate(row['rows']):
            if not metric['passed']:
                failed.append(dict(index=index,channels=[c for c in metric.get('channels',[]) if not c['passed']],other_keys={k:v for k,v in metric.items() if k not in ('channels','passed')}))
        detail.append(dict(mode=row['mode'],policy=row['policy'],passed=row['passed'],checks=len(row['rows']),failed=failed))
    audit=dict(valid=True,task='41n',complete=False,decision='INCONCLUSIVE_CANONICAL_FP32_FLOOR_FAILED',
        archive=str(archive.relative_to(ROOT)).replace('\\','/'),sha256=receipt['remote']['sha256'],bytes=archive.stat().st_size,
        original_reports_sha256={name:hashlib.sha256(z.read(name)).hexdigest() for name in reports},
        execution_eligibility=freeze,canonical_floors=detail,
        native_interface=reports.get('native_interface_report.json'),
        partial_model_summaries={k:v['summary'] for k,v in reports.get('causal_results.json',{}).items()},
        completed_fresh_execution_parity=reports.get('fresh_execution_parity.json'),
        training_updates=0,thresholds_changed=False,model_quality_judged=False,speedup_claim=False,
        missing=['autonomous model evaluation','fresh stress','full cost benchmarks'],
        interpretation='Original FP32 canonical autonomous reference fails its own registered quality floor on two checks; matched mixed64 passes. Confirmation cannot judge the learned model. No threshold waiver or automatic rerun.')
out=ROOT/'experiments/results/task41n_audited_v1';out.mkdir(exist_ok=True)
(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(dict(decision=audit['decision'],floors=[(r['mode'],r['policy'],r['passed'],len(r['failed'])) for r in detail],partial_keys=list(audit['partial_model_summaries'])),indent=2))
