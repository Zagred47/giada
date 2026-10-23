"""Seal all completed stage4 outputs; retry only the incomplete stage."""
import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/task41h_kaggle_stage4_external'
DEST=ROOT/'artifacts/task41h_kaggle_stage5_inputs';DEST.mkdir(exist_ok=True)
receipt=json.loads((D/'kaggle_final_receipt.json').read_text());source=D/receipt['archive']
with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
with zipfile.ZipFile(source) as z:
    assert z.testzip() is None
    failure=json.loads(z.read('failure_report.json'));assert 'make_core() missing 3 required' in failure['error']
    costs=json.loads(z.read('cost.json'));executions=json.loads(z.read('execution_matrix.json'))
    numeric=json.loads(z.read('selected_numerical_audit/voltage_precision.json'));compact=json.loads(z.read('compact_execution_cost.json'))
    assert len(costs)==6 and len(executions)==24 and len(numeric)==10 and len(compact)==1
    assert len(json.loads(z.read('coefficient_oracle_stress.json'))['rows'])==144
    previous=json.loads(z.read('frozen_resume_manifest.json'));freeze=json.loads(z.read('selection_freeze.json'))
    manifest=dict(valid=True,source_archive_sha256=receipt['sha256'],original_seal=previous['original_seal'],freeze_sha256=freeze['freeze_sha256'],training_updates_performed=0,thresholds_changed=False,members={})
    provenance={str(r['batch_per_seed'])+'/'+r['scope']:('Colab b9cbd5b' if (r['batch_per_seed'],r['scope']) in ((1,'ionic'),(1,'integrated'),(642,'ionic')) else 'Kaggle version4 7a40214') for r in costs}
    added={'cost_row_provenance.json':json.dumps(provenance,indent=2).encode(),'prior_failure_report.json':z.read('failure_report.json')}
    path=DEST/'remaining_evaluation_input.zip.bin'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        for name in z.namelist():
            if name in ('frozen_resume_manifest.json','failure_report.json') or not name.endswith(('.json','.pt')):continue
            raw=z.read(name);manifest['members'][name]=dict(source=source.name,sha256=hashlib.sha256(raw).hexdigest());out.writestr(name,raw)
        for name,raw in added.items():manifest['members'][name]=dict(source='recovery provenance',sha256=hashlib.sha256(raw).hexdigest());out.writestr(name,raw)
        out.writestr('frozen_resume_manifest.json',json.dumps(manifest,indent=2))
for name in ('task41b_opened_inputs.zip.bin','task41d_opened41c.zip.bin'):shutil.copy2(ROOT/'artifacts/task41h_kaggle_binary_inputs'/name,DEST/name)
shutil.copy2(ROOT/'artifacts/task41h_kaggle_binary_inputs/dataset-metadata.json',DEST/'dataset-metadata.json')
with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
result=dict(valid=True,source=receipt,input_sha256=digest,input_bytes=path.stat().st_size,completed=dict(cost_rows=6,execution_rows=24,numerical_policies=10,compact_cost_rows=1),training_updates_performed=0,selection_recomputed=False,thresholds_changed=False)
(ROOT/'artifacts/task41h_stage5_input_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
