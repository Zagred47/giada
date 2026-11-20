import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/task52_kaggle_external';dest=ROOT/'experiments/results/task52_audit_v1';dest.mkdir(exist_ok=True)
original=folder/'kaggle_receipt.original.json';path=folder/'kaggle_receipt.json'
if not original.exists():original.write_bytes(path.read_bytes())
old=json.loads(original.read_text());archive=folder/old['archive']
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==old['sha256']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    report=json.loads(z.read('final_report.json'));runtime=json.loads(z.read('runtime_receipt.json'))
assert report['training_updates']==0 and report['selection_recomputed'] is False and report['weights_unchanged']
corrected=dict(old,training_updates_performed=0,selection_recomputed=False,metadata_correction=dict(reason='Inherited notebook template fields; scientific run performed no training or reselection',original_receipt_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),archive_unchanged=True,authority='final_report.json and frozen weights contract'))
data=json.dumps(corrected,indent=2)+'\n';path.write_text(data);(dest/'kaggle_receipt.corrected.json').write_text(data);(dest/'kaggle_receipt.original.json').write_bytes(original.read_bytes())
runtime.update(training_updates_performed=0,selection_recomputed=False,metadata_correction=corrected['metadata_correction'])
(dest/'runtime_receipt.corrected.json').write_text(json.dumps(runtime,indent=2)+'\n')
print(json.dumps(dict(valid=True,archive_unchanged=True,training_updates=0,selection_recomputed=False)))
