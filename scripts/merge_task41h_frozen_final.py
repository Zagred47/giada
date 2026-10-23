"""Combine verified historical traces/checkpoints and evaluation-only results."""
import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41h_frozen_evaluation_primary_external'
DEST=ROOT/'artifacts/task41h_complete_evaluation_external'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
new_receipt=json.loads((D/'external_backup_receipt.json').read_text());new=Path(new_receipt['local'])
status=json.loads((D/'job_status.json').read_text())
assert status['state']=='finished' and status['worker_exit_code']==0
assert new_receipt['valid'] and new_receipt['remote']['label']=='final' and sha(new)==new_receipt['remote']['sha256']
old_receipt=json.loads((ROOT/'artifacts/task41h_evaluation_primary_external/external_backup_receipt.json').read_text())
old=Path(old_receipt['local']);assert old_receipt['valid'] and sha(old)==old_receipt['remote']['sha256']
DEST.mkdir(exist_ok=True);target=DEST/'recovery_final_merged.zip';tmp=target.with_suffix('.tmp')
immutable={'frozen_checkpoint.pt','selection_freeze.json','selection_parity.json','preregistration.json','training_curves.json'}
with zipfile.ZipFile(old) as a,zipfile.ZipFile(new) as b:
    assert a.testzip() is None and b.testzip() is None
    for name in immutable:assert a.read(name)==b.read(name),name
    report=json.loads(b.read('final_report.json'))
    assert report['valid'] and report['training_updates_performed']==0 and not report['selection_recomputed'] and not report['thresholds_changed']
    old_names=set(a.namelist());new_names=set(b.namelist())
    names=sorted(old_names|new_names)
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as out:
        for name in names:
            source=b if name in new_names else a
            info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_STORED if name.endswith(('.npz','.pt')) else zipfile.ZIP_DEFLATED
            with source.open(name) as f,out.open(info,'w',force_zip64=True) as g:shutil.copyfileobj(f,g,8*1024*1024)
with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
tmp.replace(target)
receipt=dict(valid=True,runtime_independent=True,local=str(target),remote=dict(label='final',sha256=sha(target),bytes=target.stat().st_size),
    source_archives=[old_receipt,new_receipt],merged_without_training_or_selection=True)
(DEST/'external_backup_receipt.json').write_text(json.dumps(receipt,indent=2))
(DEST/'job_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(dict(valid=True,archive=str(target),sha256=receipt['remote']['sha256'],bytes=target.stat().st_size)),flush=True)
