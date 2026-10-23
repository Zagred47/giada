"""Normalize a durable Kaggle result only after SHA and every ZIP CRC pass."""
import hashlib,json,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41h_kaggle_remaining_external'
receipt=json.loads((D/'kaggle_final_receipt.json').read_text())
assert receipt['valid'] and receipt['worker_exit_code']==0
archive=D/receipt['archive']
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
assert archive.stat().st_size==receipt['bytes']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    report=json.loads(z.read('final_report.json'));freeze=json.loads(z.read('selection_freeze.json'))
    assert report['valid'] and report['training_updates_performed']==0 and not report['selection_recomputed'] and not report['thresholds_changed']
    assert hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()==freeze['checkpoint_sha256']
    assert json.loads(z.read('trace_restoration.json'))['valid']
normalized=dict(valid=True,runtime_independent=True,local=str(archive),verified_at=time.time(),
    remote=dict(label='final',archive=receipt['archive'],sha256=receipt['sha256'],bytes=receipt['bytes'],platform='Kaggle persisted batch output'),
    kaggle_receipt=receipt,input_recovery=json.loads((ROOT/'artifacts/task41h_kaggle_input_receipt.json').read_text()),
    final_stage_input=json.loads((ROOT/'artifacts/task41h_stage5_input_receipt.json').read_text()))
(D/'external_backup_receipt.json').write_text(json.dumps(normalized,indent=2))
(D/'job_status.json').write_text(json.dumps(dict(state='finished',phase='done',worker_exit_code=0,technical_failure=False,scientific_valid=True,revision=receipt['revision'],decision=report['decision']),indent=2))
print(json.dumps(dict(valid=True,sha256=receipt['sha256'],bytes=receipt['bytes'],decision=report['decision'])),flush=True)
