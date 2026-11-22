import hashlib,json,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52f_resume_colab_v1';dest=ROOT/'artifacts/task52f_resume_colab_external'
r=json.loads((dest/'remote_recovery_manifest.json').read_text());archive=dest/Path(r['archive']).name
assert archive.stat().st_size==r['bytes']
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
with zipfile.ZipFile(archive) as z,zipfile.ZipFile(ROOT/'artifacts/task52f_resume_input/resume52f.zip.bin') as original:
    assert z.testzip() is None
    state=json.loads(z.read('resume_state_verified.json'));assert state['valid'] and state['checkpoint_update']==600 and state['optimizer_restored'] and state['rng_reconstructed'] and not state['encoder_calibration_repeated']
    gpu=json.loads(z.read('gpu_preflight.json'));assert gpu['valid'] and gpu['torch']=='2.11.0+cu130' and gpu['gpu']=='Tesla T4'
    assert z.read('checkpoint_600.pt')==original.read('checkpoint_600.pt')
    assert z.read('resumable_training.pt')==original.read('resumable_training.pt')
receipt=dict(valid=True,remote=r,local=str(archive),runtime_independent=True,verified_at=time.time())
(dest/'external_backup_receipt.json').write_text(json.dumps(receipt,indent=2))
(F/'execution_verified.json').write_text(json.dumps(dict(valid=True,phase='resumed_from600',state=state,gpu=gpu,backup=receipt,thresholds_changed=False,kaggle_cancelled=True),indent=2))
print(json.dumps(dict(valid=True,resumed_from=600,optimizer_restored=True,rng_reconstructed=True,backup_sha256=r['sha256'],backup_bytes=r['bytes'])))
