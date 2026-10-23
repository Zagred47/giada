"""Private, checksum-sealed remaining-evaluation input package."""
import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41h_frozen_evaluation_primary_external'
DEST=ROOT/'artifacts/task41h_kaggle_inputs';DEST.mkdir(exist_ok=True)
receipt=json.loads((D/'external_backup_receipt.json').read_text());source=Path(receipt['local'])
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert receipt['valid'] and sha(source)==receipt['remote']['sha256']
with zipfile.ZipFile(source) as z:
    assert z.testzip() is None
    previous=json.loads(z.read('frozen_resume_manifest.json'));freeze=json.loads(z.read('selection_freeze.json'))
    manifest=dict(valid=True,source_archive_sha256=receipt['remote']['sha256'],original_seal=previous['original_seal'],
        freeze_sha256=freeze['freeze_sha256'],training_updates_performed=0,thresholds_changed=False,members={})
    path=DEST/'remaining_evaluation_input.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        for name in z.namelist():
            if name=='frozen_resume_manifest.json' or not name.endswith(('.json','.pt')):continue
            raw=z.read(name);manifest['members'][name]=dict(source=source.name,sha256=hashlib.sha256(raw).hexdigest())
            out.writestr(name,raw)
        out.writestr('frozen_resume_manifest.json',json.dumps(manifest,indent=2))
for name in ('task41b_opened_inputs.zip','task41d_opened41c.zip'):shutil.copy2(ROOT/'artifacts'/name,DEST/name)
metadata=dict(title='GIADA Task41h Frozen Evaluation Inputs',id='alessandrobelli/giada-task41h-frozen-evaluation-inputs',
    licenses=[dict(name='other')],description='Private recovery inputs. All rights retained. Frozen models and completed metrics; no training or selection.')
(DEST/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
receipt=dict(valid=True,source=receipt,inputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in DEST.glob('*.zip')})
(ROOT/'artifacts/task41h_kaggle_input_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt['inputs']),flush=True)
