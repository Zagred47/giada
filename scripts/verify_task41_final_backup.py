"""Record final archive safety independently of intermediate recovery polling."""
import hashlib,json,time,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];dest=root/'artifacts/task41_retry_external'
state=json.loads((dest/'job_status.json').read_text());assert state['state']=='finished'
path=dest/Path(state['archive']).name
assert hashlib.sha256(path.read_bytes()).hexdigest()==state['archive_sha256']
with zipfile.ZipFile(path) as z:assert z.testzip() is None
receipt=dict(valid=True,remote=dict(label='final',archive=state['archive'],sha256=state['archive_sha256'],bytes=path.stat().st_size),
    local=str(path),verified_at=time.time(),runtime_independent=True)
(dest/'external_backup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
