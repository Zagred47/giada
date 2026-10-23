"""Seal a frozen-evaluation dispatch and bounded input transport parts."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41h_frozen_resume'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
for name in ('scripts/task41b_console_manager.py','scripts/resume_task41h_frozen_evaluation.py'):
    assert subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)==(ROOT/name).read_bytes()
package=json.loads((D/'package_receipt.json').read_text())
request=dict(task='41h',revision=revision,profile='primary',evaluation_only=True,frozen_evaluation_resume=True,
    recovery_seal='/content/unused_frozen_selection_seal.json', # Compatibility key; frozen resume never opens it.
    worker_sha256=sha(ROOT/'scripts/resume_task41h_frozen_evaluation.py'),manager_sha256=sha(ROOT/'scripts/task41b_console_manager.py'),
    recovery_archive='/content/task41h_frozen_resume_input.zip',recovery_archive_sha256=package['sha256'],inputs=[])
for local,remote in [('artifacts/task41b_opened_inputs.zip','/content/task41b_opened_inputs.zip'),
    ('artifacts/task41d_opened41c.zip','/content/task41d_opened41c.zip'),(package['archive'],request['recovery_archive'])]:
    path=Path(local);path=path if path.is_absolute() else ROOT/path
    item=dict(local=str(path),remote=remote,sha256=sha(path),parts=[])
    with path.open('rb') as f:
        n=0
        while block:=f.read(32*1024*1024):
            part=D/(path.name+f'.upload{n:04d}')
            part.write_bytes(block)
            item['parts'].append(dict(local=str(part),remote=remote+f'.upload{n:04d}',sha256=sha(part)));n+=1
    request['inputs'].append(item)
assert not (D/'dispatch_request.json').exists()
(D/'dispatch_request.json').write_text(json.dumps(request,indent=2));print(json.dumps(request),flush=True)
