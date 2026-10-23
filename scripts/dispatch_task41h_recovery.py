"""Pin evaluation recovery, original inputs and immutable checkpoint archive."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
folder=ROOT/'experiments/results/task41h_recovery_v1';seal=json.loads((folder/'input_seal.json').read_text())
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
for name in ('scripts/recover_task41h_evaluation.py','scripts/task41b_console_manager.py'):
    assert subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)==(ROOT/name).read_bytes()
request=dict(task='41h',revision=revision,profile='primary',evaluation_only=True,
    worker_sha256=sha(ROOT/'scripts/recover_task41h_evaluation.py'),manager_sha256=sha(ROOT/'scripts/task41b_console_manager.py'),
    recovery_archive='/content/task41h_recovery_input.zip',recovery_seal='/content/task41h_input_seal.json',inputs=[])
for local,remote in [('artifacts/task41b_opened_inputs.zip','/content/task41b_opened_inputs.zip'),('artifacts/task41d_opened41c.zip','/content/task41d_opened41c.zip'),(seal['source_backup'],request['recovery_archive']),('experiments/results/task41h_recovery_v1/input_seal.json',request['recovery_seal'])]:
    path=Path(local);path=path if path.is_absolute() else ROOT/path
    request['inputs'].append(dict(local=str(path),remote=remote,sha256=sha(path)))
assert request['inputs'][2]['sha256']==seal['source_archive_sha256']
assert not (folder/'dispatch_request.json').exists()
write_json(folder/'dispatch_request.json',request);print(json.dumps(request),flush=True)
