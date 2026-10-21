"""Pin tested source; the request is not a claim of execution."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'experiments/results/task41_protocol_v1'
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();request=dict(task='41',revision=revision)
for field,name in [('worker_sha256','scripts/run_task41_independent_confirmation.py'),('manager_sha256','scripts/task39e_console_manager.py')]:
    raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT);assert raw==(ROOT/name).read_bytes()
    request[field]=hashlib.sha256(raw).hexdigest()
pf=json.loads((folder/'gpu_preflight.json').read_text());assert pf['valid'] and not pf['fresh_confirmation_accessed']
assert pf['source_sha256']==hashlib.sha256((ROOT/'src/giada_teacher/task41_independent_confirmation.py').read_bytes()).hexdigest()
assert pf['protocol_sha256']==hashlib.sha256((ROOT/'experiments/task41_preregistration_v1.json').read_bytes()).hexdigest()
assert pf['worker_sha256']==request['worker_sha256']
assert json.loads((folder/'memory_receipt.json').read_text())['valid']
target=folder/'dispatch_request.json';assert not target.exists();target.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request))
