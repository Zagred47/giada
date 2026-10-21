"""Pin tested source, candidate and prospective registration before launch."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'experiments/results/task41c_protocol_v1'
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();request=dict(task='41c',revision=revision)
for field,name in [('worker_sha256','scripts/run_task41c_frozen_confirmation.py'),('manager_sha256','scripts/task41b_console_manager.py')]:
    raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT);assert raw==(ROOT/name).read_bytes()
    request[field]=hashlib.sha256(raw).hexdigest()
pf=json.loads((folder/'gpu_preflight.json').read_text());assert pf['valid'] and not pf['fresh_confirmation_accessed']
for field,name in [('source_sha256','src/giada_teacher/task41c_frozen_confirmation.py'),('protocol_sha256','experiments/task41c_preregistration_v1.json')]:assert pf[field]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
assert json.loads((folder/'memory_receipt.json').read_text())['valid']
target=folder/'dispatch_request.json';assert not target.exists();target.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request))
