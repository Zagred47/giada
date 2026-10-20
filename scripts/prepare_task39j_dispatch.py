"""Pin the tested committed worker; no implicit runtime allocation."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
request=dict(task='39j',revision=revision)
for field,name in [('worker_sha256','scripts/run_task39j_atomic_relaxation.py'),('manager_sha256','scripts/task39e_console_manager.py')]:
    data=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
    assert data==(ROOT/name).read_bytes()
    request[field]=hashlib.sha256(data).hexdigest()
preflight=json.loads((ROOT/'experiments/results/task39j_protocol_v1/gpu_preflight.json').read_text())
assert preflight['valid'] and preflight['selection_preflight_valid']
assert preflight['source_sha256']==hashlib.sha256((ROOT/'src/giada_teacher/task39j_atomic_relaxation.py').read_bytes()).hexdigest()
target=ROOT/'experiments/results/task39j_protocol_v1/dispatch_request.json';assert not target.exists()
target.write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8');print(json.dumps(request))
