"""Pin tested Task40b revision and generate actual launch request."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'experiments/results/task40b_protocol_v1'
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
request=dict(task='40b',revision=revision)
for field,name in [('worker_sha256','scripts/run_task40b_calcium_precision.py'),('manager_sha256','scripts/task39e_console_manager.py')]:
    data=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT);assert data==(ROOT/name).read_bytes()
    request[field]=hashlib.sha256(data).hexdigest()
preflight=json.loads((folder/'gpu_preflight.json').read_text());assert preflight['valid'] and preflight['smoke']['parameters_finite']
assert preflight['source_sha256']==hashlib.sha256((ROOT/'src/giada_teacher/task40b_calcium_precision.py').read_bytes()).hexdigest()
target=folder/'dispatch_request.json';assert not target.exists();target.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request))
