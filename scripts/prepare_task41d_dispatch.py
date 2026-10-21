import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'experiments/results/task41d_protocol_v1'
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();request=dict(task='41d',revision=revision)
for field,name in [('worker_sha256','scripts/run_task41d_precision_response.py'),('manager_sha256','scripts/task41b_console_manager.py')]:
    raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT);assert raw==(ROOT/name).read_bytes();request[field]=hashlib.sha256(raw).hexdigest()
pf=json.loads((folder/'gpu_preflight.json').read_text());assert pf['valid'] and pf['all_parameters_finite']
for field,name in [('source_sha256','src/giada_teacher/task41d_precision_response.py'),('protocol_sha256','experiments/task41d_preregistration_v1.json')]:assert pf[field]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
assert json.loads((folder/'memory_receipt.json').read_text())['valid']
path=folder/'dispatch_request.json';assert not path.exists();path.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request))
