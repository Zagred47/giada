"""Pin the tested technical-only correction; never overwrite a dispatch."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];folder=root/'experiments/results/task41_protocol_v1'
assert json.loads((folder/'execution_amendment.json').read_text())['valid']
assert json.loads((folder/'synaptic_semantics_probe.json').read_text())['valid']
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
request=dict(task='41',revision=revision,technical_retry_of='ac24e4a442267bed254521874c2dd16466e0c7d4')
for field,name in [('worker_sha256','scripts/run_task41_independent_confirmation.py'),('manager_sha256','scripts/task39e_console_manager.py')]:
    raw=subprocess.check_output(['git','show',revision+':'+name],cwd=root);assert raw==(root/name).read_bytes()
    request[field]=hashlib.sha256(raw).hexdigest()
target=folder/'retry_dispatch_request.json';assert not target.exists()
target.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request),flush=True)
