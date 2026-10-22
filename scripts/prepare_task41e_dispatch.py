"""Bind the checked scientific sources, GPU preflight, DB receipt and inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'experiments/results/task41e_protocol_v1'
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
request=dict(task='41e',revision=revision)
for field,name in [('worker_sha256','scripts/run_task41e_transfer.py'),('manager_sha256','scripts/task41b_console_manager.py')]:
    raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
    assert raw==(ROOT/name).read_bytes(),name
    request[field]=hashlib.sha256(raw).hexdigest()
pf=json.loads((folder/'gpu_preflight.json').read_text())
assert pf['valid'] and pf['all_parameters_finite'] and pf['fused_raw_bitwise_equal']
request['profile']=pf['account_profile']
for field,name in [('source_sha256','src/giada_teacher/task41e_transfer_training.py'),('protocol_sha256','experiments/task41e_preregistration_v1.json')]:
    assert pf[field]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
receipt=json.loads((folder/'memory_receipt.json').read_text())
assert receipt['valid'] and receipt['registration_before_scientific_run']
request['inputs']=[]
for name in ('artifacts/task41b_opened_inputs.zip','artifacts/task41d_opened41c.zip'):
    h=hashlib.sha256()
    with (ROOT/name).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    request['inputs'].append(dict(remote='/content/'+Path(name).name,sha256=h.hexdigest()))
path=folder/'dispatch_request.json';assert not path.exists()
path.write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps(request))
