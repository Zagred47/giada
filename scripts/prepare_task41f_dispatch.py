"""Bind the GPU-tested commit and actual pretraining DB receipt."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'experiments/results/task41f_protocol_v1'
pf=json.loads((folder/'gpu_preflight.json').read_text())
assert pf['valid'] and pf['all_parameters_finite'] and pf['whole_loop_4096_parity_passed']
assert pf['objective_probe_executed'] and pf['oracle_branches_executed']
assert json.loads((folder/'memory_receipt.json').read_text())['registration_before_scientific_run']
revision=pf['revision'];spec=json.loads((ROOT/'experiments/task41f_preregistration_v1.json').read_text())
assert hashlib.sha256((ROOT/'experiments/task41f_preregistration_v1.json').read_bytes()).hexdigest()==pf['protocol_sha256']
request=dict(task='41f',revision=revision,profile=pf['account_profile'],inputs=[])
for key,name in [('worker_sha256','scripts/run_task41f_precision_relaxation.py'),('manager_sha256','scripts/task41b_console_manager.py')]:
    raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
    assert raw==(ROOT/name).read_bytes()
    request[key]=hashlib.sha256(raw).hexdigest()
for name,sha in spec['sources'].items():
    assert hashlib.sha256(subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)).hexdigest()==sha,name
for name in ('artifacts/task41b_opened_inputs.zip','artifacts/task41d_opened41c.zip'):
    sha=hashlib.sha256()
    with (ROOT/name).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):sha.update(chunk)
    request['inputs'].append(dict(remote='/content/'+Path(name).name,sha256=sha.hexdigest()))
path=folder/'dispatch_request.json';assert not path.exists();path.write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps(request))
