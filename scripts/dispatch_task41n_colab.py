"""Create a pinned request only after verified local preregistration."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'experiments/results/task41n_protocol_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
spec=json.loads((ROOT/'experiments/task41n_preregistration_v1.json').read_text());memory=json.loads((folder/'memory_receipt.json').read_text())
assert memory['valid'] and memory['registration_before_run'] and memory['protocol_sha256']==sha(ROOT/'experiments/task41n_preregistration_v1.json')
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
request=dict(task='41n',revision=revision,session='giada-task41n-primary',profile='primary',remote_root='/content/giada_task41n_'+revision[:12],
    protocol_sha256=memory['protocol_sha256'],worker_sha256=sha(ROOT/'scripts/run_task41n_independent.py'),manager_sha256=sha(ROOT/'scripts/task41n_colab_manager.py'),training_updates=0,
    inputs=[dict(local=spec['frozen_input']['local'],remote='/content/task41n_frozen.zip.bin',sha256=spec['frozen_input']['sha256'],bytes=spec['frozen_input']['bytes'])],
    provider_reason='Kaggle GPU quota112269.406s used/108000s allowed,refresh2026-10-31T00:00:00Z. Authorized Colab primary fallback; no GPU allocated before DB preregistration.')
(folder/'dispatch_request.json').write_text(json.dumps(request,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(dict(revision=revision,session=request['session'],remote_root=request['remote_root'])))
