"""Produce a pinned console request, without allocating or launching remotely."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
old=json.loads((ROOT/'experiments/results/task41l_colab_evaluation_v1/dispatch_request.json').read_text())
reuse=json.loads((ROOT/'experiments/results/task41l_audited_v1/reuse_manifest.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
inputs=old['inputs'][1:]+[dict(local=reuse['local'],remote='/content/task41m_reuse.zip.bin',sha256=reuse['sha256'],bytes=reuse['bytes'])]
r=dict(task='41m',revision=revision,session='giada-task41m-frozen-primary',profile='primary',remote_root='/content/giada_task41m_'+revision[:12],
    training_updates=0,selection_performed=False,worker_sha256=sha(ROOT/'scripts/run_task41m_frozen_contract.py'),manager_sha256=sha(ROOT/'scripts/task41m_colab_manager.py'),
    protocol_sha256=sha(ROOT/'experiments/task41m_preregistration_v1.json'),inputs=inputs,
    environment=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',GIADA_PROVIDER='Colab',GIADA_TASK41_INPUTS='/content/task41b_opened_inputs.zip',GIADA_TASK41C_INPUTS='/content/task41d_opened41c.zip',GIADA_TASK41L_FROZEN_INPUT='/content/task41k_frozen_diagnostic.zip.bin',GIADA_TASK41M_REUSE='/content/task41m_reuse.zip.bin'))
folder=ROOT/'experiments/results/task41m_protocol_v1';folder.mkdir(exist_ok=True)
(folder/'dispatch_request.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(dict(revision=revision,remote_root=r['remote_root'])))
