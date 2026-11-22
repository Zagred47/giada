import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52f_resume_colab_v1';F.mkdir(exist_ok=True)
items=[
 ('artifacts/task41n_primary_external/recovery_final.zip','source41n.zip.bin','7e07984c6c8770fee223a83f4694922c2f8cb92529e3df899bc511cf8d5080be'),
 ('artifacts/task45c_cost_input/checkpoint45c.zip.bin','checkpoint45c.zip.bin','555133b7fc71719ecf459e2b04ffed8fd22a50af198bdb409735e0d8ca996afe'),
 ('artifacts/task45d_evaluation_input/evaluation45d.zip.bin','evaluation45d.zip.bin','daabec1f2e998bdc538c2cd7303470d895ad30bedf2124159b1bcad6e2ebc2f7'),
 ('artifacts/task52e_input/models52e.zip.bin','models52e.zip.bin','8df0d9ec12f5690e20dfee766eedbdf914a2ead984f012280302e69f1692fe49'),
 ('artifacts/task52f_resume_input/resume52f.zip.bin','resume52f.zip.bin','86012728909ac23f20bbce0cabed479d7d6546495b99e41ab095523890d5cf18')]
inputs=[]
for local,name,digest in items:
    with (ROOT/local).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    inputs.append(dict(local=local,remote='/content/'+name,sha256=digest,bytes=(ROOT/local).stat().st_size))
assert json.loads((ROOT/'experiments/results/task52f_resume_retry_v1/status.json').read_text())['status']=='cancel_acknowledged'
receipt=json.loads((ROOT/'experiments/results/task52f_resume_v1/preregistration_receipt.json').read_text());assert receipt['valid']
request=dict(task='52f',session='giada-task52f-resume600',profile='primary',revision='81f4a1600bf60965ba9661db3a7338036be6af32',remote_root='/content/giada_task52f_resume600',inputs=inputs,
 manager_sha256=hashlib.sha256((ROOT/'scripts/task52f_colab_resume_manager.py').read_bytes()).hexdigest(),resumed_from=600,target_update=800,thresholds_changed=False,abstraction_changed=False,registration_receipt=receipt)
(F/'dispatch_request.json').write_text(json.dumps(request,indent=2))
print(json.dumps(dict(valid=True,inputs_verified=len(inputs),resumed_from=600,target_update=800)))
