"""Single launch after off-cloud inputs and cancellation have been verified."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52f_resume_colab_v1'
request=json.loads((F/'dispatch_request.json').read_text())
assert json.loads((ROOT/'experiments/results/task52f_resume_retry_v1/status.json').read_text())['status']=='cancel_acknowledged'
inputs=json.loads((F/'input_verified.json').read_text());assert inputs['valid']
assert inputs['inputs']==[dict(remote=i['remote'],sha256=i['sha256']) for i in request['inputs']]
assert hashlib.sha256((ROOT/'scripts/task52f_colab_resume_manager.py').read_bytes()).hexdigest()==request['manager_sha256']
smoke=json.loads((F/'gpu_smoke.json').read_text());assert smoke['valid'] and smoke['gpu']=='Tesla T4'
assert not (F/'launch_receipt.json').exists(),'Already launched: inspect/attach, never duplicate'
dest=ROOT/'artifacts/task52f_resume_colab_external';dest.mkdir(exist_ok=True)
with (dest/'supervisor.log').open('a') as log:
    proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/task52f_colab_resume_supervisor.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
receipt=dict(valid=True,state='supervisor_dispatched',session=request['session'],profile=request['profile'],pid=proc.pid,launched_at=time.time(),revision=request['revision'],resumed_from=600,target_update=800,thresholds_changed=False,abstraction_changed=False,kaggle_cancelled=True,inputs_verified=True,db_backup_local=True)
(F/'launch_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
