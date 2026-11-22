import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52f_resume_colab_v1';request=json.loads((F/'dispatch_request.json').read_text())
code="import hashlib,json;from pathlib import Path;r=json.loads(Path('/content/task52f_resume_request.json').read_text());assert not Path(r['remote_root']).exists();assert hashlib.sha256(Path('/content/task52f_colab_resume_manager.py').read_bytes()).hexdigest()==r['manager_sha256'];print('SSH_READY_NO_WORKER')"
r=subprocess.run(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),'ssh','-s',request['session'],'--','python3','-c',code],capture_output=True,text=True,cwd=ROOT,timeout=90)
assert r.returncode==0 and 'SSH_READY_NO_WORKER' in r.stdout,('Existing worker or connection not verified: '+r.stdout[-1200:]+' '+r.stderr[-1200:])
dest=ROOT/'artifacts/task52f_resume_colab_external'
with (dest/'ssh_supervisor.log').open('a') as log:
    p=subprocess.Popen([sys.executable,str(ROOT/'scripts/task52f_colab_resume_supervisor.py'),'--ssh'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
(F/'ssh_launch_receipt.json').write_text(json.dumps(dict(valid=True,pid=p.pid,session=request['session'],at=time.time(),worker_absence_verified=True,manager_hash_verified=True,transport='official_colab_cli_ssh',new_runtime=False,resumed_from=600,thresholds_changed=False),indent=2))
print(json.dumps(dict(supervisor_pid=p.pid,transport='official_colab_cli_ssh')))
