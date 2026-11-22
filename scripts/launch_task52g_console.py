import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52g_protocol_v1'
for name in ('preregistration_receipt.json','local_preflight.json','input_verified.json','gpu_preflight.json'):assert json.loads((F/name).read_text())['valid']
assert not (F/'launch_receipt.json').exists(),'Existing52g launch: inspect instead of duplicating'
request=json.loads((F/'dispatch_request.json').read_text());code="from pathlib import Path;assert not Path("+repr(request['remote_root'])+").exists();print('NO_WORKER')"
r=subprocess.run(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),'ssh','-s',request['session'],'--','python3','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=90);assert r.returncode==0 and 'NO_WORKER' in r.stdout
dest=ROOT/'artifacts/task52g_colab_external';dest.mkdir(exist_ok=True)
with (dest/'supervisor.log').open('a') as log:proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/task52g_console_supervisor.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
(F/'launch_receipt.json').write_text(json.dumps(dict(valid=True,session=request['session'],pid=proc.pid,revision=request['revision'],at=time.time(),thresholds_changed=False,abstraction_changed=False,worker_absence_verified=True,external_backup_supervisor=True),indent=2));print('52g dispatched '+str(proc.pid))
