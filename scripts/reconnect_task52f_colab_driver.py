"""Retry transport only after proving the scientific worker never started."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52f_resume_colab_v1'
probe=subprocess.run(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),'exec','-s','giada-task52f-resume600','-f','scripts/probe_task52f_colab_resume.py'],cwd=ROOT,capture_output=True,text=True,timeout=90)
assert probe.returncode==0,'Cannot prove worker absence'
state=next(json.loads(line) for line in probe.stdout.splitlines() if line.startswith('{"root_exists"'))
assert state==dict(root_exists=False,status=None),'Existing worker directory: inspect instead of relaunch'
dest=ROOT/'artifacts/task52f_resume_colab_external'
with (dest/'console_reconnect.log').open('a') as log:
    p=subprocess.Popen(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),'exec','-s','giada-task52f-resume600','-f','scripts/colab_task52f_resume_driver.py','--timeout','18000'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
(F/'connection_recovery.json').write_text(json.dumps(dict(valid=True,worker_absence_verified=state,driver_pid=p.pid,at=time.time(),new_runtime_allocated=False,training_restarted=False),indent=2))
print(json.dumps(dict(driver_pid=p.pid,worker_absence_verified=True)))
