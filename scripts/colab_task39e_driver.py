"""CLI kernel driver: streams the active job and preserves native status files.

This remains an actual active kernel execution while the scientific worker
runs. Independent inspection uses the Contents API (download), not a second
exec request queued behind the training kernel.
"""
import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import sys
import time

request_path=Path('/content/giada_task39e_launch.json')
request=json.loads(request_path.read_text())
manager=Path('/content/task39e_console_manager.py')
assert hashlib.sha256(manager.read_bytes()).hexdigest()==request['manager_sha256']
proc=subprocess.Popen([sys.executable,'-u',str(manager),str(request_path)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
status=Path('/content')/('giada_task39e_'+request['revision'][:12])/'job_status.json';start=time.monotonic()
while True:
    events=selector.select(timeout=5)
    if events:
        line=proc.stdout.readline()
        if line:print(line.rstrip(),flush=True)
        elif proc.poll() is not None:break
    elif proc.poll() is not None:break
    else:
        state=json.loads(status.read_text()) if status.exists() else {'phase':'starting'}
        print(f'[GIADA console] active-job phase={state.get("phase")} elapsed={time.monotonic()-start:.0f}s',flush=True)
for line in proc.stdout:print(line.rstrip(),flush=True)
raise SystemExit(proc.wait())
