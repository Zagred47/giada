"""Exercise detached worker status and binary artifact transport; no training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

root=Path('/content/giada_console_contract_probe')
root.mkdir(exist_ok=True)
uploaded=Path('/content/giada_console_roundtrip_input.json')
assert json.loads(uploaded.read_text())['valid']
worker=root/'worker.py'
worker.write_text('''import json,time
from pathlib import Path
p=Path(__file__).parent
print('WORKER_STARTED',flush=True)
(p/'status.json').write_text(json.dumps({'state':'running'}))
time.sleep(3)
print('EXPECTED_FAILURE_EXIT_7',flush=True)
(p/'status.json').write_text(json.dumps({'state':'finished','exit_code':7,'expected_test_failure':True}))
raise SystemExit(7)
''')
with (root/'worker.log').open('w') as log:
    process=subprocess.Popen([sys.executable,'-u',str(worker)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'pid.json').write_text(json.dumps({'pid':process.pid}))
archive=root/'transfer_probe.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr('payload.bin',bytes(range(256))*4096)
    z.write(uploaded,'uploaded_smoke.json')
print(json.dumps({'worker_pid':process.pid,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'uploaded_sha256':hashlib.sha256(uploaded.read_bytes()).hexdigest()}),flush=True)
