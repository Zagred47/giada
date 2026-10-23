"""Benign post-evaluation kernel activity until the verified PC copy is ready."""
import json,time
from pathlib import Path
request=json.loads(Path('/content/giada_task41h_launch.json').read_text())
root=Path('/content')/('giada_task41h_'+request['revision'][:12])
status=json.loads((root/'job_status.json').read_text())
assert status['state']=='finished' and status['worker_exit_code']==0
expected=json.loads((root/'recovery_manifest.json').read_text())['sha256']
deadline=time.monotonic()+7200
while True:
    ack=root/'final_guard_ack.json'
    if ack.exists():
        receipt=json.loads(ack.read_text());assert receipt['valid'] and receipt['remote']['sha256']==expected
        print('[GIADA final guard] external final backup acknowledged',flush=True);break
    assert time.monotonic()<deadline,'Final guard expired'
    print('[GIADA final guard] evaluations finished; preserving runtime for external transfer',flush=True)
    time.sleep(20)
