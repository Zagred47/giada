"""Local-only finalization after the verified off-runtime result arrives."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41h_frozen_evaluation_primary_external'
deadline=time.monotonic()+14400
while True:
    try:
        status=json.loads((D/'job_status.json').read_text())
        receipt=json.loads((D/'external_backup_receipt.json').read_text())
    except (OSError,json.JSONDecodeError):status={};receipt={}
    if status.get('state')=='finished':
        assert status.get('worker_exit_code')==0 and not status.get('technical_failure'),status
        if receipt.get('valid') and receipt.get('remote',{}).get('label')=='final':break
    assert time.monotonic()<deadline,'Awaiting complete final external receipt'
    time.sleep(10)
print('[GIADA local recovery] verified final transfer observed; merging immutable sources',flush=True)
subprocess.run([sys.executable,'-u',str(ROOT/'scripts/merge_task41h_frozen_final.py')],cwd=ROOT,check=True)
print('[GIADA local recovery] auditing all fixed panels and paired speed comparisons',flush=True)
subprocess.run([sys.executable,'-u',str(ROOT/'scripts/compare_task41h_recovery.py'),'--external-folder',str(ROOT/'artifacts/task41h_complete_evaluation_external')],cwd=ROOT,check=True)
latest=json.loads((ROOT/'experiments/results/task41h_recovery_v1/LATEST.json').read_text())
assert latest['complete'] and latest['valid']
print('[GIADA local recovery] complete, frozen selection unchanged',flush=True)
