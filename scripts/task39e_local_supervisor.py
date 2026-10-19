"""Keep the Colab exec alive and continuously retrieve immutable recovery bundles.

Run on the PC, not the VM. No cloud credentials are handled here. Only the
official CLI is used. A missing session NEVER counts as successful completion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

def cli(repo, *args, **kwargs):
    return subprocess.run(['powershell.exe','-NoProfile','-File',str(repo/'scripts/colab.ps1'),*map(str,args)],cwd=repo,**kwargs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--revision',required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--task',choices=['39e','39f','39g','39h','39i'],default='39e');p.add_argument('--attach',action='store_true');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];dest=a.destination.resolve();dest.mkdir(parents=True,exist_ok=a.attach)
    remote='/content/giada_task'+a.task+'_'+a.revision[:12];seen=set();started=time.time()
    with (dest/'console.log').open('a',encoding='utf-8') as log:
        driver=None if a.attach else subprocess.Popen(['powershell.exe','-NoProfile','-File',str(repo/'scripts/colab.ps1'),'exec','-s',a.session,'-f','scripts/colab_task'+a.task+'_driver.py'],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        def download(name,target):
            r=cli(repo,'download','-s',a.session,remote+'/'+name,target,capture_output=True,text=True,encoding='utf-8',errors='replace')
            return r.returncode==0 and target.exists() and 'not found' not in (r.stdout+r.stderr).lower()
        finished=False;console_disconnected=False
        while True:
            download('process.log',dest/'process.log')
            got=download('job_status.json',dest/'job_status.json')
            if download('recovery_manifest.json',dest/'remote_recovery_manifest.json'):
                receipt=json.loads((dest/'remote_recovery_manifest.json').read_text())
                if receipt['sha256'] not in seen:
                    name=Path(receipt['archive']).name;target=dest/name
                    if download(name,target):
                        assert hashlib.sha256(target.read_bytes()).hexdigest()==receipt['sha256'],'External recovery SHA mismatch'
                        with zipfile.ZipFile(target) as z:assert z.testzip() is None
                        seen.add(receipt['sha256']);(dest/'external_backup_receipt.json').write_text(json.dumps(dict(valid=True,remote=receipt,local=str(target),verified_at=time.time(),runtime_independent=True),indent=2))
            # Re-read after a completed driver: the pre-poll status may still
            # say running although the final bundle has already been copied.
            if driver is not None and driver.poll() is not None:
                got=download('job_status.json',dest/'job_status.json')
            if got:
                state=json.loads((dest/'job_status.json').read_text())
                if state.get('state')=='finished':
                    # manager publishes final bundle before final status.
                    finished=True
                    if state.get('archive'):
                        target=dest/Path(state['archive']).name
                        assert download(target.name,target),'Final archive download failed'
                        assert hashlib.sha256(target.read_bytes()).hexdigest()==state['archive_sha256']
                        with zipfile.ZipFile(target) as z:assert z.testzip() is None
                    break
            if driver is not None and driver.poll() is not None:
                if got and state.get('state')=='running':
                    # The CLI can lose its shell reply while the independent
                    # manager/worker is still running. Keep Contents recovery
                    # alive; never launch a second scientific worker here.
                    if not console_disconnected:
                        print(json.dumps(dict(console_disconnected=True,driver_exit=driver.poll(),continued_external_recovery=True)),flush=True)
                        console_disconnected=True
                elif time.time()-started>120:break
            print(json.dumps(dict(recovery_poll=True,bundles=len(seen),state=state.get('phase') if got else 'unavailable')),flush=True)
            time.sleep(30)
        (dest/'supervisor_receipt.json').write_text(json.dumps(dict(finished=finished,driver_exit=driver.poll() if driver else None,external_bundles=len(seen),elapsed_seconds=time.time()-started,session=a.session,revision=a.revision),indent=2))
        return 0 if finished else 1
if __name__=='__main__':raise SystemExit(main())
