"""Finite task-scoped console driver and verified off-runtime recovery."""
import argparse,json,subprocess,time,zipfile
from pathlib import Path
from task41b_local_supervisor import cli,file_sha,recover_parts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',required=True);ap.add_argument('--destination',type=Path,required=True);ap.add_argument('--attach',action='store_true');a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];request=json.loads((repo/'experiments/results/task41m_protocol_v1/dispatch_request.json').read_text());remote=request['remote_root']
    dest=a.destination.resolve();dest.mkdir(parents=True,exist_ok=True);seen=set()
    with (dest/'console.log').open('a',encoding='utf-8') as log:
        driver=None if a.attach else subprocess.Popen(['powershell.exe','-NoProfile','-File',str(repo/'scripts/colab.ps1'),'exec','-s',a.session,'-f','scripts/colab_task41m_driver.py','--timeout','7200'],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        def download(name,target):
            result=cli(repo,'download','-s',a.session,remote+'/'+name,target,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
            return result.returncode==0 and target.exists() and 'not found' not in (result.stdout+result.stderr).lower()
        for poll in range(240):
            try:
                download('process.log',dest/'process.log');got=download('job_status.json',dest/'job_status.json')
                if download('recovery_manifest.json',dest/'remote_recovery_manifest.json'):
                    r=json.loads((dest/'remote_recovery_manifest.json').read_text());target=dest/Path(r['archive']).name
                    if r['sha256'] not in seen and recover_parts(r,target,dest,download):
                        assert file_sha(target)==r['sha256']
                        with zipfile.ZipFile(target) as z:assert z.testzip() is None
                        seen.add(r['sha256']);(dest/'external_backup_receipt.json').write_text(json.dumps(dict(valid=True,remote=r,local=str(target),runtime_independent=True,verified_at=time.time()),indent=2))
                if got:
                    state=json.loads((dest/'job_status.json').read_text())
                    if state.get('state')=='finished':
                        receipt=json.loads((dest/'external_backup_receipt.json').read_text())
                        if receipt['remote']['label']=='final' and receipt['remote']['sha256']==state['archive_sha256']:
                            (dest/'supervisor_receipt.json').write_text(json.dumps(dict(valid=True,finished=True,scientific_completed=not state.get('technical_failure'),session=a.session,revision=request['revision'],training_updates_performed=0)))
                            return
                print(json.dumps(dict(poll=poll,bundles_verified=len(seen),driver_exit=driver.poll() if driver else None)),flush=True)
            except (subprocess.TimeoutExpired,OSError,ValueError) as error:print('Read-only recovery retry '+repr(error),flush=True)
            time.sleep(45)
        raise RuntimeError('Finite backup workflow ended without final receipt; do not infer completion or restart worker.')

if __name__=='__main__':main()
