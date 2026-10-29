"""Finite off-runtime backup workflow; never starts a second scientific worker."""
import argparse,json,subprocess,time,zipfile
from pathlib import Path
from task41b_local_supervisor import cli,file_sha,recover_parts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',required=True);ap.add_argument('--destination',type=Path,required=True);ap.add_argument('--attach',action='store_true');a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];request=json.loads((repo/'experiments/results/task41l_colab_evaluation_v1/dispatch_request.json').read_text())
    revision=request['revision'];remote='/content/giada_task41l_'+revision[:12];dest=a.destination.resolve();dest.mkdir(parents=True,exist_ok=True);seen=set()
    with (dest/'console.log').open('a',encoding='utf-8') as log:
        driver=None if a.attach else subprocess.Popen(['powershell.exe','-NoProfile','-File',str(repo/'scripts/colab.ps1'),'exec','-s',a.session,'-f','scripts/colab_task41l_driver.py','--timeout','86400'],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        def download(name,target):
            r=cli(repo,'download','-s',a.session,remote+'/'+name,target,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
            return r.returncode==0 and target.exists() and 'not found' not in (r.stdout+r.stderr).lower()
        for poll in range(1440):
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
                        if state.get('technical_failure'):raise RuntimeError('Evaluation manager failed: '+repr(state))
                        receipt=json.loads((dest/'external_backup_receipt.json').read_text())
                        if receipt['remote']['label']=='final' and receipt['remote']['sha256']==state['archive_sha256']:
                            (dest/'supervisor_receipt.json').write_text(json.dumps(dict(valid=True,finished=True,session=a.session,revision=revision,training_updates_performed=0,selection_recomputed=False)));return
                print(json.dumps(dict(poll=poll,bundles_verified=len(seen),driver_exit=driver.poll() if driver else None)),flush=True)
            except (subprocess.TimeoutExpired,OSError,ValueError) as exc:print('Read-only backup transport retry: '+repr(exc),flush=True)
            time.sleep(45)
        raise RuntimeError('Finite backup workflow time limit reached; do not infer completion')

if __name__=='__main__':main()
