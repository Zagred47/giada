"""One console driver and verified off-runtime backups; never relaunch training."""
import json,subprocess,time,zipfile,sys
from pathlib import Path
from task41b_local_supervisor import cli,file_sha

ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'experiments/results/task52f_resume_colab_v1'
def main():
    request=json.loads((F/'dispatch_request.json').read_text());session=request['session'];remote=request['remote_root']
    dest=ROOT/'artifacts/task52f_resume_colab_external';dest.mkdir(exist_ok=True)
    with (dest/'console.log').open('a',encoding='utf-8') as log:
        ssh='--ssh' in sys.argv
        command=['ssh','-s',session,'--','python3','-u','/content/task52f_colab_resume_manager.py'] if ssh else ['exec','-s',session,'-f','scripts/colab_task52f_resume_driver.py','--timeout','18000']
        driver=None if '--attach' in sys.argv else subprocess.Popen(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        seen=set()
        def download(name,path):
            # Colab permits one active SSH connection. Keep it for the driver;
            # Contents transfers are independent and checksum-verified below.
            r=cli(ROOT,'download','-s',session,remote+'/'+name,str(path),capture_output=True,text=True,timeout=120)
            return r.returncode==0 and path.exists()
        for poll in range(400):
            try:
                download('process.log',dest/'process.log')
                got=download('job_status.json',dest/'job_status.json')
                for name in ('gpu_preflight.json','resume_state_verified.json'):
                    if not (dest/name).exists():download('result/'+name,dest/name)
                if download('recovery_manifest.json',dest/'remote_recovery_manifest.json'):
                    receipt=json.loads((dest/'remote_recovery_manifest.json').read_text());target=dest/Path(receipt['archive']).name
                    if receipt['sha256'] not in seen and download(Path(receipt['archive']).name,target):
                        assert file_sha(target)==receipt['sha256']
                        with zipfile.ZipFile(target) as z:assert z.testzip() is None
                        seen.add(receipt['sha256']);(dest/'external_backup_receipt.json').write_text(json.dumps(dict(valid=True,remote=receipt,local=str(target),runtime_independent=True,verified_at=time.time()),indent=2))
                if got:
                    state=json.loads((dest/'job_status.json').read_text())
                    if state.get('state')=='finished':
                        receipt=json.loads((dest/'external_backup_receipt.json').read_text())
                        if receipt['remote']['label']=='final' and receipt['remote']['sha256']==state['archive_sha256']:return
                print(json.dumps(dict(poll=poll,backups_verified=len(seen),driver_exit=driver.poll() if driver else None)),flush=True)
            except (OSError,ValueError,subprocess.TimeoutExpired) as e:print('Read-only retry '+repr(e),flush=True)
            time.sleep(45)
        raise RuntimeError('Supervisor deadline reached; inspect existing worker; do not relaunch')
if __name__=='__main__':main()
