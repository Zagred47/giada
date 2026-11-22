import argparse,json,subprocess,time,zipfile
from pathlib import Path
from task41b_local_supervisor import cli,file_sha
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52g_protocol_v1'
def rotate(dest,newest):
    """Keep two complete copies; never discard a unique immutable checkpoint."""
    archives=sorted(dest.glob('recovery_*.zip'),key=lambda p:p.stat().st_mtime,reverse=True)
    removed=[]
    with zipfile.ZipFile(newest) as z:latest={i.filename:(i.CRC,i.file_size) for i in z.infolist()}
    for old in archives[2:]:
        if old.resolve().parent!=dest.resolve() or old.is_symlink():raise RuntimeError('Unsafe backup retention target')
        with zipfile.ZipFile(old) as z:
            if z.testzip() is not None:continue
            immutable=[i for i in z.infolist() if (i.filename.endswith(('.pt','.npz')) and not i.filename.endswith('resumable_training.pt'))]
            if any(latest.get(i.filename)!=(i.CRC,i.file_size) for i in immutable):continue
        removed.append(dict(name=old.name,bytes=old.stat().st_size));old.unlink()
    if removed:
        path=dest/'retention_receipt.json';history=json.loads(path.read_text()) if path.exists() else []
        history.append(dict(at=time.time(),newest=newest.name,removed=removed,immutable_checkpoints_preserved=True));path.write_text(json.dumps(history,indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attach',action='store_true');a=ap.parse_args();request=json.loads((F/'dispatch_request.json').read_text());session=request['session'];remote=request['remote_root'];dest=ROOT/'artifacts/task52g_colab_external';dest.mkdir(exist_ok=True);seen=set()
    with (dest/'console.log').open('a') as log:
        driver=None if a.attach else subprocess.Popen(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),'ssh','-s',session,'--','python3','-u','/content/task52g_console_manager.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        def download(name,path):
            r=cli(ROOT,'download','-s',session,remote+'/'+name,str(path),capture_output=True,text=True,timeout=90);return r.returncode==0 and path.exists()
        for poll in range(500):
            try:
                download('process.log',dest/'process.log');got=download('job_status.json',dest/'job_status.json')
                if download('recovery_manifest.json',dest/'remote_recovery_manifest.json'):
                    r=json.loads((dest/'remote_recovery_manifest.json').read_text());target=dest/Path(r['archive']).name
                    if r['sha256'] not in seen and download(target.name,target):
                        assert file_sha(target)==r['sha256']
                        with zipfile.ZipFile(target) as z:assert z.testzip() is None
                        seen.add(r['sha256']);(dest/'external_backup_receipt.json').write_text(json.dumps(dict(valid=True,remote=r,local=str(target),runtime_independent=True,verified_at=time.time()),indent=2))
                        rotate(dest,target)
                if got:
                    state=json.loads((dest/'job_status.json').read_text())
                    if state.get('state')=='finished':
                        r=json.loads((dest/'external_backup_receipt.json').read_text())
                        if r['remote']['label']=='final' and r['remote']['sha256']==state['archive_sha256']:return
                print(json.dumps(dict(poll=poll,verified=len(seen),driver_exit=driver.poll() if driver else None)),flush=True)
            except (OSError,ValueError,subprocess.TimeoutExpired) as error:print('Read-only retry '+type(error).__name__,flush=True)
            time.sleep(45)
        raise RuntimeError('Monitoring deadline; inspect existing worker, never duplicate')
if __name__=='__main__':main()
