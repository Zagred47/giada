"""One frozen worker, bounded lifetime, atomically published recovery bundles."""
import hashlib,json,os,selectors,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    request=json.loads(Path('/content/giada_task41m_launch.json').read_text());root=Path(request['remote_root']);repo=root/'giada'
    out=root/'artifacts';out.mkdir(exist_ok=False);start=time.time();proc=None
    def status(**kw):
        path=root/'job_status.tmp';path.write_text(json.dumps(dict(task='41m',revision=request['revision'],elapsed_seconds=time.time()-start,**kw)));path.replace(root/'job_status.json')
    def publish(label):
        shutil.copy2(root/'process.log',out/'process.log')
        archive=root/f'recovery_{label}.zip';temp=archive.with_suffix('.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for path in out.rglob('*'):
                if path.is_file() and (label=='final' or path.suffix in ('.json','.pt','.log')):
                    z.write(path,path.relative_to(out).as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix in ('.pt','.npz') else zipfile.ZIP_DEFLATED)
        temp.replace(archive);parts=[]
        with archive.open('rb') as stream:
            while block:=stream.read(32*1024*1024):
                path=archive.with_name(archive.name+f'.part{len(parts):04d}');path.write_bytes(block)
                parts.append(dict(remote=str(path),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
        receipt=dict(label=label,archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,parts=parts)
        tmp=root/'recovery_manifest.tmp';tmp.write_text(json.dumps(receipt));tmp.replace(root/'recovery_manifest.json')
        print('[GIADA41m backup] '+label+' '+receipt['sha256'],flush=True);return receipt
    try:
        assert request['training_updates']==0 and not request['selection_performed']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==request['revision']
        assert sha(repo/'scripts/run_task41m_frozen_contract.py')==request['worker_sha256']
        preflight=json.loads((root/'preflight/gpu_preflight.json').read_text());assert preflight['valid'] and preflight['protocol_sha256']==request['protocol_sha256']
        prereg=json.loads(Path('/content/giada_task41m_registration_receipt.json').read_text())
        assert prereg['valid'] and prereg['protocol_sha256']==request['protocol_sha256'] and prereg['registration_before_run']
        shutil.copy2('/content/giada_task41m_registration_receipt.json',out/'registration_receipt.json')
        shutil.copy2(root/'preflight/gpu_preflight.json',out/'gpu_preflight.json')
        env=dict(os.environ,**request['environment']);status(state='running',phase='frozen_evaluation',training_updates_performed=0)
        with (root/'process.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/run_task41m_frozen_contract.py'),str(out)],cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            status(state='running',phase='frozen_evaluation',worker_pid=proc.pid,training_updates_performed=0)
            selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ);last=time.monotonic()-170
            while True:
                events=selector.select(timeout=10)
                if events:
                    line=proc.stdout.readline()
                    if line:log.write(line);log.flush();print(line.rstrip(),flush=True)
                    elif proc.poll() is not None:break
                elif proc.poll() is not None:break
                if time.monotonic()-last>=180:publish(str(int(time.time())));last=time.monotonic()
                if time.time()-start>3900:
                    proc.terminate();raise RuntimeError('Hard safety cap65minutes; partial artifacts preserved, no automatic restart.')
            selector.close()
            for line in proc.stdout:log.write(line);print(line.rstrip(),flush=True)
            log.flush();code=proc.wait()
        receipt=publish('final')
        report=json.loads((out/('failure_report.json' if code else 'final_report.json')).read_text())
        status(state='finished',phase='done' if not code else 'failure',technical_failure=bool(code),worker_exit_code=code,
            archive_sha256=receipt['sha256'],decision=report.get('decision'),training_updates_performed=0)
        return code
    except BaseException as error:
        traceback.print_exc()
        if proc is not None and proc.poll() is None:proc.terminate();proc.wait(timeout=30)
        receipt=publish('final') if (root/'process.log').exists() else None
        status(state='finished',phase='technical_failure',technical_failure=True,error=repr(error),archive_sha256=receipt['sha256'] if receipt else None)
        return 1

if __name__=='__main__':raise SystemExit(main())
