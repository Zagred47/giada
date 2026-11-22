"""Console-only continuation of the verified update600, with external snapshots."""
import hashlib,json,os,selectors,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    request=json.loads(Path('/content/task52f_resume_request.json').read_text())
    root=Path(request['remote_root']);root.mkdir(exist_ok=False)
    repo=root/'giada';out=root/'result';started=time.time();proc=None
    logpath=root/'process.log';logpath.touch()
    def status(**kw):
        p=root/'job_status.tmp';p.write_text(json.dumps(dict(task='52f',revision=request['revision'],resumed_from=600,thresholds_changed=False,elapsed_seconds=time.time()-started,**kw)));p.replace(root/'job_status.json')
    def publish(label):
        stage=root/'snapshot';stage.mkdir(exist_ok=True)
        if out.exists():
            for p in out.rglob('*'):
                if p.is_file():
                    dest=stage/p.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        shutil.copy2(logpath,stage/'colab_process.log')
        archive=Path(shutil.make_archive(str(root/('recovery_'+label)),'zip',stage))
        receipt=dict(label=label,archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)
        p=root/'recovery_manifest.tmp';p.write_text(json.dumps(receipt));p.replace(root/'recovery_manifest.json')
        print('[GIADA52f backup] '+label+' '+receipt['sha256'],flush=True)
        return receipt
    def run(cmd,phase,env=None):
        nonlocal proc
        status(state='running',phase=phase)
        with logpath.open('a') as log:
            log.write('[manager] '+phase+'\n');log.flush()
            proc=subprocess.Popen(cmd,cwd=repo if repo.exists() else root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            sel=selectors.DefaultSelector();sel.register(proc.stdout,selectors.EVENT_READ);last=time.monotonic()
            while True:
                if sel.select(5):
                    line=proc.stdout.readline()
                    if line:log.write(line);log.flush();print(line.rstrip(),flush=True)
                    elif proc.poll() is not None:break
                elif proc.poll() is not None:break
                if time.monotonic()-last>=180:publish(str(int(time.time())));last=time.monotonic()
                if time.time()-started>14400:proc.terminate();raise RuntimeError('Four-hour safety budget exhausted; checkpoint preserved')
            sel.close()
            for line in proc.stdout:log.write(line);print(line.rstrip(),flush=True)
            code=proc.wait()
            if code:raise RuntimeError(f'{phase} exit {code}')
    try:
        for item in request['inputs']:assert sha(item['remote'])==item['sha256']
        run(['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],'clone')
        run(['git','fetch','--quiet','--depth','1','origin',request['revision']],'pin_fetch')
        run(['git','checkout','--quiet','--detach','FETCH_HEAD'],'pin_checkout')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==request['revision']
        run([sys.executable,'-m','pip','install','--quiet','torch==2.11.0+cu130','--index-url','https://download.pytorch.org/whl/cu130'],'torch_setup')
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg')
        run([sys.executable,'-u',str(repo/'scripts/run_task52f_resume.py'),*[i['remote'] for i in request['inputs']],str(out)],'resume600_and_evaluate',env)
        report=json.loads((out/'final_report.json').read_text())
        final=publish('final');status(state='finished',technical_failure=False,archive_sha256=final['sha256'],report_valid=report.get('valid'))
        return 0
    except BaseException as error:
        traceback.print_exc()
        if proc is not None and proc.poll() is None:proc.terminate();proc.wait(timeout=30)
        with logpath.open('a') as log:log.write(traceback.format_exc())
        final=publish('final');status(state='finished',technical_failure=True,error=repr(error),archive_sha256=final['sha256']);return 1

if __name__=='__main__':raise SystemExit(main())
