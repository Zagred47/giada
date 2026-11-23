"""Pinned console execution and periodic complete recovery snapshots for52g."""
import hashlib,json,os,selectors,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    request=json.loads(Path(os.environ.get('GIADA_LAUNCH_REQUEST','/content/task52g_launch.json')).read_text());root=Path(request['remote_root']);root.mkdir(exist_ok=False);repo=root/'giada';out=root/'result';logpath=root/'process.log';logpath.touch();start=time.time();proc=None;update=request.get('resumed_from',0)
    def status(**kw):
        p=root/'job_status.tmp';p.write_text(json.dumps(dict(task='52g',revision=request['revision'],update=update,thresholds_changed=False,elapsed_seconds=time.time()-start,**kw)));p.replace(root/'job_status.json')
    def publish(label):
        stage=root/'snapshot';stage.mkdir(exist_ok=True)
        if out.exists():
            for source in out.rglob('*'):
                if source.is_file() and not source.name.endswith('.tmp'):
                    dest=stage/source.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        shutil.copy2(logpath,stage/'console_process.log');(stage/'dispatch_request.json').write_text(json.dumps(request,indent=2))
        archive=Path(shutil.make_archive(str(root/('recovery_'+label)),'zip',stage));r=dict(label=label,archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,update=update)
        p=root/'recovery_manifest.tmp';p.write_text(json.dumps(r));p.replace(root/'recovery_manifest.json');print('[GIADA52g backup] '+label+' '+r['sha256'],flush=True);return r
    def run(cmd,phase,env=None):
        nonlocal proc,update
        with logpath.open('a') as log:
            log.write('[manager] '+phase+'\n');log.flush();proc=subprocess.Popen(cmd,cwd=repo if repo.exists() else root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            status(state='running',phase=phase,worker_pid=proc.pid);sel=selectors.DefaultSelector();sel.register(proc.stdout,selectors.EVENT_READ);last=time.monotonic();buffer=b''
            while True:
                if sel.select(5):
                    block=os.read(proc.stdout.fileno(),65536)
                    if not block and proc.poll() is not None:break
                    buffer+=block
                    while b'\n' in buffer:
                        line,buffer=buffer.split(b'\n',1);text=line.decode(errors='replace');log.write(text+'\n');log.flush();print(text,flush=True)
                        if text.startswith('[GIADA52g] training '):update=int(text.split()[2].split('/')[0]);status(state='running',phase=phase,worker_pid=proc.pid)
                elif proc.poll() is not None:break
                if time.monotonic()-last>=180:publish(str(int(time.time())));last=time.monotonic()
                if time.time()-start>18000:proc.terminate();raise RuntimeError('52g total five-hour safety limit; recover checkpoint')
            if buffer:log.write(buffer.decode(errors='replace'));log.flush()
            sel.close();code=proc.wait()
            if code:raise RuntimeError(f'{phase} exit{code}')
    try:
        assert request['registration_valid'] and not request['thresholds_changed'] and not request['abstraction_changed']
        for item in request['inputs']:assert sha(item['remote'])==item['sha256']
        publish('inputs_verified')
        run(['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],'clone')
        run(['git','fetch','--quiet','--depth','1','origin',request['revision']],'pin_fetch');run(['git','checkout','--quiet','--detach','FETCH_HEAD'],'pin_checkout')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==request['revision']
        assert sha(repo/'experiments/task52g_preregistration_v1.json')==request['protocol_sha256']
        run([sys.executable,'-m','pip','install','--quiet','torch==2.11.0+cu130','--index-url','https://download.pytorch.org/whl/cu130'],'torch_setup')
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg')
        runner='run_task52g_resume.py' if request.get('resumed_from') else 'run_task52g.py'
        run([sys.executable,'-u',str(repo/'scripts'/runner),*[i['remote'] for i in request['inputs']],str(out)],'training_and_evaluation',env)
        report=json.loads((out/'final_report.json').read_text());assert report['complete'] and report['training_updates']==800
        final=publish('final');status(state='finished',technical_failure=False,archive_sha256=final['sha256'],scientific_acceptance_passed=report['scientific_acceptance_passed']);return 0
    except BaseException as error:
        traceback.print_exc()
        if proc is not None and proc.poll() is None:proc.terminate();proc.wait(timeout=30)
        with logpath.open('a') as log:log.write(traceback.format_exc())
        final=publish('final');status(state='finished',technical_failure=True,error=repr(error),archive_sha256=final['sha256']);return 1
if __name__=='__main__':raise SystemExit(main())
