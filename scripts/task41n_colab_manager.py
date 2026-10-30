"""One active console manager; isolate pinned native/CUDA workers and back up."""
import hashlib,json,os,selectors,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    request=json.loads(Path('/content/giada_task41n_launch.json').read_text());root=Path(request['remote_root']);root.mkdir(exist_ok=False)
    out=root/'artifacts';out.mkdir();repo=root/'giada';start=time.time();proc=None;scientific_start=None
    def status(**kw):
        tmp=root/'job_status.tmp';tmp.write_text(json.dumps(dict(task='41n',revision=request['revision'],elapsed_seconds=time.time()-start,**kw)));tmp.replace(root/'job_status.json')
    def publish(label):
        shutil.copy2(root/'process.log',out/'process.log');archive=root/f'recovery_{label}.zip';tmp=archive.with_suffix('.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for path in out.rglob('*'):
                if path.is_file() and (label=='final' or path.suffix in ('.json','.pt','.log','.bin') or path.name=='native_partial.npz'):
                    z.write(path,path.relative_to(out).as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix in ('.pt','.npz','.bin') else zipfile.ZIP_DEFLATED)
        tmp.replace(archive);parts=[]
        with archive.open('rb') as f:
            while block:=f.read(32*1024*1024):
                path=archive.with_name(archive.name+f'.part{len(parts):04d}');path.write_bytes(block)
                parts.append(dict(remote=str(path),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
        receipt=dict(label=label,archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,parts=parts)
        tmp=root/'recovery_manifest.tmp';tmp.write_text(json.dumps(receipt));tmp.replace(root/'recovery_manifest.json')
        print('[GIADA41n backup] '+label+' '+receipt['sha256'],flush=True);return receipt
    def run(command,phase,env=None,cwd=None):
        nonlocal proc,scientific_start
        if phase=='gpu_preflight':scientific_start=time.time()
        with (root/'process.log').open('a') as log:
            log.write('[manager] '+phase+'\n');log.flush()
            proc=subprocess.Popen(command,cwd=cwd or repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            status(state='running',phase=phase,worker_pid=proc.pid,training_updates_performed=0)
            selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ);last=time.monotonic()-170
            while True:
                events=selector.select(timeout=10)
                if events:
                    line=proc.stdout.readline()
                    if line:log.write(line);log.flush();print(line.rstrip(),flush=True)
                    elif proc.poll() is not None:break
                elif proc.poll() is not None:break
                if time.monotonic()-last>=180:publish(str(int(time.time())));last=time.monotonic()
                if time.time()-start>6600:proc.terminate();raise RuntimeError('110minute hard manager budget exhausted; no automatic restart')
                if scientific_start is not None and time.time()-scientific_start>5400:proc.terminate();raise RuntimeError('90minute scientific-worker budget exhausted; partial evidence retained')
            selector.close()
            for line in proc.stdout:log.write(line);print(line.rstrip(),flush=True)
            log.flush();code=proc.wait()
            if code:raise RuntimeError(f'Phase{phase} exit{code}; inspect preserved failure report')
        publish('phase_'+phase)
    try:
        receipt=json.loads(Path('/content/giada_task41n_registration_receipt.json').read_text())
        assert receipt['valid'] and receipt['registration_before_run'] and receipt['protocol_sha256']==request['protocol_sha256']
        shutil.copy2('/content/giada_task41n_registration_receipt.json',out/'registration_receipt.json')
        status(state='running',phase='checkout')
        with (root/'setup.log').open('w') as log:
            for cmd in (['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],['git','-C',str(repo),'fetch','--quiet','--depth','1','origin',request['revision']],['git','-C',str(repo),'checkout','--quiet','--detach','FETCH_HEAD']):subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==request['revision']
        worker=repo/'scripts/run_task41n_independent.py';assert sha(worker)==request['worker_sha256']
        for item in request['inputs']:assert sha(item['remote'])==item['sha256']
        shutil.copy2(request['inputs'][0]['remote'],out/'frozen_source.zip.bin')
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg',GIADA_TASK41N_FROZEN=request['inputs'][0]['remote'])
        run([sys.executable,'-m','pip','install','--quiet','torch==2.11.0+cu130','--index-url','https://download.pytorch.org/whl/cu130'],'torch_setup',env)
        # Python3.13 Colab cannot load the historical8.2.7 wheel. Use an isolated
        #3.12 native environment; never substitute a newer NEURON silently.
        run([sys.executable,'-m','pip','install','--quiet','uv==0.8.15'],'native_setup_uv',env)
        uv=[sys.executable,'-m','uv'];venv=root/'native312'
        run(uv+['venv','--python','3.12',str(venv)],'native_setup_python',env)
        native_python=venv/'bin/python'
        run(uv+['pip','install','--python',str(native_python),'neuron==8.2.7','numpy==1.26.4','scipy==1.13.1','pandas==2.2.2','matplotlib==3.9.2'],'native_setup_packages',env)
        code="import sys,json,neuron,numpy,scipy,pandas,matplotlib;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps(dict(python=sys.version,neuron=neuron.__version__,numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__,matplotlib=matplotlib.__version__)))"
        run([str(native_python),'-c',code,str(out/'native_environment.json')],'native_environment_audit',env)
        run([str(native_python),'-u',str(repo/'scripts/task41n_native_import_probe.py')],'native_import_preflight',env)
        teacher=root/'teacher_canonical'
        run(['git','clone','--quiet','--filter=blob:none','--no-checkout','https://github.com/SelfishGene/neuron_as_deep_net.git',str(teacher)],'teacher_checkout',env)
        run(['git','sparse-checkout','set','L5PC_NEURON_simulation/mods'],'teacher_sparse',env,teacher)
        spec=json.loads((repo/'experiments/task41n_preregistration_v1.json').read_text())
        run(['git','checkout','--quiet','--detach',spec['teacher_revision']],'teacher_pin',env,teacher)
        run([sys.executable,'-u',str(worker),str(out),'--phase','preflight'],'gpu_preflight',env)
        freeze_sha=sha(out/'execution_freeze.json');assert not (out/'native_shadow_traces.npz').exists()
        native_env=dict(env,PATH=str(venv/'bin')+os.pathsep+env.get('PATH',''))
        run([str(native_python),'-u',str(worker),str(out),'--phase','native'],'native_confirmation',native_env)
        assert sha(out/'execution_freeze.json')==freeze_sha
        run([sys.executable,'-u',str(worker),str(out),'--phase','evaluate'],'fresh_evaluation',env)
        assert sha(out/'execution_freeze.json')==freeze_sha
        result=json.loads((out/'final_report.json').read_text());assert result['complete'] and result['training_updates_performed']==0
        final=publish('final');status(state='finished',phase='done',technical_failure=False,worker_exit_code=0,archive_sha256=final['sha256'],decision=result['decision'],training_updates_performed=0)
        return 0
    except BaseException as error:
        traceback.print_exc()
        if proc is not None and proc.poll() is None:proc.terminate();proc.wait(timeout=30)
        if not (root/'process.log').exists():(root/'process.log').write_text(traceback.format_exc())
        if not (out/'failure_report.json').exists():(out/'failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(error),traceback=traceback.format_exc())))
        final=publish('final');status(state='finished',phase='technical_failure',technical_failure=True,error=repr(error),archive_sha256=final['sha256']);return 1

if __name__=='__main__':raise SystemExit(main())
