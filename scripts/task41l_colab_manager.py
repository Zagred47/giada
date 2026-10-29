"""Console-only frozen evaluation; scientific checkout remains immutable."""
import hashlib,json,os,selectors,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(request_path):
    request=json.loads(Path(request_path).read_text());revision=request['revision']
    assert request['evaluation_only'] and request['selection_recomputed'] is False
    root=Path('/content')/('giada_task41l_'+revision[:12]);root.mkdir(exist_ok=False)
    out=root/'artifacts';out.mkdir();repo=root/'giada';status=root/'job_status.json';start=time.time()
    def save(**kw):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(task='41l',revision=revision,elapsed_seconds=time.time()-start,**kw)));temp.replace(status)
    def publish(label):
        shutil.copy2(root/'process.log',out/'process.log')
        archive=root/f'recovery_{label}.zip';temp=archive.with_suffix('.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for path in out.rglob('*'):
                if not path.is_file():continue
                if label!='final' and path.suffix not in ('.json','.pt','.log'):continue
                z.write(path,path.relative_to(out).as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix in ('.pt','.npz') else zipfile.ZIP_DEFLATED)
        temp.replace(archive);parts=[]
        with archive.open('rb') as f:
            while block:=f.read(32*1024*1024):
                part=archive.with_name(archive.name+f'.part{len(parts):04d}');part.write_bytes(block)
                parts.append(dict(remote=str(part),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
        receipt=dict(label=label,archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,parts=parts,periodic_trace_arrays_omitted=label!='final')
        temp=root/'recovery_manifest.tmp';temp.write_text(json.dumps(receipt));temp.replace(root/'recovery_manifest.json')
        print('[GIADA recovery] '+json.dumps(dict(label=label,sha256=receipt['sha256'],bytes=receipt['bytes'])),flush=True)
    try:
        save(state='running',phase='checkout')
        for item in request['inputs']:assert sha(item['remote'])==item['sha256'],item['remote']
        gpu=json.loads(Path('/content/giada_task41l_gpu.json').read_text());assert gpu['valid'] and gpu['gpu']=='Tesla T4' and gpu['torch']=='2.11.0+cu130'
        with (root/'setup.log').open('w') as log:
            for command in (['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],['git','-C',str(repo),'fetch','--quiet','--depth','1','origin',revision],['git','-C',str(repo),'checkout','--quiet','--detach','FETCH_HEAD']):subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==revision
        worker=repo/'scripts/run_task41l_regression_attribution.py';assert sha(worker)==request['worker_sha256']
        archive=Path(request['archive']);assert sha(archive)==request['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            immutable={name:hashlib.sha256(z.read(name)).hexdigest() for name in ('frozen_checkpoint.pt','selection_freeze.json','training_curves.json')}
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',GIADA_PROVIDER='Colab',GIADA_TASK41_INPUTS='/content/task41b_opened_inputs.zip',GIADA_TASK41C_INPUTS='/content/task41d_opened41c.zip',GIADA_TASK41L_FROZEN_INPUT='/content/task41k_frozen_diagnostic.zip.bin')
        with (root/'process.log').open('w') as log:
            for phase in range(12):
                args=['--resume-archive',str(archive),'--resume-sha256',request['archive_sha256']] if phase==0 else ['--resume-local']
                command=[sys.executable,'-u',str(worker),str(out),*args,'--evaluation-only']
                proc=subprocess.Popen(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                save(state='running',phase='frozen_evaluation',worker_pid=proc.pid,evaluation_stage=phase,training_updates_performed=0)
                selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ);last=time.monotonic()
                while True:
                    events=selector.select(timeout=10)
                    if events:
                        line=proc.stdout.readline()
                        if line:log.write(line);log.flush();print(line.rstrip(),flush=True)
                        elif proc.poll() is not None:break
                    elif proc.poll() is not None:break
                    if time.monotonic()-last>=180 and (out/'selection_freeze.json').exists():
                        publish(str(int(time.time())));last=time.monotonic()
                selector.close()
                for line in proc.stdout:log.write(line);print(line.rstrip(),flush=True)
                log.flush();code=proc.wait()
                for name,digest in immutable.items():assert sha(out/name)==digest,'Frozen input changed: '+name
                report=json.loads((out/('failure_report.json' if code else 'final_report.json')).read_text())
                if code:raise RuntimeError('Frozen evaluation failed: '+repr(report))
                assert report.get('training_updates_performed')==0 and report.get('selection_recomputed') is False
                if report['decision']!='FROZEN_EVALUATION_STAGE_COMPLETE':break
                publish('phase_'+str(phase))
            else:raise RuntimeError('Evaluation stage budget exhausted; outputs preserved')
        publish('final')
        receipt=json.loads((root/'recovery_manifest.json').read_text())
        save(state='finished',phase='done',technical_failure=False,worker_exit_code=0,decision=report['decision'],archive=receipt['archive'],archive_sha256=receipt['sha256'],training_updates_performed=0,selection_recomputed=False)
        return 0
    except BaseException as exc:
        traceback.print_exc()
        if (root/'process.log').exists():
            try:publish('failure')
            except Exception:traceback.print_exc()
        save(state='finished',phase='technical_failure',technical_failure=True,error=repr(exc));return 1

if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
