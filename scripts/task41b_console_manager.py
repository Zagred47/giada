"""Pinned checkout + isolated scientific worker with durable job status."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('request',type=Path);args=parser.parse_args()
    request=json.loads(args.request.read_text());revision=request['revision'];task=request.get('task','39e')
    assert task in ('39e','39f','39g','39h','39i','39j','40','40b','41','41b','41c','41d','41e','41f','41g','41h')
    assert len(revision)==40 and all(c in '0123456789abcdef' for c in revision)
    root=Path('/content')/('giada_task'+task+'_'+revision[:12]);root.mkdir(exist_ok=False)
    status=root/'job_status.json';output=root/'artifacts';repo=root/'giada';started=time.time()
    def save(**fields):
        data=dict(task=task,revision=revision,started_unix=started,elapsed_seconds=time.time()-started,manager_pid=__import__('os').getpid(),**fields)
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(status)
    def publish(label):
        # Publish immutable archives atomically; a console supervisor copies them
        # OFF the ephemeral VM and verifies this SHA before acknowledging safety.
        output.mkdir(exist_ok=True)
        shutil.copy2(root/'process.log',output/'process.log')
        archive=Path(shutil.make_archive(str(root/('recovery_'+label)), 'zip', output))
        receipt=dict(label=label,archive=archive.as_posix(),sha256=file_sha(archive),bytes=archive.stat().st_size)
        if task=='41h':
            # Transport only: original ZIP and scientific files stay intact.
            parts=[]
            with archive.open('rb') as stream:
                index=0
                while block:=stream.read(32*1024*1024):
                    part=archive.with_name(archive.name+f'.part{index:04d}')
                    part.write_bytes(block)
                    parts.append(dict(remote=part.as_posix(),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
                    index+=1
            receipt['parts']=parts
        temp=root/'recovery_manifest.tmp';temp.write_text(json.dumps(receipt));temp.replace(root/'recovery_manifest.json')
        print('[GIADA recovery] '+json.dumps(receipt),flush=True)
    try:
        save(state='running',phase='checkout');print('[GIADA console] cloning pinned repository',flush=True)
        with (root/'setup.log').open('w') as log:
            subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['git','fetch','--depth','1','origin',revision],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==revision
        for item in request.get('inputs',[]):
            assert file_sha(Path(item['remote']))==item['sha256'],'Input transport SHA mismatch: '+item['remote']
        workers={'39e':'run_task39e_tail_support_heads.py','39f':'run_task39f_temporal_recovery.py','39g':'run_task39g_objective_horizon.py','39h':'run_task39h_relaxation_identification.py','39i':'run_task39i_coupled_relaxation.py','39j':'run_task39j_atomic_relaxation.py'}
        workers['40']='run_task40_causal_reinsertion.py'
        workers['40b']='run_task40b_calcium_precision.py'
        workers['41']='run_task41_independent_confirmation.py'
        workers['41b']='run_task41b_local_precision.py'
        workers['41c']='run_task41c_frozen_confirmation.py'
        workers['41d']='run_task41d_precision_response.py'
        workers['41e']='run_task41e_transfer.py'
        workers['41f']='run_task41f_precision_relaxation.py'
        workers['41g']='run_task41g_preservation.py'
        workers['41h']='run_task41h_path_repair.py'
        worker_source=repo/'scripts'/workers[task]
        assert hashlib.sha256(worker_source.read_bytes()).hexdigest()==request['worker_sha256']
        save(state='running',phase='scientific_worker');print('[GIADA console] pinned checkout verified; starting worker',flush=True)
        with (root/'process.log').open('w') as log:
            worker=subprocess.Popen([sys.executable,'-u',str(worker_source),str(output)],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            last_publish=0.
            for line in worker.stdout:
                log.write(line);log.flush()
                print(line.rstrip(),flush=True)
                if '[GIADA Task' in line:
                    save(state='running',phase='scientific_worker',worker_pid=worker.pid,last_progress=line.rstrip())
                    checkpoint='checkpoint_saved ' in line or 'checkpoint ' in line or 'preflights passed' in line
                    if checkpoint and (task not in ('41d','41e','41f','41g','41h') or 'checkpoint_saved ' in line or 'preflights passed' in line or time.time()-last_publish>=120):
                        publish(str(int(time.time()*1000)))
                        last_publish=time.time()
            code=worker.wait()
        output.mkdir(exist_ok=True);shutil.copy2(root/'process.log',output/'process.log')
        report_path=output/('final_report.json' if (output/'final_report.json').exists() else 'failure_report.json')
        report=json.loads(report_path.read_text()) if report_path.exists() else dict(valid=False,technical_failure=True,error='worker returned without report')
        if not report_path.exists():report_path.write_text(json.dumps(report,indent=2))
        if code:print('[GIADA failure] '+json.dumps(report),flush=True)
        publish('final')
        save(state='running',phase='archive',worker_exit_code=code)
        archive=Path(shutil.make_archive(str(root/('giada_task'+task+'_result')),'zip',output))
        save(state='finished',phase='done',worker_exit_code=code,report=report_path.as_posix(),scientific_valid=report.get('valid',False),
             decision=report.get('decision'),technical_failure=bool(code or report.get('technical_failure')),archive=archive.as_posix(),archive_sha256=file_sha(archive))
        print('[GIADA console] '+json.dumps(json.loads(status.read_text())),flush=True)
        return code if code else (0 if report.get('valid') else 1)
    except BaseException as exc:
        save(state='finished',phase='technical_failure',technical_failure=True,error=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc();return 1
if __name__=='__main__':raise SystemExit(main())
