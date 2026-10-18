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

def main():
    parser=argparse.ArgumentParser();parser.add_argument('request',type=Path);args=parser.parse_args()
    request=json.loads(args.request.read_text());revision=request['revision']
    assert len(revision)==40 and all(c in '0123456789abcdef' for c in revision)
    root=Path('/content')/('giada_task39e_'+revision[:12]);root.mkdir(exist_ok=False)
    status=root/'job_status.json';output=root/'artifacts';repo=root/'giada';started=time.time()
    def save(**fields):
        data=dict(task='39e',revision=revision,started_unix=started,elapsed_seconds=time.time()-started,manager_pid=__import__('os').getpid(),**fields)
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(status)
    def publish(label):
        # Publish immutable archives atomically; a console supervisor copies them
        # OFF the ephemeral VM and verifies this SHA before acknowledging safety.
        output.mkdir(exist_ok=True)
        shutil.copy2(root/'process.log',output/'process.log')
        archive=Path(shutil.make_archive(str(root/('recovery_'+label)), 'zip', output))
        receipt=dict(label=label,archive=archive.as_posix(),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)
        temp=root/'recovery_manifest.tmp';temp.write_text(json.dumps(receipt));temp.replace(root/'recovery_manifest.json')
        print('[GIADA recovery] '+json.dumps(receipt),flush=True)
    try:
        save(state='running',phase='checkout');print('[GIADA console] cloning pinned repository',flush=True)
        with (root/'setup.log').open('w') as log:
            subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['git','fetch','--depth','1','origin',revision],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==revision
        worker_source=repo/'scripts/run_task39e_tail_support_heads.py'
        assert hashlib.sha256(worker_source.read_bytes()).hexdigest()==request['worker_sha256']
        save(state='running',phase='scientific_worker');print('[GIADA console] pinned checkout verified; starting worker',flush=True)
        with (root/'process.log').open('w') as log:
            worker=subprocess.Popen([sys.executable,'-u',str(worker_source),str(output)],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in worker.stdout:
                log.write(line);log.flush()
                print(line.rstrip(),flush=True)
                if '[GIADA Task' in line:
                    save(state='running',phase='scientific_worker',worker_pid=worker.pid,last_progress=line.rstrip())
                    if 'checkpoint_saved ' in line or 'checkpoint ' in line or 'preflights passed' in line:
                        publish(str(int(time.time()*1000)))
            code=worker.wait()
        output.mkdir(exist_ok=True);shutil.copy2(root/'process.log',output/'process.log')
        report_path=output/('final_report.json' if (output/'final_report.json').exists() else 'failure_report.json')
        report=json.loads(report_path.read_text()) if report_path.exists() else dict(valid=False,technical_failure=True,error='worker returned without report')
        if not report_path.exists():report_path.write_text(json.dumps(report,indent=2))
        if code:print('[GIADA failure] '+json.dumps(report),flush=True)
        publish('final')
        save(state='running',phase='archive',worker_exit_code=code)
        archive=Path(shutil.make_archive(str(root/'giada_task39e_result'),'zip',output))
        save(state='finished',phase='done',worker_exit_code=code,report=report_path.as_posix(),scientific_valid=report.get('valid',False),
             decision=report.get('decision'),technical_failure=bool(code or report.get('technical_failure')),archive=archive.as_posix(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        print('[GIADA console] '+json.dumps(json.loads(status.read_text())),flush=True)
        return code if code else (0 if report.get('valid') else 1)
    except BaseException as exc:
        save(state='finished',phase='technical_failure',technical_failure=True,error=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc();return 1
if __name__=='__main__':raise SystemExit(main())
