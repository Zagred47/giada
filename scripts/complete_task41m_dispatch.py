"""Finite local continuation: DB preregistration -> one launch -> local DB backup.

No retraining/retries, no Airtable/Drive, no second worker on transport failures.
The already-running DB writer is not interrupted or bypassed.
"""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from task41b_local_supervisor import cli
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'experiments/results/task41m_protocol_v1'
DEST=ROOT/'artifacts/task41m_primary_external'

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n');tmp.replace(path)

def main():
    request=json.loads((FOLDER/'dispatch_request.json').read_text());start=time.time()
    def status(state,**kw):save(FOLDER/'dispatch_workflow.json',dict(state=state,elapsed_seconds=time.time()-start,revision=request['revision'],**kw))
    def command(*args):
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    def upload(source,target):
        # Keep local paths relative to the declared cwd across Windows/WSL.
        # This does not replace checking that the remote session still exists.
        local=Path(source).resolve().relative_to(ROOT).as_posix()
        result=cli(ROOT,'upload','-s',request['session'],local,target,timeout=300)
        if result.returncode:raise RuntimeError('Console upload failed; scientific worker not duplicated')
    try:
        assert not (FOLDER/'launch.json').exists(),'Existing launch receipt: inspect, never duplicate.'
        os.environ['GIADA_COLAB_PROFILE']='primary'
        status('waiting_for_verified_preregistration')
        deadline=time.monotonic()+90*60
        while not (FOLDER/'memory_receipt.json').exists():
            if time.monotonic()>deadline:raise RuntimeError('Preregistration deadline exhausted; no launch attempted.')
            time.sleep(10)
        registered=json.loads((FOLDER/'memory_receipt.json').read_text())
        spec_path=ROOT/'experiments/task41m_preregistration_v1.json'
        assert registered['valid'] and registered['registration_before_run']
        assert registered['protocol_sha256']==request['protocol_sha256']==hashlib.sha256(spec_path.read_bytes()).hexdigest()
        upload(FOLDER/'memory_receipt.json','/content/giada_task41m_registration_receipt.json')
        upload(ROOT/'scripts/task41m_colab_manager.py','/content/task41m_colab_manager.py')
        DEST.mkdir(parents=True,exist_ok=True)
        log=(DEST/'supervisor_stdout.log').open('a',encoding='utf-8')
        child=subprocess.Popen([sys.executable,'-u','scripts/task41m_colab_supervisor.py','--session',request['session'],'--destination',str(DEST)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        status('launched_awaiting_external_copy',supervisor_pid=child.pid)
        deadline=time.monotonic()+10*60
        while not (DEST/'external_backup_receipt.json').exists():
            if child.poll() is not None:raise RuntimeError('Supervisor ended before external copy; inspect logs, do not relaunch.')
            if time.monotonic()>deadline:raise RuntimeError('First external copy not verified within10minutes; worker may still run. Inspect, do not relaunch.')
            time.sleep(10)
        backup=json.loads((DEST/'external_backup_receipt.json').read_text());assert backup['valid'] and backup['runtime_independent']
        job=json.loads((DEST/'job_status.json').read_text());assert not job.get('technical_failure'),job
        save(FOLDER/'launch.json',dict(valid=True,request=request,observed=job,supervisor_pid=child.pid,external_backup=backup,
            provider_reason='Kaggle GPU quota checked:112269.406s used/108000s allowed, reset2026-10-31T00:00:00Z; use authorized Colab primary.',training_updates=0))
        status('recording_verified_launch',supervisor_pid=child.pid)
        command('-m','research_memory.register_task41m','--running')
        status('local_db_backup_in_progress',supervisor_pid=child.pid)
        command('-m','research_memory.prepare_drive_backup')
        command('-m','research_memory.record_local_backup','--raw-sqlite','--scope',
            'Complete41l original245JSON reports,paired audit,24completed runs,scoped findings and stop policy;sealed41m frozen no-training protocol,actualT4preflight and first external archive. No41m scientific result claimed. DBlocalonly.')
        db_backup=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());assert db_backup['valid'] and not db_backup['drive_updated']
        for path in [ROOT/'experiments/results/task41l_audited_v1/memory_receipt.json',FOLDER/'memory_receipt.json',FOLDER/'launch_memory_receipt.json']:
            row=json.loads(path.read_text());row['DB_backup']=db_backup
            if 'backup' in row:row['backup']=db_backup
            save(path,row)
        files=['research_memory/register_task41m.py','scripts/complete_task41m_dispatch.py',
            'experiments/results/task41l_audited_v1/memory_receipt.json',
            'experiments/results/task41m_protocol_v1/gpu_preflight.json',
            'experiments/results/task41m_protocol_v1/preflight_failure_9bddcb8.json',
            'experiments/results/task41m_protocol_v1/dispatch_request.json',
            'experiments/results/task41m_protocol_v1/memory_receipt.json',
            'experiments/results/task41m_protocol_v1/launch.json',
            'experiments/results/task41m_protocol_v1/launch_memory_receipt.json',
            'research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
        # Only known small evidence files may enter this commit; refuse unrelated staging.
        staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())
        assert staged<=set(files),staged-set(files)
        subprocess.run(['git','commit','-m','Record complete41l evidence and verified frozen41m launch with local DB backup'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        status('dispatch_complete',DB_backup=db_backup,supervisor_pid=child.pid,scientific_revision=request['revision'],
            audit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scientific_results_pending=True)
    except BaseException as exc:
        status('attention_required',error=repr(exc),traceback=traceback.format_exc(),automatic_restart=False)
        raise

if __name__=='__main__':main()
