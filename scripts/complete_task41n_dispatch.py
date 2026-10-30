"""Finite authorized launch, verified off-VM backup, local-only DB backup."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from task41b_local_supervisor import cli
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'experiments/results/task41n_protocol_v1';DEST=ROOT/'artifacts/task41n_primary_external'

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n');temp.replace(path)

def main():
    start=time.time();request=json.loads((FOLDER/'dispatch_request.json').read_text())
    def status(state,**kw):save(FOLDER/'dispatch_workflow.json',dict(state=state,elapsed_seconds=time.time()-start,revision=request['revision'],**kw))
    def command(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    def bridge(*args,**kwargs):
        result=cli(ROOT,*args,**kwargs)
        if result.returncode:raise RuntimeError('Console command failed; no automatic retry/duplicate worker: '+str(args[0]))
        return result
    def upload(source,target):bridge('upload','-s',request['session'],Path(source).resolve().relative_to(ROOT).as_posix(),target,timeout=300)
    try:
        assert request['profile']=='primary' and not (FOLDER/'launch.json').exists()
        memory=json.loads((FOLDER/'memory_receipt.json').read_text());assert memory['valid'] and memory['registration_before_run']
        os.environ['GIADA_COLAB_PROFILE']='primary'
        identity=bridge('whoami',capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        assert 'bellialessandro39@gmail.com' in identity.stdout,'Unexpected account; do not allocate'
        sessions=bridge('sessions',capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        reuse=os.environ.get('GIADA_TASK41N_REUSE_FAILED_SETUP')
        if reuse:
            previous=Path(reuse).resolve();prior=json.loads(previous.read_text())
            assert prior['state']=='finished' and prior['technical_failure'] and prior['phase']=='technical_failure'
            assert prior['error']=="RuntimeError('Phasenative_environment_audit exit1; inspect preserved failure report')"
            assert request['session'] in sessions.stdout,'Expected existing session missing'
        else:
            assert request['session'] not in sessions.stdout,'Named session already exists; inspect rather than duplicate'
        status('allocating_after_verified_preregistration')
        if not reuse:bridge('new','-s',request['session'],'--gpu','T4',timeout=180)
        for item in request['inputs']:upload(ROOT/item['local'],item['remote'])
        upload(FOLDER/'dispatch_request.json','/content/giada_task41n_launch.json')
        upload(FOLDER/'memory_receipt.json','/content/giada_task41n_registration_receipt.json')
        upload(ROOT/'scripts/task41n_colab_manager.py','/content/task41n_colab_manager.py')
        DEST.mkdir(parents=True,exist_ok=True);log=(DEST/'supervisor_stdout.log').open('a',encoding='utf-8')
        child=subprocess.Popen([sys.executable,'-u','scripts/task41n_colab_supervisor.py','--session',request['session'],'--destination',str(DEST)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        status('launched_awaiting_external_copy',supervisor_pid=child.pid);deadline=time.monotonic()+10*60
        while not (DEST/'external_backup_receipt.json').exists():
            if child.poll() is not None:raise RuntimeError('Supervisor ended before backup; inspect existing process, do not relaunch')
            if time.monotonic()>deadline:raise RuntimeError('First external copy timeout; manager may still run, inspect without relaunch')
            time.sleep(10)
        backup=json.loads((DEST/'external_backup_receipt.json').read_text());observed=json.loads((DEST/'job_status.json').read_text());assert backup['valid'] and backup['runtime_independent'] and not observed.get('technical_failure'),observed
        save(FOLDER/'launch.json',dict(valid=True,request=request,observed=observed,supervisor_pid=child.pid,external_backup=backup,training_updates=0))
        status('recording_verified_launch',supervisor_pid=child.pid)
        command('-m','research_memory.register_task41n','--running')
        status('local_db_backup_in_progress',supervisor_pid=child.pid)
        command('-m','research_memory.prepare_drive_backup')
        command('-m','research_memory.record_local_backup','--raw-sqlite','--scope','Complete41m64original JSON reports,full audit,ninecompleted frozen evaluations;sealed41n independent confirmation/exact execution,CPUverification andactual console launch/first off-runtime backup. No41n scientific result yet audited. DBlocalonly.')
        db_backup=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());assert db_backup['valid'] and not db_backup['drive_updated']
        for file in [ROOT/'experiments/results/task41m_audited_v1/memory_receipt.json',FOLDER/'memory_receipt.json',FOLDER/'launch_memory_receipt.json']:
            row=json.loads(file.read_text());row['DB_backup']=db_backup;save(file,row)
        files=['experiments/results/task41m_audited_v1/memory_receipt.json',*['experiments/results/task41n_protocol_v1/'+name for name in ('local_verification.json','memory_receipt.json','dispatch_request.json','launch.json','launch_memory_receipt.json')],'research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
        staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines());assert staged<=set(files),staged-set(files)
        subprocess.run(['git','commit','-m','Record41m results and verified41n launch with local SQLite backup'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        status('dispatch_complete',DB_backup=db_backup,scientific_revision=request['revision'],scientific_result_not_yet_audited=True,supervisor_pid=child.pid)
    except BaseException as error:
        status('attention_required',error=repr(error),traceback=traceback.format_exc(),automatic_restart=False);raise

if __name__=='__main__':main()
