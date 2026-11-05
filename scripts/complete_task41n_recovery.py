"""Finite authorized lifecycle: launch once, monitor, register, back up and push."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
FOLDER=ROOT/'experiments/results/task41n_recovery_v1'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume-format-fix',action='store_true');args=ap.parse_args()
    start=time.time()
    def status(state,**kw):write_json(FOLDER/'workflow.json',dict(state=state,elapsed_seconds=time.time()-start,**kw))
    def command(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    try:
        assert json.loads((FOLDER/'preregistration_receipt.json').read_text())['valid']
        if args.resume_format_fix:
            assert json.loads((FOLDER/'kaggle_launch.json').read_text())['version']==2 and (FOLDER/'technical_attempt_v1.json').exists()
            command('scripts/task41n_recovery_kaggle.py','status')
        else:
            status('launching_once');command('scripts/task41n_recovery_kaggle.py','launch');command('scripts/task41n_recovery_kaggle.py','status')
        status('recording_actual_launch');command('-m','research_memory.register_task41n_recovery','--running')
        status('monitoring_existing_evaluation');command('scripts/task41n_recovery_kaggle.py','supervise')
        status('registering_recovered_output');command('-m','research_memory.register_task41n_recovery','--result','--validate-only');command('-m','research_memory.register_task41n_recovery','--result')
        status('local_only_payload_database_backup');command('-m','research_memory.prepare_drive_backup');command('-m','research_memory.record_local_backup','--raw-sqlite','--scope','Complete authoritative memory including41n evaluation-only diagnostic recovery. Original FP32 floor failures retained; no training/selection/threshold changes or primary confirmation reinstatement. Raw report payloads included; two local verified generations.')
        backup=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());assert backup['valid'] and not backup['single_file'] and not backup['drive_updated']
        for name in ('preregistration_receipt.json','launch_memory_receipt.json','result_memory_receipt.json'):
            path=FOLDER/name;value=json.loads(path.read_text());value['DB_backup']=backup;write_json(path,value)
        files=['experiments/results/task41n_recovery_v1/'+name for name in ('preregistration_receipt.json','launch_memory_receipt.json','result_memory_receipt.json','kaggle_launch.json','input_verified.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
        staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines());assert staged<=set(files),staged
        subprocess.run(['git','commit','-m','Record41n diagnostic recovery and verified compact local backup'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        result=json.loads((ROOT/'artifacts/task41n_recovery_kaggle_external/external_backup_receipt.json').read_text())
        status('complete',technical_failure=result['technical_failure'],decision=result['report'].get('decision'),DB_backup=backup,external_backup_verified=True)
    except BaseException as error:
        status('attention_required',error=repr(error),traceback=traceback.format_exc(),automatic_restart=False);raise
if __name__=='__main__':main()
