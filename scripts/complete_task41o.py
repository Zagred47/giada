"""Single submission, finite supervision and verified result/DB backups."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
FOLDER=ROOT/'experiments/results/task41o_causal_v1'

def main():
    start=time.time()
    def status(state,**kw):write_json(FOLDER/'workflow.json',dict(state=state,elapsed_seconds=time.time()-start,**kw))
    def command(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    try:
        assert json.loads((FOLDER/'preregistration_receipt.json').read_text())['valid']
        status('launching_once');command('scripts/task41o_kaggle.py','launch')
        status('monitoring');command('scripts/task41o_kaggle.py','supervise')
        status('recording_results');command('-m','research_memory.register_task41o','--result','--validate-only');command('-m','research_memory.register_task41o','--result')
        status('local_database_backup');command('-m','research_memory.prepare_drive_backup');command('-m','research_memory.record_local_backup','--raw-sqlite','--scope','Task41o causal diagnostic controls, all thresholds unchanged, no training or oracle release. SQLite plus compressed raw reports, two verified local generations; no Drive/Airtable.')
        files=['experiments/results/task41o_causal_v1/'+s for s in ('preregistration_receipt.json','result_memory_receipt.json','kaggle_launch.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
        staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines());assert staged<=set(files)
        subprocess.run(['git','commit','-m','Record41o causal diagnostic results and compact verified backups'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        receipt=json.loads((ROOT/'artifacts/task41o_kaggle_external/external_backup_receipt.json').read_text());db=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());assert db['valid']
        status('complete',technical_failure=receipt['technical_failure'],external_backup_verified=receipt['valid'],database_backup_verified=db['valid'],report=receipt['report'].get('decision'))
    except BaseException as error:
        status('attention_required',error=repr(error),traceback=traceback.format_exc(),automatic_restart=False);raise
if __name__=='__main__':main()
