"""One43 run; verify persisted output, register results and rotate local DB backup."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
FOLDER=ROOT/'experiments/results/task43_protocol_v1'

def main():
    start=time.time()
    def state(name,**kw):write_json(FOLDER/'workflow.json',dict(state=name,elapsed_seconds=time.time()-start,**kw))
    def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    try:
        assert json.loads((FOLDER/'preregistration_receipt.json').read_text())['valid']
        state('launching_once');run('scripts/task43_kaggle.py','launch')
        state('monitoring');run('scripts/task43_kaggle.py','supervise')
        state('recording_results');run('-m','research_memory.register_task43','--result','--validate-only');run('-m','research_memory.register_task43','--result')
        state('local_DB_backup');run('-m','research_memory.prepare_drive_backup');run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','Task43 frozen41r paired heterogeneous transfer; unchanged thresholds and abstraction. Full DB and payload bundle local only, last2 verified.')
        files=['experiments/results/task43_protocol_v1/'+s for s in ('preregistration_receipt.json','result_memory_receipt.json','kaggle_launch.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
        staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines());assert staged<=set(files)
        subprocess.run(['git','commit','-m','Record Task43 paired transfer output and verified local DB backup'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        receipt=json.loads((ROOT/'artifacts/task43_kaggle_external/external_backup_receipt.json').read_text())
        state('complete',technical_failure=receipt['technical_failure'],external_backup_verified=True,DB_backup_verified=True)
    except BaseException as error:state('attention_required',error=repr(error),traceback=traceback.format_exc(),automatic_restart=False);raise
if __name__=='__main__':main()
