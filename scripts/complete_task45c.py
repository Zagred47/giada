"""One45c GPU run after mounted-input proof, verified external and local backups."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
FOLDER=ROOT/'experiments/results/task45c_protocol_v1'
def main():
    started=time.time()
    def state(name,**kw):write_json(FOLDER/'workflow.json',dict(state=name,elapsed_seconds=time.time()-started,**kw))
    def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    try:
        assert json.loads((FOLDER/'preregistration_receipt.json').read_text())['valid']
        assert json.loads((FOLDER/'input_verified.json').read_text())['valid']
        pre=json.loads((FOLDER/'local_preflight.json').read_text());assert pre['valid']
        for name,digest in pre['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        state('launching_once');run('scripts/task45c_kaggle.py','launch');state('monitoring');run('scripts/task45c_kaggle.py','supervise')
        state('recording_results');run('-m','research_memory.register_task45c','--result','--validate-only');run('-m','research_memory.register_task45c','--result')
        state('local_DB_backup');run('-m','research_memory.prepare_drive_backup');run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','45c same-head preservation matrix from f011, all original thresholds. Complete SQLite/payload backup local only, last2 verified; external final/partial ZIP SHA/CRC verified.')
        files=['experiments/results/task45c_protocol_v1/'+n for n in ('preregistration_receipt.json','result_memory_receipt.json','kaggle_launch.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True);assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())<=set(files)
        subprocess.run(['git','commit','-m','Record Task45c output and local DB backup'],cwd=ROOT,check=True);subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        receipt=json.loads((ROOT/'artifacts/task45c_kaggle_external/external_backup_receipt.json').read_text());state('complete',technical_failure=receipt['technical_failure'],external_backup_verified=True,DB_backup_verified=True)
    except BaseException as error:state('attention_required',error=repr(error),traceback=traceback.format_exc(),automatic_restart=False);raise
if __name__=='__main__':main()
