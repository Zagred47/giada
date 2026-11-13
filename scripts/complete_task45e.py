"""Bounded single-run supervision and verified external/local-only DB backups."""
import json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task45e_protocol_v1'
def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
def main():
    def state(s,**kw):write_json(F/'workflow.json',dict(state=s,**kw))
    try:
        state('monitoring');run('scripts/task45e_kaggle.py','supervise')
        state('comparative_audit');run('scripts/compare_task45e.py')
        state('recording_results');run('-m','research_memory.register_task45e','--result')
        state('local_DB_backup');run('-m','research_memory.prepare_drive_backup');run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','45e C1-anchored ionic repair output and paired audit; unchanged thresholds/abstraction, archived45d limitations retained. Local SQLite+lossless reports, last2verified rotation; runtime-independent output hash/CRC verified.')
        names=('kaggle_launch.json','comparison.json','result_memory_receipt.json')
        files=['experiments/results/task45e_protocol_v1/'+n for n in names]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True);assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())<=set(files)
        subprocess.run(['git','commit','-m','Record45e anchored repair output and verified local backup'],cwd=ROOT,check=True);subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        r=json.loads((ROOT/'artifacts/task45e_kaggle_external/external_backup_receipt.json').read_text());state('complete',technical_failure=r['technical_failure'],external_backup_verified=True,DB_backup_verified=True)
    except BaseException as ex:state('attention_required',error=repr(ex),traceback=traceback.format_exc(),automatic_restart=False);raise
if __name__=='__main__':main()
