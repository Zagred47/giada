"""Monitor one technical retry; preserve output and local-only DB backup."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task45d_retry_v1'
def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
def main():
    def state(name,**kw):write_json(F/'workflow.json',dict(state=name,**kw))
    try:
        state('monitoring');run('scripts/task45d_retry_kaggle.py','supervise')
        for _ in range(120):
            if (F/'memory_receipt.json').exists():break
            time.sleep(10)
        else:raise RuntimeError('Retry DB registration pending; do not duplicate')
        state('recording_results');run('-m','research_memory.register_task45d_retry','--result');run('-m','research_memory.prepare_drive_backup');run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','45d same-protocol cache retry, original failure retained. SQLite and compressed reports local only; runtime-independent output SHA/CRC verified.')
        files=['experiments/results/task45d_retry_v1/'+n for n in ('kaggle_launch.json','memory_receipt.json','result_memory_receipt.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True);assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())<=set(files)
        subprocess.run(['git','commit','-m','Record45d cache retry output and local backup'],cwd=ROOT,check=True);subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        receipt=json.loads((ROOT/'artifacts/task45d_retry_external/external_backup_receipt.json').read_text());state('complete',technical_failure=receipt['technical_failure'],external_backup_verified=True,DB_backup_verified=True)
    except BaseException as e:state('attention_required',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
