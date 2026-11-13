"""One corrected46run: persist technical history, results and local backups."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task46_retry_v1'
def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
def main():
    def state(s,**kw):write_json(F/'workflow.json',dict(state=s,**kw))
    try:
        state('monitoring');run('scripts/task46_retry_kaggle.py','supervise');run('scripts/compare_task46_retry.py')
        for _ in range(120):
            if (F/'memory_receipt.json').exists():break
            time.sleep(10)
        else:raise RuntimeError('Technical retry memory registration pending')
        state('recording_results');run('-m','research_memory.register_task46_retry','--result')
        state('local_DB_backup');run('-m','research_memory.prepare_drive_backup');run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','Task46 same-protocol isolated-cache retry, original technical failure and native references preserved; frozenP, no training/selection/new payload upload.45e incomplete; original thresholds. External archive hash/CRC verified, SQLite+raw reports local, last2verified bundles.')
        files=['experiments/results/task46_retry_v1/'+name for name in ('kaggle_launch.json','comparison.json','memory_receipt.json','result_memory_receipt.json','preregistration_receipt.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True);assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())<=set(files)
        subprocess.run(['git','commit','-m','Record46 isolated-cache audit and verified local backup'],cwd=ROOT,check=True);subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        r=json.loads((ROOT/'artifacts/task46_retry_external/external_backup_receipt.json').read_text());state('complete',technical_failure=r['technical_failure'],external_backup_verified=True,DB_backup_verified=True)
    except BaseException as ex:state('attention_required',error=repr(ex),traceback=traceback.format_exc(),automatic_restart=False);raise
if __name__=='__main__':main()
