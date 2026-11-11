"""Supervise existing recovery once; verify external output and local DB backup."""
import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
def main():
    run('scripts/task45c_cost_recovery_kaggle.py','supervise')
    receipt=ROOT/'experiments/results/task45c_cost_recovery_v1/memory_receipt.json'
    for _ in range(120):
        if receipt.exists():break
        time.sleep(10)
    else:raise RuntimeError('Wait for DB preregistration verification; do not duplicate')
    run('-m','research_memory.register_task45c_recovery','--result')
    run('-m','research_memory.prepare_drive_backup')
    run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','45c cost-only recovery: original failure and original accuracy retained; no retraining/reselection. SQLite and raw reports local only.')
    files=['experiments/results/task45c_cost_recovery_v1/'+n for n in ('kaggle_launch.json','memory_receipt.json','result_memory_receipt.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())<=set(files)
    subprocess.run(['git','commit','-m','Record45c cost-only recovery and verified local backup'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
if __name__=='__main__':main()
