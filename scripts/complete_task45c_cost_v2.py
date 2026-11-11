"""Recover cost-only output, register and locally back up memory; never retry run."""
import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
def main():
    run('scripts/task45c_cost_transport_v2.py','supervise')
    receipt=ROOT/'experiments/results/task45c_cost_recovery_v2/memory_receipt.json'
    for _ in range(120):
        if receipt.exists():break
        time.sleep(10)
    else:raise RuntimeError('DB verification pending; do not duplicate')
    run('-m','research_memory.register_task45c_recovery','--v2','--result')
    run('-m','research_memory.prepare_drive_backup')
    run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','45c cost-only verified-input recovery, original45c failures and accuracy retained. No training, selection or threshold changes. DB backup local only.')
    files=['experiments/results/task45c_cost_recovery_v2/'+n for n in ('kaggle_launch.json','memory_receipt.json','result_memory_receipt.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())<=set(files)
    subprocess.run(['git','commit','-m','Record45c cost recovery verified-input result and backup'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
if __name__=='__main__':main()
