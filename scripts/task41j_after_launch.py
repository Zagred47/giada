"""Finite launch-record/LOCAL-DB-backup completion, no scientific submission."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
folder=ROOT/'experiments/results/task41j_protocol_v1'
status=ROOT/'artifacts/task41j_after_launch_status.json'


def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)


def main():
    deadline=time.monotonic()+90*60
    while not (folder/'launch.json').exists():
        failure=ROOT/'artifacts/task41j_pipeline.stderr.log'
        if failure.exists() and 'Traceback' in failure.read_text(errors='replace'):
            raise RuntimeError('Scientific phase supervisor reported an error; no DB success fabricated')
        assert time.monotonic()<deadline,'First verified external checkpoint not available within bounded launch audit'
        time.sleep(20)
    export=ROOT/'research_memory/data/airtable_snapshot.json'
    prospective=folder/'memory_receipt.json'
    # Ignore old abandoned exports without deleting user data. Only an export
    # newer than the last atomically completed snapshot can still be active.
    while (not export.exists() or export.stat().st_mtime<prospective.stat().st_mtime or
           any(path.stat().st_mtime>export.stat().st_mtime for path in export.parent.glob('airtable_snapshot.json.*.tmp'))):
        assert time.monotonic()<deadline,'Prior consistent export still pending'
        time.sleep(10)
    write_json(status,dict(state='registering actual external checkpoint',scientific_worker_interrupted=False))
    run('-m','research_memory.register_task41j','--running','--validate-only')
    run('-m','research_memory.register_task41j','--running')
    write_json(status,dict(state='creating consistent local DB backup',scientific_worker_interrupted=False))
    run('-m','research_memory.prepare_drive_backup')
    run('-m','research_memory.record_local_backup','--raw-sqlite','--scope',
        'Complete authoritative GIADA memory: all60original Task41i JSON reports and audited NO-GO; Task41j four arms x3seeds preregistered and actually launched on pinned Kaggle revision, finite first checkpoint verified off-runtime. Scientific Task41j outcomes pending. Thresholds and abstraction unchanged. DB backup LOCAL ONLY.')
    run('scripts/link_task41j_memory_backup.py')
    paths=['scripts/link_task41j_memory_backup.py','scripts/task41j_after_launch.py',
        'experiments/results/task41j_protocol_v1/launch.json','experiments/results/task41j_protocol_v1/launch_memory_receipt.json',
        'experiments/results/task41j_protocol_v1/memory_receipt.json','experiments/results/task41j_protocol_v1/kaggle_active_launch.json',
        'experiments/results/task41i_recovery_v1/memory_receipt.json',
        'research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
    for name in paths:assert (ROOT/name).stat().st_size<8_000_000
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Record actual Task41j external checkpoint and verified local memory backup'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
    write_json(status,dict(state='complete',DB_backup_valid=True,DB_backup_local_only=True,scientific_results_pending=True,
                          scientific_worker_interrupted=False,training_or_selection_repeated=False))


if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(status,dict(state='error',error=traceback.format_exc(),scientific_worker_interrupted=False));raise
