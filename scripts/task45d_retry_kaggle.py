"""Explicitly authorized technical retry; original failed attempt retained."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task45d_kaggle as old
from research_memory.mirror import write_json
k=old.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task45d_retry_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task45d_retry_external';k.SLUG='alessandrobelli/giada-task45d-cache-retry';k.STATUS_LABEL='45d cache retry'
k.LAUNCH_SCOPE=dict(exploratory=True,thresholds_changed=False,abstraction_changed=False,technical_retry=True,original_registered_training_budget=True)
def bundle(revision):
    folder=ROOT/'artifacts/task45d_retry_notebook';folder.mkdir(exist_ok=True)
    write_json(folder/'notebook.ipynb',old.notebook(revision,old.plans()[0]['source_sha256']))
    write_json(folder/'kernel-metadata.json',old.metadata(folder,True,k.SLUG,'GIADA Task45d cache retry'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='prepare':
        source=ROOT/'experiments/results/task45d_protocol_v1'
        assert json.loads((source/'result_memory_receipt.json').read_text())['valid']
        receipt=json.loads((ROOT/'artifacts/task45d_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['technical_failure']
        write_json(F/'preregistration_receipt.json',dict(valid=True,same_protocol=True,source_preregistration='experiments/results/task45d_protocol_v1/preregistration_receipt.json',original_failure_preserved=True,original_archive_sha256=receipt['remote']['sha256'],reason='Resident constants cached as inference tensors; original worker stopped before first backward/update. Fix cache tensor lifetime, no equation/value/budget/threshold changes.'))
        proof=json.loads((source/'input_verified.json').read_text());assert proof['valid'];write_json(F/'input_verified.json',proof)
        write_json(ROOT/'notebooks/giada_task45d_cache_retry.ipynb',old.notebook('PINNED_BY_CONSOLE_DISPATCH',old.plans()[0]['source_sha256']))
    else:k.main()
