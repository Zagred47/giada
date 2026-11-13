"""Same46protocol technical retry after preflight/GPU source-cache collision."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task46_kaggle as previous
from research_memory.mirror import write_json
k=previous.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task46_retry_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task46_retry_external';k.SLUG='alessandrobelli/giada-task46-cache-retry';k.STATUS_LABEL='46 cache retry'
k.LAUNCH_SCOPE=dict(exploratory=True,technical_retry=True,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,frozen_candidate='45e/P',task45e_passed=False)
def bundle(revision):
    folder=ROOT/'artifacts/task46_retry_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',previous.notebook(revision,previous.plans()[0]['source_sha256']));write_json(folder/'kernel-metadata.json',previous.metadata(True,k.SLUG,'GIADA Task46 cache retry'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='prepare':
        rec=json.loads((ROOT/'artifacts/task46_kaggle_external/external_backup_receipt.json').read_text());assert rec['valid'] and rec['technical_failure']
        assert 'FileExistsError' in rec['report']['error'] and rec['report']['phase']=='gpu'
        for name in ('input_verified.json','native_environment_verified.json','reconstruction_receipt.json','local_preflight.json'):
            p=ROOT/'experiments/results/task46_protocol_v1'/name;obj=json.loads(p.read_text());assert obj['valid'];write_json(F/name,obj)
        write_json(F/'preregistration_receipt.json',dict(valid=True,same_protocol=True,source_preregistration='experiments/results/task46_protocol_v1/preregistration_receipt.json',original_failure_preserved=True,original_archive_sha256=rec['remote']['sha256'],reason='Preflight and mainGPU both extracted to exclusive verified_source46 cache. Preflight now uses a separate sibling workspace. Weights/inputs/thresholds/numerics unchanged; no training/selection. Native data generation is deterministic and rerun only because failed-version output cannot be mounted without new payload upload.'))
        write_json(ROOT/'notebooks/giada_task46_cache_retry.ipynb',previous.notebook('PINNED_BY_CONSOLE_DISPATCH',previous.plans()[0]['source_sha256']))
    else:k.main()
