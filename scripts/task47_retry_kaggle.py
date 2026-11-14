"""Same preregistered47; fix child entrypoint, preserve misrouted original job."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task47_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task47_retry_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task47_retry_external';k.SLUG='alessandrobelli/giada-task47-protocol-retry';k.STATUS_LABEL='47 protocol retry'
k.LAUNCH_SCOPE.update(technical_retry=True,protocol_identity_checked_per_phase=True)
def bundle(revision):
    folder=ROOT/'artifacts/task47_retry_notebook';folder.mkdir(exist_ok=True)
    write_json(folder/'notebook.ipynb',prior.notebook(revision));write_json(folder/'kernel-metadata.json',prior.prior.metadata(True,k.SLUG,'GIADA Task47 protocol retry'));return folder
k.bundle=bundle
def prepare():
    F.mkdir(exist_ok=True);old=ROOT/'experiments/results/task47_protocol_v1'
    launch=json.loads((old/'kaggle_launch.json').read_text());receipt=json.loads((ROOT/'artifacts/task47_kaggle_external/kaggle_receipt.json').read_text());log=(ROOT/'artifacts/task47_kaggle_external/process.log').read_text()
    assert receipt['revision']==launch['revision'] and receipt['worker_exit_code']==0
    assert '[GIADA47] phase native' in log and '[GIADA42] native chain4 complete' in log and '[GIADA46] chain4/autonomous' in log
    p=json.loads((ROOT/'experiments/task47_preregistration_v1.json').read_text())
    evidence=dict(valid=True,technical_failure=True,scientific_task47_valid=False,original_launch=launch,original_provider_receipt=receipt,original_log=log,archive_local_verification_pending=True,cause='Parent47 context was not inherited when child processes ran run_task46.py via worker.__file__. Native and GPU phases ran46geometries; finalizer relabelled the result47.',fix='Explicit47entrypoint on each child; per-phase full-plan hash/geometry/input-seed receipts checked by parent and finalizer. Reject misrouted completed report.',same_preregistration_sha256=hashlib.sha256((ROOT/'experiments/task47_preregistration_v1.json').read_bytes()).hexdigest(),frozen_P=p['freeze']['logical_weights_sha256'],training_updates=0,selection_performed=False,thresholds_changed=False,abstraction_changed=False,no_new_payload_upload=True)
    write_json(F/'technical_attempt_v1.json',evidence)
    for name in ('input_verified.json','native_environment_verified.json','local_preflight.json'):
        obj=json.loads((old/name).read_text());assert obj['valid'];write_json(F/name,obj)
    write_json(ROOT/'notebooks/giada_task47_protocol_retry.ipynb',prior.notebook('PINNED_BY_CONSOLE_DISPATCH'))
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        if sys.argv[1]=='launch':
            for name in ('technical_attempt_v1.json','input_verified.json','native_environment_verified.json','local_preflight.json','preregistration_receipt.json'):assert json.loads((F/name).read_text())['valid'],name
            proof=json.loads((F/'local_preflight.json').read_text())
            for name,digest in proof['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Unverified code change: '+name
        k.main()
