import hashlib,json,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task52c_kaggle as prior
import task46_kaggle as base
import task52d_input as inputs
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task52d_protocol_v2'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task52d_kaggle_external';k.SLUG='alessandrobelli/giada-task52d-balance-trajectory';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task52d_final\.zip|task52d_checkpoint_latest\.zip|task52d_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='52d';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,abstraction_changed=False,decoder_recalibration_authorized=True,training_updates=800)
def notebook(revision):
    doc=prior.notebook(revision);old=json.loads((ROOT/'experiments/results/task52c_protocol_v1/package.json').read_text())['sha256'];new=json.loads((F/'package.json').read_text())['sha256']
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task52c','giada_task52d').replace('run_task52c.py','run_task52d.py').replace('task52c_final','task52d_final').replace('GIADA52c','GIADA52d').replace('models52c.zip.bin','models52d.zip.bin').replace(old,new) for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task52d — balance × autonomous trajectory\nUser-authorized decoder recalibration after unresolved frozen-decoder feasibility. Four paired arms, frozen encoder, same10states and thresholds. All channel losses retained; conditional net-current/charge and autonomous-voltage/calcium terms.']
    return doc
def bundle(revision):
    dest=ROOT/'artifacts/task52d_notebook';dest.mkdir(exist_ok=True);write_json(dest/'notebook.ipynb',notebook(revision));meta=base.metadata(True,k.SLUG,'GIADA Task52d Balance Trajectory');meta['dataset_sources'].append(inputs.DATASET);write_json(dest/'kernel-metadata.json',meta);return dest
k.bundle=bundle
def supervise():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();k.DEST.mkdir(exist_ok=True);launch=json.loads((F/'kaggle_launch.json').read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version']);partial=k.DEST/'partial_download';partial.mkdir(exist_ok=True)
    for _ in range(320):
        try:
            state=api.kernels_status(ref);name=str(state.status).split('.')[-1].lower();write_json(F/'status.json',dict(status=name,observed_at=time.time(),launch=launch,failure_message=state.failure_message));print('[52d] '+name,flush=True)
            try:
                api.kernels_output(ref,str(partial),file_pattern=r'(task52d_checkpoint_latest\.zip|task52d_checkpoint_receipt\.json|process\.log)$',force=True,quiet=True)
                rp=partial/'task52d_checkpoint_receipt.json';cp=partial/'task52d_checkpoint_latest.zip'
                if rp.exists() and cp.exists():
                    receipt=json.loads(rp.read_text())
                    with cp.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
                    with zipfile.ZipFile(cp) as z:assert z.testzip() is None
                    cp.replace(k.DEST/'verified_checkpoint_latest.zip');write_json(k.DEST/'partial_backup_receipt.json',dict(valid=True,runtime_independent=True,local='verified_checkpoint_latest.zip',remote=receipt,verified_at=time.time()))
            except Exception as ex:print('[52d] partial unavailable: '+type(ex).__name__,flush=True)
            if any(word in name for word in ('complete','error','fail','cancel')):sys.argv[1]='retrieve';k.main();return
        except Exception as ex:print('[52d] read-only retry: '+type(ex).__name__,flush=True)
        time.sleep(45)
    raise RuntimeError('Monitoring deadline; do not restart automatically')
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task52d_balance_trajectory.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    elif sys.argv[1]=='supervise':supervise()
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json','frozen_input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        k.main()
