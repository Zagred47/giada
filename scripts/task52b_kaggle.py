import hashlib,json,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task52_kaggle as prior
import task46_kaggle as base
import task52_input as inputs
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task52b_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task52b_kaggle_external';k.SLUG='alessandrobelli/giada-task52b-decoder-memory';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task52b_final\.zip|task52b_checkpoint_latest\.zip|task52b_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='52b';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,abstraction_changed=False,training_updates=1200)

def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task52','giada_task52b').replace('run_task52.py','run_task52b.py').replace('task52_final','task52b_final').replace('GIADA52','GIADA52b').replace("worker_report.get('training_updates',0)","worker_report.get('training_updates')") for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task52b — decoder × memory\nFour paired arms, fixed 10-dimensional family state, shared streams and loss; sigmoid/log decoder × standard/multiscale initialization. Frozen52/P controls; post-freeze h0 oracle and teacher re-encoding are diagnostic only. Thresholds unchanged.']
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task52b_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));meta=base.metadata(True,k.SLUG,'GIADA Task52b Decoder Memory');meta['dataset_sources'].append(inputs.DATASET);write_json(folder/'kernel-metadata.json',meta);return folder
k.bundle=bundle

def supervise():
    """Read-only polls, checked partial downloads, then standard final verification."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();k.DEST.mkdir(exist_ok=True)
    launch=json.loads((F/'kaggle_launch.json').read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    partial=k.DEST/'partial_download';partial.mkdir(exist_ok=True)
    for _ in range(320):
        try:
            state=api.kernels_status(ref);name=str(state.status).split('.')[-1].lower()
            write_json(F/'status.json',dict(status=name,observed_at=time.time(),launch=launch,failure_message=state.failure_message));print('[52b] '+name,flush=True)
            try:
                api.kernels_output(ref,str(partial),file_pattern=r'(task52b_checkpoint_latest\.zip|task52b_checkpoint_receipt\.json|process\.log)$',force=True,quiet=True)
                rp=partial/'task52b_checkpoint_receipt.json';cp=partial/'task52b_checkpoint_latest.zip'
                if rp.exists() and cp.exists():
                    receipt=json.loads(rp.read_text())
                    with cp.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
                    with zipfile.ZipFile(cp) as z:assert z.testzip() is None
                    cp.replace(k.DEST/'verified_checkpoint_latest.zip')
                    write_json(k.DEST/'partial_backup_receipt.json',dict(valid=True,runtime_independent=True,local='verified_checkpoint_latest.zip',remote=receipt,verified_at=time.time()))
            except Exception as ex:print('[52b] partial not yet downloadable: '+type(ex).__name__,flush=True)
            if any(word in name for word in ('complete','error','fail','cancel')):
                sys.argv[1]='retrieve';k.main();return
        except Exception as ex:print('[52b] read-only retry: '+type(ex).__name__,flush=True)
        time.sleep(45)
    raise RuntimeError('Monitoring deadline; do not restart training automatically')

if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task52b_decoder_memory.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    elif sys.argv[1]=='supervise':supervise()
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json','frozen_input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        k.main()
