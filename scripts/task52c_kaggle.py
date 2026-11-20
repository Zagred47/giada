import hashlib,json,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task52b_kaggle as prior
import task46_kaggle as base
import task52c_input as inputs
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task52c_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task52c_kaggle_external';k.SLUG='alessandrobelli/giada-task52c-component-repair';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task52c_final\.zip|task52c_checkpoint_latest\.zip|task52c_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='52c';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,abstraction_changed=False,training_updates=2400)
def notebook(revision):
    doc=prior.notebook(revision);old=json.loads((ROOT/'experiments/results/task52_protocol_v1/package.json').read_text())['sha256'];new=json.loads((F/'package.json').read_text())['sha256']
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task52b','giada_task52c').replace('run_task52b.py','run_task52c.py').replace('task52b_final','task52c_final').replace('GIADA52b','GIADA52c').replace('models52.zip.bin','models52c.zip.bin').replace(old,new).replace('valid=code==0,revision',"valid=code==0,execution_valid=code==0,scientific_acceptance_passed=worker_report.get('scientific_acceptance_passed',False),revision") for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task52c — D/DE/DR/DER\nFrozen52b core, common calibrated decoder, paired encoder/recurrence repair. Same10-state abstraction and quality thresholds. Passive negative control, original controls and postfreeze diagnostics.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task52c_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));meta=base.metadata(True,k.SLUG,'GIADA Task52c Component Repair');meta['dataset_sources'].append(inputs.DATASET);write_json(folder/'kernel-metadata.json',meta);return folder
k.bundle=bundle
def supervise():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();k.DEST.mkdir(exist_ok=True);launch=json.loads((F/'kaggle_launch.json').read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    partial=k.DEST/'partial_download';partial.mkdir(exist_ok=True)
    for _ in range(320):
        try:
            state=api.kernels_status(ref);name=str(state.status).split('.')[-1].lower();write_json(F/'status.json',dict(status=name,observed_at=time.time(),launch=launch,failure_message=state.failure_message));print('[52c] '+name,flush=True)
            try:
                api.kernels_output(ref,str(partial),file_pattern=r'(task52c_checkpoint_latest\.zip|task52c_checkpoint_receipt\.json|process\.log)$',force=True,quiet=True)
                rp=partial/'task52c_checkpoint_receipt.json';cp=partial/'task52c_checkpoint_latest.zip'
                if rp.exists() and cp.exists():
                    receipt=json.loads(rp.read_text())
                    with cp.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
                    with zipfile.ZipFile(cp) as z:assert z.testzip() is None
                    cp.replace(k.DEST/'verified_checkpoint_latest.zip');write_json(k.DEST/'partial_backup_receipt.json',dict(valid=True,runtime_independent=True,local='verified_checkpoint_latest.zip',remote=receipt,verified_at=time.time()))
            except Exception as ex:print('[52c] partial not yet downloadable: '+type(ex).__name__,flush=True)
            if any(word in name for word in ('complete','error','fail','cancel')):sys.argv[1]='retrieve';k.main();return
        except Exception as ex:print('[52c] read-only retry: '+type(ex).__name__,flush=True)
        time.sleep(45)
    raise RuntimeError('Monitoring deadline; do not restart training automatically')
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task52c_component_repair.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json','frozen_input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        if sys.argv[1]=='supervise':supervise()
        else:k.main()
