import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51e_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51f_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51f_kaggle_external';k.SLUG='alessandrobelli/giada-task51f-local-targets';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51f_final\.zip|task51f_checkpoint_latest\.zip|task51f_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51f';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,training_updates=5000)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task51e','giada_task51f').replace('run_task51e.py','run_task51f.py').replace('task51e_final','task51f_final').replace('GIADA51e','GIADA51f') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51f local capacity / exact log targets\nLocal8/local24 x floored/exact log labels, legacy8 control; same stream and paired weights. Static full-state map, thresholds unchanged. Fit-only local linear controls quantify target distortion, not temporal success. P frozen.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51f_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.base.metadata(True,k.SLUG,'GIADA Task51f local targets'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51f_local_targets.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,d in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d
        k.main()
