import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51b_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51c_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51c_kaggle_external';k.SLUG='alessandrobelli/giada-task51c-hybrid-openings';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51c_final\.zip|task51c_checkpoint_latest\.zip|task51c_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51c';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,training_updates=3000)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task51b','giada_task51c').replace('run_task51b.py','run_task51c.py').replace('task51b_final','task51c_final').replace('GIADA51b','GIADA51c').replace('training_updates_performed=2000','training_updates_performed=3000') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51c hybrid small/large opening loss\nSigmoid/log output xlambda0/1/10; identical balanced sampling and weights perseed. Static physical-state mapping only. Common development selection before fresh. FrozenP, unchanged amplitude thresholds, no temporal claim.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51c_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.prior.base.metadata(True,k.SLUG,'GIADA Task51c hybrid openings'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51c_hybrid_openings.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,d in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d
        k.main()
