import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51c_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51d_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51d_kaggle_external';k.SLUG='alessandrobelli/giada-task51d-local-relative';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51d_final\.zip|task51d_checkpoint_latest\.zip|task51d_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51d';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,training_updates=5000)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task51c','giada_task51d').replace('run_task51c.py','run_task51d.py').replace('task51c_final','task51d_final').replace('GIADA51c','GIADA51d').replace('training_updates_performed=3000','training_updates_performed=5000') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51d local readout / relative loss\nFixed log-output, shared vs channel-local masks x hybrid/relative-hybrid loss. Three seeds, identical matrices, balanced sample stream; no temporal dynamics or changed thresholds. Nonselectable analytic log-linear control. Common development selection before fresh.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51d_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.prior.prior.base.metadata(True,k.SLUG,'GIADA Task51d local relative'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51d_local_relative.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,d in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d
        k.main()
