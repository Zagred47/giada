import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51i_kaggle as prior
import task46_kaggle as base
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51j_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51j_kaggle_external';k.SLUG='alessandrobelli/giada-task51j-confirmation';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51j_final\.zip|task51j_checkpoint_latest\.zip|task51j_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51j';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,training_updates=5000)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task51i','giada_task51j').replace('run_task51i.py','run_task51j.py').replace('task51i_final','task51j_final').replace('GIADA51i','GIADA51j') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51j Confirmation\nIndependent confirmation of fixed SGD and Adam epsilon1e-4, new streams/seeds,48freshhistories; frozen and defaultAdam controls. Same architecture, thresholds and frozen P. Static readout only.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51j_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',base.metadata(True,k.SLUG,'GIADA Task51j Confirmation'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51j_confirmation.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,d in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d
        k.main()
