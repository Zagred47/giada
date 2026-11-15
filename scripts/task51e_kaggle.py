import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51d_kaggle as prior
import task46_kaggle as base
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51e_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51e_kaggle_external';k.SLUG='alessandrobelli/giada-task51e-readout-conditioning';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51e_final\.zip|task51e_checkpoint_latest\.zip|task51e_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51e';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,training_updates=5000)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task51d','giada_task51e').replace('run_task51d.py','run_task51e.py').replace('task51d_final','task51e_final').replace('GIADA51d','GIADA51e') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51e readout capacity/conditioning\nShared32/shared88/local8/learned linear x hybrid/raw-log objectives x3seeds. Same balanced sampling. Separate fit-only least-squares control. Static full-state map, no recurrence or changed thresholds.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51e_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',base.metadata(True,k.SLUG,'GIADA Task51e readout conditioning'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51e_readout_conditioning.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,d in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d
        k.main()
