import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task49_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task50_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task50_kaggle_external';k.SLUG='alessandrobelli/giada-task50-compact-family-memory';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task50_final\.zip|process\.log)$';k.STATUS_LABEL='50'
k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,latent_training=True)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task49','giada_task50').replace('run_task49.py','run_task50.py').replace('task49_final','task50_final').replace('GIADA49','GIADA50').replace('training_updates_performed=0','training_updates_performed=2000').replace('selection_recomputed=False','selection_recomputed=True') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task50 — compact recurrent family memory\nFive fixed architectures x three seeds, common data/minibatches, checkpoints250/1000/2000. Physical-state initialization only; no gate decoder or canonical rates during recurrence. Canonical targets, frozenP comparator, development selection before fresh evaluation. Unchanged quality thresholds, no coupled release.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task50_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.prior.prior.metadata(True,k.SLUG,'GIADA Task50 compact family memory'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task50_compact_family_memory.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid'],name
            for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        k.main()
