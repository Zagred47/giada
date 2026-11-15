import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51b_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51b_kaggle_external';k.SLUG='alessandrobelli/giada-task51b-small-openings';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51b_final\.zip|task51b_checkpoint_latest\.zip|task51b_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51b'
k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,temporal_dynamics_tested=False,training_updates=2000)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task51','giada_task51b').replace('run_task51.py','run_task51b.py').replace('task51_final','task51b_final').replace('GIADA51','GIADA51b').replace('training_updates_performed=600','training_updates_performed=2000') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51b — static small-opening learnability\nThree output parameterizations x two objective/sampling packages x3seeds. Full physical gates supplied; no recurrence. Fit-only bias, held-out development selection and fresh evaluation, analytic identity/zero controls. No threshold changes or compressed-memory claim. P frozen.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51b_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.base.metadata(True,k.SLUG,'GIADA Task51b small openings'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51b_small_openings.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        k.main()
