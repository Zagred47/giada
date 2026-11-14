"""Private47 console dispatch using only already verified mounted inputs."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task46_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task47_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task47_kaggle_external';k.SLUG='alessandrobelli/giada-task47-independent-confirmation';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task47_final\.zip|process\.log)$';k.STATUS_LABEL='47'
k.LAUNCH_SCOPE=dict(exploratory=True,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,frozen_candidate='45e/P',task45e_passed=False)
def notebook(revision):
    doc=prior.notebook(revision,prior.plans()[0]['source_sha256'])
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task46','giada_task47').replace('run_task46.py','run_task47.py').replace('task46_final','task47_final').replace('GIADA46','GIADA47') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task47 — independent level-one confirmation\nFrozenP, new chain5/branch9, disjoint input seeds and waveforms. Same native/formula/C1/P/R41 paired gates, no training/selection or altered thresholds. Quality and runtime verdicts separate;45e remains incomplete.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task47_notebook';folder.mkdir(exist_ok=True)
    write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.metadata(True,k.SLUG,'GIADA Task47 independent confirmation'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task47_independent_confirmation.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('native_environment_verified.json','input_verified.json','local_preflight.json','preregistration_receipt.json'):assert json.loads((F/name).read_text())['valid'],name
        k.main()
