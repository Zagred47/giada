"""Private Task49 dispatch reusing the three verified existing Kaggle inputs."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task48_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task49_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task49_kaggle_external';k.SLUG='alessandrobelli/giada-task49-causal-initialization';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task49_final\.zip|process\.log)$';k.STATUS_LABEL='49'
k.LAUNCH_SCOPE=dict(exploratory=True,gradient_training_updates=0,codec_fit_only=True,selection_recomputed=False,thresholds_changed=False,frozen_candidate='45e/P',task47_passed=False)

def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task48','giada_task49').replace('run_task48.py','run_task49.py').replace('task48_final','task49_final').replace('GIADA48','GIADA49') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task49 — causal initialization\nFrozenP and canonical references. Before/after-rest cuts300/350/400ms; state-available fixed-width fit-only family PCA versus physical identity, past-only warmup0/1/8/40/100/300/400ms, restart and shuffled-state controls. Explicit disjoint fit/development/fresh histories, no arm selection. P unchanged; PCA decoder is only an initialization probe, not compressed online recurrence. No modified thresholds or Task47 requalification. All outputs and partial panels preserved; no new input upload.']
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task49_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.prior.metadata(True,k.SLUG,'GIADA Task49 causal initialization'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task49_causal_initialization.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid'],name
            for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        k.main()
