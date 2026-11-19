import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51f_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51g_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51g_kaggle_external';k.SLUG='alessandrobelli/giada-task51g-optimization-gap';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51g_final\.zip|task51g_checkpoint_latest\.zip|task51g_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51g';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,training_updates=15000)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task51f','giada_task51g').replace('run_task51f.py','run_task51g.py').replace('task51f_final','task51g_final').replace('GIADA51f','GIADA51g') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51g optimization gap\nFit-only initialization x learning-rate schedule; paired seeds, exact labels, 0/5k/10k/15k budget audit. Same architecture, thresholds and frozen P. Static readout only.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51g_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.prior.base.metadata(True,k.SLUG,'GIADA Task51g optimization gap'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51g_optimization_gap.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,d in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d
        k.main()
