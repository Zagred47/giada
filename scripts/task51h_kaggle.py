import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51g_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51h_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51h_kaggle_external';k.SLUG='alessandrobelli/giada-task51h-reach-preserve';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51h_final\.zip|task51h_checkpoint_latest\.zip|task51h_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51h';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,training_updates=15000)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task51g','giada_task51h').replace('run_task51g.py','run_task51h.py').replace('task51g_final','task51h_final').replace('GIADA51g','GIADA51h') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51h optimization gap\nProjection x late learning-rate schedule; paired seeds, exact labels, pre/post intervention audit. Same architecture, thresholds and frozen P. Static readout only.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51h_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.prior.prior.base.metadata(True,k.SLUG,'GIADA Task51h optimization gap'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51h_reach_preserve.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,d in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d
        k.main()
