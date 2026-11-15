import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task50_kaggle as prior
import task46_kaggle as base
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task51_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task51_kaggle_external';k.SLUG='alessandrobelli/giada-task51-initial-loss-memory';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task51_final\.zip|task51_checkpoint_latest\.zip|task51_checkpoint_receipt\.json|process\.log)$';k.STATUS_LABEL='51'
k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,factorial='2x2x2',training_updates=600)
def notebook(revision):
    doc=prior.notebook(revision)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('giada_task50','giada_task51').replace('run_task50.py','run_task51.py').replace('task50_final','task51_final').replace('GIADA50','GIADA51').replace('training_updates_performed=2000','training_updates_performed=600') for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task51 — initial/loss/memory factorial\nEight fixedL10/T16 arms x3seeds with identical initialization perseed. Explicit startup supervision x compound calibrated loss x6.4/102.4ms stateful training(TBPTT64). Same starts/updates, not same FLOPs. Development-only freeze then fresh. P unchanged, original quality thresholds, no coupled release. Full checkpoint/optimizer backups at100/300/600.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task51_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',base.metadata(True,k.SLUG,'GIADA Task51 initial loss memory'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task51_initial_loss_memory.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid'],name
            for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        k.main()
