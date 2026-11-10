"""Single private Task43 submission, paired frozen controls and verified input."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task42_kaggle as previous
from research_memory.mirror import write_json
launcher=previous.launcher;ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task43_protocol_v1'
launcher.DEST=ROOT/'artifacts/task43_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task43-frozen-transfer'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task43_final\.zip|process\.log)$'
launcher.STATUS_LABEL='GIADA43'
launcher.LAUNCH_SCOPE=dict(training_updates_performed=0,selection_recomputed=False,exploratory=True,frozen41r=True,thresholds_changed=False,paired_node_orders=True)

def notebook(revision,source_sha):
    doc=previous.notebook(revision,source_sha)
    for cell in doc['cells']:
        cell['source']=[s.replace('run_task42.py','run_task43.py').replace('task42_final','task43_final').replace('giada_task42','giada_task43').replace('GIADA42','GIADA43').replace('GIADA Task42','GIADA Task43') for s in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task43 — frozen level-one transfer\n','Paired physical node order controls, heterogeneous channels and local conditioning; unchanged41r and thresholds. No training/selection. Native200ms references generated only for this new protocol.\n']
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task43_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task43 frozen heterogeneous transfer',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=[],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if sys.argv[1:]==['build']:
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task43_frozen_transfer.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        launcher.FOLDER.mkdir(exist_ok=True)
        write_json(launcher.FOLDER/'input_verified.json',json.loads((ROOT/'experiments/results/task41v_protocol_v1/input_verified.json').read_text()))
    else:launcher.main()
