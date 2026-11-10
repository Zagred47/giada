"""One pinned console44 run with verified frozen source and external retrieval."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task43_kaggle as previous
from research_memory.mirror import write_json
launcher=previous.launcher;ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task44_protocol_v1'
launcher.DEST=ROOT/'artifacts/task44_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task44-composition-localization'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task44_final\.zip|process\.log)$'
launcher.STATUS_LABEL='GIADA44'
launcher.LAUNCH_SCOPE=dict(training_updates_performed=0,selection_recomputed=False,exploratory=True,frozen41r=True,thresholds_changed=False,paired_spatial_family_masks=True)

def notebook(revision,source_sha):
    doc=previous.notebook(revision,source_sha)
    for cell in doc['cells']:
        cell['source']=[s.replace('run_task43.py','run_task44.py').replace('task43_final','task44_final').replace('giada_task43','giada_task44').replace('GIADA43','GIADA44') for s in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task44 — composition localization\n','Frozen41r; eight paired spatial/channel substitution arms, imposed and autonomous; original thresholds and physics. Batched arms, no training/selection or speedup claim.\n']
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task44_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task44 composition localization',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=[],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if sys.argv[1:]==['build']:
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task44_composition_localization.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        launcher.FOLDER.mkdir(exist_ok=True)
        write_json(launcher.FOLDER/'input_verified.json',json.loads((ROOT/'experiments/results/task43_protocol_v1/input_verified.json').read_text()))
    else:launcher.main()
