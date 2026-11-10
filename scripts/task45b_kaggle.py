"""One private45b console submission, no duplicate retry or threshold mutation."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task45_kaggle as previous
from research_memory.mirror import write_json
launcher=previous.launcher;ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task45b_protocol_v1';launcher.DEST=ROOT/'artifacts/task45b_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task45b-rate-coverage'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task45b_final\.zip|process\.log)$'
launcher.STATUS_LABEL='GIADA45b'
launcher.LAUNCH_SCOPE=dict(training_authorized=True,training_updates_per_support_seed=2000,sidecar_selection_FIT_DEV_only=True,exploratory=True,frozen41r_reference=True,thresholds_changed=False)

def notebook(revision,source_sha):
    doc=previous.notebook(revision,source_sha)
    for cell in doc['cells']:
        cell['source']=[line.replace('run_task45.py','run_task45b.py').replace('task45_final','task45b_final').replace('giada_task45','giada_task45b').replace('GIADA45','GIADA45b').replace('training_updates_performed=0,selection_recomputed=False',"training_updates_performed=max([int(p.stem.split('_')[-1]) for p in out.glob('training_*.pt')]+[0]),selection_recomputed=(out/'repair_freeze.json').exists()") for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task45b — coefficient attribution and coverage\n',
        'Native-only calibration; atomic inf/rate oracles (diagnostic only); nominal/tail ×500/2000 updates, FIT/DEV freeze;8learned factorial arms + formula control, imposed/autonomous; original45 and44/stress regressions.41r reference immutable, all original thresholds and18states. One pinned console run, partial/final ZIP persisted and independently recovered.\n']
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task45b_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task45b rate coverage',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=[],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if sys.argv[1:]==['build']:
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task45b_rate_coverage.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        launcher.FOLDER.mkdir(exist_ok=True)
        write_json(launcher.FOLDER/'input_verified.json',json.loads((ROOT/'experiments/results/task45_protocol_v1/input_verified.json').read_text()))
    elif sys.argv[1:]==['supervise']:previous.supervise()
    else:launcher.main()
