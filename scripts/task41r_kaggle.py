"""Private41r console dispatcher; no stochastic training or duplicate launch."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as launcher
from research_memory.mirror import write_json
ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task41r_protocol_v1';launcher.DEST=ROOT/'artifacts/task41r_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task41r-direct-timing-factorial'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task41r_final\.zip|process\.log)$'
launcher.LAUNCH_SCOPE=dict(stochastic_training_updates=0,deterministic_weights_frozen=True,development_nomination=True,thresholds_changed=False)
launcher.STATUS_LABEL='GIADA41r'
original_notebook=launcher.notebook
def notebook(revision,source_sha):
    value=original_notebook(revision,source_sha)
    for cell in value['cells']:
        cell['id']=cell['id'].replace('41n','41r')
        cell['source']=[line.replace('run_task41n_recovery.py','run_task41r.py').replace('task41n_recovery_final','task41r_final').replace('GIADA41n recovery','GIADA41r').replace('giada_task41n_recovery','giada_task41r').replace('# GIADA41n — evaluation-only recovery','# GIADA41r — direct timing factorial').replace('No training/selection/native regeneration. Original failed FP32 floors retained; no reinstated primary confirmation.','Frozen deterministic fits:2x2 K_Pst update x timing objective, plus baseline and saved equilibrium control. No stochastic training; unchanged quality gates; new-development nomination before opened data.').replace('selection_recomputed=False','development_nomination=True') for line in cell['source']]
    return value
def bundle(revision):
    folder=ROOT/'artifacts/task41r_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task41r direct timing factorial',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=['alessandrobelli/giada-task41n-evaluation-only-recovery'],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='build':
        spec=json.loads((ROOT/'experiments/task41r_preregistration_v1.json').read_text());assert json.loads((launcher.FOLDER/'fit_manifest.json').read_text())['valid']
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());write_json(ROOT/'notebooks/giada_task41r_direct_timing.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        receipt=json.loads((ROOT/'experiments/results/task41o_causal_v1/input_verified.json').read_text());write_json(launcher.FOLDER/'input_verified.json',receipt)
    else:launcher.main()
