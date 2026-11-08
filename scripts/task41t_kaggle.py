"""Private41t console dispatcher; no stochastic training or duplicate launch."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as launcher
from research_memory.mirror import write_json
ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task41t_protocol_v1';launcher.DEST=ROOT/'artifacts/task41t_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task41t-numerical-memory-factorial'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task41t_final\.zip|process\.log)$'
launcher.LAUNCH_SCOPE=dict(stochastic_training_updates=0,deterministic_weights_frozen=True,development_nomination=True,thresholds_changed=False)
launcher.STATUS_LABEL='GIADA41t'
original_notebook=launcher.notebook
def notebook(revision,source_sha):
    value=original_notebook(revision,source_sha)
    for cell in value['cells']:
        cell['id']=cell['id'].replace('41n','41t')
        cell['source']=[line.replace('run_task41n_recovery.py','run_task41t.py').replace('task41n_recovery_final','task41t_final').replace('GIADA41n recovery','GIADA41t').replace('giada_task41n_recovery','giada_task41t').replace('# GIADA41n — evaluation-only recovery','# GIADA41t — direct timing factorial').replace('No training/selection/native regeneration. Original failed FP32 floors retained; no reinstated primary confirmation.','Frozen same-head fits: K_Pst identification, Nap geometry x preservation, paired memory and development-only recomposition;41r incumbent. No stochastic training; unchanged quality gates; new-development nomination before opened data.').replace('selection_recomputed=False','development_nomination=True') for line in cell['source']]
    return value
def bundle(revision):
    folder=ROOT/'artifacts/task41t_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task41t numerical memory factorial',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=['alessandrobelli/giada-task41n-evaluation-only-recovery'],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='build':
        spec=json.loads((ROOT/'experiments/task41t_preregistration_v1.json').read_text());assert json.loads((launcher.FOLDER/'fit_manifest.json').read_text())['valid']
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());write_json(ROOT/'notebooks/giada_task41t_direct_timing.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        receipt=json.loads((ROOT/'experiments/results/task41o_causal_v1/input_verified.json').read_text());write_json(launcher.FOLDER/'input_verified.json',receipt)
    else:launcher.main()
