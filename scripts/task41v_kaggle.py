"""Private41v console dispatcher; single bounded GPU training/evaluation submission; no duplicate launch."""
import json,sys,threading,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as launcher
from research_memory.mirror import write_json
ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task41v_protocol_v1';launcher.DEST=ROOT/'artifacts/task41v_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task41v-targeted-coefficient-factorial'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task41v_final\.zip|process\.log|checkpoint_receipt\.json|checkpoint_backup\.zip)$'
launcher.LAUNCH_SCOPE=dict(registered_training_updates_per_arm=768,training_requested_on_gpu=True,frozen_incumbent=True,development_nomination=True,thresholds_changed=False)
launcher.STATUS_LABEL='GIADA41v'
original_notebook=launcher.notebook
def notebook(revision,source_sha):
    value=original_notebook(revision,source_sha)
    for cell in value['cells']:
        cell['id']=cell['id'].replace('41n','41v')
        cell['source']=[line.replace('run_task41n_recovery.py','run_task41v.py').replace('task41n_recovery_final','task41v_final').replace('GIADA41n recovery','GIADA41v').replace('giada_task41n_recovery','giada_task41v').replace('# GIADA41n — evaluation-only recovery','# GIADA41v — targeted coefficient factorial').replace('No training/selection/native regeneration. Original failed FP32 floors retained; no reinstated primary confirmation.','Targeted K_u12 and joint Nap_u8/z9: direct coefficient vs indirect trajectory supervision, existing global heads vs existing local correction bank, isolated controls and recomposition. Nonselectable GPU causal probes. GPU training64/256/768updates; all training finishes before development selection. Original quality gates and abstraction unchanged; original failed FP32 floors retained.').replace('selection_recomputed=False','development_nomination=True').replace('training_updates_performed=0','registered_training_updates_per_arm=768') for line in cell['source']]
    return value
def bundle(revision):
    folder=ROOT/'artifacts/task41v_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task41v targeted coefficient factorial',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=['alessandrobelli/giada-task41n-evaluation-only-recovery'],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='build':
        spec=json.loads((ROOT/'experiments/task41v_preregistration_v1.json').read_text());assert json.loads((launcher.FOLDER/'fit_manifest.json').read_text())['valid']
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());write_json(ROOT/'notebooks/giada_task41v_targeted_coefficients.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        receipt=json.loads((ROOT/'experiments/results/task41o_causal_v1/input_verified.json').read_text());write_json(launcher.FOLDER/'input_verified.json',receipt)
    else:
        stopped=threading.Event()
        def checkpoint_reader():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi();api.authenticate();stage=ROOT/'artifacts/task41v_checkpoint_readback';stage.mkdir(exist_ok=True)
            while not stopped.is_set():
                try:
                    launch=json.loads((launcher.FOLDER/'kaggle_launch.json').read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
                    api.kernels_output(ref,str(stage),file_pattern=r'(checkpoint_receipt\.json|checkpoint_backup\.zip)$',force=True,quiet=True)
                    receipt=json.loads((stage/'checkpoint_receipt.json').read_text());path=stage/'checkpoint_backup.zip'
                    assert launcher.sha(path)==receipt['sha256'] and path.stat().st_size==receipt['bytes']
                    target=launcher.DEST/f'verified_checkpoint_{receipt["step"]}.zip';launcher.DEST.mkdir(exist_ok=True)
                    if not target.exists():shutil.copy2(path,target)
                    assert launcher.sha(target)==receipt['sha256']
                    write_json(launcher.DEST/'intermediate_backup_receipt.json',dict(valid=True,runtime_independent=True,local=str(target),**receipt))
                except Exception:pass # Running versions may not expose output yet; never claim a backup without verification.
                stopped.wait(90)
        if len(sys.argv)>1 and sys.argv[1]=='supervise':threading.Thread(target=checkpoint_reader,daemon=True).start()
        try:launcher.main()
        finally:stopped.set()
