"""Single GPU-only recovery on Kaggle using persisted42 output as kernel source."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as launcher
from research_memory.mirror import write_json
ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task42_recovery_v1';launcher.DEST=ROOT/'artifacts/task42_recovery_external'
launcher.SLUG='alessandrobelli/giada-task42-gpu-evaluation-recovery'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task42_recovery_final\.zip|process\.log)$'
launcher.LAUNCH_SCOPE=dict(training_updates_performed=0,selection_recomputed=False,native_regenerated=False,thresholds_changed=False)
launcher.STATUS_LABEL='GIADA42 recovery'
original=launcher.notebook

def notebook(revision,source_sha):
    doc=original(revision,source_sha)
    for cell in doc['cells']:
        cell['id']=cell['id'].replace('41n','42recovery')
        cell['source']=[line.replace('run_task41n_recovery.py','run_task42_gpu_recovery.py').replace('task41n_recovery_final','task42_recovery_final').replace('GIADA41n recovery','GIADA42 recovery').replace('giada_task41n_recovery','giada_task42_gpu_recovery').replace('# GIADA41n — evaluation-only recovery','# GIADA42 — GPU-only recovery').replace('No training/selection/native regeneration. Original failed FP32 floors retained; no reinstated primary confirmation.','Restore saved60 native cases and15 passive checks. Fix seed-axis adapter; complete original GPU evaluations only. No native phase, teacher checkout or NEURON installation.41r and original thresholds unchanged.') for line in cell['source']]
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task42_recovery_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task42 GPU evaluation recovery',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=['alessandrobelli/giada-task42-exploratory-axial-port'],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if sys.argv[1:]==['build']:
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task42_gpu_recovery.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        launcher.FOLDER.mkdir(exist_ok=True)
        write_json(launcher.FOLDER/'input_verified.json',dict(valid=True,source_archive_sha256=json.loads((ROOT/'experiments/task42_gpu_recovery_v1.json').read_text())['source_sha256'],input_role='Saved native traces and frozen41r; whole archive/member checks mandatory in GPU worker.'))
    else:launcher.main()
