"""Private41o console dispatch; reuse verified41n inputs and persisted output."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as launcher
from research_memory.mirror import write_json
ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task41o_causal_v1'
launcher.DEST=ROOT/'artifacts/task41o_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task41o-causal-localization'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task41o_final\.zip|process\.log)$'
old_notebook=launcher.notebook

def notebook(revision,source_sha):
    value=old_notebook(revision,source_sha)
    for cell in value['cells']:
        cell['id']=cell['id'].replace('41n','41o')
        cell['source']=[line.replace('run_task41n_recovery.py','run_task41o.py').replace('task41n_recovery_final','task41o_final').replace('GIADA41n recovery','GIADA41o').replace('giada_task41n_recovery','giada_task41o').replace('# GIADA41n — evaluation-only recovery','# GIADA41o — causal localization').replace('No training/selection/native regeneration. Original failed FP32 floors retained; no reinstated primary confirmation.','Frozen causal controls, not a released repair. No training or selection; original FP32 failures retained.') for line in cell['source']]
    return value

def bundle(revision):
    folder=ROOT/'artifacts/task41o_notebook';folder.mkdir(exist_ok=True)
    plan=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,plan['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task41o causal localization',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=['alessandrobelli/giada-task41n-evaluation-only-recovery'],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='build':
        plan=json.loads((ROOT/'experiments/task41o_preregistration_v1.json').read_text())
        plan['quality']=json.loads((ROOT/'experiments/task41n_preregistration_v1.json').read_text())['quality']
        write_json(ROOT/'experiments/task41o_preregistration_v1.json',plan)
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task41o_causal_localization.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        receipt=json.loads((ROOT/'experiments/results/task41n_recovery_v1/input_verified.json').read_text())
        receipt.update(persisted_kernel_source='alessandrobelli/giada-task41n-evaluation-only-recovery/2',recovery_sha256=plan['recovery_sha256'],local_recovery_crc_verified=True)
        write_json(launcher.FOLDER/'input_verified.json',receipt)
    else:launcher.main()
