"""41p private console dispatch with frozen41n mounted inputs."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as launcher
from research_memory.mirror import write_json
ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task41p_repair_v1'
launcher.DEST=ROOT/'artifacts/task41p_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task41p-targeted-tail-repair'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task41p_final\.zip|process\.log)$'
launcher.LAUNCH_SCOPE=dict(training_budget_per_arm=12000,development_selection=True,scientific_thresholds_changed=False)
original_notebook=launcher.notebook

def notebook(revision,source_sha):
    value=original_notebook(revision,source_sha)
    for cell in value['cells']:
        cell['id']=cell['id'].replace('41n','41p')
        cell['source']=[line.replace('run_task41n_recovery.py','run_task41p.py').replace('task41n_recovery_final','task41p_final').replace('GIADA41n recovery','GIADA41p').replace('giada_task41n_recovery','giada_task41p').replace('# GIADA41n — evaluation-only recovery','# GIADA41p — targeted tail repair').replace('No training/selection/native regeneration. Original failed FP32 floors retained; no reinstated primary confirmation.','Two paired repair arms, frozen baseline, same architecture and thresholds. New-development selection only; teacher labels offline. FP32 floor failures retained.') for line in cell['source']]
        # Runtime receipts must describe the actual training, not inherited recovery semantics.
        cell['source']=[line.replace('training_updates_performed=0,selection_recomputed=False','training_budget_per_arm=12000,development_selection=True') for line in cell['source']]
    return value

def bundle(revision):
    folder=ROOT/'artifacts/task41p_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task41p targeted tail repair',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=['alessandrobelli/giada-task41n-evaluation-only-recovery'],model_sources=[]))
    return folder
launcher.bundle=bundle
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='build':
        plan=json.loads((ROOT/'experiments/task41p_preregistration_v1.json').read_text());prior=json.loads((ROOT/'experiments/task41n_preregistration_v1.json').read_text())
        assert not set(plan['development_seeds'])&set(prior['native_seeds']+prior['stress_seeds'])
        plan['quality']=prior['quality'];write_json(ROOT/'experiments/task41p_preregistration_v1.json',plan)
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task41p_targeted_tail_repair.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        launcher.FOLDER.mkdir(exist_ok=True)
        receipt=json.loads((ROOT/'experiments/results/task41o_causal_v1/input_verified.json').read_text());write_json(launcher.FOLDER/'input_verified.json',receipt)
    else:launcher.main()
