"""Single private42 console submission using the verified output retrieval path."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as launcher
from research_memory.mirror import write_json
ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task42_protocol_v1'
launcher.DEST=ROOT/'artifacts/task42_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task42-exploratory-axial-port'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task42_final\.zip|process\.log)$'
launcher.STATUS_LABEL='GIADA42'
launcher.LAUNCH_SCOPE=dict(training_updates_performed=0,selection_recomputed=False,exploratory=True,frozen41r=True,thresholds_changed=False)
original=launcher.notebook

def notebook(revision,source_sha):
    doc=original(revision,source_sha)
    for cell in doc['cells']:
        cell['id']=cell['id'].replace('41n','42')
        cell['source']=[line.replace('run_task41n_recovery.py','run_task42.py').replace('task41n_recovery_final','task42_final').replace('GIADA41n recovery','GIADA42').replace('giada_task41n_recovery','giada_task42').replace('# GIADA41n — evaluation-only recovery','# GIADA42 — exploratory axial port').replace('No training/selection/native regeneration. Original failed FP32 floors retained; no reinstated primary confirmation.','Frozen41r; paired native/formula/candidate references on chains2/4/8/16 and bifurcation5. No training or selection. Original thresholds and41 limitations retained. New native floors must pass before model judgment.') for line in cell['source']]
    lines=doc['cells'][1]['source'];point=next(i for i,line in enumerate(lines) if line.startswith(' sources='))
    lines[point:point]=[
        " setup([sys.executable,'-m','pip','install','--quiet','neuron==8.2.7'])\n",
        " teacher=WORK/'neuron_as_deep_net'\n",
        " setup(['git','clone','--quiet','--depth','1','https://github.com/SelfishGene/neuron_as_deep_net.git',str(teacher)])\n",
        " setup(['git','-C',str(teacher),'fetch','--quiet','--depth','1','origin','074c4666300a8ad246601dab179a97a6942f0f29'])\n",
        " setup(['git','-C',str(teacher),'checkout','--quiet','--detach','FETCH_HEAD'])\n"]
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task42_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task42 exploratory axial port',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=[],model_sources=[]))
    return folder
launcher.bundle=bundle

if __name__=='__main__':
    if sys.argv[1:] == ['build']:
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task42_exploratory_axial_port.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        receipt=json.loads((ROOT/'experiments/results/task41v_protocol_v1/input_verified.json').read_text())
        launcher.FOLDER.mkdir(exist_ok=True);write_json(launcher.FOLDER/'input_verified.json',receipt)
    else:launcher.main()
