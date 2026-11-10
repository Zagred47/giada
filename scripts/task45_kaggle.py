"""One pinned45 console run; frozen41r, no optional refit executed."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task44_kaggle as previous
from research_memory.mirror import write_json
launcher=previous.launcher;ROOT=launcher.ROOT
launcher.FOLDER=ROOT/'experiments/results/task45_protocol_v1';launcher.DEST=ROOT/'artifacts/task45_kaggle_external'
launcher.SLUG='alessandrobelli/giada-task45-controlled-coverage'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task45_final\.zip|process\.log)$';launcher.STATUS_LABEL='GIADA45'
launcher.LAUNCH_SCOPE=dict(training_updates_performed=0,selection_recomputed=False,exploratory=True,frozen41r=True,thresholds_changed=False,conditional_refit_performed=False)

def notebook(revision,source_sha):
    doc=previous.notebook(revision,source_sha)
    for cell in doc['cells']:
        cell['source']=[s.replace('run_task44.py','run_task45.py').replace('task44_final','task45_final').replace('giada_task44','giada_task45').replace('GIADA44','GIADA45') for s in cell['source']]
    doc['cells'][0]['source']=['# GIADA Task45 — controlled coverage\n','Frozen41r versus matched formula controls; prospective1s histories, anchor/new diagnostic geometries,100/500/1000ms prefixes. Original thresholds. No optimizer/selection: optional roadmap refit deferred pending evidence and explicit decision.\n']
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task45_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task45 controlled coverage',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=[],model_sources=[]))
    return folder
launcher.bundle=bundle

def supervise(api=None):
    """Queue tolerance only: no extension of GPU runtime or scientific budget."""
    if api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi()
    api.authenticate()
    launch=json.loads((launcher.FOLDER/'kaggle_launch.json').read_text())
    ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    for poll in range(720):
        try:
            state=api.kernels_status(ref);name=str(state.status).split('.')[-1].lower()
        except Exception as error:
            print('Read-only status retry '+type(error).__name__+'; run untouched',flush=True);time.sleep(30);continue
        write_json(launcher.FOLDER/'status.json',dict(status=name,observed_at=time.time(),launch=launch,failure_message=state.failure_message))
        print('[GIADA45] '+name,flush=True)
        if any(word in name for word in ('complete','error','fail','cancel')):
            sys.argv=[__file__,'retrieve'];launcher.main();return
        time.sleep(30)
    raise RuntimeError('Six-hour read-only monitoring window ended; inspect existing run, never auto-restart')

if __name__=='__main__':
    if sys.argv[1:]==['build']:
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task45_controlled_coverage.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
        launcher.FOLDER.mkdir(exist_ok=True)
        write_json(launcher.FOLDER/'input_verified.json',json.loads((ROOT/'experiments/results/task44_protocol_v1/input_verified.json').read_text()))
    elif sys.argv[1:]==['supervise']:supervise()
    else:launcher.main()
