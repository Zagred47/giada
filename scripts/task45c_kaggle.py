"""Verified private fixture transport, CPU mount preflight and one GPU run."""
import json,sys,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as launcher
from research_memory.mirror import write_json
ROOT=launcher.ROOT;FOLDER=ROOT/'experiments/results/task45c_protocol_v1'
launcher.FOLDER=FOLDER;launcher.DEST=ROOT/'artifacts/task45c_kaggle_external';launcher.SLUG='alessandrobelli/giada-task45c-preserving-kinetics'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task45c_final\.zip|process\.log)$';launcher.STATUS_LABEL='GIADA45c'
launcher.LAUNCH_SCOPE=dict(exploratory=True,baseline='45b/f011',thresholds_changed=False,abstraction_changed=False,refinement_checkpoints=[1,4])
base_notebook=launcher.notebook

def plan():return json.loads((ROOT/'experiments/task45c_transport_v1.json').read_text())
def notebook(revision,source_sha):
    doc=base_notebook(revision,source_sha);p=plan()
    for cell in doc['cells']:
        src=[]
        for line in cell['source']:
            line=line.replace('run_task41n_recovery.py','run_task45c.py').replace('task41n_recovery_final','task45c_final').replace('giada_task41n_recovery','giada_task45c').replace('GIADA41n recovery','GIADA45c')
            line=line.replace('str(sources[0]),str(out)',"str(sources[0]),str(fixtures[0]),str(out)")
            line=line.replace('training_updates_performed=0,selection_recomputed=False',"refinement_checkpoint=max([4 if list(out.glob('*_4.pt')) else 0,1 if list(out.glob('*_1.pt')) else 0]),selection_recomputed=(out/'nomination_freeze.json').exists()")
            src.append(line)
            if line.startswith(" with sources[0].open('rb')"):
                src.extend([" fixtures=list(Path('/kaggle/input').rglob('task45b_final.zip'));assert len(fixtures)==1\n",f" with fixtures[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={p['source45b_sha256']!r}\n"])
        cell['source']=src
    doc['cells'][0]['source']=['# GIADA Task45c — preserving kinetic repair\n','Frozen f011; Nap2x2 preservation, Ih crossing, Ca_LVAst.h tail and NaTs2_t.h negative attribution control; DEV-only freeze and coupled/stress regression audits. Same thresholds,18states, physics and abstraction. Reused native fixtures, no NEURON load.\n']
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task45c_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task45c preserving kinetics',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=[plan()['kernel_source']],model_sources=[]))
    return folder
launcher.bundle=bundle

def publish():
    raise RuntimeError('Local payload upload was denied. Do not retry or export indirectly; use existing Kaggle45b output read-only.')

def mount_preflight():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();p=plan();source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    folder=ROOT/'artifacts/task45c_mount_preflight';folder.mkdir(exist_ok=True);marker=FOLDER/'mount_preflight_launch.json'
    code="""import hashlib,json,zipfile
from pathlib import Path
p=%r
sources=list(Path('/kaggle/input').rglob('source41n.zip.bin'));assert len(sources)==1
with sources[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==%r
files=list(Path('/kaggle/input').rglob('task45b_final.zip'));assert len(files)==1
with files[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==p['source45b_sha256']
with zipfile.ZipFile(files[0]) as z:
 assert set(p['members'])<=set(z.namelist())
 for name,digest in p['members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
Path('/kaggle/working/input_mount_preflight.json').write_text(json.dumps(dict(valid=True,sha256=p['source45b_sha256'],bytes=files[0].stat().st_size,members=len(p['members']),GPU_used=False,local_payload_uploaded=False)))
print('Immutable mounted inputs verified; no GPU used')
"""%(p,source['source_sha256'])
    if not marker.exists():
        assert not (FOLDER/'mount_submission_pending.json').exists(),'Inspect uncertain CPU preflight; never duplicate'
        write_json(folder/'notebook.ipynb',dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='code',id='mount45c',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))]))
        slug='alessandrobelli/giada-task45c-input-preflight'
        write_json(folder/'kernel-metadata.json',dict(id=slug,title='GIADA Task45c input preflight',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,dataset_sources=[launcher.DATASET],competition_sources=[],kernel_sources=[p['kernel_source']],model_sources=[]))
        write_json(FOLDER/'mount_submission_pending.json',dict(phase='CPU only'))
        result=api.kernels_push(str(folder),timeout='1800');assert not result.error,result.error
        write_json(marker,dict(ref=result.ref,version=result.version_number));(FOLDER/'mount_submission_pending.json').unlink()
    launch=json.loads(marker.read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    for poll in range(80):
        state=api.kernels_status(ref);status=str(state.status).split('.')[-1].lower();print('[GIADA45c CPU mount] '+status,flush=True)
        if 'complete' in status:break
        if any(word in status for word in ('error','fail','cancel')):raise RuntimeError('CPU mount preflight failed; inspect existing run, no GPU started')
        time.sleep(15)
    else:raise RuntimeError('CPU preflight monitoring window ended; inspect existing run')
    dest=ROOT/'artifacts/task45c_mount_readback';dest.mkdir(exist_ok=True)
    api.kernels_output(ref,str(dest),file_pattern=r'input_mount_preflight\.json$',force=True,quiet=True)
    proof=json.loads((dest/'input_mount_preflight.json').read_text());assert proof['valid'] and proof['sha256']==p['source45b_sha256'] and proof['bytes']==p['source45b_bytes']
    write_json(FOLDER/'input_verified.json',dict(valid=True,private=True,kernel_source=p['kernel_source'],mounted_preflight=proof,source41n_sha256=source['source_sha256']))

def supervise():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();launch=json.loads((FOLDER/'kaggle_launch.json').read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    for poll in range(720):
        try:state=api.kernels_status(ref);name=str(state.status).split('.')[-1].lower()
        except Exception as error:print('Read-only status retry '+type(error).__name__,flush=True);time.sleep(30);continue
        write_json(FOLDER/'status.json',dict(status=name,observed_at=time.time(),launch=launch,failure_message=state.failure_message));print('[GIADA45c] '+name,flush=True)
        if any(word in name for word in ('complete','error','fail','cancel')):sys.argv=[__file__,'retrieve'];launcher.main();return
        time.sleep(30)
    raise RuntimeError('Monitoring ended; inspect existing run, never auto-restart')

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True)
    mode=sys.argv[1]
    if mode=='build':
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());write_json(ROOT/'notebooks/giada_task45c_preserving_kinetics.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
    elif mode=='publish-fixtures':publish()
    elif mode=='mount-preflight':mount_preflight()
    elif mode=='supervise':supervise()
    else:launcher.main()
