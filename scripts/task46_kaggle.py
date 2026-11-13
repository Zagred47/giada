"""Task46 console dispatch: reuse only three existing private input datasets."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task45d_kaggle as previous
from research_memory.mirror import write_json
k=previous.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task46_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task46_kaggle_external';k.SLUG='alessandrobelli/giada-task46-spatial-interventions';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task46_final\.zip|process\.log)$';k.STATUS_LABEL='46'
k.LAUNCH_SCOPE=dict(exploratory=True,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,frozen_candidate='45e/P',task45e_passed=False)
def plans():
    return [json.loads((ROOT/name).read_text()) for name in ('experiments/task41n_recovery_v1.json','experiments/task45c_cost_transport_v2.json','experiments/task45d_evaluation_recovery_v1.json')]
def metadata(gpu,slug,title):
    return dict(id=slug,title=title,code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=gpu,enable_internet=gpu,dataset_sources=[k.DATASET,plans()[1]['dataset'],plans()[2]['dataset']],kernel_sources=[],competition_sources=[],model_sources=[])
def notebook(revision,sha):
    doc=previous.original(revision,sha);c1,nd=plans()[1:]
    for cell in doc['cells']:
        if cell['cell_type']!='code':continue
        lines=[]
        for line in cell['source']:
            line=line.replace('giada_task41n_recovery','giada_task46').replace('run_task41n_recovery.py','run_task46.py').replace('task41n_recovery_final','task46_final').replace('GIADA41n recovery','GIADA46').replace('threading.Timer(5400','threading.Timer(10800').replace('str(sources[0]),str(out)','str(sources[0]),str(checkpoints[0]),str(saved[0]),str(out)')
            lines.append(line)
            if "--index-url" in line:
                lines.extend([" setup([sys.executable,'-m','pip','install','--quiet','uv==0.9.5'])\n"," native_env=WORK/'native_py312'\n"," setup([sys.executable,'-m','uv','venv','--python','3.12.11',str(native_env)])\n"," native_python=native_env/'bin/python'\n"," setup([sys.executable,'-m','uv','pip','install','--python',str(native_python),'numpy==1.26.4','neuron==8.2.7'])\n"," teacher=WORK/'neuron_as_deep_net'\n"," setup(['git','clone','--quiet','--depth','1','https://github.com/SelfishGene/neuron_as_deep_net.git',str(teacher)])\n"," setup(['git','-C',str(teacher),'fetch','--quiet','--depth','1','origin','074c4666300a8ad246601dab179a97a6942f0f29'])\n"," setup(['git','-C',str(teacher),'checkout','--quiet','--detach','FETCH_HEAD'])\n"])
            if line.startswith(' with sources[0].open'):
                lines.extend([" checkpoints=list(Path('/kaggle/input').rglob('checkpoint45c.zip.bin'));saved=list(Path('/kaggle/input').rglob('evaluation45d.zip.bin'));assert len(checkpoints)==len(saved)==1\n",f" with checkpoints[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={c1['sha256']!r}\n",f" with saved[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={nd['sha256']!r}\n"])
            if line.startswith(' env='):lines.append(" env['GIADA_NATIVE_PYTHON']=str(native_python)\n")
        cell['source']=lines
    doc['cells'][0]['source']=['# GIADA Task46 — spatial interventions and cost frontier\nFrozen45eP reconstructed bitwise from existing45c/45d inputs; no new upload, training or selection.45e remains incomplete. Paired native/formula/C1/P/41r, proximal/distal inputs, sodium/Ih block, uniform/graded densities, disconnected/coupled, imposed/autonomous. Native andTorch isolated. Same quality thresholds and level-one abstraction; no47authorization.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task46_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision,plans()[0]['source_sha256']));write_json(folder/'kernel-metadata.json',metadata(True,k.SLUG,'GIADA Task46 spatial interventions'));return folder
k.bundle=bundle
def mount():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();F.mkdir(exist_ok=True);folder=ROOT/'artifacts/task46_input_preflight';folder.mkdir(exist_ok=True)
    checks=[('source41n.zip.bin',plans()[0]['source_sha256'])]+[(q['filename'],q['sha256']) for q in plans()[1:]];marker=F/'mount_launch.json';slug='alessandrobelli/giada-task46-input-preflight'
    if not marker.exists():
        assert not (F/'mount_pending.json').exists(),'Inspect uncertain CPU submission'
        code="import hashlib,json\nfrom pathlib import Path\nchecks="+repr(checks)+"\nfor name,digest in checks:\n paths=list(Path('/kaggle/input').rglob(name));assert len(paths)==1,(name,len(paths))\n with paths[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest\nPath('/kaggle/working/mounted46.json').write_text(json.dumps(dict(valid=True,GPU_used=False,new_payload_upload=False,checks=checks)))\nprint('Task46 existing input hashes verified; no payload upload')\n"
        write_json(folder/'notebook.ipynb',dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='code',id='verify46',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))]));write_json(folder/'kernel-metadata.json',metadata(False,slug,'GIADA Task46 input preflight'))
        write_json(F/'mount_pending.json',dict(CPU_only=True,no_payload_upload=True));r=api.kernels_push(str(folder),timeout='1800');assert not r.error,r.error
        write_json(marker,dict(ref=r.ref,version=r.version_number));(F/'mount_pending.json').unlink()
    launch=json.loads(marker.read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    for _ in range(100):
        state=str(api.kernels_status(ref).status).lower();print('[46input] '+state,flush=True)
        if 'complete' in state:break
        if any(v in state for v in ('error','fail','cancel')):raise RuntimeError('CPU mount failed; noGPUlaunch')
        time.sleep(15)
    else:raise RuntimeError('CPU verification window ended; inspect existing run, no duplicate')
    dest=ROOT/'artifacts/task46_input_readback';dest.mkdir(exist_ok=True);api.kernels_output(ref,str(dest),file_pattern=r'mounted46\.json$',force=True,quiet=True)
    proof=json.loads((dest/'mounted46.json').read_text());assert proof['valid'];write_json(F/'input_verified.json',proof)
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='mount':mount()
    elif sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task46_spatial_interventions.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',plans()[0]['source_sha256']))
    else:
        if sys.argv[1]=='launch':
            assert json.loads((F/'native_environment_verified.json').read_text())['valid'],'Native environment must pass onCPU first'
            assert json.loads((F/'local_preflight.json').read_text())['valid']
            assert json.loads((F/'reconstruction_receipt.json').read_text())['valid']
        k.main()
