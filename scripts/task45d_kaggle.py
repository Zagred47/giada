"""CPU mounted-input proof then one private45d T4 notebook, no payload upload."""
import json,sys,time,ast,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as k
from research_memory.mirror import write_json
ROOT=k.ROOT;F=ROOT/'experiments/results/task45d_protocol_v1';k.FOLDER=F;k.DEST=ROOT/'artifacts/task45d_kaggle_external';k.SLUG='alessandrobelli/giada-task45d-learnability-execution'
k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task45d_final\.zip|process\.log)$';k.STATUS_LABEL='45d'
k.LAUNCH_SCOPE=dict(exploratory=True,thresholds_changed=False,abstraction_changed=False,baseline='f011 and45cJ',training_performed=True)
original=k.notebook
def plans():return [json.loads((ROOT/name).read_text()) for name in ('experiments/task41n_recovery_v1.json','experiments/task45c_transport_v1.json','experiments/task45c_cost_transport_v2.json')]
def metadata(folder,gpu,slug,title):
    return dict(id=slug,title=title,code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=gpu,enable_internet=gpu,dataset_sources=[k.DATASET,plans()[2]['dataset']],kernel_sources=[plans()[1]['kernel_source']],competition_sources=[],model_sources=[])
def notebook(revision,source_sha):
    source,native,ck=plans();doc=original(revision,source_sha)
    for cell in doc['cells']:
        if cell['cell_type']!='code':continue
        lines=[]
        for line in cell['source']:
            line=line.replace('giada_task41n_recovery','giada_task45d').replace('run_task41n_recovery.py','run_task45d.py').replace('task41n_recovery_final','task45d_final').replace('threading.Timer(5400','threading.Timer(10800')
            line=line.replace('str(sources[0]),str(out)',"str(sources[0]),str(native[0]),str(checkpoints[0]),str(out)")
            line=line.replace('training_updates_performed=0,selection_recomputed=False',"training_performed=bool(list(out.glob('A*_*.pt'))),selection_recomputed=(out/'nomination_freeze.json').exists()")
            lines.append(line)
            if line.startswith(' with sources[0].open'):
                lines.extend([" native=list(Path('/kaggle/input').rglob('task45b_final.zip'));checkpoints=list(Path('/kaggle/input').rglob('checkpoint45c.zip.bin'));assert len(native)==len(checkpoints)==1\n",f" with native[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={native['source45b_sha256']!r}\n",f" with checkpoints[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={ck['sha256']!r}\n"])
        cell['source']=lines
    doc['cells'][0]['source']=['# GIADA Task45d — learnability, support and execution\nFrozen f011/45cJ controls; support×feature matrix; Nap budget/line-search; Ih timing pair. No altered thresholds,18states,precision or physics. No NEURON loading. CPU mounted-input proof beforeGPU.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task45d_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision,plans()[0]['source_sha256']));write_json(folder/'kernel-metadata.json',metadata(folder,True,k.SLUG,'GIADA Task45d learnability execution'));return folder
k.bundle=bundle
# Preserve the established submission/retrieval guards while aligning the
# provider deadline with the preregistered10800s worker budget.
tree=ast.parse(inspect.getsource(k.main))
for node in ast.walk(tree):
    if isinstance(node,ast.keyword) and node.arg=='timeout' and isinstance(node.value,ast.Constant) and node.value.value=='7200':node.value.value='12000'
    if isinstance(node,ast.Constant) and node.value==240:node.value=420
exec(compile(tree,'45d_console_dispatch','exec'),k.__dict__)
def mount():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();source,native,ck=plans();folder=ROOT/'artifacts/task45d_mount';folder.mkdir(exist_ok=True)
    slug='alessandrobelli/giada-task45d-input-preflight';marker=F/'mount_launch.json'
    if not marker.exists():
        assert not (F/'mount_pending.json').exists(),'Inspect uncertain CPU submission'
        code="""import json,hashlib,zipfile
from pathlib import Path
checks=%r
for filename,digest in checks:
 paths=list(Path('/kaggle/input').rglob(filename));assert len(paths)==1,(filename,len(paths))
 with paths[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
with zipfile.ZipFile(next(Path('/kaggle/input').rglob('checkpoint45c.zip.bin'))) as z:
 assert z.testzip() is None
 assert all(k+'.pt' in z.namelist() for k in ('C0','J','R41','R111'))
Path('/kaggle/working/mounted45d.json').write_text(json.dumps(dict(valid=True,GPU_used=False,checks=checks)))
print('All3 mounted45d sources verified')
"""%([('source41n.zip.bin',source['source_sha256']),('task45b_final.zip',native['source45b_sha256']),('checkpoint45c.zip.bin',ck['sha256'])],)
        write_json(folder/'notebook.ipynb',dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='code',id='verify45d',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))]))
        write_json(folder/'kernel-metadata.json',metadata(folder,False,slug,'GIADA Task45d input preflight'));write_json(F/'mount_pending.json',dict(CPU_only=True))
        r=api.kernels_push(str(folder),timeout='1800');assert not r.error,r.error;write_json(marker,dict(ref=r.ref,version=r.version_number));(F/'mount_pending.json').unlink()
    launch=json.loads(marker.read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    for _ in range(100):
        state=str(api.kernels_status(ref).status).lower();print('[45d CPU mount] '+state,flush=True)
        if 'complete' in state:break
        if any(s in state for s in ('error','fail','cancel')):raise RuntimeError('CPU mounted-input check failed; no GPU launch')
        time.sleep(15)
    else:raise RuntimeError('CPU monitoring ended, do not duplicate')
    dest=ROOT/'artifacts/task45d_mount_readback';dest.mkdir(exist_ok=True);api.kernels_output(ref,str(dest),file_pattern=r'mounted45d\.json$',force=True,quiet=True)
    proof=json.loads((dest/'mounted45d.json').read_text());assert proof['valid'];write_json(F/'input_verified.json',proof)
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='mount':mount()
    elif sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task45d_learnability_execution.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',plans()[0]['source_sha256']))
    else:k.main()
