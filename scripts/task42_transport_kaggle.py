"""Verified private binary input transport; fail before GPU worker if absent."""
import hashlib,json,shutil,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task42_recovery_kaggle as recovery
launcher=recovery.launcher;ROOT=launcher.ROOT
from research_memory.mirror import write_json
FOLDER=ROOT/'experiments/results/task42_transport_recovery_v1'
INPUT='alessandrobelli/giada-task42-native-recovery-input'
NAME='native42.zip.bin'
launcher.FOLDER=FOLDER;launcher.DEST=ROOT/'artifacts/task42_transport_recovery_external'
launcher.SLUG='alessandrobelli/giada-task42-verified-input-gpu-recovery'
launcher.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task42_transport_final\.zip|process\.log|input_transport_receipt\.json)$'
launcher.STATUS_LABEL='GIADA42 verified recovery'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_payload(path,plan):
    assert path.stat().st_size==plan['source_bytes'] and sha(path)==plan['source_sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,expected in plan['members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==expected,name

def publish():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    plan=json.loads((ROOT/'experiments/task42_gpu_recovery_v1.json').read_text())
    source=ROOT/'artifacts/task42_kaggle_external/task42_final.zip';verify_payload(source,plan)
    FOLDER.mkdir(exist_ok=True);upload=ROOT/'artifacts/task42_native_input_upload';upload.mkdir(exist_ok=True)
    readback=ROOT/'artifacts/task42_native_input_readback';readback.mkdir(exist_ok=True)
    if not (FOLDER/'input_created.json').exists():
        assert not (FOLDER/'input_creation_requested.json').exists(),'Uncertain input creation: inspect existing dataset, do not duplicate.'
        shutil.copy2(source,upload/NAME);verify_payload(upload/NAME,plan)
        write_json(upload/'dataset-metadata.json',dict(id=INPUT,title='GIADA Task42 native recovery input',licenses=[dict(name='other')],description='Private immutable original42 saved native traces and frozen checkpoint. Binary extension prevents archive filename dependence on extraction. No generated data, training or native replay.'))
        write_json(FOLDER/'input_creation_requested.json',dict(dataset=INPUT,sha256=plan['source_sha256'],private_requested=True))
        api.dataset_create_new(str(upload),public=False,quiet=True,convert_to_csv=False)
        write_json(FOLDER/'input_created.json',dict(dataset=INPUT,sha256=plan['source_sha256'],private_requested=True))
    for attempt in range(16):
        try:
            api.dataset_download_file(INPUT,NAME,path=str(readback),force=True,quiet=True)
            verify_payload(readback/NAME,plan)
            receipt=dict(valid=True,dataset=INPUT,filename=NAME,bytes=plan['source_bytes'],sha256=plan['source_sha256'],
                api_readback_verified=True,zip_crc_verified=True,member_hashes_verified=len(plan['members']),
                private_requested=True,GPU_launched=False,native_regenerated=False,thresholds_changed=False)
            write_json(FOLDER/'input_verified.json',receipt)
            print(json.dumps(receipt),flush=True);return
        except Exception:
            if attempt==15:raise
            time.sleep(10)

def notebook(revision,source_sha):
    doc=recovery.notebook(revision,source_sha)
    plan=json.loads((ROOT/'experiments/task42_gpu_recovery_v1.json').read_text())
    for cell in doc['cells']:
        cell['id']=cell['id'].replace('42recovery','42transport')
        cell['source']=[line.replace('giada_task42_gpu_recovery','giada_task42_transport_recovery').replace('task42_recovery_final','task42_transport_final') for line in cell['source']]
    lines=doc['cells'][1]['source'];point=next(i for i,line in enumerate(lines) if "pinned Torch/GPU setup" in line)
    lines[point:point]=[
        " native_candidates=sorted(Path('/kaggle/input').rglob('native42.zip.bin'))\n",
        f" native_verified=[p for p in native_candidates if p.is_file() and p.stat().st_size=={plan['source_bytes']} and hashlib.sha256(p.read_bytes()).hexdigest()=={plan['source_sha256']!r}]\n",
        " assert native_verified,'Saved42 input missing/hash mismatch: GPU worker not started; inspect mounted input dataset'\n",
        " native_archive=native_verified[0]\n",
        f" (public/'input_transport_receipt.json').write_text(json.dumps(dict(valid=True,mounted_path=str(native_archive),bytes=native_archive.stat().st_size,sha256={plan['source_sha256']!r},native_regenerated=False,worker_started=False)))\n",
        " print('[GIADA42 recovery] mounted saved archive SHA256 verified before GPU worker',flush=True)\n"]
    doc['cells'][1]['source']=[line.replace("MPLBACKEND='Agg')","MPLBACKEND='Agg',GIADA_TASK42_NATIVE_ARCHIVE=str(native_archive))") for line in lines]
    return doc

def probe():
    """Verify the actual Kaggle mount on CPU before requesting any GPU run."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    plan=json.loads((ROOT/'experiments/task42_gpu_recovery_v1.json').read_text())
    readback=json.loads((FOLDER/'input_verified.json').read_text());assert readback['api_readback_verified']
    folder=ROOT/'artifacts/task42_input_probe';folder.mkdir(exist_ok=True)
    slug='alessandrobelli/giada-task42-native-input-cpu-preflight'
    code='''import hashlib,json,zipfile
from pathlib import Path
plan=%r
files=sorted(Path('/kaggle/input').rglob('native42.zip.bin'))
valid=[]
for path in files:
 if path.stat().st_size==plan['source_bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==plan['source_sha256']:valid.append(path)
receipt=dict(valid=False,GPU_used=False,native_regenerated=False,candidate_count=len(files),input_files=[str(x) for x in files])
try:
 assert valid,'No saved42 archive with the required size/SHA256 was mounted'
 with zipfile.ZipFile(valid[0]) as z:
  assert z.testzip() is None
  for name,sha in plan['members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
 receipt.update(valid=True,mounted_path=str(valid[0]),sha256=plan['source_sha256'],bytes=plan['source_bytes'],member_hashes_verified=len(plan['members']),zip_crc_verified=True)
finally:
 Path('/kaggle/working/input_probe.json').write_text(json.dumps(receipt))
assert receipt['valid']
print('CPU input mount and all hashes verified; no GPU or native simulation.')
'''%plan
    write_json(folder/'notebook.ipynb',dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='code',id='verify42input',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))]))
    write_json(folder/'kernel-metadata.json',dict(id=slug,title='GIADA Task42 native input CPU preflight',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,dataset_sources=[INPUT],competition_sources=[],kernel_sources=[],model_sources=[]))
    marker=FOLDER/'cpu_probe_launch.json'
    if not marker.exists():
        result=api.kernels_push(str(folder),timeout='600');assert not result.error,result.error
        write_json(marker,dict(ref=result.ref,version=result.version_number,url=result.url))
    launch=json.loads(marker.read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    for attempt in range(90):
        state=api.kernels_status(ref);status=str(state.status).split('.')[-1].lower()
        if any(x in status for x in ('complete','error','fail','cancel')):break
        time.sleep(5)
    else:raise RuntimeError('CPU input probe still pending: no GPU launch allowed.')
    output=ROOT/'artifacts/task42_input_probe_readback';output.mkdir(exist_ok=True)
    api.kernels_output(ref,str(output),file_pattern=r'input_probe\.json$',force=True,quiet=True)
    result=json.loads((output/'input_probe.json').read_text())
    assert result['valid'] and result['GPU_used'] is False and result['sha256']==plan['source_sha256']
    write_json(FOLDER/'cpu_input_verified.json',dict(**result,probe_launch=launch,provider_status=status))
    print('Actual Kaggle CPU mount verified; GPU launch may proceed after preregistration.',flush=True)

def bundle(revision):
    receipt=json.loads((FOLDER/'input_verified.json').read_text());assert receipt['valid'] and receipt['api_readback_verified']
    cpu=json.loads((FOLDER/'cpu_input_verified.json').read_text());assert cpu['valid'] and cpu['GPU_used'] is False
    launcher.LAUNCH_SCOPE.update(input_api_readback_verified=True,input_cpu_mount_verified=True,input_sha256=receipt['sha256'],input_dataset=INPUT)
    folder=ROOT/'artifacts/task42_transport_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=launcher.SLUG,title='GIADA Task42 verified input GPU recovery',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[launcher.DATASET,INPUT],competition_sources=[],kernel_sources=[],model_sources=[]))
    return folder
launcher.bundle=bundle

if __name__=='__main__':
    if sys.argv[1:]==['publish']:publish()
    elif sys.argv[1:]==['probe']:probe()
    elif sys.argv[1:]==['build']:
        source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
        write_json(ROOT/'notebooks/giada_task42_verified_input_gpu_recovery.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',source['source_sha256']))
    else:launcher.main()
