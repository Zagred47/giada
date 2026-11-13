"""One45e eval-only run, reusing already persisted Kaggle output. No uploads."""
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task45d_evaluation_kaggle as previous
from src.giada_teacher.task45d_learnability_execution import remap
from research_memory.mirror import write_json
k=previous.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task45e_evaluation_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task45e_evaluation_external';k.SLUG='alessandrobelli/giada-task45e-evaluation-completion';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task45e_eval_final\.zip|process\.log)$';k.STATUS_LABEL='45e evaluation recovery'
k.LAUNCH_SCOPE=dict(evaluation_only=True,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False)
def plan():return json.loads((ROOT/'experiments/task45e_evaluation_recovery_v1.json').read_text())
def notebook(revision,sha):
    doc=remap(previous.notebook,plan=plan)(revision,sha)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[line.replace('run_task45d_evaluation_recovery.py','run_task45e_evaluation_recovery.py').replace('task45d_eval_final','task45e_eval_final').replace('giada_task45d_eval_recovery','giada_task45e_eval_recovery').replace('evaluation45d.zip.bin',plan()['filename']) for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA45e — missing evaluations only\n10 completed blocks reused,8 missing blocks evaluated from saved frozen checkpoints. Training, selection, thresholds and abstraction unchanged. Original full archive preserved; no native regeneration.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task45e_eval_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision,previous.old.plans()[0]['source_sha256']))
    meta=previous.old.metadata(folder,True,k.SLUG,'GIADA Task45e evaluation completion');meta['kernel_sources'].append(plan()['kernel_source']);write_json(folder/'kernel-metadata.json',meta);return folder
k.bundle=bundle
def mount():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();p=plan();assert p['transport']=='existing_private_kernel_output' and not p['new_upload']
    folder=ROOT/'artifacts/task45e_eval_mount';folder.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
    source,native,ck=previous.old.plans();checks=[('source41n.zip.bin',source['source_sha256']),('task45b_final.zip',native['source45b_sha256']),('checkpoint45c.zip.bin',ck['sha256']),(p['filename'],p['sha256'])]
    marker=F/'mount_launch.json';slug='alessandrobelli/giada-task45e-recovery-input-preflight'
    if not marker.exists():
        assert not (F/'mount_pending.json').exists(),'Inspect uncertain CPU submission'
        code="import json,hashlib\nfrom pathlib import Path\nchecks="+repr(checks)+"\nfor name,digest in checks:\n paths=list(Path('/kaggle/input').rglob(name));assert len(paths)==1,(name,len(paths))\n with paths[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest\nPath('/kaggle/working/mounted45e_eval.json').write_text(json.dumps(dict(valid=True,new_upload=False,GPU_used=False,checks=checks)))\nprint('All4 existing Kaggle sources verified; no checkpoint upload')\n"
        write_json(folder/'notebook.ipynb',dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='code',id='mount45e',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))]))
        meta=previous.old.metadata(folder,False,slug,'GIADA Task45e recovery input preflight');meta['kernel_sources'].append(p['kernel_source']);write_json(folder/'kernel-metadata.json',meta)
        write_json(F/'mount_pending.json',dict(CPU_only=True,no_payload_upload=True));r=api.kernels_push(str(folder),timeout='1800');assert not r.error,r.error
        write_json(marker,dict(ref=r.ref,version=r.version_number));(F/'mount_pending.json').unlink()
    launch=json.loads(marker.read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    for _ in range(100):
        state=str(api.kernels_status(ref).status).lower();print('[45e input preflight] '+state,flush=True)
        if 'complete' in state:break
        if any(q in state for q in ('error','fail','cancel')):raise RuntimeError('Existing-output CPU mount failed; no GPU launched')
        time.sleep(15)
    else:raise RuntimeError('CPU preflight deadline; inspect existing launch, never duplicate')
    dest=ROOT/'artifacts/task45e_eval_mount_readback';dest.mkdir(exist_ok=True);api.kernels_output(ref,str(dest),file_pattern=r'mounted45e_eval\.json$',force=True,quiet=True)
    proof=json.loads((dest/'mounted45e_eval.json').read_text());assert proof['valid'] and not proof['new_upload'];write_json(F/'input_verified.json',proof)
    write_json(ROOT/'notebooks/giada_task45e_evaluation_recovery.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',previous.old.plans()[0]['source_sha256']))
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='mount':mount()
    elif sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task45e_evaluation_recovery.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',previous.old.plans()[0]['source_sha256']))
    else:k.main()
