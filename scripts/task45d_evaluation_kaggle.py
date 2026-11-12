"""Private compact evaluation-only transport with readback before one GPU run."""
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task45d_kaggle as old
from research_memory.mirror import write_json
k=old.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task45d_evaluation_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task45d_evaluation_external';k.SLUG='alessandrobelli/giada-task45d-evaluation-completion';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task45d_eval_final\.zip|process\.log)$';k.STATUS_LABEL='45d evaluation recovery'
k.LAUNCH_SCOPE=dict(evaluation_only=True,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False)
def plan():return json.loads((ROOT/'experiments/task45d_evaluation_recovery_v1.json').read_text())
def notebook(revision,sha):
    doc=old.notebook(revision,sha)
    for cell in doc['cells']:
        if cell['cell_type']!='code':continue
        lines=[]
        for line in cell['source']:
            line=line.replace('run_task45d.py','run_task45d_evaluation_recovery.py').replace('task45d_final','task45d_eval_final').replace('giada_task45d','giada_task45d_eval_recovery')
            line=line.replace('str(sources[0]),str(native[0]),str(checkpoints[0]),str(out)','str(sources[0]),str(native[0]),str(saved[0]),str(checkpoints[0]),str(out)')
            line=line.replace("training_performed=bool(list(out.glob('A*_*.pt'))),selection_recomputed=(out/'nomination_freeze.json').exists()",'training_performed=False,selection_recomputed=False,evaluation_only=True')
            lines.append(line)
            if line.startswith(' with checkpoints[0].open'):
                lines.extend([" saved=list(Path('/kaggle/input').rglob('evaluation45d.zip.bin'));assert len(saved)==1\n",f" with saved[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={plan()['sha256']!r}\n"])
        cell['source']=lines
    doc['cells'][0]['source']=['# GIADA45d — missing evaluations only\nSaved final checkpoints and byte-identical nomination;9 completed blocks reused. No training, selection, changed thresholds or native generation. Original trace archive retained separately.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task45d_eval_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision,old.plans()[0]['source_sha256']))
    meta=old.metadata(folder,True,k.SLUG,'GIADA Task45d evaluation completion');meta['dataset_sources'].append(plan()['dataset']);write_json(folder/'kernel-metadata.json',meta);return folder
k.bundle=bundle
def publish():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();p=plan();folder=ROOT/'artifacts/task45d_evaluation_input';F.mkdir(exist_ok=True)
    write_json(folder/'dataset-metadata.json',dict(id=p['dataset'],title='GIADA Task45d evaluation recovery',licenses=[dict(name='other')],description='Private saved learned checkpoints, unchanged selection and completed aggregate case metrics only. No native trajectories or database. Evaluation-only recovery.'))
    if not (F/'dataset_created.json').exists():
        assert not (F/'upload_pending.json').exists(),'Inspect uncertain upload; never duplicate'
        write_json(F/'upload_pending.json',dict(dataset=p['dataset']));api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False)
        write_json(F/'dataset_created.json',dict(dataset=p['dataset'],private=True));(F/'upload_pending.json').unlink()
    dest=ROOT/'artifacts/task45d_eval_readback';dest.mkdir(exist_ok=True)
    for attempt in range(20):
        try:api.dataset_download_file(p['dataset'],p['filename'],path=str(dest),force=True,quiet=True);break
        except Exception:
            if attempt==19:raise
            print('Waiting for private checkpoint input; read-only retry',flush=True);time.sleep(15)
    with (dest/p['filename']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==p['sha256']
    assert (dest/p['filename']).stat().st_size==p['bytes'];write_json(F/'input_verified.json',dict(valid=True,private=True,sha256=p['sha256'],bytes=p['bytes'],readback_verified=True,old3inputs_proof='experiments/results/task45d_protocol_v1/input_verified.json'))
    write_json(F/'preregistration_receipt.json',dict(valid=True,same_protocol=True,evaluation_only=True,training_performed=False,selection_recomputed=False,source_archive_sha256=p['source_archive_sha256']))
    write_json(ROOT/'notebooks/giada_task45d_evaluation_recovery.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',old.plans()[0]['source_sha256']))
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='publish':publish()
    else:k.main()
