"""Private checkpoint-only Dataset with readback hash before any GPU launch."""
import sys,json,hashlib,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task45c_cost_recovery_kaggle as old
k=old.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task45c_cost_recovery_v2'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task45c_cost_recovery_v2_external';k.SLUG='alessandrobelli/giada45c-cost-recovery-verified-input'
def plan():return json.loads((ROOT/'experiments/task45c_cost_transport_v2.json').read_text())
def notebook(revision,sha):
    doc=old.notebook(revision,sha)
    for cell in doc['cells']:
        if cell['cell_type']=='code':cell['source']=[s.replace("rglob('task45c_final.zip')","rglob('checkpoint45c.zip.bin')") for s in cell['source']]
    return doc
def bundle(revision):
    from research_memory.mirror import write_json
    d=ROOT/'artifacts/task45c_cost_v2_notebook';d.mkdir(exist_ok=True)
    source=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(d/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(d/'kernel-metadata.json',dict(id=k.SLUG,title='GIADA45c cost recovery verified input',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[k.DATASET,plan()['dataset']],kernel_sources=[],competition_sources=[],model_sources=[]))
    return d
k.bundle=bundle
def publish():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from research_memory.mirror import write_json
    api=KaggleApi();api.authenticate();p=plan();folder=ROOT/'artifacts/task45c_cost_input';F.mkdir(exist_ok=True)
    write_json(folder/'dataset-metadata.json',dict(id=p['dataset'],title='GIADA Task45c cost checkpoints',licenses=[dict(name='other')],description='Private frozen model weights and aggregate metrics only; no native traces, no database. Evaluation-only transport recovery.'))
    marker=F/'dataset_created.json'
    if not marker.exists():
        assert not (F/'submission_pending_input.json').exists(),'Inspect uncertain publication; do not duplicate'
        write_json(F/'submission_pending_input.json',dict(dataset=p['dataset']))
        api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False)
        write_json(marker,dict(dataset=p['dataset'],private=True));(F/'submission_pending_input.json').unlink()
    dest=ROOT/'artifacts/task45c_cost_readback';dest.mkdir(exist_ok=True)
    for attempt in range(20):
        try:
            api.dataset_download_file(p['dataset'],p['filename'],path=str(dest),force=True,quiet=True)
            break
        except Exception as error:
            if attempt==19:raise
            print('Waiting for private Dataset availability; read-only retry '+type(error).__name__,flush=True)
            time.sleep(15)
    with (dest/p['filename']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==p['sha256']
    assert (dest/p['filename']).stat().st_size==p['bytes']
    receipt=dict(valid=True,private=True,sha256=p['sha256'],bytes=p['bytes'],download_readback_verified=True,native_traces_uploaded=False,DB_uploaded=False,training_performed=False,selection_recomputed=False,thresholds_changed=False)
    write_json(F/'input_verified.json',receipt);write_json(F/'preregistration_receipt.json',receipt)
    write_json(ROOT/'notebooks/giada_task45c_cost_recovery_v2.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text())['source_sha256']))
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='publish':publish()
    else:k.main()
