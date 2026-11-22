import hashlib,json,sys,time,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'_kaggle_cli_deps'))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task52f_resume_v1';DEST=ROOT/'artifacts/task52f_resume_input';DATASET='alessandrobelli/giada-task52f-resume600-input'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def build():
    spec=json.loads((ROOT/'experiments/task52f_resume_v1.json').read_text());source=ROOT/'artifacts/task52f_kaggle_external/task52f_final.zip';assert sha(source)==spec['source_sha256'];F.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
    target=DEST/'resume52f.zip.bin';shutil.copyfile(source,target);write_json(DEST/'dataset-metadata.json',dict(id=DATASET,title='GIADA52f Resume600',licenses=[dict(name='other')],description='Private verified600optimizer checkpoint andsaved history.'))
    proof=dict(valid=True,private=True,sha256=sha(target),bytes=target.stat().st_size,filename=target.name,dataset=DATASET);write_json(F/'recovery_package.json',proof);return proof
def publish():
    from kaggle.api.kaggle_api_extended import KaggleApi
    proof=build();api=KaggleApi();api.authenticate()
    if not(F/'input_created.json').exists():
        assert not(F/'input_pending.json').exists();write_json(F/'input_pending.json',proof);api.dataset_create_new(str(DEST),public=False,quiet=True,convert_to_csv=False);write_json(F/'input_created.json',proof);(F/'input_pending.json').unlink()
    target=ROOT/'artifacts/task52f_resume_readback';target.mkdir(exist_ok=True)
    for attempt in range(30):
        try:
            api.dataset_download_file(DATASET,proof['filename'],path=str(target),force=True,quiet=True);assert sha(target/proof['filename'])==proof['sha256'];write_json(F/'recovery_input_verified.json',proof);print('52f recovery input hash verified',flush=True);return
        except Exception:
            if attempt==29:raise
            time.sleep(10)
if __name__=='__main__':publish()
