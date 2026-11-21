import hashlib,json,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'_kaggle_cli_deps'))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task52e_protocol_v1';DEST=ROOT/'artifacts/task52e_input';DATASET='alessandrobelli/giada-task52e-frozen-inputs'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def build():
    F.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);p=json.loads((ROOT/'experiments/task52e_preregistration_v1.json').read_text());source=ROOT/'artifacts/task52d_kaggle_external/task52d_final.zip';assert sha(source)==p['source52d_sha256'];target=DEST/'models52e.zip.bin'
    with zipfile.ZipFile(ROOT/'artifacts/task52d_input/models52d.zip.bin') as old,zipfile.ZipFile(source) as c,zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
        for name in old.namelist():z.writestr(zipfile.ZipInfo(name,(2020,1,1,0,0,0)),old.read(name))
        for name in ('selected.pt','calibration.pt','preregistration.json'):z.writestr(zipfile.ZipInfo('source52d/'+name,(2020,1,1,0,0,0)),c.read(name))
    proof=dict(valid=True,dataset=DATASET,private=True,filename=target.name,sha256=sha(target),bytes=target.stat().st_size,source52d_sha256=p['source52d_sha256']);write_json(F/'package.json',proof);write_json(DEST/'dataset-metadata.json',dict(id=DATASET,title='GIADA Task52e Frozen Inputs',licenses=[dict(name='other')],description='Private52d trajectory starting weights and original frozen controls.'));return proof
def publish():
    from kaggle.api.kaggle_api_extended import KaggleApi
    proof=build();api=KaggleApi();api.authenticate();marker=F/'input_created.json'
    if not marker.exists():
        assert not(F/'input_pending.json').exists(),'Inspect pending upload before retry';write_json(F/'input_pending.json',proof);api.dataset_create_new(str(DEST),public=False,quiet=True,convert_to_csv=False);write_json(marker,proof);(F/'input_pending.json').unlink()
    dest=ROOT/'artifacts/task52e_input_readback';dest.mkdir(exist_ok=True)
    for attempt in range(30):
        try:
            api.dataset_download_file(DATASET,proof['filename'],path=str(dest),force=True,quiet=True);assert sha(dest/proof['filename'])==proof['sha256'];write_json(F/'frozen_input_verified.json',proof);print('52e input verified',flush=True);return
        except Exception:
            if attempt==29:raise
            time.sleep(10)
if __name__=='__main__':publish() if sys.argv[1]=='publish' else print(build())
