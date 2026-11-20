import hashlib,json,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'_kaggle_cli_deps'))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task52c_protocol_v1';DEST=ROOT/'artifacts/task52c_input';DATASET='alessandrobelli/giada-task52c-frozen-inputs'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def build():
    F.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);p=json.loads((ROOT/'experiments/task52c_preregistration_v1.json').read_text());source=ROOT/'artifacts/task52b_kaggle_external/task52b_final.zip'
    assert sha(source)==p['source52b_sha256'];target=DEST/'models52c.zip.bin'
    with zipfile.ZipFile(ROOT/'artifacts/task52_input/models52.zip.bin') as old,zipfile.ZipFile(source) as b,zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
        for name in old.namelist():z.writestr(zipfile.ZipInfo(name,(2020,1,1,0,0,0)),old.read(name))
        for name in ('selected.pt','preregistration.json','fit_calibration.json','selection_freeze.json'):z.writestr(zipfile.ZipInfo('source52b/'+name,(2020,1,1,0,0,0)),b.read(name))
    proof=dict(valid=True,dataset=DATASET,private=True,filename=target.name,sha256=sha(target),bytes=target.stat().st_size,source52b_sha256=p['source52b_sha256'])
    write_json(F/'package.json',proof);write_json(DEST/'dataset-metadata.json',dict(id=DATASET,title='GIADA Task52c Frozen Inputs',licenses=[dict(name='other')],description='Frozen52b checkpoint and original51/51j controls, immutable diagnostic input.'))
    return proof
def publish():
    from kaggle.api.kaggle_api_extended import KaggleApi
    proof=build();api=KaggleApi();api.authenticate();marker=F/'input_created.json'
    if not marker.exists():
        assert not(F/'input_pending.json').exists(),'Inspect pending upload; no duplicate'
        write_json(F/'input_pending.json',proof);api.dataset_create_new(str(DEST),public=False,quiet=True,convert_to_csv=False);write_json(marker,proof);(F/'input_pending.json').unlink()
    dest=ROOT/'artifacts/task52c_input_readback';dest.mkdir(exist_ok=True)
    for i in range(30):
        try:
            api.dataset_download_file(DATASET,proof['filename'],path=str(dest),force=True,quiet=True);assert sha(dest/proof['filename'])==proof['sha256'];write_json(F/'frozen_input_verified.json',proof);print('52c private input readback verified',flush=True);return
        except Exception:
            if i==29:raise
            time.sleep(10)
if __name__=='__main__':publish() if sys.argv[1]=='publish' else print(build())
