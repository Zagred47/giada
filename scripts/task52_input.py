import hashlib,json,shutil,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'_kaggle_cli_deps'))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task52_protocol_v1';DEST=ROOT/'artifacts/task52_input';DATASET='alessandrobelli/giada-task52-frozen-inputs'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def build():
    F.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);p=json.loads((ROOT/'experiments/task52_preregistration_v1.json').read_text());target=DEST/'models52.zip.bin'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
        for key,row in p['inputs'].items():
            path=ROOT/row['path'];assert sha(path)==row['sha256'];z.write(path,key+'.zip')
    proof=dict(valid=True,dataset=DATASET,private=True,filename=target.name,sha256=sha(target),bytes=target.stat().st_size)
    write_json(F/'package.json',proof);write_json(DEST/'dataset-metadata.json',dict(id=DATASET,title='GIADA Task52 Frozen Inputs',licenses=[dict(name='other')],description='Private immutable Task51 compact and Task51j static readout checkpoints for exploratory Task52.'))
    return proof
def publish():
    from kaggle.api.kaggle_api_extended import KaggleApi
    proof=build();api=KaggleApi();api.authenticate();mark=F/'input_created.json'
    if not mark.exists():
        assert not (F/'input_pending.json').exists(),'Inspect pending creation before retry'
        write_json(F/'input_pending.json',proof);api.dataset_create_new(str(DEST),public=False,quiet=True,convert_to_csv=False);write_json(mark,proof);(F/'input_pending.json').unlink()
    readback=ROOT/'artifacts/task52_input_readback';readback.mkdir(exist_ok=True)
    for attempt in range(30):
        try:
            api.dataset_download_file(DATASET,proof['filename'],path=str(readback),force=True,quiet=True)
            path=readback/proof['filename'];assert sha(path)==proof['sha256'] and path.stat().st_size==proof['bytes'];write_json(F/'frozen_input_verified.json',proof);print('Task52 private package readback verified',flush=True);return
        except Exception:
            if attempt==29:raise
            time.sleep(10)
if __name__=='__main__':publish() if sys.argv[1]=='publish' else print(build())
