import hashlib,json,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task52i_kaggle as i
r=i.r;ROOT=i.ROOT;F=ROOT/'experiments/results/task52j_protocol_v1';F.mkdir(exist_ok=True)
DATASET='alessandrobelli/giada-task52j-warmstart';SLUG='alessandrobelli/giada-task52j-final-guard'
i.F=F;i.DATASET=DATASET;i.SLUG=SLUG;r.F=F;r.DEST=ROOT/'artifacts/task52j_kaggle_external';r.PREFIX='task52j'
def publish():
    source=ROOT/'artifacts/task52i_kaggle_external/task52i_final.zip';p=json.loads((ROOT/'experiments/task52j_preregistration_v1.json').read_text());assert r.sha(source)==p['warmstart']['source_sha256']
    folder=ROOT/'artifacts/task52j_input';folder.mkdir(exist_ok=True);target=folder/'warmstart52i.zip.bin'
    if not target.exists():
        with zipfile.ZipFile(source) as z,zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as out:
            assert z.testzip() is None
            for name in ('selected.pt','selection_freeze.json','preregistration.json'):out.writestr(name,z.read(name))
            out.writestr('source_manifest.json',json.dumps(dict(source_sha256=r.sha(source),selected_sha256=hashlib.sha256(z.read('selected.pt')).hexdigest())))
    r.save(folder/'dataset-metadata.json',dict(id=DATASET,title='GIADA Task52j Warmstart',licenses=[dict(name='other')],description='Private selected52i weights for final52j diagnostic; no evaluation traces.'))
    a=r.api()
    if not(F/'input_created.json').exists():
        assert not(F/'input_pending.json').exists();r.save(F/'input_pending.json',dict(dataset=DATASET));a.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False);r.save(F/'input_created.json',dict(dataset=DATASET,private=True));(F/'input_pending.json').unlink()
    dest=ROOT/'artifacts/task52j_readback';dest.mkdir(exist_ok=True)
    for attempt in range(30):
        try:
            a.dataset_download_file(DATASET,target.name,path=str(dest),force=True,quiet=True);assert r.sha(dest/target.name)==r.sha(target);r.save(F/'input_verified.json',dict(valid=True,dataset=DATASET,sha256=r.sha(target),bytes=target.stat().st_size));print('52j warmstart verified',flush=True);return
        except Exception:
            if attempt==29:raise
            time.sleep(10)
def build(revision):
    folder=i.build(revision);doc=json.loads((folder/'notebook.ipynb').read_text())
    for cell in doc['cells']:
        if cell['cell_type']=='code':
            cell['source']=[line.replace('task52i','task52j').replace('GIADA52i','GIADA52j').replace('warmstart52h.zip.bin','warmstart52i.zip.bin') for line in cell['source']];compile(''.join(cell['source']),'52j','exec')
        else:cell['source']=['# GIADA52j — final guard diagnostic\n2x2: current guard128/512ticks x coupledV/Ca guard.400updates, same selected52i warmstarts,10states and original thresholds. After audit proceed53; no52k.']
    dest=ROOT/'artifacts/task52j_notebook';dest.mkdir(exist_ok=True);r.save(dest/'notebook.ipynb',doc);meta=json.loads((folder/'kernel-metadata.json').read_text());meta.update(id=SLUG,title='GIADA Task52j Final Guard');r.save(dest/'kernel-metadata.json',meta);r.save(ROOT/'notebooks/giada_task52j_final_guard.ipynb',doc);return dest
def launch():
    for n in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/n).read_text())['valid']
    for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert not(F/'kaggle_launch.json').exists() and not(F/'launch_pending.json').exists();rev=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();folder=build(rev);a=r.api();r.save(F/'launch_pending.json',dict(revision=rev))
    result=a.kernels_push(str(folder),timeout='43200',acc='NvidiaTeslaT4');assert not result.error,result.error;r.save(F/'kaggle_launch.json',dict(ref=result.ref,version=result.version_number,url=result.url,revision=rev,thresholds_changed=False));(F/'launch_pending.json').unlink();print(result.url,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='publish':publish()
    elif sys.argv[1]=='build':build('PINNED_BY_CONSOLE_DISPATCH')
    elif sys.argv[1]=='launch':launch()
    else:r.supervise()
