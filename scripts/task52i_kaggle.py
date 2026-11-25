import hashlib,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task52h_kaggle as h
r=h.r;ROOT=h.ROOT;F=ROOT/'experiments/results/task52i_protocol_v1';F.mkdir(exist_ok=True)
r.F=F;r.DEST=ROOT/'artifacts/task52i_kaggle_external';r.PREFIX='task52i'
SLUG='alessandrobelli/giada-task52i-solution-retention';DATASET='alessandrobelli/giada-task52i-warmstart'
def publish():
    source=ROOT/'artifacts/task52h_kaggle_external/task52h_final.zip';p=json.loads((ROOT/'experiments/task52i_preregistration_v1.json').read_text());assert r.sha(source)==p['warmstart']['source_sha256']
    folder=ROOT/'artifacts/task52i_input';folder.mkdir(exist_ok=True);target=folder/'warmstart52h.zip.bin'
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as out:
        assert z.testzip() is None
        for name in ('selected.pt','selection_freeze.json','preregistration.json'):out.writestr(name,z.read(name))
        out.writestr('source_manifest.json',json.dumps(dict(source_sha256=r.sha(source),selected_sha256=hashlib.sha256(z.read('selected.pt')).hexdigest())))
    r.save(folder/'dataset-metadata.json',dict(id=DATASET,title='GIADA Task52i Warmstart',licenses=[dict(name='other')],description='Private verified selected52h weights, no evaluation traces.'))
    a=r.api()
    if not(F/'input_created.json').exists():
        assert not(F/'input_pending.json').exists();r.save(F/'input_pending.json',dict(dataset=DATASET));a.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False);r.save(F/'input_created.json',dict(dataset=DATASET,private=True));(F/'input_pending.json').unlink()
    dest=ROOT/'artifacts/task52i_readback';dest.mkdir(exist_ok=True)
    for attempt in range(30):
        try:
            a.dataset_download_file(DATASET,target.name,path=str(dest),force=True,quiet=True);assert r.sha(dest/target.name)==r.sha(target);r.save(F/'input_verified.json',dict(valid=True,dataset=DATASET,sha256=r.sha(target),bytes=target.stat().st_size));print('52i warmstart verified',flush=True);return
        except Exception:
            if attempt==29:raise
            time.sleep(10)
def build(revision):
    folder=h.build(revision);doc=json.loads((folder/'notebook.ipynb').read_text());checks=[(Path(i['remote']).name,i['sha256']) for i in json.loads((ROOT/'experiments/results/task52g_resume620_v1/dispatch_request.json').read_text())['inputs'][:4]];checks.append(('warmstart52h.zip.bin',json.loads((F/'input_verified.json').read_text())['sha256']))
    for cell in doc['cells']:
        if cell['cell_type']=='code':
            cell['source']=[('CHECKS='+repr(checks)+'\n') if line.startswith('CHECKS=') else line.replace('task52h','task52i').replace('GIADA52h','GIADA52i').replace('all four input hashes','all five input hashes') for line in cell['source']];compile(''.join(cell['source']),'52i-notebook','exec')
        else:cell['source']=['# GIADA52i — solution retention\nSame52h selected warmstart for four paired arms: constant/decaying step × fit-only current step acceptance. Original52d development guards,10states and thresholds unchanged. Cap52j.']
    dest=ROOT/'artifacts/task52i_notebook';dest.mkdir(exist_ok=True);r.save(dest/'notebook.ipynb',doc);meta=json.loads((ROOT/'artifacts/task52f_notebook/kernel-metadata.json').read_text());meta.update(id=SLUG,title='GIADA Task52i Solution Retention');meta['dataset_sources'].append(DATASET);r.save(dest/'kernel-metadata.json',meta);r.save(ROOT/'notebooks/giada_task52i_solution_retention.ipynb',doc);return dest
def launch():
    for n in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/n).read_text())['valid']
    for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert not(F/'kaggle_launch.json').exists() and not(F/'launch_pending.json').exists();rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();folder=build(rev);a=r.api();r.save(F/'launch_pending.json',dict(revision=rev))
    result=a.kernels_push(str(folder),timeout='43200',acc='NvidiaTeslaT4');assert not result.error,result.error;r.save(F/'kaggle_launch.json',dict(ref=result.ref,version=result.version_number,url=result.url,revision=rev,thresholds_changed=False));(F/'launch_pending.json').unlink();print(result.url,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='publish':publish()
    elif sys.argv[1]=='build':build('PINNED_BY_CONSOLE_DISPATCH')
    elif sys.argv[1]=='launch':launch()
    else:r.supervise()
