import hashlib,json,subprocess,sys,zipfile,time
from pathlib import Path
import task52g_resume_kaggle as r
ROOT=r.ROOT; F=ROOT/'experiments/results/task53_exploratory_v1';F.mkdir(exist_ok=True)
SLUG='alessandrobelli/giada-task53-exploratory'; DATA='alessandrobelli/giada-task53-frozen52j'
r.F=F;r.DEST=ROOT/'artifacts/task53_kaggle_external';r.PREFIX='task53'
SOURCE=ROOT/'artifacts/task52j_kaggle_external/task52j_final.zip'
def package():
    receipt=json.loads((SOURCE.parent/'kaggle_receipt.json').read_text())
    assert r.sha(SOURCE)==receipt['sha256']
    folder=ROOT/'artifacts/task53_input';folder.mkdir(exist_ok=True)
    path=folder/'frozen52j.zip.bin'
    with zipfile.ZipFile(SOURCE) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'))
        assert report['complete'] and not report['scientific_acceptance_passed']
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as w:
            for name in ('selected.pt','selection_freeze.json','calibration.pt'):w.writestr(zipfile.ZipInfo(name),z.read(name))
            w.writestr(zipfile.ZipInfo('source_manifest.json'),json.dumps(dict(source_sha256=receipt['sha256'],selected_sha256=hashlib.sha256(z.read('selected.pt')).hexdigest())))
        r.save(F/'source_audit.json',dict(report=report,backup=receipt,selection=json.loads(z.read('selection_freeze.json'))))
    return folder,path
def publish():
    folder,path=package();a=r.api()
    r.save(folder/'dataset-metadata.json',dict(id=DATA,title='GIADA Task53 Frozen52j',licenses=[dict(name='other')]))
    if not (F/'input_created.json').exists():
        assert not (F/'input_pending.json').exists()
        r.save(F/'input_pending.json',dict(dataset=DATA))
        a.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False)
        r.save(F/'input_created.json',dict(dataset=DATA));(F/'input_pending.json').unlink()
    dest=ROOT/'artifacts/task53_readback';dest.mkdir(exist_ok=True)
    for attempt in range(20):
        try:
            a.dataset_download_file(DATA,path.name,path=str(dest),force=True,quiet=True)
            assert r.sha(dest/path.name)==r.sha(path)
            r.save(F/'input_verified.json',dict(valid=True,sha256=r.sha(path)));return
        except Exception:
            if attempt==19:raise
            time.sleep(5)
def build(rev):
    path=ROOT/'notebooks/giada_task52j_final_guard.ipynb'
    doc=json.loads(path.read_text(encoding='utf-8'));proof=json.loads((F/'input_verified.json').read_text());assert proof['valid']
    checks=[(Path(i['remote']).name,i['sha256']) for i in json.loads((ROOT/'experiments/results/task52g_resume620_v1/dispatch_request.json').read_text())['inputs'][:4]]+[('frozen52j.zip.bin',proof['sha256'])]
    doc['cells'][0]['source']=['# GIADA Task 53 — exploratory frozen transfer\n','52j long_current frozen; no training or selection. New stimulus timing/amplitude variants within existing families. No promotion; quality thresholds unchanged.']
    code=doc['cells'][1]['source'];doc['cells'][1]['source']=[f'REVISION={rev!r}\n' if x.startswith('REVISION=') else f'CHECKS={checks!r}\n' if x.startswith('CHECKS=') else x.replace('task52j','task53').replace('GIADA52j','GIADA53') for x in code]
    compile(''.join(doc['cells'][1]['source']),'task53','exec')
    dest=ROOT/'artifacts/task53_notebook';dest.mkdir(exist_ok=True)
    meta=json.loads((ROOT/'artifacts/task52f_notebook/kernel-metadata.json').read_text());meta.update(id=SLUG,title='GIADA Task53 Exploratory');meta['dataset_sources'].append(DATA)
    r.save(dest/'notebook.ipynb',doc);r.save(dest/'kernel-metadata.json',meta)
    r.save(ROOT/'notebooks/giada_task53_exploratory.ipynb',doc);return dest
def launch():
    assert json.loads((F/'preregistration_receipt.json').read_text())['valid']
    assert not (F/'kaggle_launch.json').exists() and not (F/'launch_pending.json').exists()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    folder=build(rev);a=r.api();r.save(F/'launch_pending.json',dict(revision=rev))
    res=a.kernels_push(str(folder),timeout='43200',acc='NvidiaTeslaT4');assert not res.error,res.error
    r.save(F/'kaggle_launch.json',dict(ref=res.ref,version=res.version_number,url=res.url,revision=rev));(F/'launch_pending.json').unlink();print(res.url,flush=True)
if __name__=='__main__':
    {'publish':publish,'build':lambda:build('PINNED_BY_CONSOLE_DISPATCH'),'launch':launch,'supervise':r.supervise}[sys.argv[1]]()
