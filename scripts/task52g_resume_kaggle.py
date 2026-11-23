"""Private verified640 input, pinned Kaggle continuation, external backup polling."""
import hashlib,io,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'_kaggle_cli_deps'))
F=ROOT/'experiments/results/task52g_resume640_v1';F.mkdir(exist_ok=True)
DEST=ROOT/'artifacts/task52g_kaggle_external';DATASET='alessandrobelli/giada-task52g-resume640-input';SLUG='alessandrobelli/giada-task52g-resume640'
SOURCE=ROOT/'artifacts/task52g_resume620_external/recovery_1795427655.zip'
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a=KaggleApi();a.authenticate();return a
def preflight():
    import torch
    spec=json.loads((ROOT/'experiments/task52g_resume640_v1.json').read_text());assert sha(SOURCE)==spec['source_sha256']
    with zipfile.ZipFile(SOURCE) as z:
        assert z.testzip() is None
        d=torch.load(io.BytesIO(z.read('recovery_latest.pt')),map_location='cpu',weights_only=True)
        assert d['update']==640 and d['recovery_only'] and 'rng' in d and 'diagnostic_sums' in d
        assert all(torch.isfinite(t).all() for t in d['model'].values())
        assert {float(v['step']) for v in d['optimizer']['state'].values()}=={640.0}
    save(F/'local_preflight.json',dict(valid=True,checkpoint_update=640,source_sha256=sha(SOURCE),model_optimizer_rng_verified=True,selection_unchanged=True))
def publish():
    preflight();a=api();folder=ROOT/'artifacts/task52g_kaggle_input';folder.mkdir(exist_ok=True)
    target=folder/'resume52g.zip.bin';shutil.copyfile(SOURCE,target)
    save(folder/'dataset-metadata.json',dict(id=DATASET,title='GIADA Task52g Resume640 Input',licenses=[dict(name='other')],description='Private recovery checkpoint640 with optimizer RNG and original history.'))
    if not (F/'input_created.json').exists():
        assert not (F/'input_pending.json').exists(),'Reconcile pending creation first'
        save(F/'input_pending.json',dict(dataset=DATASET,sha256=sha(target)))
        a.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False)
        save(F/'input_created.json',dict(dataset=DATASET,private=True));(F/'input_pending.json').unlink()
    readback=ROOT/'artifacts/task52g_kaggle_input_readback';readback.mkdir(exist_ok=True)
    for attempt in range(30):
        try:
            a.dataset_download_file(DATASET,target.name,path=str(readback),force=True,quiet=True)
            assert sha(readback/target.name)==sha(SOURCE)
            save(F/'input_verified.json',dict(valid=True,dataset=DATASET,sha256=sha(SOURCE),private=True));print('Private input readback SHA verified',flush=True);return
        except Exception:
            if attempt==29:raise
            time.sleep(10)
def build(revision):
    inputs=json.loads((ROOT/'experiments/results/task52g_resume620_v1/dispatch_request.json').read_text())['inputs']
    inputs[-1].update(sha256=sha(SOURCE))
    checks=[(Path(i['remote']).name,i['sha256']) for i in inputs]
    code='''import hashlib,json,os,shutil,subprocess,sys,time,threading,zipfile,traceback
from pathlib import Path
REVISION=__REV__
CHECKS=__CHECKS__
work=Path('/tmp/giada_task52g_resume640');work.mkdir(exist_ok=False)
repo=work/'giada';out=work/'result';public=Path('/kaggle/working');log=public/'process.log';started=time.time();returncode=None
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def setup(cmd):
 with log.open('a') as stream:subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
lock=threading.Lock();stop=threading.Event()
def snapshot(final=False):
 with lock:
  temporary=public/'snapshot_build.zip';name='task52g_final.zip' if final else 'task52g_checkpoint_latest.zip'
  with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
   if out.exists():
    for p in out.rglob('*'):
     if p.is_file() and not p.name.endswith('.tmp'):z.write(p,p.relative_to(out))
   if log.exists():z.write(log,'console_process.log')
  temporary.replace(public/name)
  receipt=dict(valid=True,archive=name,sha256=sha(public/name),bytes=(public/name).stat().st_size,revision=REVISION,at=time.time(),worker_exit_code=returncode,thresholds_changed=False)
  rp=public/('kaggle_receipt.json' if final else 'task52g_checkpoint_receipt.json');tmp=rp.with_suffix('.tmp');tmp.write_text(json.dumps(receipt));tmp.replace(rp)
def periodic():
 while not stop.wait(180):
  try:snapshot()
  except Exception as e:print('Backup retry: '+type(e).__name__,flush=True)
try:
 paths=[]
 for name,digest in CHECKS:
  found=list(Path('/kaggle/input').rglob(name));assert len(found)==1,(name,len(found));assert sha(found[0])==digest;paths.append(found[0])
 print('[GIADA52g] all five input hashes verified',flush=True)
 setup([sys.executable,'-m','pip','install','--quiet','torch==2.11.0+cu130','--index-url','https://download.pytorch.org/whl/cu130'])
 for cmd in (['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],['git','-C',str(repo),'fetch','--quiet','--depth','1','origin',REVISION],['git','-C',str(repo),'checkout','--quiet','--detach','FETCH_HEAD']):setup(cmd)
 env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg',GIADA_RESUME_SPEC='experiments/task52g_resume640_v1.json')
 thread=threading.Thread(target=periodic,daemon=True);thread.start()
 with log.open('a') as stream:
  proc=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/run_task52g_resume.py'),*map(str,paths),str(out)],cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in proc.stdout:
   stream.write(line);stream.flush()
   if line.startswith('[GIADA') or 'Error' in line or 'Traceback' in line:print(line.rstrip(),flush=True)
  returncode=proc.wait()
finally:
 stop.set()
 out.mkdir(exist_ok=True)
 (out/'kaggle_execution.json').write_text(json.dumps(dict(revision=REVISION,resumed_from=640,returncode=returncode,thresholds_changed=False,elapsed_seconds=time.time()-started)))
 snapshot(final=True)
assert returncode==0,'Worker failed: inspect saved logs and checkpoint, do not retrain from zero'
'''.replace('__REV__',repr(revision)).replace('__CHECKS__',repr(checks))
    compile(code,'52g-kaggle','exec')
    doc=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# GIADA52g — recovery640 on Kaggle\nOriginal thresholds, Adam, RNG, anchor and selection points preserved. Continue641–800, then original evaluations.']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    folder=ROOT/'artifacts/task52g_kaggle_notebook';folder.mkdir(exist_ok=True);save(folder/'notebook.ipynb',doc)
    meta=json.loads((ROOT/'artifacts/task52f_notebook/kernel-metadata.json').read_text());meta.update(id=SLUG,title='GIADA Task52g Resume640');meta['dataset_sources'].append(DATASET);save(folder/'kernel-metadata.json',meta)
    save(ROOT/'notebooks/giada_task52g_resume640.ipynb',doc);return folder
def launch():
    for n in ('local_preflight.json','input_verified.json','preregistration_receipt.json'):assert json.loads((F/n).read_text())['valid']
    assert not (F/'kaggle_launch.json').exists() and not (F/'launch_pending.json').exists()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();folder=build(revision);a=api()
    save(F/'launch_pending.json',dict(revision=revision,at=time.time()))
    r=a.kernels_push(str(folder),timeout='43200',acc='NvidiaTeslaT4');assert not r.error,r.error
    save(F/'kaggle_launch.json',dict(ref=r.ref,version=r.version_number,url=r.url,revision=revision,resumed_from=640,thresholds_changed=False));(F/'launch_pending.json').unlink();print(r.url,flush=True)
def supervise():
    a=api();r=json.loads((F/'kaggle_launch.json').read_text());ref=r['ref'].lstrip('/').removeprefix('code/')+'/'+str(r['version']);DEST.mkdir(exist_ok=True)
    for _ in range(960):
        try:
            s=a.kernels_status(ref);state=str(s.status).split('.')[-1].lower();save(F/'status.json',dict(status=state,failure_message=s.failure_message,at=time.time()));print(state,flush=True)
            a.kernels_output(ref,str(DEST),file_pattern=r'(task52g_final\.zip|task52g_checkpoint_latest\.zip|task52g_checkpoint_receipt\.json|kaggle_receipt\.json|process\.log)$',force=True,quiet=True)
            for receipt in ('task52g_checkpoint_receipt.json','kaggle_receipt.json'):
                rp=DEST/receipt
                if rp.exists():
                    p=json.loads(rp.read_text());archive=DEST/p['archive']
                    if archive.exists():
                        assert sha(archive)==p['sha256']
                        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
                        save(DEST/'external_backup_receipt.json',dict(valid=True,runtime_independent=True,remote=p,local=str(archive),at=time.time()))
            if state in ('complete','error','cancelled','failed'):return
        except Exception as e:print('Read-only retry '+type(e).__name__,flush=True)
        time.sleep(45)
if __name__=='__main__':
    if sys.argv[1]=='publish':publish()
    elif sys.argv[1]=='launch':launch()
    elif sys.argv[1]=='supervise':supervise()
    else:build('PINNED_BY_CONSOLE_DISPATCH')
