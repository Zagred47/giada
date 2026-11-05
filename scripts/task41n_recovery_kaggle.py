"""One private console-run recovery, immutable input and verified off-runtime ZIP."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'_kaggle_cli_deps'));sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
FOLDER=ROOT/'experiments/results/task41n_recovery_v1';DEST=ROOT/'artifacts/task41n_recovery_kaggle_external'
SLUG='alessandrobelli/giada-task41n-evaluation-only-recovery'
DATASET='alessandrobelli/giada-task41n-evaluation-recovery-input'
OUTPUT_PATTERN=r'(kaggle_receipt\.json|task41n_recovery_final\.zip|process\.log)$'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def notebook(revision,source_sha):
    code='''import hashlib,json,os,shutil,subprocess,sys,zipfile,time,threading
from pathlib import Path
REVISION=%r
SOURCE_SHA=%r
WORK=Path('/tmp/giada_task41n_recovery');WORK.mkdir(exist_ok=False)
repo=WORK/'giada';out=WORK/'result';public=Path('/kaggle/working');log=public/'process.log'
code=None;started=time.time()
def setup(cmd):
 with log.open('a') as stream:subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
try:
 assert len(REVISION)==40,'Use console dispatch to pin the exact commit'
 print('[GIADA41n recovery] pinned Torch/GPU setup',flush=True)
 setup([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0+cu130','--index-url','https://download.pytorch.org/whl/cu130'])
 for cmd in (['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],['git','-C',str(repo),'fetch','--quiet','--depth','1','origin',REVISION],['git','-C',str(repo),'checkout','--quiet','--detach','FETCH_HEAD']):setup(cmd)
 sources=list(Path('/kaggle/input').rglob('source41n.zip.bin'));assert len(sources)==1
 with sources[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==SOURCE_SHA
 env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg')
 with log.open('a') as f:
  proc=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/run_task41n_recovery.py'),str(sources[0]),str(out)],cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  timer=threading.Timer(5400,proc.terminate);timer.daemon=True;timer.start()
  for line in proc.stdout:
   f.write(line);f.flush()
   if line.startswith('[GIADA') or 'Error' in line or 'Traceback' in line:print(line.rstrip(),flush=True)
  code=proc.wait();timer.cancel()
finally:
 out.mkdir(exist_ok=True)
 if log.exists():shutil.copy2(log,out/'process.log')
 (out/'runtime_receipt.json').write_text(json.dumps(dict(revision=REVISION,source_sha256=SOURCE_SHA,worker_exit_code=code,elapsed_seconds=time.time()-started,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False)))
 archive=Path(shutil.make_archive(str(public/'task41n_recovery_final'),'zip',out))
 with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 (public/'kaggle_receipt.json').write_text(json.dumps(dict(valid=code==0,revision=REVISION,source_sha256=SOURCE_SHA,archive=archive.name,sha256=digest,bytes=archive.stat().st_size,worker_exit_code=code,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False)))
 print('[GIADA41n recovery] final archive persisted '+digest,flush=True)
assert code==0,'Evaluation recovery failed technically; preserve output and logs'
'''%(revision,source_sha)
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python')),cells=[dict(cell_type='markdown',id='scope41n',metadata={},source=['# GIADA41n — evaluation-only recovery\nNo training/selection/native regeneration. Original failed FP32 floors retained; no reinstated primary confirmation.']),dict(cell_type='code',id='recovery41n',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])

def bundle(revision):
    plan=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());folder=ROOT/'artifacts/task41n_recovery_notebook';folder.mkdir(exist_ok=True)
    write_json(folder/'notebook.ipynb',notebook(revision,plan['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=SLUG,title='GIADA Task41n evaluation-only recovery',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[DATASET],competition_sources=[],kernel_sources=[],model_sources=[]))
    return folder

def publish(api):
    plan=json.loads((ROOT/'experiments/task41n_recovery_v1.json').read_text());source=ROOT/plan['source_local'];assert sha(source)==plan['source_sha256']
    folder=ROOT/'artifacts/task41n_recovery_upload';folder.mkdir(exist_ok=True)
    marker=FOLDER/'input_created.json';readback=ROOT/'artifacts/task41n_recovery_input_readback';readback.mkdir(exist_ok=True)
    if not marker.exists():
        shutil.copy2(source,folder/'source41n.zip.bin')
        write_json(folder/'dataset-metadata.json',dict(id=DATASET,title='GIADA Task41n frozen evaluation recovery input',licenses=[dict(name='other')],description='Private immutable existing41n evaluation state. No public sharing, new training or new selection.'))
        api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False)
        write_json(marker,dict(dataset=DATASET,source_sha256=plan['source_sha256'],private=True))
    for attempt in range(30):
        try:
            api.dataset_download_file(DATASET,'source41n.zip.bin',path=str(readback),force=True,quiet=True)
            path=readback/'source41n.zip.bin'
            assert path.stat().st_size==plan['source_bytes'] and sha(path)==plan['source_sha256']
            write_json(FOLDER/'input_verified.json',dict(valid=True,dataset=DATASET,private=True,sha256=plan['source_sha256'],bytes=path.stat().st_size))
            for p in (folder/'source41n.zip.bin',path):
                assert p.resolve().is_relative_to(ROOT/'artifacts') and p.name=='source41n.zip.bin';p.unlink(missing_ok=True)
            return
        except TypeError:
            raise RuntimeError('SDK signature mismatch; inspect API before retry') from None
        except Exception:
            if attempt==29:raise RuntimeError('Private input readback unavailable; no GPU launched, no automatic resubmission')
            time.sleep(15)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['publish','launch','retry-notebook-format','status','supervise','retrieve']);a=ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();FOLDER.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
    if a.mode=='publish':publish(api);return
    launchfile=FOLDER/'kaggle_launch.json'
    if a.mode in ('launch','retry-notebook-format'):
        assert not (FOLDER/'submission_pending.json').exists(),'Inspect uncertain launch; never duplicate'
        if a.mode=='retry-notebook-format':
            old=json.loads(launchfile.read_text());assert old['version']==1
            status=api.kernels_status(old['ref'].lstrip('/').removeprefix('code/')+'/1');assert 'error' in str(status.status).lower()
            log=DEST/'giada-task41n-evaluation-only-recovery.log';text=log.read_text(encoding='utf-8')
            assert 'No kernel name found in notebook' in text and not (DEST/'kaggle_receipt.json').exists()
            preserved=DEST/'attempt1_notebook_metadata.log'
            if not preserved.exists():shutil.copy2(log,preserved)
            assert sha(preserved)==sha(log)
            write_json(FOLDER/'technical_attempt_v1.json',dict(valid=True,launch=old,phase='provider_notebook_metadata_before_cells',cause='Papermill requires kernelspec.name; no worker/cell executed',log_local=preserved.relative_to(ROOT).as_posix(),log_sha256=sha(preserved),training_updates=0,selection_recomputed=False,source_unchanged=True))
        else:assert not launchfile.exists(),'Inspect existing launch; never duplicate'
        memory=json.loads((FOLDER/'preregistration_receipt.json').read_text());assert memory['valid']
        assert json.loads((FOLDER/'input_verified.json').read_text())['valid']
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(FOLDER/'submission_pending.json',dict(revision=revision,submitted_at=time.time()))
        result=api.kernels_push(str(bundle(revision)),timeout='7200',acc='NvidiaTeslaT4');assert not result.error,result.error
        launch=dict(ref=result.ref,version=result.version_number,url=result.url,revision=revision,training_updates_performed=0,selection_recomputed=False)
        write_json(launchfile,launch);(FOLDER/'submission_pending.json').unlink();print(json.dumps(launch),flush=True);return
    launch=json.loads(launchfile.read_text());ref=launch['ref'].lstrip('/').removeprefix('code/')+'/'+str(launch['version'])
    if a.mode in ('status','supervise'):
        for poll in range(240 if a.mode=='supervise' else 1):
            try:state=api.kernels_status(ref);name=str(state.status).split('.')[-1].lower()
            except Exception as error:
                if a.mode=='status':raise
                print('Read-only status retry '+type(error).__name__+'; run untouched',flush=True);time.sleep(30);continue
            write_json(FOLDER/'status.json',dict(status=name,observed_at=time.time(),launch=launch,failure_message=state.failure_message))
            print('[41n recovery] '+name,flush=True)
            if a.mode=='status':return
            if any(word in name for word in ('complete','error','fail','cancel')):break
            time.sleep(30)
        else:raise RuntimeError('Finite monitoring deadline reached; inspect existing run, never auto-restart')
    for attempt in range(12):
        try:
            api.kernels_output(ref,str(DEST),file_pattern=OUTPUT_PATTERN,force=True,quiet=True);break
        except Exception:
            if attempt==11:raise
            time.sleep(30)
    receipt=json.loads((DEST/'kaggle_receipt.json').read_text());archive=DEST/receipt['archive']
    assert receipt['revision']==launch['revision'] and sha(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json' if 'final_report.json' in z.namelist() else ('failure_report.json' if 'failure_report.json' in z.namelist() else 'runtime_receipt.json')))
    verified=dict(valid=True,runtime_independent=True,local=str(archive),remote=dict(label='final',sha256=receipt['sha256'],bytes=receipt['bytes'],platform='Kaggle persisted output'),launch=launch,technical_failure=not(receipt['valid'] and report.get('complete')),report=report)
    write_json(DEST/'external_backup_receipt.json',verified)
    print(json.dumps(dict(backup_verified=True,technical_failure=verified['technical_failure'],decision=report.get('decision'))),flush=True)
if __name__=='__main__':main()
