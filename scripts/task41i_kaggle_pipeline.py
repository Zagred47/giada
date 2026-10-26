"""Finite Kaggle console pipeline: pinned phases and verified local backups.

No credentials are read or printed here: official SDK authentication uses the
existing process environment. A failed/uncertain operation never launches a
duplicate worker. One private notebook, versioned private resume dataset.
"""
import argparse,hashlib,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/kaggle_cli_deps'));sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
SLUG='alessandrobelli/giada-task41i-selective-repair'
PROBE='alessandrobelli/giada-task41i-preflight'
RESUME='alessandrobelli/giada-task41i-resume-stages'
FOLDER=ROOT/'experiments/results/task41i_protocol_v1'
DEST=ROOT/'artifacts/task41i_kaggle_external'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def notebook_source(revision,mode,stop=None,resume_sha=None):
    # No tokens, signed URLs or credentials belong in notebook source.
    return '''import hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
REVISION=%r
MODE=%r
STOP=%r
RESUME_SHA=%r
WORK=Path('/tmp/giada_task41i');WORK.mkdir(exist_ok=False)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0+cu130','--index-url','https://download.pytorch.org/whl/cu130'],check=True)
repo=WORK/'giada'
for command in (['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],['git','-C',str(repo),'fetch','--quiet','--depth','1','origin',REVISION],['git','-C',str(repo),'checkout','--quiet','--detach','FETCH_HEAD']):subprocess.run(command,check=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==REVISION
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',GIADA_PROVIDER='Kaggle')
out=WORK/'result';log=Path('/kaggle/working/process.log');code=None
try:
    if MODE=='preflight':
        env.update(GIADA_REMOTE_ROOT=str(WORK/'probe'),GIADA_PREFLIGHT_REVISION=REVISION,GIADA_PREFLIGHT_OUTPUT='/kaggle/working/task41i_gpu_preflight.json')
        command=[sys.executable,'-u',str(repo/'scripts/task41i_gpu_probe.py')]
    else:
        hashes={'task41b_opened_inputs.zip':'a8b261ed8bd742c18e66bfb2f92d0e06a6c875403545d94480e82ef9521934bb','task41d_opened41c.zip':'172ec87c4b1135d4f3421d63343ae399368f8426a68fadcf4c7ca788c6e97aff'}
        for name,digest in hashes.items():
            matches=list(Path('/kaggle/input').rglob(name+'.bin'));assert len(matches)==1,(name,len(matches))
            target=WORK/name;shutil.copy2(matches[0],target)
            with target.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest
        env.update(GIADA_TASK41_INPUTS=str(WORK/'task41b_opened_inputs.zip'),GIADA_TASK41C_INPUTS=str(WORK/'task41d_opened41c.zip'))
        command=[sys.executable,'-u',str(repo/'scripts/run_task41i_selective_repair.py'),str(out)]
        if STOP is not None:command+=['--stop-after',str(STOP)]
        if RESUME_SHA:
            matches=list(Path('/kaggle/input').rglob('resume.zip.bin'));assert len(matches)==1
            command+=['--resume-archive',str(matches[0]),'--resume-sha256',RESUME_SHA]
        if MODE=='evaluate':command+=['--evaluation-only']
    with log.open('w') as stream:
        process=subprocess.Popen(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            stream.write(line);stream.flush()
            if line.startswith('[GIADA') or 'Traceback' in line or 'Error' in line:print(line.rstrip(),flush=True)
        code=process.wait()
finally:
    receipt=dict(valid=code==0,worker_exit_code=code,revision=REVISION,mode=MODE,stop_after=STOP,input_resume_sha256=RESUME_SHA,thresholds_changed=False)
    if MODE!='preflight':
        out.mkdir(exist_ok=True)
        if log.exists():shutil.copy2(log,out/'process.log')
        archive=Path('/kaggle/working/task41i_stage.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for path in out.rglob('*'):
                if path.is_file():z.write(path,path.relative_to(out).as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix in ('.pt','.npz') else zipfile.ZIP_DEFLATED)
        with archive.open('rb') as stream:receipt.update(archive=archive.name,sha256=hashlib.file_digest(stream,'sha256').hexdigest(),bytes=archive.stat().st_size)
    Path('/kaggle/working/kaggle_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('[GIADA Task41i] persisted phase '+json.dumps(receipt),flush=True)
if code:raise RuntimeError('Task41i technical failure; durable diagnostic outputs retained')
'''%(revision,mode,stop,resume_sha)


def bundle(revision,mode,stop=None,resume_sha=None):
    folder=ROOT/'artifacts/task41i_kaggle_bundles'/('preflight' if mode=='preflight' else f'{mode}_{stop or "frozen"}')
    folder.mkdir(parents=True,exist_ok=True)
    code=notebook_source(revision,mode,stop,resume_sha)
    write_json(folder/'notebook_source.ipynb',dict(cells=[dict(cell_type='code',id='task41i-pinned',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))],metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    meta=json.loads((ROOT/'artifacts/task41h_kaggle_metadata_verified/kernel-metadata.json').read_text())
    meta.pop('id_no',None);meta.update(id=PROBE if mode=='preflight' else SLUG,title='GIADA Task41i Preflight' if mode=='preflight' else 'GIADA Task41i Selective Repair',code_file='notebook_source.ipynb',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=['alessandrobelli/giada-task41h-frozen-evaluation-inputs']+([RESUME] if resume_sha else []))
    write_json(folder/'kernel-metadata.json',meta);return folder


def launch(api,revision,mode,stop=None,resume_sha=None):
    pending=FOLDER/'kaggle_submission_pending.json'
    assert not pending.exists(),'Uncertain submission must be reconciled before retry'
    write_json(pending,dict(revision=revision,mode=mode,stop_after=stop,resume_sha256=resume_sha,submitted_at=time.time()))
    result=api.kernels_push(str(bundle(revision,mode,stop,resume_sha)),timeout='43200',acc='NvidiaTeslaT4')
    assert not result.error,result.error
    value=dict(ref=result.ref,version=result.version_number,url=result.url,revision=revision,mode=mode,stop_after=stop,input_resume_sha256=resume_sha)
    assert value['ref'] and value['version']
    path=FOLDER/('kaggle_preflight_launch.json' if mode=='preflight' else 'kaggle_active_launch.json')
    write_json(path,value);pending.unlink();print(json.dumps(value),flush=True);return value


def status(api,launch):
    ref=launch['ref'].lstrip('/').removeprefix('code/')
    result=api.kernels_status(ref+'/'+str(launch['version']))
    return str(result.status).split('.')[-1].lower(),result.failure_message


def retrieve(api,launch):
    target=DEST/(f"preflight_v{launch['version']}" if launch['mode']=='preflight' else f"v{launch['version']}");target.mkdir(parents=True,exist_ok=True)
    # Small explicit set avoids duplicating notebook logs and caches.
    ref=launch['ref'].lstrip('/').removeprefix('code/')
    api.kernels_output(ref+'/'+str(launch['version']),str(target),file_pattern=r'(kaggle_receipt\.json|task41i_stage\.zip|task41i_gpu_preflight\.json|process\.log)$',force=True,quiet=True)
    receipt=json.loads((target/'kaggle_receipt.json').read_text());assert receipt['revision']==launch['revision']
    assert receipt['mode']==launch['mode'] and receipt['stop_after']==launch['stop_after']
    if launch['mode']=='preflight':
        if not receipt['valid']:
            history=FOLDER/'preflight_history.json'
            rows=json.loads(history.read_text()) if history.exists() else []
            if not any(row['launch']==launch for row in rows):
                rows.append(dict(launch=launch,receipt=receipt,scientific_training_performed=False,
                                 log_path=str(target/'process.log'),log_sha256=sha(target/'process.log'),
                                 error_tail=(target/'process.log').read_text(errors='replace').splitlines()[-30:]))
                write_json(history,rows)
            raise RuntimeError('Technical preflight failed; complete local logs preserved; no scientific training')
        pf=json.loads((target/'task41i_gpu_preflight.json').read_text());assert pf['valid'] and pf['revision']==launch['revision']
        shutil.copy2(target/'task41i_gpu_preflight.json',FOLDER/'gpu_preflight.json')
        return receipt,target
    archive=target/receipt['archive'];assert sha(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'))
        import torch
        import io
        checkpoints=[name for name in z.namelist() if name.startswith('checkpoint_') and name.endswith('.pt')]
        latest=max(checkpoints,key=lambda name:int(name.split('_')[1].split('.')[0])) if checkpoints else None
        state=torch.load(io.BytesIO(z.read(latest)),map_location='cpu',weights_only=False) if latest else {}
        def finite(value):
            if torch.is_tensor(value):return bool(torch.isfinite(value).all())
            if isinstance(value,dict):return all(finite(item) for item in value.values())
            if isinstance(value,(list,tuple)):return all(finite(item) for item in value)
            return True
        tensors_finite=finite(state)
    verified=dict(valid=True,runtime_independent=True,local=str(archive),remote=dict(label='phase',sha256=receipt['sha256'],bytes=receipt['bytes'],platform='Kaggle persisted output'),kaggle=launch,latest_checkpoint=latest,all_recovered_tensors_finite=bool(latest and tensors_finite),report=report,technical_failure=not(receipt['valid'] and report.get('valid') and tensors_finite),verified_at=time.time())
    write_json(target/'external_backup_receipt.json',verified);write_json(DEST/'external_backup_receipt.json',verified)
    if not verified['technical_failure'] and latest and not (FOLDER/'launch.json').exists():
        write_json(FOLDER/'launch.json',dict(valid=True,request=dict(revision=launch['revision'],platform='Kaggle',notebook=launch['url']),
                  status_observed=launch,external_backup=verified,last_recovered_checkpoint=latest,
                  all_recovered_tensors_finite=True,scientific_result_pending=True,thresholds_changed=False,DB_backup='local only'))
    return verified,target


def publish_resume(api,receipt):
    folder=ROOT/'artifacts/task41i_resume_dataset';folder.mkdir(exist_ok=True)
    shutil.copy2(receipt['local'],folder/'resume.zip.bin')
    assert sha(folder/'resume.zip.bin')==receipt['remote']['sha256']
    write_json(folder/'dataset-metadata.json',dict(id=RESUME,title='GIADA Task41i Resume Stages',licenses=[dict(name='other')],description='Private sealed experiment state. Prior versions retained; no public sharing.'))
    marker=DEST/'resume_dataset_created.json'
    if marker.exists():api.dataset_create_version(str(folder),'Verified Task41i resume '+receipt['latest_checkpoint'],quiet=True,convert_to_csv=False,delete_old_versions=False)
    else:
        api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False)
        write_json(marker,dict(created=True,id=RESUME))
    deadline=time.monotonic()+1800
    while True:
        state=str(api.dataset_status(RESUME)).lower()
        if 'ready' in state or 'complete' in state:break
        assert time.monotonic()<deadline,('Resume dataset processing',state)
        time.sleep(30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','status','logs','retrieve','launch','supervise']);args=ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();DEST.mkdir(parents=True,exist_ok=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if args.mode=='preflight':launch(api,revision,'preflight');return
    if args.mode in ('status','logs','retrieve'):
        path=FOLDER/('kaggle_active_launch.json' if (FOLDER/'kaggle_active_launch.json').exists() else 'kaggle_preflight_launch.json')
        value=json.loads(path.read_text())
        if args.mode=='logs':
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
            ref=value['ref'].lstrip('/').removeprefix('code/');owner,slug=ref.split('/')
            request=ApiGetKernelSessionLogsStreamRequest();request.user_name=owner;request.kernel_slug=slug;request.version_label='v'+str(value['version']);request.wait_for_logs_url_seconds=60
            with api.build_kaggle_client() as client:
                response=client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
                try:
                    with (DEST/('preflight_live.log' if value['mode']=='preflight' else 'training_live.log')).open('a',encoding='utf-8') as stream:
                        for line in response.iter_lines(chunk_size=1,decode_unicode=True):
                            if isinstance(line,bytes):line=line.decode('utf-8',errors='replace')
                            if line:stream.write(line+'\n');stream.flush()
                            if line and 'END_OF_LOG' in line:break
                finally:response.close()
        elif args.mode=='status':print(json.dumps(dict(launch=value,status=status(api,value))),flush=True)
        else:retrieve(api,value);print('Verified phase retrieved',flush=True)
        return
    if args.mode=='launch':
        pf=json.loads((FOLDER/'gpu_preflight.json').read_text());memory=json.loads((FOLDER/'memory_receipt.json').read_text())
        assert pf['valid'] and memory['valid'] and memory['registration_before_scientific_run']
        assert pf['protocol_sha256']==sha(ROOT/'experiments/task41i_preregistration_v1.json')
        assert not (FOLDER/'kaggle_active_launch.json').exists()
        launch(api,pf['revision'],'train',512);return
    # Finite workflow, not an open-ended monitor. Never submit after uncertainty.
    while True:
        active=json.loads((FOLDER/'kaggle_active_launch.json').read_text());state,error=status(api,active)
        write_json(DEST/'pipeline_status.json',dict(launch=active,state=state,error=error,observed_at=time.time()))
        if 'complete' in state:
            receipt,target=retrieve(api,active);report=receipt['report']
            assert not receipt['technical_failure'],'Technical failure; outputs preserved; next phase blocked'
            if report['decision'] not in ('TRAINING_STAGE_COMPLETE','FROZEN_TRAINING_COMPLETE','FROZEN_EVALUATION_STAGE_COMPLETE'):
                write_json(DEST/'pipeline_final.json',receipt);return
            publish_resume(api,receipt)
            if not report.get('training_complete'):
                stop=next(k for k in (8192,30000,60000) if k>report['updates_completed']);mode='train'
            else:stop=None;mode='evaluate'
            launch(api,active['revision'],mode,stop,receipt['remote']['sha256'])
        elif any(word in state for word in ('error','cancel','fail')):
            try:retrieve(api,active)
            finally:raise RuntimeError((state,error,'No next phase launched; available outputs retrieved'))
        time.sleep(45)

if __name__=='__main__':main()
