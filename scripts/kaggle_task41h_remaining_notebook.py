"""Single Kaggle notebook cell: frozen recovery, no training or selection."""
import hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path

REVISION='751f59ac94a0d17b04eef5e1e841b9eea46a93ab'
INPUTS={
    'remaining_evaluation_input.zip':'5fcd9f68b968fc0d9caf80d5a12f27b73deba0c7e4a62b37e1ab302a82c27a02',
    'task41b_opened_inputs.zip':'a8b261ed8bd742c18e66bfb2f92d0e06a6c875403545d94480e82ef9521934bb',
    'task41d_opened41c.zip':'172ec87c4b1135d4f3421d63343ae399368f8426a68fadcf4c7ca788c6e97aff',
}
WORK=Path('/tmp/giada_task41h_remaining');WORK.mkdir(exist_ok=False)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0+cu130','--index-url','https://download.pytorch.org/whl/cu130'],check=True)
for name,digest in INPUTS.items():
    matches=list(Path('/kaggle/input').rglob(name+'.bin'));assert len(matches)==1,(name,len(matches))
    path=WORK/name;shutil.copy2(matches[0],path)
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    print('[GIADA Task41h] exact input verified '+name,flush=True)
repo=WORK/'giada'
subprocess.run(['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],check=True)
subprocess.run(['git','fetch','--quiet','--depth','1','origin',REVISION],cwd=repo,check=True)
subprocess.run(['git','checkout','--quiet','--detach','FETCH_HEAD'],cwd=repo,check=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==REVISION
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',GIADA_TASK41_INPUTS=str(WORK/'task41b_opened_inputs.zip'),GIADA_TASK41C_INPUTS=str(WORK/'task41d_opened41c.zip'))
output=WORK/'artifacts';log=WORK/'process.log';code=None
try:
    with log.open('w') as stream:
        worker=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/finish_task41h_remaining_evaluation.py'),str(output),
            '--archive',str(WORK/'remaining_evaluation_input.zip'),'--sha256',INPUTS['remaining_evaluation_input.zip']],
            cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in worker.stdout:
            stream.write(line);stream.flush()
            if line.startswith('[GIADA') or 'Traceback' in line or 'Error:' in line:print(line.rstrip(),flush=True)
        code=worker.wait()
finally:
    output.mkdir(exist_ok=True)
    if log.exists():shutil.copy2(log,output/'process.log')
    (output/'recovery_dispatch.json').write_text(json.dumps(dict(revision=REVISION,inputs=INPUTS,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False),indent=2))
    archive=Path('/kaggle/working/giada_task41h_remaining_evaluation.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for path in output.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(output).as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix in ('.npz','.pt') else zipfile.ZIP_DEFLATED)
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    receipt=dict(valid=code==0,worker_exit_code=code,revision=REVISION,archive=archive.name,sha256=digest,bytes=archive.stat().st_size,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False)
    Path('/kaggle/working/kaggle_final_receipt.json').write_text(json.dumps(receipt,indent=2));print('[GIADA Task41h] durable output '+json.dumps(receipt),flush=True)
if code:
    print('\n'.join(log.read_text().splitlines()[-20:]));raise RuntimeError('Remaining-stage evaluation failed; download durable diagnostic archive')
