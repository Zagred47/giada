import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52g_resume_v1'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert json.loads((F/'preregistration_receipt.json').read_text())['valid'];test=json.loads((F/'local_preflight.json').read_text());assert test['valid']
for n,d in test['files_sha256'].items():assert sha(ROOT/n)==d
old=json.loads((ROOT/'experiments/results/task52g_protocol_v1/dispatch_request.json').read_text());inputs=old['inputs'][:4]
for item in inputs:assert sha(ROOT/item['local'])==item['sha256']
source=Path(json.loads((ROOT/'artifacts/task52g_colab_external/external_backup_receipt.json').read_text())['local']);digest=json.loads((ROOT/'experiments/task52g_resume_v1.json').read_text())['source_sha256'];assert sha(source)==digest
inputs.append(dict(local=source.relative_to(ROOT).as_posix(),remote='/content/resume52g.zip.bin',sha256=digest,bytes=source.stat().st_size))
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
request=dict(task='52g-resume',session='giada-task52g-resume600',profile='primary',remote_root='/content/giada_task52g_resume_'+revision[:12],local_output='artifacts/task52g_resume_colab_external',revision=revision,inputs=inputs,protocol_sha256=sha(ROOT/'experiments/task52g_preregistration_v1.json'),manager_sha256=sha(ROOT/'scripts/task52g_console_manager.py'),registration_valid=True,thresholds_changed=False,abstraction_changed=False,last_allowed_subtask='52j',resumed_from=600,detached_manager=True,recovery_only_snapshot_every=20,selection_checkpoints_unchanged=True)
(F/'dispatch_request.json').write_text(json.dumps(request,indent=2))
code="""import os,json,subprocess,sys
from pathlib import Path
request=__PAYLOAD__
for item in request['inputs']:assert Path(item['remote']).is_file(),item['remote']
bootstrap=Path('/content/giada52g_resume_bootstrap');assert not bootstrap.exists(),'Do not duplicate a running recovery'
subprocess.run(['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(bootstrap)],check=True)
subprocess.run(['git','fetch','--quiet','--depth','1','origin',request['revision']],cwd=bootstrap,check=True)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=bootstrap,check=True)
launch=Path('/content/task52g_launch.json');launch.write_text(json.dumps(request))
subprocess.run([sys.executable,str(bootstrap/'scripts/task52g_console_manager.py')],env=dict(os.environ,GIADA_LAUNCH_REQUEST=str(launch)),check=True)
""".replace('__PAYLOAD__',repr(request))
doc=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# GIADA52g — ripresa esatta da600\n','Completare601–800 e valutazioni originali. Adam/RNG/riferimento iniziale conservati; soglie e astrazione invariate. Non eseguire in parallelo al recupero da console.']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(ROOT/'notebooks/giada_task52g_resume.ipynb').write_text(json.dumps(doc,indent=2));compile(code,'52g-resume','exec');print(json.dumps(dict(revision=revision,session=request['session'],resumed_from=600)))
