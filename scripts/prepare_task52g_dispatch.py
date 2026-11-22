import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52g_protocol_v1';F.mkdir(exist_ok=True)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
registration=json.loads((F/'preregistration_receipt.json').read_text());assert registration['valid']
preflight=json.loads((F/'local_preflight.json').read_text());assert preflight['valid']
for name,digest in preflight['files_sha256'].items():assert sha(ROOT/name)==digest
old=json.loads((ROOT/'experiments/results/task52f_resume_colab_v1/dispatch_request.json').read_text());inputs=old['inputs'][:4]
for item in inputs:assert sha(ROOT/item['local'])==item['sha256']
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
request=dict(task='52g',session='giada-task52g-preservation',profile='primary',remote_root='/content/giada_task52g_'+revision[:12],revision=revision,inputs=inputs,protocol_sha256=sha(ROOT/'experiments/task52g_preregistration_v1.json'),manager_sha256=sha(ROOT/'scripts/task52g_console_manager.py'),registration_valid=True,thresholds_changed=False,abstraction_changed=False,last_allowed_subtask='52j')
(F/'dispatch_request.json').write_text(json.dumps(request,indent=2))
code="""import os,json,subprocess,sys
from pathlib import Path
request=__REQUEST_PAYLOAD__
for item in request['inputs']:
    assert Path(item['remote']).is_file(), 'Upload the registered input: '+item['remote']
bootstrap=Path('/content/giada52g_notebook_bootstrap')
assert not bootstrap.exists(), 'Use the existing console run; do not duplicate it'
subprocess.run(['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(bootstrap)],check=True)
subprocess.run(['git','fetch','--quiet','--depth','1','origin',request['revision']],cwd=bootstrap,check=True)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=bootstrap,check=True)
launch=Path('/content/task52g_launch.json');launch.write_text(json.dumps(request))
subprocess.run([sys.executable,str(bootstrap/'scripts/task52g_console_manager.py')],env=dict(os.environ,GIADA_LAUNCH_REQUEST=str(launch)),check=True)
""".replace('__REQUEST_PAYLOAD__',repr(request))
doc=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=[dict(cell_type='markdown',metadata={},source=['# GIADA52g — preservazione delle correnti e conflitto dei gradienti\n','Quattro bracci appaiati, stessi512tick,800aggiornamenti,10stati. Soglie originali; limite52j.\n','Il lancio operativo avviene via console con backup esterni; NON eseguire una seconda copia di questo notebook. Gli input verificati devono essere caricati negli stessi percorsi/content.']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(ROOT/'notebooks/giada_task52g_gradient_preservation.ipynb').write_text(json.dumps(doc,indent=2))
print(json.dumps(dict(valid=True,revision=revision,session=request['session'],inputs=len(inputs))))
