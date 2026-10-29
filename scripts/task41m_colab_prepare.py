"""Pinned checkout and actual GPU preflight only, never the scientific suite."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
request=json.loads(Path('/content/giada_task41m_launch.json').read_text())
root=Path(request['remote_root']);repo=root/'giada'
root.mkdir(exist_ok=True)
if not repo.exists():
    subprocess.run(['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(repo)],check=True)
subprocess.run(['git','fetch','--quiet','--depth','1','origin',request['revision']],cwd=repo,check=True)
subprocess.run(['git','checkout','--quiet','--detach','FETCH_HEAD'],cwd=repo,check=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==request['revision']
for row in request['inputs']:
    with Path(row['remote']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
env=dict(os.environ,**request['environment'])
subprocess.run([sys.executable,'-u',str(repo/'scripts/run_task41m_frozen_contract.py'),str(root/'preflight'),'--preflight'],cwd=repo,env=env,check=True)
report=json.loads((root/'preflight/gpu_preflight.json').read_text());assert report['valid']
print(json.dumps(report),flush=True)
