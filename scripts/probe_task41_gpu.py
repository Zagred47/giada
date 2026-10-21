"""Technical preflight only; no fresh Task41 confirmation inputs accessed."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,subprocess,sys,zipfile
from pathlib import Path
root=Path('/content/giada41_preflight')
assert not root.exists(),'Inspect previous preflight instead of overwriting'
subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],check=True,stdout=subprocess.DEVNULL)
with zipfile.ZipFile('/content/task41_preflight_payload.zip') as z:
    for name in z.namelist():
        path=root/name;assert path.resolve().is_relative_to(root.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
sys.path.insert(0,str(root))
import torch
from src.giada_teacher import task41_independent_confirmation as c
output=Path('/content/task41_smoke');output.mkdir(exist_ok=False)
with torch.inference_mode():
    report=c.evaluate(torch,root,output,smoke=True)
    legacy=c.t.preflight(torch,root,'cuda');assert legacy['valid']
report.update(device=torch.cuda.get_device_name(),torch=torch.__version__,source_sha256=c.p.digest(root/'src/giada_teacher/task41_independent_confirmation.py'),
    worker_sha256=c.p.digest(root/'scripts/run_task41_independent_confirmation.py'),protocol_sha256=c.p.digest(root/c.CONFIG),legacy_formula_preflight=legacy)
Path('/content/task41_gpu_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
