"""No fresh confirmation inputs; actual GPU graph and legacy numerical probe."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,subprocess,sys,time,zipfile
from pathlib import Path
root=Path('/content/giada41c_preflight_'+str(time.time_ns()))
subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],check=True,stdout=subprocess.DEVNULL)
with zipfile.ZipFile('/content/task41c_preflight_payload.zip') as z:
    for name in z.namelist():
        path=root/name;assert path.resolve().is_relative_to(root.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
sys.path.insert(0,str(root));import torch
from src.giada_teacher import task41c_frozen_confirmation as c
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available();out=root/'smoke';out.mkdir()
with torch.inference_mode():
    report=c.smoke(torch,root,out,'cuda')
    legacy=c.t.preflight(torch,root,'cuda');assert legacy['valid']
report.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,legacy_formula_preflight=legacy,
    source_sha256=c.p.digest(root/'src/giada_teacher/task41c_frozen_confirmation.py'),
    protocol_sha256=c.p.digest(root/c.CONFIG),worker_sha256=c.p.digest(root/'scripts/run_task41c_frozen_confirmation.py'))
Path('/content/task41c_gpu_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
