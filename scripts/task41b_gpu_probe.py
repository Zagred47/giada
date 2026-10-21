"""Preflight actual T4 before preregistered training; no opened41 outcomes."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,subprocess,sys,time,zipfile
from pathlib import Path
root=Path('/content/giada41b_preflight_'+str(time.time_ns()))
subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],check=True,stdout=subprocess.DEVNULL)
with zipfile.ZipFile('/content/task41b_preflight_payload.zip') as z:
    for name in z.namelist():
        path=root/name;assert path.resolve().is_relative_to(root.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
sys.path.insert(0,str(root));import torch
from src.giada_teacher import task41b_local_precision as c
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available();spec,_,_,_=c.contract(root);out=root/'smoke';out.mkdir()
report=c.preflight(torch,root,out,'cuda',spec)
parent,models,ds,data,sel,w,norm,weights=c.prepare(torch,root,'cuda',spec)
idx=torch.arange(spec['batch'],device='cuda')[None].expand(24,-1);opt=torch.optim.Adam([w],lr=spec['lr'],eps=1e-8)
for _ in range(3):c.step(torch,w,opt,data,idx,weights,norm)
torch.cuda.synchronize();start=time.perf_counter()
for _ in range(10):c.step(torch,w,opt,data,idx,weights,norm)
torch.cuda.synchronize();report['seconds_per_full_update']=(time.perf_counter()-start)/10
report['projected_training_seconds']=report['seconds_per_full_update']*spec['checkpoints'][-1]
report['full_parameters_finite']=bool(torch.isfinite(w).all());assert report['full_parameters_finite']
report['gpu']=torch.cuda.get_device_name();report['worker_sha256']=c.p.digest(root/'scripts/run_task41b_local_precision.py')
report['preflight_root']=str(root);report['opened41_accessed']=False
Path('/content/task41b_gpu_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
