"""Actual GPU preflight on the pinned code; no opened regression selection."""
import json
from pathlib import Path
import subprocess
import sys
import time
revision=json.loads(Path('/content/task41e_preflight_request.json').read_text())['revision']
root=Path('/content/giada41e_preflight_'+str(time.time_ns()))
subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['git','fetch','--depth','1','origin',revision],cwd=root,check=True,stdout=subprocess.DEVNULL)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=root,check=True,stdout=subprocess.DEVNULL)
code=r'''
import os,sys,json
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root));import torch
from src.giada_teacher import task41e_transfer_model as m
from src.giada_teacher import task41e_transfer_training as training
from src.giada_teacher import task41e_transfer_evaluation as evaluation
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available()
out=root/'smoke';out.mkdir();spec,*_=m.contract(root)
report=training.preflight(torch,root,out,'cuda',spec)
with torch.inference_mode():
    report['legacy_formula_preflight']=m.t.preflight(torch,root,'cuda')
    controls=m.frozen_controls(torch,root,spec,'cuda')
    v=torch.linspace(-100,60,641,device='cuda')[None].expand(3,-1)
    ca=torch.logspace(-7,-2,641,device='cuda')[None].expand(3,-1)
    aa=controls['D_combined'].raw(v,ca);bb=controls['D_fused'].raw(v,ca)
    report['fused_raw_bitwise_equal']=all(torch.equal(a,b) for a,b in zip(aa,bb))
    assert report['fused_raw_bitwise_equal']
    report['new_recurrence_preflight']=m.c.j.i.old.recurrence_preflight(out,{'recurrence_chunk':32,'numeric_atol':1e-5},[controls['P110'],controls['D_fused']],torch,'cuda') is not None
assert report['legacy_formula_preflight']['valid']
report['revision']=sys.argv[2]
Path('/content/task41e_gpu_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
'''
result=subprocess.run([sys.executable,'-u','-c',code,str(root),revision],cwd=root,capture_output=True,text=True)
Path('/content/task41e_preflight_process.log').write_text(result.stdout+'\n'+result.stderr)
print(result.stdout,flush=True)
if result.returncode:raise RuntimeError(result.stderr[-16000:])
