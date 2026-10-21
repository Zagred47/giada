"""Actual full factorial GPU preflight; no opened regression access."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,subprocess,sys,time
from pathlib import Path
revision=json.loads(Path('/content/task41d_preflight_request.json').read_text())['revision']
root=Path('/content/giada41d_preflight_'+str(time.time_ns()))
subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['git','fetch','--depth','1','origin',revision],cwd=root,check=True,stdout=subprocess.DEVNULL)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=root,check=True,stdout=subprocess.DEVNULL)
sys.path.insert(0,str(root));import torch
from src.giada_teacher import task41d_precision_response as d
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available();out=root/'smoke';out.mkdir();spec,*_=d.contract(root)
report=d.preflight(torch,root,out,'cuda',spec)
from src.giada_teacher.task41d_evaluation import compose
with torch.inference_mode():
    report['legacy_formula_preflight']=d.t.preflight(torch,root,'cuda')
    parent=d.prior.load_models(torch,root,'cuda')['P110']
    ca=d.make_model(torch,parent,[d.arms()[0]],[17,29,43],'cuda');nap=d.make_model(torch,parent,[d.arms()[4]],[17,29,43],'cuda')
    combined=compose(torch,ca,nap,parent)
    report['composition_recurrence_preflight']=d.c.j.i.old.recurrence_preflight(out,{'recurrence_chunk':32,'numeric_atol':1e-5},[parent,ca,nap,combined],torch,'cuda') is not None
assert report['legacy_formula_preflight']['valid']
Path('/content/task41d_gpu_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
