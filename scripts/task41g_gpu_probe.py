"""Pinned CUDA checks, including auxiliary update and precision-policy graphs."""
import json
from pathlib import Path
import subprocess
import sys
import time
request=json.loads(Path('/content/task41g_preflight_request.json').read_text())
root=Path('/content/giada41g_preflight_'+str(time.time_ns()))
for command in (['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],
                ['git','-C',str(root),'fetch','--depth','1','origin',request['revision']],
                ['git','-C',str(root),'checkout','--detach','FETCH_HEAD']):
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
code=r'''
import os,sys,json
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
import numpy as np
import torch
from src.giada_teacher import task41g_preservation as g
from src.giada_teacher import task41g_evaluation as ev
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
spec,cfg,calcium,base=g.contract(root);out=root/'smoke';out.mkdir()
report=g.preflight(torch,root,out,'cuda',spec)
print('[GIADA Task41g preflight] graph and independent optimizer passed',flush=True)
controls=g.frozen_controls(torch,root,spec,'cuda');parent=controls['E_Q0']
n=4;steps=180;v=np.full((n,steps),-45.);ca=np.full_like(v,1e-4)
initial,_=g.p.coefficients(v[:,0],ca[:,0],.1)
data=dict(voltage=v,calcium=ca,gates=np.repeat(initial[:,None],steps,1),injection=np.full_like(v,.004),base_g_us=np.zeros((n,steps,4)),multipliers=np.ones((n,11)),area_um2=100*np.pi)
with torch.inference_mode():
    for policy in __import__('itertools').product((False,True),repeat=3):
        paths=ev.policy_rollout(torch,parent,data,cfg,calcium,base,'cuda',policy)
        assert np.isfinite(paths).all()
        __import__('gc').collect();torch.cuda.empty_cache()
    print('[GIADA Task41g preflight] eight precision policies passed',flush=True)
    ordinary=g.fe.typed_rollout(torch,parent,data,cfg,calcium,base,'cuda')['current']
    actual=ev.policy_rollout(torch,parent,data,cfg,calcium,base,'cuda',(False,False,False))
    assert np.array_equal(actual,ordinary)
    small=dict(spec,cost=dict(spec['cost'],batches=[4096],repetitions=2))
    g.fe.execution_matrix(torch,root,out,{'E_Q0':parent},data,small,cfg,calcium,base,'cuda')
    print('[GIADA Task41g preflight] whole-loop4096 passed',flush=True)
g.f.ee.differentiable_constants()
model=g.make_model(torch,parent,g.arms(),'cuda')
atomic,_=g.atomic_data(torch,parent,spec,'cuda',count=16)
idx=torch.arange(8,device='cuda')[None].expand(36,-1)
norm=g.f.atomic_terms(torch,model.f_correction,model.active,atomic,idx).detach().clamp_min(1e-12)
tail=g.tail_data(torch,parent,spec,'cuda',count=32)
fb,support=g.f.feedback_data(torch,spec,cfg,calcium,base,'cuda',small=True)
view=g.FeedbackView(torch,parent,model);opt=g.Adam(torch,model.f_correction,g.arms(),spec)
diag=g.auxiliary_step(torch,model,opt,atomic,idx,norm,atomic,idx,tail,torch.arange(8,device='cuda'),view,fb,support,torch.arange(2,device='cuda'),cfg,calcium,base)
assert torch.isfinite(model.f_correction).all()
assert torch.equal(model.f_correction*(1-model.active),torch.zeros_like(model.f_correction))
report.update(revision=sys.argv[2],account_profile=sys.argv[3],arms=[a['name'] for a in g.arms()],
    precision_branches_executed=True,whole_loop_4096_parity_passed=True,auxiliary_update_executed=True,
    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),scientific_training_performed=False)
Path('/content/task41g_gpu_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
'''
with Path('/content/task41g_preflight_process.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,'-u','-c',code,str(root),request['revision'],request['profile']],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout:
        log.write(line);log.flush();print(line.rstrip(),flush=True)
    result=proc.wait()
if result:raise RuntimeError(f'GPU preflight process failed with exit={result}; see saved process log')
