"""Pinned CUDA preflight and synthetic precision/evaluation stress paths."""
import json
from pathlib import Path
import subprocess
import sys
import time
request=json.loads(Path('/content/task41f_preflight_request.json').read_text())
root=Path('/content/giada41f_preflight_'+str(time.time_ns()))
subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['git','fetch','--depth','1','origin',request['revision']],cwd=root,check=True,stdout=subprocess.DEVNULL)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=root,check=True,stdout=subprocess.DEVNULL)
code=r'''
import os,sys,json,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
import numpy as np
import torch
from src.giada_teacher import task41f_precision_relaxation as f
from src.giada_teacher import task41f_evaluation as ev
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
spec,cfg,calcium,base=f.contract(root);out=root/'smoke';out.mkdir()
report=f.preflight(torch,root,out,'cuda',spec)
n=4;steps=180;v=np.full((n,steps),-45.);ca=np.full_like(v,1e-4)
g,_=f.p.coefficients(v[:,0],ca[:,0],.1)
data=dict(voltage=v,calcium=ca,gates=np.repeat(g[:,None],steps,1),injection=np.full_like(v,.004),
    base_g_us=np.zeros((n,steps,4)),multipliers=np.ones((n,11)),area_um2=100*np.pi)
data['multipliers'][2,4]=0
controls=f.frozen_controls(torch,root,spec,'cuda')
with torch.inference_mode():
    for policy in ('FP32','mixed_state64','FP64'):
        for name in ('P110','D_combined','E_Q0','E_H10'):
            paths=ev.typed_rollout(torch,controls[name],data,cfg,calcium,base,'cuda',policy)
            assert all(np.isfinite(v).all() for v in paths.values())
    for u,z in [((0,1),()),((),(0,1)),((0,1,8),(0,1,8))]:
        oracle=ev.CoefficientOracle(torch,controls['P110'],u,z)
        cast=ev.cast_model(torch,oracle,torch.float64)
        paths=ev.typed_rollout(torch,cast,data,cfg,calcium,base,'cuda','FP64')
        assert all(np.isfinite(v).all() for v in paths.values())
    small=dict(spec,cost=dict(spec['cost'],batches=[4096],repetitions=2))
    ev.execution_matrix(torch,root,out,{'P110':controls['P110']},data,small,cfg,calcium,base,'cuda')
    print('[GIADA Task41f preflight] reproducing original frozen failures and all attribution paths',flush=True)
    ev.frozen_audit(torch,root,out,controls,spec,cfg,calcium,base,'cuda')
f.ee.differentiable_constants()
model=f.make_model(torch,controls['P110'],f.arms(),'cuda')
atomic,_=f.atomic_data(torch,controls['P110'],spec,'cuda',count=16)
idx=torch.arange(8,device='cuda')[None].expand(36,-1)
norm=f.atomic_terms(torch,model.f_correction,model.active,atomic,idx).detach().clamp_min(1e-12)
fb,support=f.feedback_data(torch,spec,cfg,calcium,base,'cuda',small=True)
view=f.FeedbackView(torch,controls['P110'],model)
probe=f.objective_probe(torch,model,view,atomic,norm,fb,support,fb,support,cfg,calcium,base)
assert probe['optimizer_and_training_weights_unchanged']
report.update(revision=sys.argv[2],account_profile=sys.argv[3],precision_branches_executed=True,
    arms=[a['name'] for a in f.arms()],
    oracle_branches_executed=True,whole_loop_4096_parity_passed=True,objective_probe_executed=True,
    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),opened_regressions_accessed=True,
    opened_regressions_used_for_selection=False,frozen_failure_reproduction_passed=True,scientific_training_performed=False)
Path('/content/task41f_gpu_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
'''
result=subprocess.run([sys.executable,'-u','-c',code,str(root),request['revision'],request['profile']],cwd=root,capture_output=True,text=True)
Path('/content/task41f_preflight_process.log').write_text(result.stdout+'\n'+result.stderr)
print(result.stdout,flush=True)
if result.returncode:raise RuntimeError(result.stderr[-18000:])
