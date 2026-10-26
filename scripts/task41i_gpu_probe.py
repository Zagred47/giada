"""Pinned CUDA graph, gradients, feedback and exact-reference preflight."""
import json,os,subprocess,sys,time
from pathlib import Path
request=json.loads(Path('/content/task41i_preflight_request.json').read_text())
root=Path('/content/giada41i_preflight_'+str(time.time_ns()))
for cmd in (['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],['git','-C',str(root),'fetch','--depth','1','origin',request['revision']],['git','-C',str(root),'checkout','--detach','FETCH_HEAD']):
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
code=r'''
import os,sys,json,copy
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
import torch,numpy as np
from src.giada_teacher import task41i_selective_repair as i
from src.giada_teacher import task41i_evaluation as e
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
spec,cfg,ca,base=i.contract(root);out=root/'probe';out.mkdir()
r=i.preflight(torch,root,out,'cuda',spec)
parent=i.frozen_controls(torch,root,spec,'cuda')['H_nominee']
raw=i.d.anchors(spec['new_seeds']['fit'],64);knots=i.fit_knots(raw)
model=i.make_model(torch,parent,i.arms(),knots,'cuda');pool=i.atomic_pool(torch,parent,raw,'cuda')
ids=torch.arange(8,device='cuda')[None].expand(24,-1).clone();norm=i.atomic_loss(torch,model,pool,ids)[0].detach().clamp_min(1e-12)
opt=i.g.Adam(torch,model.correction,i.arms(),spec);graph=i.capture(torch,model,opt,pool,ids,norm,pool,ids);graph.replay()
tail=i.path_pool(torch,parent,spec,'cuda',count=8);i.f.ee.differentiable_constants()
fb,support=i.f.feedback_data(torch,spec,cfg,ca,base,'cuda',small=True)
view=i.FeedbackView(torch,model)
diag=i.loss_step(torch,model,opt,pool,ids,norm,pool,ids,tail,torch.arange(4,device='cuda'),view,fb,support,torch.arange(2,device='cuda'),(cfg,ca,base))
assert torch.isfinite(model.correction).all()
one=i.extract(torch,model,7,parent,knots,'cuda')
v=torch.linspace(-95,55,33,device='cuda')[None].expand(3,-1);cc=torch.full_like(v,1e-4)
compiled=e.RuntimeModel(torch,copy.deepcopy(one),'compiled')
with torch.no_grad():
    a=one.raw(v,cc);b=compiled.raw(v,cc)
    err=max(float((x-y).abs().max()) for x,y in zip(a,b));assert err<=spec['computational_parity_atol'],err
r.update(revision=sys.argv[2],gpu=torch.cuda.get_device_name(),torch=torch.__version__,auxiliary_update_executed=True,
         point_graph_executed=True,whole_chain_compile_tested=True,compiled_raw_max_error=err,
         peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),scientific_training_performed=False)
Path('/content/task41i_gpu_preflight.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
'''
with Path('/content/task41i_preflight_process.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,'-u','-c',code,str(root),request['revision']],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    result=proc.wait()
if result:raise RuntimeError(f'GPU preflight failed exit={result}; inspect saved log')
