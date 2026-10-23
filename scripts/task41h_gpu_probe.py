"""Actual pinned GPU checks before prospective registration and launch."""
import json,subprocess,sys,time
from pathlib import Path
request=json.loads(Path('/content/task41h_preflight_request.json').read_text())
root=Path('/content/giada41h_preflight_'+str(time.time_ns()))
for cmd in (['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],['git','-C',str(root),'fetch','--depth','1','origin',request['revision']],['git','-C',str(root),'checkout','--detach','FETCH_HEAD']):
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
code=r'''
import os,sys,json
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
import torch,numpy as np
from src.giada_teacher import task41h_path_repair as h
from src.giada_teacher import task41h_evaluation as e
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
spec,cfg,ca,base=h.contract(root);out=root/'probe';out.mkdir()
r=h.preflight(torch,root,out,'cuda',spec)
parent=h.frozen_controls(torch,root,spec,'cuda')['E_Q0'];model=h.make_model(torch,parent,h.arms(),'cuda')
data,_=h.atomic_data(torch,parent,spec,'cuda',count=16);ids=torch.arange(8,device='cuda')[None].expand(30,-1)
norm=h.f.atomic_terms(torch,model.f_correction,model.active,data,ids).detach().clamp_min(1e-12)
opt=h.g.Adam(torch,model.f_correction,h.arms(),spec);graph=h.g.capture(torch,model,opt,data,ids,norm,data,ids);graph.replay()
tail=h.path_tail_data(torch,parent,spec,'cuda',count=8)
h.f.ee.differentiable_constants();fb,support=h.f.feedback_data(torch,spec,cfg,ca,base,'cuda',small=True)
view=h.FeedbackView(torch,parent,model)
diag=h.auxiliary_step(torch,model,opt,data,ids,norm,data,ids,tail,torch.arange(4,device='cuda'),view,fb,support,torch.arange(2,device='cuda'),cfg,ca,base,16)
assert torch.isfinite(model.f_correction).all()
v=torch.full((3,4),-45.,device='cuda');cc=torch.full_like(v,1e-4);gg=torch.full((3,4,18),.1,device='cuda');mm=torch.ones(3,4,11,device='cuda');inj=torch.zeros_like(v);syn=torch.zeros(3,4,4,device='cuda')
with torch.no_grad():
    wanted=h.t.make_core(torch,cfg,ca,base,100*np.pi,parent,h.t.FAMILIES['all'])(v,cc,gg,mm,inj,syn)
    actual=e.precision_core(torch,parent,cfg,ca,base,100*np.pi)(v,cc,gg,mm,inj,syn)
    assert all(torch.equal(a,b) for a,b in zip(actual,wanted))
    for policy in ((0,1,0),(1,0,0),(1,1,0),(0,0,1)):
        values=e.precision_core(torch,parent,cfg,ca,base,100*np.pi,*policy)(v,cc,gg,mm,inj,syn)
        assert all(torch.isfinite(a).all() for a in values)
r.update(revision=sys.argv[2],gpu=torch.cuda.get_device_name(),auxiliary_update_executed=True,point_graph_executed=True,precision_seams_executed=True,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),scientific_training_performed=False)
Path('/content/task41h_gpu_preflight.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
'''
with Path('/content/task41h_preflight_process.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,'-u','-c',code,str(root),request['revision']],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    result=proc.wait()
if result:raise RuntimeError(f'GPU preflight failed exit={result}; see saved log')
