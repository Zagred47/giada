"""Actual T4 diagnostic; all optimizer updates are rolled back, no scientific fit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path('/content/giada39h_preflight');ROOT=BASE/'repo'
BASE.mkdir(exist_ok=True)
if not ROOT.exists():
    subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(ROOT)],check=True,stdout=subprocess.DEVNULL)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='f22ae19da0ed238ce977df975b8957f28506c295'
import shutil
shutil.copy2('/content/task39h_relaxation_identification.py',ROOT/'src/giada_teacher/task39h_relaxation_identification.py')
shutil.copy2('/content/task39h_preregistration_v1.json',ROOT/'experiments/task39h_preregistration_v1.json')
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.giada_teacher import task39h_relaxation_identification as h
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
spec=json.loads((ROOT/h.CONFIG).read_text());quality=json.loads((ROOT/h.p.CONFIG).read_text())
print('[GIADA39h preflight] loading identical original FIT pools',flush=True)
arrays=[h.windows(ROOT/spec['dataset'],role,list(quality['families']),512,256,spec['data_seed']+off)[0] for role,off in [('fit',0),('fit_extra',1009)]]
anchors=h.c.anchor_pool(np.concatenate(arrays),spec['data_seed']+2,True)
cal=json.loads((ROOT/spec['parent']/'fit_loss_calibration.json').read_text());normal=json.loads((ROOT/spec['normalizers']).read_text())
scales=[torch.tensor(v,device='cuda',dtype=torch.float32) for v in (normal['gate'],normal['current'],cal['u_scales'],cal['log_z_scales'])]
cohorts=h.groups(torch,ROOT,spec,'cuda');static=[]
for group in cohorts:
    n=group['model'].a.shape[0]
    weights=[[.5,.5,cal['parent_weights'][j][0],cal['parent_weights'][j][1],0] for arm in group['definitions'] for j in range(3)]
    static.append(dict(batch=torch.tensor(arrays[0][:2].reshape(8,64,71),device='cuda')[None].expand(n,-1,-1,-1).clone(),
        anchor=torch.tensor(anchors[:64],device='cuda')[None].expand(n,-1,-1).clone(),ticks=torch.ones(n,64,1,device='cuda')*1000,weights=torch.tensor(weights,device='cuda')))
probe=h.gradient_probe(torch,cohorts[0],static[0],scales,0)
norm=np.asarray(probe['term_norms'])[:3];cos=np.asarray(probe['cosines'])[:3];main=static[0]['weights'].cpu().numpy()[:3,:4]*norm[:,:4]
lam=np.clip(np.sqrt(np.maximum(np.einsum('bi,bij,bj->b',main,cos[:,:4,:4],main),1e-30))/np.maximum(norm[:,4],1e-12),.001,1e6)
for group,s in zip(cohorts,static):
    for i,arm in enumerate(group['definitions']):
        if arm['auxiliary']:s['weights'][i*3:(i+1)*3,4]=torch.tensor(lam,device='cuda',dtype=torch.float32)
parity=[]
for group,s in zip(cohorts,static):
    value=h.optimizer_preflight(torch,group,s,scales,spec['seeds']);parity.append(value);print('[GIADA39h preflight] '+json.dumps(value),flush=True)
graph,clips,execution=h.capture_training(torch,cohorts,static,scales,spec)
result=dict(valid=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,training_updates=0,optimizer_updates_rolled_back=True,
    parity=parity,capture=execution,auxiliary_weights=lam.tolist(),parent_revision='f22ae19da0ed238ce977df975b8957f28506c295',
    source_sha256=h.p.digest(ROOT/'src/giada_teacher/task39h_relaxation_identification.py'),config_sha256=h.p.digest(ROOT/h.CONFIG))
Path('/content/task39h_gpu_preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
