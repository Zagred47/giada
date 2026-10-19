"""Isolated actual CUDA parity/capture checks; every optimizer update reset."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,shutil,subprocess,sys
from pathlib import Path
BASE=Path('/content/giada39i_preflight');ROOT=BASE/'repo';BASE.mkdir(exist_ok=True)
if not ROOT.exists():subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(ROOT)],check=True,stdout=subprocess.DEVNULL)
for src,dst in [('task39i_coupled_relaxation.py','src/giada_teacher/task39i_coupled_relaxation.py'),('task39i_preregistration_v1.json','experiments/task39i_preregistration_v1.json')]:shutil.copy2('/content/'+src,ROOT/dst)
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.giada_teacher import task39i_coupled_relaxation as i
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
spec=json.loads((ROOT/i.CONFIG).read_text());quality=json.loads((ROOT/i.p.CONFIG).read_text());device='cuda'
arrays=[i.q.windows(ROOT/spec['dataset'],role,list(quality['families']),spec['windows_per_family'],256,spec['data_seed']+off)[0] for role,off in [('fit',0),('fit_extra',1009)]]
anchors=i.c.anchor_pool(np.concatenate(arrays),spec['data_seed']+2,True)
cal=json.loads((ROOT/spec['parent']/'fit_loss_calibration.json').read_text());normal=json.loads((ROOT/spec['normalizers']).read_text())
scales=[torch.tensor(x,device=device,dtype=torch.float32) for x in (normal['gate'],normal['current'],cal['u_scales'],cal['log_z_scales'])]
cohorts=i.groups(torch,ROOT,spec,device);parent=i.load_parents(torch,ROOT,spec,device,torch.float32)[0]
transfer=i.transfer_sk(torch,cohorts,parent,spec,device);static=[]
batch=i.labelled(arrays[0][:2]).reshape(8,64,89);del arrays
for group in cohorts:
    n=group['model'].a.shape[0]
    static.append(dict(batch=torch.tensor(batch,device=device)[None].expand(n,-1,-1,-1).clone(),
        anchor=torch.tensor(anchors[:64],device=device)[None].expand(n,-1,-1).clone(),
        ticks=torch.ones(n,64,1,device=device)*1000,weights=torch.ones(n,7,device=device)))
parity=[]
for group,s in zip(cohorts,static):
    value=i.optimizer_preflight(torch,group,s,scales,spec['seeds']);parity.append(value);print('[GIADA Task39i] GPU optimizer '+json.dumps(value),flush=True)
graph,clips,execution=i.capture(torch,cohorts,static,scales,spec)
report=dict(valid=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,training_updates=0,optimizer_updates_rolled_back=True,
    parity=parity,capture=execution,sk_transfer=transfer,synthetic=i.synthetic_preflight(torch,device),
    source_sha256=i.p.digest(ROOT/'src/giada_teacher/task39i_coupled_relaxation.py'),config_sha256=i.p.digest(ROOT/i.CONFIG))
Path('/content/task39i_gpu_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
