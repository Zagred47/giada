"""Nontraining diagnostic on the retained failed39g runtime."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json
from pathlib import Path
import numpy as np
import torch
BASE=Path('/content/giada_task39g_8f8fa556a1d9');ROOT=BASE/'giada';sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39g_objective_horizon as g
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
spec=json.loads((ROOT/g.CONFIG).read_text());cal=json.loads((BASE/'artifacts/fit_loss_calibration.json').read_text());norm=json.loads((ROOT/spec['normalizers']).read_text())
pool,_=g.windows(ROOT/spec['dataset'],'fit',['steps'],512,256,spec['data_seed'])
dummy=np.zeros((512*12*2,1,71),dtype='float32');dummy[:512,0]=pool[:,0,:71]
anchor=g.c.anchor_pool(dummy,spec['data_seed']+2,True)[:64]
cohort=g.groups(torch,ROOT,spec,'cuda')[0];model=cohort['model'];n=model.a.shape[0]
s=dict(batch=torch.tensor(pool[:2].reshape(8,64,126),device='cuda')[None].expand(n,-1,-1,-1).clone(),anchor=torch.tensor(anchor,device='cuda')[None].expand(n,-1,-1).clone(),ticks=torch.full((n,64,1),1000.,device='cuda'),weights=torch.tensor(cal['weights']['h64'],device='cuda'))
scales=[torch.tensor(a,device='cuda',dtype=torch.float32) for a in (norm['gate'],norm['current'],cal['u_scales'],cal['log_z_scales'])]
g.objective(torch,cohort,s,scales).sum().backward();raw={k:v.grad.clone() for k,v in model.named_parameters()};g.old.clip(torch,model);batched={k:v.grad.clone() for k,v in model.named_parameters()};rows=[]
for j in range(n):
    iso=g.f.model_class(torch)('wide',[spec['seeds'][j%3]]).cuda();iso.load_state_dict({k:v.clone() if k=='gbar' else v[j:j+1].clone() for k,v in model.state_dict().items()})
    gg=dict(cohort,model=iso,flags=cohort['flags'][j:j+1]);ss={k:v[j:j+1] for k,v in s.items()}
    g.objective(torch,gg,ss,scales).sum().backward();rawerror=max(float((v.grad[0]-raw[k][j]).abs().max()) for k,v in iso.named_parameters());g.old.clip(torch,iso)
    errors={str(eps):max(float((.001*(v.grad[0]/(v.grad[0].abs()+eps)-batched[k][j]/(batched[k][j].abs()+eps))).abs().max()) for k,v in iso.named_parameters()) for eps in (1e-8,1e-6,1e-5,1e-4,1e-3)}
    rows.append(dict(member=j,arm=cohort['definitions'][j//3]['name'],seed=spec['seeds'][j%3],raw_gradient_max_error=rawerror,adam_first_step_error=errors))
    print(json.dumps(rows[-1]),flush=True)
(BASE/'parity_diagnosis.json').write_text(json.dumps(dict(rows=rows,training_updates=0),indent=2))
