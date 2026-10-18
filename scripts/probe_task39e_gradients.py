"""FIT-only finite-weight gradient probes from immutable step30000."""
import io,json,zipfile
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task39e_tail_support_heads as e
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[1];spec=json.loads((root/e.CONFIG).read_text());quality=json.loads((root/e.p.CONFIG).read_text())
out=root/'experiments/results/task39e_instability_v1';data=[]
for role,extra in [('fit',0),('fit_extra',1009)]:
    a,_=e.q.windows(root/spec['dataset'],role,quality['families'],spec['windows_per_family'],256,spec['data_seed']+extra);data.append(a)
emp=e.c.anchor_pool(np.concatenate(data),spec['data_seed']+2);strat=e.c.anchor_pool(np.concatenate(data),spec['data_seed']+2,True)
normal=json.loads((root/spec['normalizers']).read_text());scales=[torch.tensor(a,dtype=torch.float32) for a in (normal['gate'],normal['current'],np.std(emp[:,71:89],0).clip(.05),np.std(np.log(emp[:,89:107].clip(1e-30)),0).clip(.5))]
with zipfile.ZipFile(root/'artifacts/task39e_retry1_external/recovery_final.zip') as z:
    weights=torch.load(io.BytesIO(z.read('checkpoint_30000.pt')),weights_only=True,map_location='cpu')
    loss_weights=torch.tensor(json.loads(z.read('fit_loss_calibration.json'))['per_seed_weights'])
cohorts=e.groups(torch,e.arms(),[17,29,43],'cpu')
for g in cohorts:g['model'].load_state_dict(weights[g['key']])
rng=np.random.default_rng(spec['data_seed']+99);rows=[]
for step in range(30100):
    idx=rng.integers(len(data[0]),size=2);extra=rng.random(2)<.5;ai=rng.integers(len(emp),size=64);di=rng.integers(6,size=64)
    if step<30000:continue
    batch=torch.tensor(np.where(extra[:,None,None],data[1][idx],data[0][idx]));tt=torch.tensor(np.asarray(spec['held_ticks'])[di],dtype=torch.float32)[None,:,None]
    for g in cohorts:
        n=g['model'].a.shape[0];s=dict(batch=(batch if g['long'] else batch.reshape(8,64,-1))[None].expand(n,-1,-1,-1),anchor=torch.where(g['flags'][:,1,None,None].bool(),torch.tensor(strat[ai])[None],torch.tensor(emp[ai])[None]),ticks=tt.expand(n,-1,-1),weights=torch.cat((torch.full((n,2),.5),loss_weights.repeat(n//3,1)),1))
        g['model'].zero_grad();loss=e.objective(torch,g,s,scales);loss.sum().backward()
        bad=[k for k,x in g['model'].named_parameters() if not torch.isfinite(x.grad).all()]
        rows.append(dict(step=step,cohort=g['key'],loss_finite=bool(torch.isfinite(loss).all()),bad_gradients=bad,max_loss=float(loss.max())))
        if bad:
            print(json.dumps(rows[-1]),flush=True)
            with torch.autograd.detect_anomaly():e.objective(torch,g,s,scales).sum().backward()
    if step%10==0:print('probe',step,flush=True)
(out/'fit_gradient_probes.json').write_text(json.dumps(rows,indent=2));print('ALL_FINITE',flush=True)
