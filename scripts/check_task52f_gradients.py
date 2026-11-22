import json
from pathlib import Path
import torch
from src.giada_teacher import task52f_encoder_credit as f
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1);p=f.plan(ROOT);p['support_protocol']['count_per_role']=2
net,cal,state=f.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');before=f.a.n.weights_hash(net)
with torch.no_grad():data=f.support_data(torch,p,'cpu');aut=f.d.coupled_data(torch,p,ROOT,'cpu')
cal=dict(cal,maxop=data['op'][:,:2].amax((0,1)).clamp_min(1e-30));E=len(net.members);ids=torch.zeros(E,2,dtype=torch.long);starts=torch.full_like(ids,2500)
f.teacher(torch,p,net,data,ids,starts,cal,torch.ones(E),True)
f.trajectory(torch,p,net,aut,torch.tensor([0,1]),torch.tensor([0,1500]),512,torch.ones(E),True)
vec=torch.cat([v.grad.reshape(E,-1) for v in net.parameters() if v.grad is not None],1);assert torch.isfinite(vec).all()
assert torch.allclose(vec[:3],vec[3:6],atol=1e-6,rtol=1e-5) and torch.allclose(vec[6:9],vec[9:12],atol=1e-6,rtol=1e-5)
assert net.enc.grad is None and net.eb.grad is None and before==f.a.n.weights_hash(net)
out=ROOT/'experiments/results/task52f_protocol_v1/full_gradient_preflight.json';out.write_text(json.dumps(dict(valid=True,CPU_only=True,teacher_ticks=512,autonomous_ticks=512,weights_unchanged=True,encoder_gradients_absent=True,finite_gradients=True,paired_same_horizon_gradients_match=True,short_long_gradient_distance=float((vec[:3]-vec[6:9]).norm())),indent=2));print(out.read_text())
