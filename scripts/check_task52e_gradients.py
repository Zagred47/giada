import copy,json
from pathlib import Path
import torch
from src.giada_teacher import task52e_protected_currents as e
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1);p=e.plan(ROOT);p['support_protocol']['count_per_role']=2
net,cal,state=e.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');before=e.a.n.weights_hash(net)
with torch.no_grad():data=e.support_data(torch,p,'cpu');aut=e.d.coupled_data(torch,p,ROOT,'cpu')
cal=dict(cal,maxop=data['op'][:,:2].amax((0,1)).clamp_min(1e-30))
E=len(net.members);ids=torch.zeros(E,2,dtype=torch.long);starts=torch.full_like(ids,2500)
e.teacher(torch,p,net,data,ids,starts,cal,torch.ones(E),True)
e.trajectory(torch,p,net,aut,torch.tensor([0,1]),torch.tensor([0,1500]),512,torch.ones(E),True)
assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
assert net.enc.grad is None and net.eb.grad is None and before==e.a.n.weights_hash(net)
out=ROOT/'experiments/results/task52e_protocol_v1/full_gradient_preflight.json';out.write_text(json.dumps(dict(valid=True,CPU_only=True,teacher_ticks=128,autonomous_ticks=512,weights_unchanged=True,encoder_gradients_absent=True,finite_gradients=True),indent=2))
print(out.read_text())
