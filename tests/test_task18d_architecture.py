import json
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher.hh_family_transfer import update
from src.giada_teacher.hh_potassium_architecture import initialize,architecture_factory

def test_affine_gate_error_bound_for_arbitrary_initial_states():
    rng=np.random.default_rng(42);i=rng.uniform(0,1,(100,2));t=rng.uniform(.1,100,(100,2));ip=rng.uniform(0,1,(100,2));tp=rng.uniform(.1,100,(100,2));dt=rng.uniform(.025,100,100)
    endpoint=np.maximum(abs(update(np.zeros_like(i),i,t,dt)-update(np.zeros_like(i),ip,tp,dt)),abs(update(np.ones_like(i),i,t,dt)-update(np.ones_like(i),ip,tp,dt)))
    for _ in range(20):
        x=rng.uniform(0,1,(100,2));assert np.all(abs(update(x,i,t,dt)-update(x,ip,tp,dt))<=endpoint+1e-14)

def test_same_parent_function_and_independent_adam_for_every_architecture():
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'experiments/task18d_potassium_architecture.json').read_text());model,parent,ds=initialize(torch,cfg,root/'experiments/fixtures/task18c_frozen_parent.zip','cpu')
    x=torch.tensor(np.c_[np.linspace(-135,75,64),np.full(64,.3),np.full(64,.7),np.full(64,25)],dtype=torch.float32)[None].expand(9,-1,-1)
    initial={k:v.clone() for k,v in model.state_dict().items()};p=model(x)[0]
    assert sum(v.numel() for v in parent.parameters())//3==1252
    assert sum(v.numel() for v in model.parameters())//9==1317
    ref=parent(x[:3])[0]
    for j in range(3):assert torch.max(abs(p[j*3:(j+1)*3]-ref))<1e-5
    from src.giada_teacher.joint_gate_symmetric_confirmation import _clip_per_seed
    opt=torch.optim.Adam(model.parameters(),lr=.003);p.square().mean((1,2)).sum().backward();_clip_per_seed(model,torch,1);opt.step()
    for n in (0,3,6):
        single=architecture_factory(torch,32,[ds[n]]);single.load_state_dict({k:v[n:n+1] for k,v in initial.items()});so=torch.optim.Adam(single.parameters(),lr=.003);single(x[n:n+1])[0].square().mean().backward();_clip_per_seed(single,torch,1);so.step()
        for k,v in model.named_parameters():assert torch.max(abs(v[n:n+1]-single.state_dict()[k]))<1e-5
