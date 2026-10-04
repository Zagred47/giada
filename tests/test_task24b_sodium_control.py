import numpy as np
import torch
from pathlib import Path
from src.giada_teacher import sodium_family_composition as s
from src.giada_teacher import sodium_control_diagnosis as d


def test_schedule_and_no_fresh_reuse():
    cfg=d.config(Path(__file__).resolve().parents[1])
    assert d.learning_rate('constant_bundle',60000,cfg)==.003
    assert d.learning_rate('decay_channel',15000,cfg)==.003
    assert abs(d.learning_rate('decay_channel',60000,cfg)-.0001)<1e-12
    assert set(cfg['fresh_seeds']).isdisjoint({240401,240402,240403,240404})
    a=s.state_grid(True);b=s.state_grid(True,node_offset=.71)
    assert not np.array_equal(a,b)


def test_per_channel_clipping_and_independent_adam():
    torch.set_num_threads(1)
    model=s.model_factory(torch,16,'independent',[17,29,43]);before={k:v.detach().clone() for k,v in model.named_parameters()}
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    for p in model.parameters():p.grad=torch.ones_like(p)*torch.tensor([.1,1,10])[None,:,*([None]*(p.ndim-2))]
    grads={k:p.grad.clone() for k,p in model.named_parameters()}
    d.clip(model,torch,'channel')
    torch.testing.assert_close(d.channel_norms(model,torch),torch.ones((3,3)))
    opt.step()
    for seed in range(3):
        for channel in range(3):
            params=[torch.nn.Parameter(v[seed,channel].clone()) for v in before.values()]
            single=torch.optim.Adam(params,lr=.003)
            for p,g in zip(params,grads.values()):p.grad=g[seed,channel].clone()
            torch.nn.utils.clip_grad_norm_(params,1.)
            single.step()
            for p,v in zip(params,model.parameters()):torch.testing.assert_close(p,v[seed,channel],atol=1e-7,rtol=1e-6)


def test_composed_hold_is_labelled_and_matches_direct_solver():
    torch.set_num_threads(1);model=s.model_factory(torch,16,'independent',[17])
    row=d.composed_hold(model,[17],torch,'cpu',241205)[0]
    assert set(row)=={'1000','10000'}
    assert all(x[c]['finite'] for x in row.values() for c in s.CHANNELS)


def test_runtime_channel_preflight():
    torch.set_num_threads(1)
    assert d.channel_step_equivalence(torch,16,[17,29,43],'cpu')['valid']
