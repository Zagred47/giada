import json
from pathlib import Path

import numpy as np
import torch

from src.giada_teacher import current_supervision_comparison as c
from src.giada_teacher import joint_heterogeneous_comparison as j

ROOT=Path(__file__).resolve().parents[1]


def test_parent_result_pinned_and_notebook_compiles():
    cfg=c.config(ROOT)
    widths,hashes=c.verify_parent(ROOT,cfg)
    assert widths=={'independent':32,'shared_heads':16}
    assert hashes==cfg['parent_result_hashes']
    assert set(cfg['supervision_arms'])=={'none','individual','total','both'}
    notebook=json.loads((ROOT/'notebooks/27_roadmap_current_supervision.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<task27-cell>','exec')


def test_current_losses_do_not_confuse_cancellation_with_channel_accuracy():
    panels=c.panel_tensors(torch,'cpu')
    truth=torch.full((3,5,8,2),.5)
    pred=truth.clone();pred[:,0,:,0]=.7;pred[:,2,:,0]=.7
    voltage=torch.full((3,8),70.)
    individual,total=c.current_terms(pred,truth,voltage,panels,torch)
    assert individual.shape==total.shape==(3,)
    assert torch.isfinite(individual).all() and torch.isfinite(total).all()
    assert (individual>0).all() and (total>0).all()
    assert torch.allclose(individual,individual[:1].expand_as(individual))


def test_all_four_arms_have_identical_initialization_and_finite_gradients():
    cfg=c.config(ROOT)
    vals=j.h.data(270991,32,[.025,1,1000])
    inf,tau,target=j.h.targets(vals)
    x=torch.tensor(vals,dtype=torch.float32)[None].expand(3,-1,-1)
    ti=torch.tensor(inf,dtype=torch.float32)[None]
    tt=torch.tensor(tau,dtype=torch.float32)[None]
    ty=torch.tensor(target,dtype=torch.float32)[None]
    panel=c.panel_tensors(torch,'cpu')
    for family,width in [('independent',8),('shared_heads',8)]:
        models=[j.model_factory(torch,width,family,[17,29,43]) for _ in cfg['supervision_arms']]
        for model in models[1:]:
            assert all(torch.equal(a,b) for a,b in zip(models[0].parameters(),model.parameters()))
        values=[]
        for arm,model in zip(cfg['supervision_arms'],models):
            loss=c.arm_loss(model,arm,x,ti,tt,ty,cfg,panel,torch)
            assert loss.shape==(3,) and torch.isfinite(loss).all()
            loss.sum().backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            values.append(loss.detach().numpy())
        assert np.all(values[1]>=values[0])
        assert np.all(values[2]>=values[0])
        assert np.all(values[3]>=values[1])
        assert np.all(values[3]>=values[2])
