import json
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import heterogeneous_mechanism_composition as h

ROOT=Path(__file__).resolve().parents[1]


def test_state_contract_and_formula_floor():
    values=h.data(250001,64,[.025,1,1000]);i,t,y=h.targets(values)
    assert y.shape==(5,64,2) and np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
    metrics=h.measurements(values,y,i,t)
    assert all(metrics[c]['gate_rmse']==0 for c in h.CHANNELS)
    assert metrics['currents']['total_current_normalized_max']==0


def test_cancellation_cannot_hide_individual_errors():
    v=np.full(4,65.);truth=np.zeros((5,4,2));pred=truth.copy()
    truth[0,:,0]=truth[2,:,0]=1;pred[:]=truth;pred[0,:,1]=.01;pred[2,:,1]=.055
    metrics=h.current_metrics(v,pred,truth)
    assert metrics['individual_current_normalized_rmse']>.01
    assert len(metrics['panels'])==20
    # At nominal reversal a sodium error can cancel calcium exactly.
    assert metrics['panels'][0]['total_current_normalized_rmse']<1e-12
    assert metrics['total_current_normalized_rmse']>0


def test_adapter_matches_separate_modules_and_seeds():
    cfg=json.loads((ROOT/'experiments/task25_heterogeneous_mechanisms.json').read_text());models,_=h.load_frozen(ROOT,cfg,torch,'cpu')
    pair=h.bundle(models,'independent');x=torch.tensor(h.data(9,24,[1.]),dtype=torch.float32)[None].expand(3,-1,-1)
    p,i,t=h.forward(pair,x,torch)
    assert p.shape==(3,5,24,2)
    assert torch.equal(p[:,:2],pair[0](torch.cat((x[...,:5],x[...,-1:]),-1))[0])
    assert torch.equal(p[:,2:],pair[1](torch.cat((x[...,:1],x[...,5:11],x[...,-1:]),-1))[0])
    assert not any(v.requires_grad for m in pair for v in m.parameters())
    for n,seed in enumerate(cfg['seeds']):
        cp=[h.ca.model_factory(torch,16,'independent',[seed]),h.na.model_factory(torch,16,'independent',[seed])]
        for single,model in zip(cp,pair):single.load_state_dict({k:v[n:n+1] for k,v in model.state_dict().items()})
        assert float((h.forward(cp,x[n:n+1],torch)[0]-p[n:n+1]).detach().abs().max())<1e-6


def test_structured_extrema_and_wrong_routing():
    cfg={'extrema_nodes':3,'extrema_node_offset':.83,'dt_values_ms':[1.]};v=h.extrema(cfg)
    assert len(np.unique(v[:,1:11],axis=0))==22
    i,t,y=h.targets(v);state=v[:,1:11].reshape(-1,5,2).transpose(1,0,2)
    wrong=h.update(state,i[[1,2,3,4,0]],t[[1,2,3,4,0]],v[:,-1])
    assert np.sqrt(np.mean((wrong-y)**2))>.01
