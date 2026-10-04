import json
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import joint_heterogeneous_comparison as j
ROOT=Path(__file__).resolve().parents[1]


def test_parameter_counts_and_matched_initialization():
    for width,ind_count,joint_count in ((16,1860,644),(32,6260,1780)):
        ind=j.model_factory(torch,width,'independent',[17,29,43])
        joint=j.model_factory(torch,width,'shared_heads',[17,29,43])
        assert sum(p.numel() for p in ind.parameters())//3 == ind_count
        assert sum(p.numel() for p in joint.parameters())//3 == joint_count
        for key in ('w1','b1','w2','b2'):
            assert torch.equal(getattr(ind,key)[:,0],getattr(joint,key))
        assert torch.equal(ind.wo,joint.wo)
        x=torch.tensor(j.h.data(269902,32,[.025,1,1000]),dtype=torch.float32)[None].expand(3,-1,-1)
        assert torch.allclose(ind(x)[0],joint(x)[0],atol=1e-7)


def test_vectorized_adam_equivalence():
    for family in ('independent','shared_heads'):
        assert j.equivalence(torch,8,family,[17,29,43],'cpu')['valid']


def test_evaluation_adapter_preserves_direct_outputs():
    x=torch.tensor(j.h.data(269903,24,[1.]),dtype=torch.float32)[None].expand(3,-1,-1)
    for family in ('independent','shared_heads'):
        model=j.model_factory(torch,8,family,[17,29,43])
        # Distinguish channel heads before testing routing.
        with torch.no_grad():model.bo.copy_(torch.arange(20).reshape(1,5,4).expand(3,-1,-1)*.05)
        direct=model(x);adapted=j.h.forward(j.as_pair(model,torch),x,torch)
        assert all(torch.equal(a,b) for a,b in zip(direct,adapted))
        assert ((direct[0]>=0)&(direct[0]<=1)).all()


def test_shared_gradient_probe_and_no_parameter_mutation():
    cfg=j.config(ROOT);model=j.model_factory(torch,8,'shared_heads',[17,29,43])
    vals=j.h.data(269904,24,[1.]);i,t,y=j.h.targets(vals)
    x,ti,tt,ty=[torch.tensor(v,dtype=torch.float32) for v in (vals,i,t,y)]
    state={k:v.clone() for k,v in model.state_dict().items()}
    probe=j.gradient_probe(model,x[None].expand(3,-1,-1),ti[None],tt[None],ty[None],cfg,torch)
    cosine=np.array(probe['cosines'])
    assert cosine.shape==(3,5,5) and np.isfinite(cosine).all()
    assert np.allclose(cosine,cosine.transpose(0,2,1))
    assert np.allclose(np.diagonal(cosine,axis1=1,axis2=2),1,atol=1e-6)
    assert all(torch.equal(value,state[key]) for key,value in model.state_dict().items())
    assert not probe['selection_eligible']


def test_prerequisite_fresh_and_scope():
    cfg=j.config(ROOT);assert len(j.validate_prerequisite(ROOT,cfg))==3
    parent=json.loads((ROOT/cfg['parent_contract']).read_text())
    assert set(cfg['fresh_seeds']).isdisjoint(parent['fresh_seeds'])
    assert cfg['gates']==parent['gates'] and cfg['current_gates']==parent['current_gates']
    assert cfg['families']==['independent','shared_heads']
    assert not any(key in cfg for key in ('arms','freeze','calcium_source','sodium_source'))
    for cell in json.loads((ROOT/'notebooks/26_roadmap_joint_vs_independent.ipynb').read_text())['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<cell>','exec')


def test_foreach_optimizer_does_not_couple_model_moments():
    families=('independent','shared_heads')
    models=[j.model_factory(torch,4,f,[17]) for f in families]
    singles=[j.model_factory(torch,4,f,[17]) for f in families]
    combined=torch.optim.Adam([p for m in models for p in m.parameters()],lr=.003,foreach=True)
    separate=[torch.optim.Adam(m.parameters(),lr=.003,foreach=True) for m in singles]
    x=torch.tensor(j.h.data(269905,24,[1.]),dtype=torch.float32)[None]
    for _ in range(2):
        combined.zero_grad();sum(m(x)[0].square().mean()*(k+1) for k,m in enumerate(models)).backward()
        for model in models:j._clip_per_seed(model,torch,1.)
        combined.step()
        for k,(model,opt) in enumerate(zip(singles,separate)):
            opt.zero_grad();(model(x)[0].square().mean()*(k+1)).backward();j._clip_per_seed(model,torch,1.);opt.step()
        assert all(torch.allclose(a,b,atol=1e-7,rtol=0) for m,s in zip(models,singles) for a,b in zip(m.parameters(),s.parameters()))
