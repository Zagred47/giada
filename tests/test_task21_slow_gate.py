import numpy as np
from src.giada_teacher import slow_gate_transfer as slow
from src.giada_teacher import single_gate_transfer as engine


def test_rates_slow_and_bounded():
    i,t=slow.rates('short_dt',np.r_[np.linspace(-135,75,1001),-38,-17,-64.4])
    assert np.all((i>=0)&(i<=1)) and t.min()>300 and t.max()>2000
    assert np.isfinite(t).all()


def test_training_intervention_and_target_agreement():
    import torch
    values=slow.fit({'pool_size':128})
    descriptors=[('short_dt','transition_only',17),('multiscale_dt','transition_only',17)]
    batch=slow.training_batch(torch.tensor(values),torch.arange(128),descriptors)
    assert torch.equal(batch[0,:,:2],batch[1,:,:2])
    assert batch[0,:,2].max()<=1 and batch[1,:,2].max()>1000
    i,t=slow.rates('short_dt',values[:,0])
    targets=slow.fit_targets(values,descriptors,np.stack([i,i]),np.stack([t,t]))
    for n in (0,1):
        np.testing.assert_allclose(targets[n],engine.update(values[:,1],i,t,batch[n,:,2].numpy()))


def test_equivalence_and_grid():
    import torch
    d=[(c,o,s) for c in slow.CHANNELS for o in ('transition_only','rate_supervised') for s in (17,29,43)]
    model=slow.model_factory(torch,16,d)
    x=torch.tensor(slow.data(11,32),dtype=torch.float32)[None].expand(len(d),-1,-1)
    assert slow.equivalent(torch,model,16,d,x)['valid']
    assert slow.numerical_grid(513)[0]==-135
    assert sum(p.numel() for p in model.parameters())==len(d)*slow.parameter_count(16)


def test_rollout_floor_and_persistence():
    import torch
    d=[('multiscale_dt','rate_supervised',17)]
    m=slow.model_factory(torch,16,d)
    report=slow.rollout(m,d,slow.roll_inputs()[:8],[1,100],torch,'cpu')[0]
    assert report['100']['formula_f32_rmse']<1e-5
    assert report['100']['persistence_rmse']>report['1']['persistence_rmse']
