import numpy as np
import torch
from src.giada_teacher import sodium_family_composition as sodium


def test_canonical_shift_and_slow_nap():
    v=np.array([-90.,-70.,-40.,0.,50.])
    i,t=sodium.rates('NaTs2_t',v)
    ii,tt=sodium.rates('NaTa_t',v-6)
    np.testing.assert_allclose(i,ii,atol=1e-13)
    np.testing.assert_allclose(t,tt,atol=1e-13)
    i,t=sodium.rates('Nap_Et2',np.array(sodium.SEAMS))
    assert np.isfinite(t).all() and np.all(t>0)
    assert np.all(t[:,1]>100*t[:,0])
    values=sodium.state_grid(True)
    assert values.shape[1]==8
    i,t,truth=sodium.targets(values)
    assert np.all((truth>=0)&(truth<=1))
    metrics=sodium.measurements(values,truth,i,t)
    assert metrics['pair']['total_current_normalized_max']==0


def test_vectorized_adam_and_input_contract():
    torch.set_num_threads(1)
    for family in sodium.FAMILIES:
        assert sodium.equivalence(torch,16,family,[17,29,43],'cpu')['valid']
        model=sodium.model_factory(torch,16,family,[17,29,43])
        x=torch.tensor(sodium.data(5,16),dtype=torch.float32)[None].expand(3,-1,-1)
        pred,inf,tau=model(x)
        assert pred.shape==(3,3,16,2)
        altered=x.clone();altered[...,1:7]=1-x[...,1:7];altered[...,7]*=2
        _,ii,tt=model(altered)
        torch.testing.assert_close(ii,inf);torch.testing.assert_close(tt,tau)
        assert sum(p.numel() for p in model.parameters())//3=={'independent':1116,'shared_heads':508,'conditioned':420}[family]


def test_currents_attribute_each_channel():
    x=sodium.data(7,16);i,t,y=sodium.targets(x)
    for k,c in enumerate(sodium.CHANNELS):
        p=y.copy();p[k,:,0]=np.minimum(p[k,:,0]+.1,1)
        metrics=sodium.measurements(x,p,i,t)
        assert metrics[c]['current_rmse_ma_cm2']>0
        for other in sodium.CHANNELS:
            if other!=c:assert metrics[other]['current_rmse_ma_cm2']==0
        assert metrics['pair']['total_current_normalized_rmse']>0


def test_fast_slow_paths_and_long_holds():
    torch.set_num_threads(1)
    model=sodium.model_factory(torch,16,'shared_heads',[17])
    for dt in (.025,2.5):
        path=sodium.path_rollout(model,[17],torch,'cpu',dt)[0]
        assert path['finite'] and path['occupancy_violations']==0
        assert path['duration_ms']==dt*1000
        assert set(path['per_channel'])==set(sodium.CHANNELS)
    held=sodium.held_rollout(model,[17],torch,'cpu',[1,10])
    assert set(held[0])=={'1','10'}
