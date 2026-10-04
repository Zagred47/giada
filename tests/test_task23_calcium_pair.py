import numpy as np
import torch
from src.giada_teacher import calcium_pair_composition as pair


def test_calva_temperature_shift_and_limits():
    inf,tau=pair.rates('Ca_LVAst',np.array([-90.,-40.]))
    assert abs(inf[0,1]-.5)<1e-14
    assert abs(inf[1,0]-.5)<1e-14
    assert np.all((tau>0)&(inf>=0)&(inf<=1))
    values=pair.state_grid(True);i,t,target=pair.targets(values)
    assert np.isfinite(target).all() and np.all((target>=0)&(target<=1))
    m=pair.measurements(values,target,i,t)
    assert m['pair']['total_current_normalized_max']==0


def test_vectorized_adam_and_information_contract():
    torch.set_num_threads(1)
    for f in pair.FAMILIES:
        assert pair.equivalence(torch,16,f,[17,29,43],'cpu')['valid']
        model=pair.model_factory(torch,16,f,[17,29,43])
        values=pair.data(5,16);x=torch.tensor(values,dtype=torch.float32)[None].expand(3,-1,-1)
        pred,inf,tau=model(x)
        assert pred.shape==(3,2,16,2)
        assert torch.all((pred>=0)&(pred<=1))
        altered=x.clone();altered[...,1:5]=1-x[...,1:5];altered[...,5]*=2
        _,ii,tt=model(altered)
        torch.testing.assert_close(ii,inf);torch.testing.assert_close(tt,tau)
        assert sum(p.numel() for p in model.parameters())//3=={'independent':744,'shared_heads':440,'conditioned':404}[f]


def test_currents_do_not_hide_component_error():
    x=pair.data(7,16);i,t,y=pair.targets(x)
    p=y.copy();p[0,:,0]=np.minimum(p[0,:,0]+.1,1)
    m=pair.measurements(x,p,i,t)
    assert m['Ca_HVA']['current_rmse_ma_cm2']>0
    assert m['Ca_LVAst']['current_rmse_ma_cm2']==0
    assert m['pair']['individual_current_normalized_rmse']>0
    assert m['pair']['total_current_normalized_rmse']>0


def test_imposed_path_and_hold_rollout_shapes():
    torch.set_num_threads(1)
    model=pair.model_factory(torch,16,'shared_heads',[17])
    path=pair.path_rollout(model,[17],torch,'cpu')
    assert path[0]['finite'] and path[0]['occupancy_violations']==0
    assert len(path[0]['per_path_gate_rmse'])==5
    held=pair.held_rollout(model,[17],torch,'cpu',[1,10])
    assert set(held[0])=={'1','10'}
