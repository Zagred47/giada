import numpy as np
from src.giada_teacher.hh_family_transfer import CHANNELS,rates,update,data,targets,measurements,passes

def test_rates_domains_and_singularities():
    for c in CHANNELS:
        inf,tau=rates(c,np.r_[np.linspace(-155,95,501),[-66,-60,-38,-27]])
        assert np.isfinite(tau).all() and np.all(tau>0)
        assert np.all((inf>=0)&(inf<=1))

def test_convex_update_semigroup():
    for c in CHANNELS:
        i,t=rates(c,np.array([-100.,-60.,-20.,40.]))
        s=np.array([[0,1]]*4)
        a=update(s,i,t,.5);b=update(a,i,t,.5)
        assert np.max(abs(b-update(s,i,t,1)))<1e-14
        assert np.all((a>=0)&(a<=1))

def test_formula_metrics_and_split_disjointness():
    x=data(180401,100)
    inf,tau,y=targets(x)
    for n,c in enumerate(CHANNELS):
        assert passes(measurements(x,y[n],inf[n],tau[n],c))
    assert not set(map(tuple,x))&set(map(tuple,data(180201,100)))

def test_channel_specific_conductance_power():
    x=data(180401,100);i,t=rates('NaTa_t',x[:,0]);p=update(x[:,1:3],i,t,x[:,3])
    altered=p.copy();altered[:,0]*=.8
    m=measurements(x,altered,i,t,'NaTa_t')
    expected=np.sqrt(np.mean((altered[:,0]**3*altered[:,1]-p[:,0]**3*p[:,1])**2))
    assert m['open_rmse']==expected
