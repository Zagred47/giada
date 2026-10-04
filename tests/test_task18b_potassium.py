import numpy as np
from src.giada_teacher.hh_potassium_diagnosis import pools,descriptors,score

def test_only_voltage_enriched_and_domains_preserved():
    p=pools(100)
    assert np.array_equal(p['uniform'][:,1:],p['tail_enriched'][:,1:])
    assert np.array_equal(p['uniform'][25:],p['tail_enriched'][25:])
    assert np.all((p['tail_enriched'][:25,0]>=-135)&(p['tail_enriched'][:25,0]<=-115))

def test_full_paired_factorial():
    ds=descriptors([17,29,43]);assert len(ds)==12
    assert len(set(ds))==12
    for a in {o for _,o,_ in ds}:assert [s for _,o,s in ds if o==a]==[17,29,43]

def test_score_checks_tail_and_max_error():
    m={'gate_rmse':0.,'gate_max_error':0.,'inf_rmse':0.,'log_tau_rmse':0.}
    tail=dict(m,gate_max_error=.03)
    assert score({'uniform':m,'negative_tail':tail})==3.
