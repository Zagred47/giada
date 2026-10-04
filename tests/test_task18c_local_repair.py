import json
from pathlib import Path
import numpy as np
from src.giada_teacher.hh_potassium_local_repair import arms,pools,initialize

def test_pairing_and_preserved_tail():
    p=pools(100);a,b=p['tail_only'],p['tail_and_kink']
    assert np.array_equal(a[:25],b[:25])
    assert np.array_equal(a[:,1:],b[:,1:])
    assert np.array_equal(a[50:],b[50:])
    assert np.all((b[25:50,0]>=-65)&(b[25:50,0]<=-55))

def test_cloned_parent_and_factorial():
    import torch
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];c=json.loads((root/'experiments/task18c_potassium_local_repair.json').read_text())
    assert len(arms(c))==8
    model,parent,ds=initialize(torch,c,root/'experiments/fixtures/task18b_frozen_parent.zip','cpu');assert len(ds)==24
    for k,v in model.state_dict().items():
        for n in range(8):assert torch.equal(v[n*3:(n+1)*3],parent.state_dict()[k])
