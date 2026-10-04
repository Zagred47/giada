import numpy as np
import torch
from src.giada_teacher import controlled_gate_sharing as sharing


def test_counts_shapes_identity():
    counts=[]
    for f in sharing.FAMILIES:
        m=sharing.model_factory(torch,16,f,[17,29,43]);x=torch.tensor(sharing.data(1,8),dtype=torch.float32)[None].expand(3,-1,-1)
        p,i,t=m(x);assert p.shape==(3,3,8)
        assert torch.isfinite(p).all() and torch.all((p>=0)&(p<=1))
        counts.append(sum(a.numel() for a in m.parameters())//3)
        swapped=m(x,[1,2,0])[0]
        if f!='conditioned':torch.testing.assert_close(swapped,p[:,[1,2,0]])
    assert counts==[1014,406,386]


def test_vectorized_adam():
    torch.set_num_threads(1)
    for f in sharing.FAMILIES:
        assert sharing.equivalence(torch,16,f,[17,29,43],'cpu')['valid']


def test_teacher_targets_and_rollout():
    values=sharing.data(1,32)
    for c in sharing.CHANNELS:
        i,t=sharing.rates(c,values[:,0]);assert np.isfinite(t).all() and (t>0).all()
    model=sharing.model_factory(torch,16,'shared_heads',[17])
    out=sharing.rollout(model,np.c_[values[:4,:2],np.ones(4)],[17],torch,'cpu',[1,10])
    assert set(out[0])==set(sharing.CHANNELS)


def test_native_subprocess_sets_repository_cwd(tmp_path,monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    calls=[]
    def fake_run(command,**kwargs):
        calls.append(kwargs['cwd'])
        for label in ('single','slow'):
            sub=tmp_path/('oracle_'+label)
            if sub.is_dir():(sub/'native_audit.json').write_text(json.dumps({'valid':True}))
    monkeypatch.setattr(subprocess,'run',fake_run)
    assert sharing.native_audit('teacher',tmp_path)['valid']
    assert calls==[Path(sharing.__file__).resolve().parents[2]]*2
