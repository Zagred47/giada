import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.giada_teacher import task39_learnability as t

ROOT = Path(__file__).resolve().parents[1]


class Task39Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec = json.loads((ROOT/t.CONFIG).read_text())
        cls.data, cls.support = t.windows(ROOT/cls.spec['dataset'], 'fit', ['steps'], 16, 8, 4)

    def test_windows_are_real_contiguous_targets(self):
        self.assertEqual(self.data.shape, (16,8,71))
        np.testing.assert_array_equal(self.data[:,1:,2:20],self.data[:,:-1,20:38])
        self.assertEqual(self.support[0]['independent_trajectories'],4)

    def test_all_paired_optimizer_types(self):
        model = t.model_class(torch)(t.arms(), [17,29,43])
        batch = torch.tensor(self.data[:5])[None].expand(30,-1,-1,-1)
        result = t.preflight(torch,self.spec,batch,torch.ones(18),torch.ones(11),'cpu')
        self.assertTrue(result['valid'])

    def test_context_masks_and_pairing(self):
        defs = t.arms(); m = t.model_class(torch)(defs,[17,29,43])
        self.assertTrue(torch.equal(m.a[:3],m.a[3:6]))
        x = torch.rand(30,5,18); v=torch.full((30,5),-50.); ca=torch.full_like(v,.0001)
        a=m(v,ca,x); xx=x.clone();xx[...,1]=1-xx[...,1]; b=m(v,ca,xx)
        for k,d in enumerate(defs):
            if d['bias']=='factorized':self.assertTrue(torch.equal(a[k*3:k*3+3,:,0],b[k*3:k*3+3,:,0]))
        c=m(v,ca*2,x)
        self.assertTrue(torch.equal(a[...,:16],c[...,:16]))
        self.assertTrue(torch.equal(a[...,17:],c[...,17:]))

    def test_chunk_continuity_all_masks(self):
        definitions=t.arms(); m=t.model_class(torch)(definitions,[17,29,43])
        models=t.split_models(torch,m,definitions,[17,29,43],'cpu')
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            t.old.recurrence_preflight(Path(folder),self.spec,models,torch,'cpu')

    def test_teacher_force_uses_eight_targets(self):
        m=t.model_class(torch)([t.arms()[0]],[17])
        b=torch.tensor(self.data[:5])[None]
        tf=torch.ones(1,1,1,dtype=torch.bool)
        got=t.objective(torch,m,b,torch.ones(18),torch.ones(11),tf,torch.ones(1),torch.zeros(1))
        expected=sum(t.old.loss(torch,m,b[:,:,tick][0],torch.ones(18),torch.ones(11)) for tick in range(8))/8
        torch.testing.assert_close(got,expected)


if __name__=='__main__': unittest.main()
