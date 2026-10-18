import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.giada_teacher import task39b_architecture_diagnosis as t

ROOT=Path(__file__).resolve().parents[1]

class Task39bTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.defs=t.arms();cls.seeds=[17,29,43];cls.spec=json.loads((ROOT/t.CONFIG).read_text())
        cls.data,_=t.q.windows(ROOT/cls.spec['dataset'],'fit',['steps'],4,64,91)

    def test_matrix_and_equal_parameters(self):
        self.assertEqual(len(self.defs),12)
        counts=[]
        for d in self.defs:
            m=t.model_class(torch)([d],[17],list(range(8)));counts.append(sum(p.numel() for p in m.parameters()))
        self.assertEqual(len(set(counts)),1)
        m=t.model_class(torch)(self.defs,self.seeds,list(range(8)))
        self.assertTrue(torch.equal(m.a[:3],m.a[3:6]))

    def test_collocation_alignment_and_oracle(self):
        a=t.collocation(self.data,99)
        np.testing.assert_array_equal(a[:,1:,2:20],a[:,:-1,20:38])
        u,z=t.p.coefficients(a[...,0],a[...,1],.1)
        np.testing.assert_allclose(a[...,20:38],a[...,2:20]+z*(u-a[...,2:20]),atol=1e-7)
        np.testing.assert_array_equal(a[...,:2],self.data[...,:2])
        self.assertFalse(np.array_equal(a[:,0,2:20],self.data[:,0,2:20]))

    def test_all_optimizer_members(self):
        m=t.model_class(torch)(self.defs,self.seeds,list(range(8)))
        b=torch.tensor(self.data[:2,:16])[None].expand(36,-1,-1,-1)
        result=t.optimizer_preflight(torch,m,b,torch.ones(18),torch.ones(11),self.defs,self.seeds)
        self.assertTrue(result['valid']);self.assertEqual(result['members_tested'],36)

    def test_bounds_dependency_masks_and_chunk_continuity(self):
        m=t.model_class(torch)(self.defs,self.seeds,list(range(8)))
        x=torch.rand(36,4,18);v=torch.full((36,4),-50.);ca=torch.full_like(v,.0001)
        a=m(v,ca,x);changed=x.clone();changed[...,1:]=1-changed[...,1:]
        torch.testing.assert_close(a[...,0],m(v,ca,changed)[...,0],rtol=0,atol=0)
        b=m(v,ca*10,x);torch.testing.assert_close(a[...,:16],b[...,:16],rtol=0,atol=0)
        self.assertGreaterEqual(a.min(),0);self.assertLessEqual(a.max(),1)
        models=t.split_models(torch,m,self.defs,self.seeds,'cpu')
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:t.old.recurrence_preflight(Path(folder),self.spec,models,torch,'cpu')

    def test_short_reset_and_multiscale_anchor(self):
        defs=[self.defs[0],self.defs[4]];m=t.model_class(torch)(defs,[17],list(range(8)))
        b=torch.tensor(self.data[:2,:16])[None].expand(2,-1,-1,-1)
        sh,_,rel=t.flags(torch,defs,[17],'cpu')
        got=t.objective(torch,m,b,torch.ones(18),torch.ones(11),sh,rel)
        ms=t.split_models(torch,m,defs,[17],'cpu')
        tf=torch.zeros(1,1,1,dtype=torch.bool)
        expected=sum(t.q.objective(torch,ms[0],b[:1,:,s:s+8],torch.ones(18),torch.ones(11),tf,torch.ones(1),torch.zeros(1)) for s in (0,8))/2
        torch.testing.assert_close(got[0],expected[0])
        self.assertAlmostEqual(float(torch.sigmoid(torch.tensor(-3.))),float(torch.sigmoid(torch.tensor(-3.)+torch.tensor([0.,-3.,-6.,-9.])+t.multiscale_shift()).mean()),places=6)

    def test_selection_returns_all_members(self):
        m=t.model_class(torch)(self.defs,self.seeds,list(range(8)))
        spec=dict(self.spec,development_length=64)
        values=t.selection(torch,m,self.data,np.ones(18),np.ones(11),spec,'cpu')
        self.assertEqual(values.shape,(36,));self.assertTrue(np.isfinite(values).all())

    def test_evaluation_and_contrasts_smoke(self):
        m=t.model_class(torch)(self.defs,self.seeds,list(range(8)))
        models=t.split_models(torch,m,self.defs,self.seeds,'cpu')
        base=json.loads((ROOT/t.old.CONFIG).read_text());quality=json.loads((ROOT/t.p.CONFIG).read_text())
        spec=dict(self.spec,architectures=[d['name'] for d in self.defs])
        for key in ('baseline_parent','baseline_checkpoint'):spec[key]=base[key]
        quality['families']=['seams']
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            summary=t.old.evaluate(ROOT,Path(folder),spec,models,torch,'cpu',quality)
            rows=json.loads((Path(folder)/'recurrence_evaluation.json').read_text())
            self.assertEqual(len(summary),13)
            self.assertEqual(len(t.planned_contrasts(rows,self.defs)['contrasts']),16)

if __name__=='__main__':unittest.main()
