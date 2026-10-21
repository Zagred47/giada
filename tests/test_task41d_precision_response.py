import json,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41d_precision_response as d
ROOT=Path(__file__).resolve().parents[1]
class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.spec,*_=d.contract(ROOT)
    def test_thresholds_and_design(self):
        self.assertEqual(self.spec['quality'],json.loads((ROOT/d.prior.CONFIG).read_text())['quality'])
        self.assertEqual(len(d.arms()),12);self.assertEqual(sum(a['panel']=='Ca' for a in d.arms()),4)
        self.assertEqual(sum(a['feedback'] for a in d.arms()),4)
    def test_separate_rng_support(self):
        a=d.anchors(5001,64,'uniform',0);b=d.anchors(5001,64,'sensitive',0)
        self.assertTrue(np.array_equal(a['eps'],b['eps']))
        self.assertFalse(np.array_equal(a['source_indices'],b['source_indices']))
        for support in ('uniform','sensitive'):
            a=d.anchors(5001,64,support,0);b=d.anchors(5001,64,support,0)
            for k in a:self.assertTrue(np.array_equal(a[k],b[k]))
    def test_basis_and_single_function(self):
        parent=d.prior.load_models(torch,ROOT,'cpu')['P110'];a=d.arms()[0];model=d.make_model(torch,parent,[a],[17,29,43],'cpu')
        v=torch.linspace(-95,55,101)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
        with torch.no_grad():
            before=parent.raw(v,ca);model.local_equilibrium.fill_(.0003);after=model.raw(v,ca)
        self.assertTrue(torch.equal(before[0][...,1:],after[0][...,1:]));self.assertTrue(torch.equal(before[1],after[1]))
        self.assertTrue((before[0][...,0]-after[0][...,0]).abs().max()>0)
        for shape in ('linear','smooth'):
            f=d.feature(torch,v,64,shape);self.assertTrue(torch.isfinite(f).all());self.assertTrue((f>=0).all());self.assertLess(float((f.sum(-1)-1).abs().max()),1e-5)
    def test_smooth_basis_continuous_derivative(self):
        v=torch.tensor([-37.4415],dtype=torch.float64,requires_grad=True)
        f=d.feature(torch,v,64,'smooth');w=torch.linspace(-.1,.1,64,dtype=torch.float64)
        deriv=torch.autograd.grad((f*w).sum(),v)[0]
        eps=1e-4;fd=((d.feature(torch,v.detach()+eps,64,'smooth')*w).sum()-(d.feature(torch,v.detach()-eps,64,'smooth')*w).sum())/(2*eps)
        self.assertAlmostEqual(deriv.item(),fd.item(),places=8)
    def test_composition_scanner_interface(self):
        from src.giada_teacher.task41d_evaluation import compose
        parent=d.prior.load_models(torch,ROOT,'cpu')['P110'];ca=d.make_model(torch,parent,[d.arms()[0]],[17,29,43],'cpu');nap=d.make_model(torch,parent,[d.arms()[4]],[17,29,43],'cpu')
        model=compose(torch,ca,nap,parent);self.assertEqual(model.a.shape,parent.a.shape)
        with torch.no_grad():
            ca.local_equilibrium.fill_(.001);nap.local_equilibrium.fill_(.002)
            v=torch.full((3,2),-45.);calcium=torch.full_like(v,1e-4);before=parent.raw(v,calcium);after=model.raw(v,calcium)
            self.assertTrue(torch.equal(after[0][...,:1],ca.raw(v,calcium)[0][...,:1]))
            self.assertTrue(torch.equal(after[0][...,8:9],nap.raw(v,calcium)[0][...,8:9]))
            self.assertTrue(torch.equal(after[1],before[1]))
if __name__=='__main__':unittest.main()
