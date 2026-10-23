import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41h_path_repair as h
from src.giada_teacher import task41h_evaluation as e
ROOT=Path(__file__).resolve().parents[1]

class Task41hTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.spec,cls.cfg,cls.ca,cls.base=h.contract(ROOT)
        cls.parent=h.frozen_controls(torch,ROOT,cls.spec,'cpu')['E_Q0']
    def test_factorial_and_acceptance(self):
        old=json.loads((ROOT/h.g.CONFIG).read_text())
        for key in ('quality','compartment_gates','cost'):self.assertEqual(self.spec[key],old[key])
        self.assertEqual(len(h.arms()),10)
        for axis,a,b in h.contrasts():
            aa=next(x for x in h.arms() if x['name']==a);bb=next(x for x in h.arms() if x['name']==b)
            self.assertEqual([k for k in ('history','aux_every','fit_count','basis') if aa[k]!=bb[k]],[axis])
    def test_sparse_features_forward_and_gradient(self):
        for n in (64,128):
            v=torch.cat((torch.linspace(-110,70,501,dtype=torch.float64),torch.linspace(-100,60,n))).repeat(3,1)
            w=torch.randn(3,128,dtype=torch.float64,requires_grad=True)
            dense=(h.d.feature(torch,v,n)*w[:,:n,None].transpose(1,2)).sum(-1)
            sparse=h.sparse_sum(torch,v,w,n)
            self.assertLess(float((dense-sparse).abs().max()),1e-12)
            a=torch.autograd.grad(dense.square().sum(),w)[0];b=torch.autograd.grad(sparse.square().sum(),w)[0]
            self.assertLess(float((a-b).abs().max()),1e-10)
    def test_precision_original_identity(self):
        v=torch.full((3,7),-45.);ca=torch.full_like(v,1e-4);g=torch.full((3,7,18),.1)
        mult=torch.ones(3,7,11);inj=torch.full_like(v,.004);syn=torch.zeros(3,7,4)
        original=h.t.make_core(torch,self.cfg,self.ca,self.base,100*np.pi,self.parent,h.t.FAMILIES['all'])(v,ca,g,mult,inj,syn)
        actual=e.precision_core(torch,self.parent,self.cfg,self.ca,self.base,100*np.pi)(v,ca,g,mult,inj,syn)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(original,actual)))
        for policy in ((0,1,0),(1,0,0),(1,1,0),(0,0,1)):
            values=e.precision_core(torch,self.parent,self.cfg,self.ca,self.base,100*np.pi,*policy)(v,ca,g,mult,inj,syn)
            self.assertTrue(all(torch.isfinite(a).all() for a in values))
    def test_path_training_and_protected_state(self):
        model=h.make_model(torch,self.parent,h.arms(),'cpu')
        tail=h.path_tail_data(torch,self.parent,self.spec,'cpu',count=8)
        before=model.raw(torch.full((30,4),-45.),torch.full((30,4),1e-4))
        with torch.no_grad():model.f_correction.normal_(0,.001);model.f_correction.mul_(model.active)
        after=model.raw(torch.full((30,4),-45.),torch.full((30,4),1e-4))
        self.assertTrue(torch.equal(before[1],after[1]));self.assertTrue(torch.equal(before[0][...,1:],after[0][...,1:]))
        loss=h.path_loss(torch,model,tail,torch.arange(4));grad=torch.autograd.grad(loss.sum(),model.f_correction)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertTrue((grad*(1-model.active)==0).all())
        self.assertLessEqual(float(tail['ticks'].max()+tail['prefix_ticks']),12000)
        # A constant prefix composed from its frozen coefficients reproduces the stored endpoint.
        recomposed=tail['prefix_initial'][0]+(tail['prefix_weight'][0]*torch.sigmoid(tail['prefix_u'][0])).sum(-1)
        self.assertLess(float((recomposed-tail['end_gates'][0,...,0]).abs().max()),1e-12)
if __name__=='__main__':unittest.main()
