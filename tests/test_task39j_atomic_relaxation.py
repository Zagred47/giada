import unittest
from pathlib import Path
import json
import torch
from src.giada_teacher import task39j_atomic_relaxation as j

class AtomicPrecision(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model=j.model_class(torch)('local',[17,29,43])
        self.data=j.tensors(torch,j.anchors(42,16),'cpu')
    def test_labels_and_powers(self):
        self.assertEqual([int(j.p.ionic.POWERS[k]) for k in (4,7)],[3,2])
        self.assertTrue(torch.isfinite(self.data['u']).all())
    def test_gradient_finite(self):
        loss=j.loss_terms(torch,self.model,self.data,torch.arange(16))
        self.assertEqual(loss.shape,(3,3));loss.sum().backward()
        self.assertTrue(torch.isfinite(self.model.correction.grad).all())
    def test_zero_and_channel_preservation(self):
        v=self.data['v'][None].expand(3,-1);ca=self.data['ca'][None].expand(3,-1)
        before=self.model.raw(v,ca)
        with torch.no_grad():self.model.correction.fill_(.01)
        after=self.model.raw(v,ca);other=[k for k in range(18) if k not in j.GATES]
        for a,b in zip(before,after):self.assertTrue(torch.equal(a[...,other],b[...,other]))
    def test_adam_member_independence(self):
        initial={k:v.clone() for k,v in self.model.state_dict().items()}
        weights=torch.ones(3,3,dtype=torch.float64);idx=torch.arange(16)
        opt=torch.optim.Adam([self.model.correction],lr=1e-4)
        j.step(torch,self.model,opt,self.data,idx,weights)
        for member,seed in enumerate([17,29,43]):
            single=j.model_class(torch)('local',[seed])
            single.load_state_dict({k:v if k=='gbar' else v[member:member+1] for k,v in initial.items()})
            one=torch.optim.Adam([single.correction],lr=1e-4)
            j.step(torch,single,one,self.data,idx,weights[member:member+1])
            self.assertLess(float((single.correction-self.model.correction[member:member+1]).abs().max()),1e-6)
    def test_roles_disjoint(self):
        a=j.anchors(390701,1024);b=j.anchors(390702,256)
        self.assertFalse(set(a['v'].tolist()) & set(b['v'].tolist()))

if __name__=='__main__':unittest.main()
