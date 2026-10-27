import copy,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41k_learning_alignment as k

class LearningAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.root=Path(__file__).resolve().parents[1]
        cls.spec,cls.cfg,cls.calcium,cls.base=k.contract(cls.root)
        cls.cc,cls.models,cls.data,_,cls.paths,_,cls.fb,_,_=k.setup(torch,cls.root,None,'cpu',cls.spec,cls.cfg,cls.calcium,cls.base,True)
        cls.physics=(cls.cfg,cls.calcium,cls.base)
    def test_unchanged_contract(self):
        old=k.j.contract(self.root)[0]
        for key in ('quality','cost','compartment_gates'):self.assertEqual(self.spec[key],old[key])
        self.assertEqual(sum(a['task']=='repair' for a in k.arms()),6)
    def test_initial_pairing_and_full_trainability(self):
        v=torch.full((3,8),-60.);ca=torch.full_like(v,1e-4)
        for a in k.arms()[:6]:
            m=self.models[a['name']]
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(m.raw(v,ca),self.cc['H_nominee'].raw(v,ca))))
            self.assertTrue(all(p.requires_grad for p in m.parameters()))
        a,b=[self.models[x] for x in ('K_copy_coeff','K_copy_dynamic')]
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
    def test_parent_targets_zero_and_canonical_nonzero(self):
        pi=torch.arange(2);fi=torch.arange(2);parent=self.models['K_point']
        loss,_=k.dynamic_losses(torch,parent,self.paths,self.fb,pi,fi,self.physics,True)
        self.assertLess(float(loss.max()),1e-10)
        canonical,_=k.dynamic_losses(torch,parent,self.paths,self.fb,pi,fi,self.physics)
        self.assertGreater(float(canonical.max()),1e-12)
    def test_path_objective_unchanged(self):
        ids=torch.arange(2);m=self.models['K_point']
        expected=k.j.path_loss(torch,m,self.paths,ids)
        actual=k.path_errors(torch,k.path_prediction(torch,m,self.paths,ids),self.paths,ids,reduce=True)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-10),(actual,expected))
    def test_paired_sampling(self):
        cdf=k.targeted_cdf(np.arange(16));self.assertAlmostEqual(cdf[-1],1.)
        a,b=k.paired_ids(torch,np.random.default_rng(1),16,10000,cdf,'cpu')
        self.assertGreater(float((b>=12).float().mean()),.60)
        self.assertLess(float((a>=12).float().mean()),.28)
    def test_feedback_objective_unchanged(self):
        ids=torch.arange(2);m=self.models['K_point']
        expected=k.j.feedback_loss(torch,m,self.fb,ids,self.physics)
        actual=k.feedback_errors(torch,k.feedback_prediction(torch,m,self.fb,ids,self.physics),self.fb,ids).mean(1)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-10))
    def test_compression_ignores_canonical_labels(self):
        paths=dict(self.paths,target=self.paths['target']+1000)
        fb=dict(self.fb,target=self.fb['target']+1000)
        m=self.models['K_copy_dynamic'];ids=torch.arange(2)
        a,_=k.dynamic_losses(torch,m,self.paths,self.fb,ids,ids,self.physics,True)
        b,_=k.dynamic_losses(torch,m,paths,fb,ids,ids,self.physics,True)
        self.assertTrue(torch.equal(a,b))
    def test_all_objectives_finite_resume_equal(self):
        models=copy.deepcopy(self.models);opts={n:torch.optim.Adam(k.j.parameters(m),lr=1e-5) for n,m in models.items()}
        ids=torch.arange(2);pair=(ids,ids)
        k.update(torch,models,opts,self.data,ids,self.paths,self.fb,pair,pair,self.physics,16)
        copies=copy.deepcopy(models);co={n:torch.optim.Adam(k.j.parameters(m),lr=1e-5) for n,m in copies.items()}
        for n in co:co[n].load_state_dict(copy.deepcopy(opts[n].state_dict()))
        for ms,os in ((models,opts),(copies,co)):
            k.update(torch,ms,os,self.data,ids,self.paths,self.fb,pair,pair,self.physics,17);k.j.finite(torch,ms,os)
        self.assertTrue(all(torch.equal(a,b) for n in models for a,b in zip(models[n].parameters(),copies[n].parameters())))
    def test_retention_mask_zero_disables_replay(self):
        ids=torch.arange(2);paths=dict(self.paths,retain_mask=torch.zeros_like(self.paths['retain_mask']));fb=dict(self.fb,retain_mask=torch.zeros_like(self.fb['retain_mask']))
        _,r=k.dynamic_losses(torch,self.models['K_copy_dynamic'],paths,fb,ids,ids,self.physics,retain=True)
        self.assertEqual(float(r.max()),0.)

if __name__=='__main__':unittest.main()
