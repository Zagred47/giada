import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41i_selective_repair as i
from src.giada_teacher import task41i_evaluation as e
ROOT=Path(__file__).resolve().parents[1]

class Task41iTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.spec,cls.cfg,cls.ca,cls.base=i.contract(ROOT)
        cls.parent=i.frozen_controls(torch,ROOT,cls.spec,'cpu')['H_nominee']
        cls.raw=i.d.anchors(cls.spec['new_seeds']['fit'],64);cls.knots=i.fit_knots(cls.raw)
    def model(self):return i.make_model(torch,self.parent,i.arms(),self.knots,'cpu')
    def test_factorial_and_thresholds(self):
        old=json.loads((ROOT/i.h.CONFIG).read_text())
        for key in ('quality','compartment_gates','cost'):self.assertEqual(self.spec[key],old[key])
        self.assertEqual(len(i.arms()),8);self.assertEqual(len(i.contrasts()),12)
        for axis,a,b in i.contrasts():
            aa=next(x for x in i.arms() if x['name']==a);bb=next(x for x in i.arms() if x['name']==b)
            self.assertEqual([key for key in ('nap','official_loss','adaptive') if aa[key]!=bb[key]],[axis])
    def test_uniform_feature_and_gradient(self):
        v=torch.linspace(-102,62,201,dtype=torch.float64)[None].expand(3,-1)
        knots=torch.linspace(-100,60,128,dtype=torch.float64)[None].expand(3,-1)
        a=i.feature(torch,v,knots);b=i.d.feature(torch,v,128)
        self.assertLess(float((a-b).abs().max()),1e-12)
    def test_initial_parent_and_mask(self):
        m=self.model();v=torch.full((24,9),-43.);ca=torch.full_like(v,1e-4)
        old=m.raw(v,ca)
        for j in range(8):
            expected=self.parent.raw(v[:3],ca[:3]);self.assertTrue(all(torch.equal(x[3*j:3*j+3],y) for x,y in zip(old,expected)))
        with torch.no_grad():m.correction.normal_(0,.001);m.correction.mul_(m.mask)
        new=m.raw(v,ca);self.assertTrue(torch.equal(old[1],new[1]))
        self.assertTrue(torch.equal(old[0][:12,:,8],new[0][:12,:,8]))
        self.assertFalse(torch.equal(old[0][12:,:,8],new[0][12:,:,8]))
        for k in range(18):
            if k not in (0,8):self.assertTrue(torch.equal(old[0][...,k],new[0][...,k]))
    def test_atomic_gradients(self):
        m=self.model();pool=i.atomic_pool(torch,self.parent,self.raw,'cpu');ids=torch.arange(8)[None].expand(24,-1)
        loss,_=i.atomic_loss(torch,m,pool,ids);grad=torch.autograd.grad(loss.sum(),m.correction)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertTrue((grad*(1-m.mask)==0).all())
        self.assertGreater(float(grad[12:,1].abs().max()),0)
    def test_capture_safe_censoring_matches_original(self):
        target=torch.tensor([0.,1e-9,.2,.8,1-1e-9,1.],dtype=torch.float64)
        raw=torch.linspace(-20,20,6,dtype=torch.float64)
        original=i.h.prior.clipped_or_censored_error(torch,raw,target,True)
        residual=raw-torch.logit(target.clamp(1e-7,1-1e-7))
        safe=torch.where(target<=1e-7,residual.clamp_min(0),torch.where(target>=1-1e-7,residual.clamp_max(0),residual))
        self.assertTrue(torch.equal(original,safe))
    def test_full_path_queries_and_affine_identity(self):
        m=self.model();tail=i.path_pool(torch,self.parent,self.spec,'cpu',count=4)
        pred,gates=i.path_loss(torch,m,tail,torch.arange(4),return_predictions=True)
        expected=tail['base_query'].repeat(8,1,1,1)
        self.assertLess(float((gates-expected).abs().max()),1e-12)
        self.assertEqual(tuple(tail['crossings'].shape),(4,2,2))
        self.assertTrue(bool((tail['times']>=0).all() and (tail['times']<12000).all()))
        self.assertTrue(bool((tail['threshold']>=0).all()))
        with torch.no_grad():m.correction.normal_(0,.0001);m.correction.mul_(m.mask)
        for method in (0,1):
            loss=i.path_loss(torch,m,tail,torch.arange(4),method=method)
            grad=torch.autograd.grad(loss.sum(),m.correction)[0]
            self.assertTrue(torch.isfinite(grad).all());self.assertTrue((grad*(1-m.mask)==0).all())
    def test_runtime_adapters_preserve_reference(self):
        original=i.t.make_core;roll=i.t.rollout
        with e.runtime_adapters():
            v=torch.full((3,4),-45.);ca=torch.full_like(v,1e-4);gg=torch.full((3,4,18),.1);mult=torch.ones(3,4,11);inj=torch.zeros_like(v);syn=torch.zeros(3,4,4)
            expected=original(torch,self.cfg,self.ca,self.base,100*np.pi)(v,ca,gg,mult,inj,syn)
            actual=i.t.make_core(torch,self.cfg,self.ca,self.base,100*np.pi)(v,ca,gg,mult,inj,syn)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,actual)))
            for policy in ('joint64','increment32'):
                net=e.RuntimeModel(torch,copy.deepcopy(self.parent),policy)
                ionic=i.t.make_core(torch,self.cfg,self.ca,self.base,100*np.pi,net,i.t.FAMILIES['all'],True)
                parent_ionic=original(torch,self.cfg,self.ca,self.base,100*np.pi,self.parent,i.t.FAMILIES['all'],True)
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(ionic(v,ca,gg,mult,inj,syn),parent_ionic(v,ca,gg,mult,inj,syn))))
                core=i.t.make_core(torch,self.cfg,self.ca,self.base,100*np.pi,net,i.t.FAMILIES['all'])
                result=core(v.double(),ca,gg,mult,inj.double(),syn.double())
                self.assertEqual(result[0].dtype,torch.float64)
                self.assertTrue(all(torch.isfinite(x).all() for x in result))
        self.assertIs(i.t.make_core,original);self.assertIs(i.t.rollout,roll)
    def test_component_metric_visibility(self):
        self.assertTrue(i.metric_visibility(self.spec['quality'])['valid'])
    def test_optimizer_and_rng_resume_are_exact(self):
        rng=np.random.default_rng(991);weights=torch.zeros(24,2,128,requires_grad=True)
        opt=i.g.Adam(torch,weights,i.arms(),self.spec)
        for step in range(8):
            opt.rate(step);weights.grad=torch.tensor(rng.normal(size=weights.shape),dtype=weights.dtype);opt.step()
        saved=copy.deepcopy(opt.state_dict());saved_weights=weights.detach().clone();saved_rng=copy.deepcopy(rng.bit_generator.state)
        for step in range(8,20):
            opt.rate(step);weights.grad=torch.tensor(rng.normal(size=weights.shape),dtype=weights.dtype);opt.step()
        other=saved_weights.clone().requires_grad_();resumed=i.g.Adam(torch,other,i.arms(),self.spec)
        i.restore_optimizer(torch,resumed,saved);rng.bit_generator.state=saved_rng
        for step in range(8,20):
            resumed.rate(step);other.grad=torch.tensor(rng.normal(size=other.shape),dtype=other.dtype);resumed.step()
        self.assertTrue(torch.equal(weights,other));self.assertTrue(torch.equal(opt.m,resumed.m));self.assertTrue(torch.equal(opt.v,resumed.v))
    @torch.no_grad()
    def test_mixed_policy_rollout_matches_existing(self):
        data=dict(voltage=np.full((2,9),-45.,dtype='float32'),calcium=np.full((2,9),1e-4,dtype='float32'),
                  gates=np.full((2,9,18),.1,dtype='float32'),multipliers=np.ones((2,11),dtype='float32'),
                  injection=np.zeros((2,8),dtype='float32'),base_g_us=np.zeros((2,8,4),dtype='float32'),area_um2=100*np.pi)
        expected=e.he.rollout(torch,self.parent,data,self.cfg,self.ca,self.base,'cpu',(True,True,False))
        with e.runtime_adapters():
            net=e.RuntimeModel(torch,copy.deepcopy(self.parent),'joint64')
            core=i.t.make_core(torch,self.cfg,self.ca,self.base,data['area_um2'],net,i.t.FAMILIES['all'])
            actual=i.t.rollout(torch,core,data,'cpu')
        for key in expected:self.assertTrue(np.array_equal(expected[key],actual[key]),key)

if __name__=='__main__':unittest.main()
