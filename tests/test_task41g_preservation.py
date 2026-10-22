import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.giada_teacher import task41g_preservation as g
from src.giada_teacher import task41g_evaluation as ev
ROOT=Path(__file__).resolve().parents[1]


class Task41gTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec,cls.cfg,cls.calcium,cls.base=g.contract(ROOT)
        cls.controls=g.frozen_controls(torch,ROOT,cls.spec,'cpu');cls.parent=cls.controls['E_Q0']

    def test_matrix_and_unchanged_acceptance(self):
        old=json.loads((ROOT/g.f.CONFIG).read_text())
        for key in ('quality','compartment_gates','cost'):self.assertEqual(self.spec[key],old[key])
        self.assertEqual(len(g.arms()),12)
        for axis,aa,bb in g.contrasts():
            a=next(a for a in g.arms() if a['name']==aa);b=next(a for a in g.arms() if a['name']==bb)
            self.assertEqual([k for k in ('heads','optimizer','objective','support') if a[k]!=b[k]],[axis])

    def test_zero_parity_protected_heads_and_compact(self):
        model=g.make_model(torch,self.parent,g.arms(),'cpu')
        v=torch.linspace(-95,55,23)[None].expand(36,-1);ca=torch.full_like(v,1e-4)
        with torch.no_grad():
            original=model.raw(v,ca);expected=self.parent.raw(v[:3],ca[:3])
            for j in range(12):
                for a,b in zip(original,expected):self.assertTrue(torch.equal(a[3*j:3*j+3],b))
            model.f_correction.normal_(0,.001);actual=model.raw(v,ca)
            self.assertTrue(torch.equal(original[1],actual[1]))
            for j,a in enumerate(g.arms()):
                if a['heads']=='M':self.assertTrue(torch.equal(original[0][3*j:3*j+3,:,1:],actual[0][3*j:3*j+3,:,1:]))
                small=g.extract(torch,model,j,self.parent,'cpu').raw(v[:3],ca[:3])
                for x,y in zip(actual,small):self.assertLess(float((x[3*j:3*j+3]-y).abs().max()),1e-6)

    def test_member_adam_matches_torch_adam(self):
        torch.manual_seed(71);w=torch.nn.Parameter(torch.randn(36,6,128)*.001)
        opt=g.Adam(torch,w,g.arms(),self.spec)
        refs=[torch.nn.Parameter(w[k:k+1].detach().clone()) for k in (0,6,35)]
        oo=[torch.optim.Adam([r],lr=.001) for r in refs]
        for step in (0,1,30000,59999):
            gradient=torch.randn_like(w)*.01;w.grad=gradient.clone();opt.rate(step)
            for k,r,o in zip((0,6,35),refs,oo):
                o.param_groups[0]['lr']=float(opt.lr[k]);r.grad=gradient[k:k+1].clone();o.step()
            opt.step()
            for k,r in zip((0,6,35),refs):self.assertLess(float((w[k:k+1]-r).abs().max()),2e-7)

    def test_tail_labels_and_gradient(self):
        model=g.make_model(torch,self.parent,g.arms(),'cpu')
        tail=g.tail_data(torch,self.parent,self.spec,'cpu',count=32)
        loss=g.tail_loss(torch,model,tail,torch.arange(8))
        grad=torch.autograd.grad(loss.sum(),model.f_correction)[0]
        self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(grad).all())
        self.assertTrue((tail['ticks']>=1).all());self.assertTrue((tail['ticks']<=4000).all())
        self.assertTrue(bool((grad*(1-model.active)==0).all()))

    def test_precision_fp32_identity_and_finite_mixed(self):
        n=4;steps=10;v=np.full((n,steps),-45.);ca=np.full_like(v,1e-4)
        initial,_=g.p.coefficients(v[:,0],ca[:,0],.1)
        data=dict(voltage=v,calcium=ca,gates=np.repeat(initial[:,None],steps,1),
            injection=np.full_like(v,.004),base_g_us=np.zeros((n,steps,4)),multipliers=np.ones((n,11)),area_um2=100*np.pi)
        with torch.inference_mode():
            actual=ev.policy_rollout(torch,self.parent,data,self.cfg,self.calcium,self.base,'cpu',(False,False,False))
            original=g.fe.typed_rollout(torch,self.parent,data,self.cfg,self.calcium,self.base,'cpu')['current']
            self.assertTrue(np.array_equal(actual,original))
            for policy in ((True,False,False),(False,True,False),(False,False,True),(True,True,True)):
                actual=ev.policy_rollout(torch,self.parent,data,self.cfg,self.calcium,self.base,'cpu',policy)
                self.assertTrue(np.isfinite(actual).all())

    def test_all_atomic_and_feedback_gradients(self):
        model=g.make_model(torch,self.parent,g.arms(),'cpu')
        atomic,_=g.atomic_data(torch,self.parent,self.spec,'cpu',count=16)
        idx=torch.arange(8)[None].expand(36,-1)
        norm=g.f.atomic_terms(torch,model.f_correction,model.active,atomic,idx).detach().clamp_min(1e-12)
        point,_=g.point_loss(torch,model,atomic,idx,norm,atomic,idx)
        gp=torch.autograd.grad(point.sum(),model.f_correction)[0]
        self.assertTrue(torch.isfinite(gp).all())
        fb,support=g.f.feedback_data(torch,self.spec,self.cfg,self.calcium,self.base,'cpu',small=True)
        view=g.FeedbackView(torch,self.parent,model)
        response=g.feedback_terms(torch,model,view,fb,support,torch.arange(2),self.cfg,self.calcium,self.base)
        grad=torch.autograd.grad(response.sum(),model.f_correction)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad*(1-model.active)==0).all())

if __name__=='__main__':unittest.main()
