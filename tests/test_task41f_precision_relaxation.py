import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.giada_teacher import task41f_precision_relaxation as f
from src.giada_teacher import task41f_evaluation as ev
ROOT=Path(__file__).resolve().parents[1]


class Task41fTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec,cls.cfg,cls.calcium,cls.base=f.contract(ROOT)
        cls.parent=f.d.prior.load_models(torch,ROOT,'cpu')['P110']

    def test_threshold_contract_and_capacity_contrasts(self):
        old=json.loads((ROOT/f.prior.CONFIG).read_text())
        self.assertEqual(self.spec['quality'],old['quality'])
        self.assertEqual(self.spec['compartment_gates'],old['compartment_gates'])
        self.assertEqual(len(f.arms()),12)
        mask=f.active_mask(torch,f.arms(),'cpu')
        self.assertEqual([int(mask[k*3].sum()) for k in (0,4,8)],[192,384,384])
        for axis,aa,bb in f.contrasts():
            a=next(a for a in f.arms() if a['name']==aa);b=next(a for a in f.arms() if a['name']==bb)
            if axis in ('support','objective'):
                self.assertEqual([k for k in a if k not in ('name','feedback') and a[k]!=b[k]],[axis])

    def test_boundary_selection_uses_official_pass(self):
        q=self.spec['quality'];truth=np.zeros((30,11));truth[5:15,0]=1e-4;pred=truth.copy();pred[15:18,0]=1e-4
        metric=f.p.evaluate_currents(truth,pred,.1,{**q,'current_nrmse_max':1.,'current_nmax_max':1.,'signed_charge_error_max':1.,'absolute_charge_error_max':1.})
        self.assertTrue(metric['channels'][0]['passed'])
        qq={**q,'current_nrmse_max':1.,'current_nmax_max':1.,'signed_charge_error_max':1.,'absolute_charge_error_max':1.}
        self.assertLessEqual(f.channel_severity(metric,qq,(0,)),1.)

    def test_parent_and_unchanged_coefficients(self):
        model=f.make_model(torch,self.parent,f.arms(),'cpu')
        v=torch.linspace(-90,40,21)[None].expand(36,-1);ca=torch.full_like(v,1e-4)
        with torch.no_grad():
            a=model.raw(v,ca);expected=self.parent.raw(v[:3],ca[:3])
            for j in range(12):
                for aa,b in zip(a,expected):self.assertTrue(torch.equal(aa[3*j:3*j+3],b))
            model.f_correction.fill_(.003);b=model.raw(v,ca)
        untouched=[g for g in range(18) if g not in f.GATES]
        self.assertTrue(torch.equal(a[0][...,untouched],b[0][...,untouched]))
        self.assertTrue(torch.equal(a[1][...,untouched],b[1][...,untouched]))
        self.assertTrue(torch.equal(a[1][:24],b[1][:24]))

    def test_compact_padded_forward(self):
        model=f.make_model(torch,self.parent,f.arms(),'cpu')
        with torch.no_grad():model.f_correction.normal_(0,.001)
        v=torch.linspace(-95,55,23)[None].expand(36,-1);ca=torch.full_like(v,1e-4)
        with torch.no_grad():
            allout=model.raw(v,ca)
            for k in (0,4,8,11):
                compact=f.extract(torch,model,k,self.parent,'cpu').raw(v[:3],ca[:3])
                for aa,bb in zip(allout,compact):self.assertLess(float((aa[3*k:3*k+3]-bb).abs().max()),1e-5)

    def test_atomic_isolation_and_mask(self):
        model=f.make_model(torch,self.parent,f.arms(),'cpu')
        data,_=f.atomic_data(torch,self.parent,self.spec,'cpu',count=16)
        idx=torch.arange(8)[None].expand(36,-1)
        norm=f.atomic_terms(torch,model.f_correction,model.active,data,idx).detach().clamp_min(1e-12)
        opt=torch.optim.Adam([model.f_correction],lr=self.spec['lr']);f.atomic_step(torch,model,opt,data,idx,norm)
        self.assertTrue(torch.equal(model.f_correction*(1-model.active),torch.zeros_like(model.f_correction)))
        for k in (0,14,35):
            w=torch.nn.Parameter(torch.zeros(1,6,128));oo=torch.optim.Adam([w],lr=self.spec['lr']);oo.zero_grad()
            ds={n:v[k:k+1] for n,v in data.items()}
            (f.atomic_terms(torch,w,model.active[k:k+1],ds,idx[:1])/norm[k:k+1]).sum().backward()
            f.clip(torch,w.grad);oo.step()
            self.assertLess(float((w[0]-model.f_correction[k]).abs().max()),1e-6)

    def test_feedback_gradient_and_oracle_dtype(self):
        model=f.make_model(torch,self.parent,f.arms(),'cpu')
        fb,support=f.feedback_data(torch,self.spec,self.cfg,self.calcium,self.base,'cpu',small=True)
        view=f.FeedbackView(torch,self.parent,model)
        loss=f.feedback_terms(torch,model,view,fb,support,torch.arange(2),self.cfg,self.calcium,self.base)
        grad=torch.autograd.grad(loss.sum(),model.f_correction)[0]
        self.assertTrue(torch.isfinite(grad).all())
        v=torch.full((3,2),-45.,dtype=torch.float64);ca=torch.full_like(v,1e-4);state=torch.full((3,2,18),.4,dtype=torch.float64)
        oracle=ev.CoefficientOracle(torch,copy.deepcopy(self.parent),(0,1),(0,1))
        clone=ev.cast_model(torch,oracle,torch.float64)
        out=clone(v,ca,state)
        self.assertTrue(torch.isfinite(out).all());self.assertTrue(((out>=0)&(out<=1)).all())
        mixed=ev.MixedModel(torch,self.parent)
        self.assertEqual(mixed(v,ca,state).dtype,torch.float64)


if __name__=='__main__':unittest.main()
