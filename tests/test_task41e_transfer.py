import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.giada_teacher import task41e_transfer_model as m
from src.giada_teacher import task41e_transfer_data as data
from src.giada_teacher import task41e_transfer_training as train
from src.giada_teacher import task41e_transfer_evaluation as evaluation

ROOT=Path(__file__).resolve().parents[1]


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec,cls.cfg,cls.calcium,cls.base=m.contract(ROOT)
        cls.parent=m.d.prior.load_models(torch,ROOT,'cpu')['P110']

    def test_shared_matrix_and_thresholds(self):
        old=json.loads((ROOT/m.d.CONFIG).read_text())
        self.assertEqual(self.spec['quality'],old['quality'])
        self.assertEqual(self.spec['compartment_gates'],old['compartment_gates'])
        self.assertEqual(self.spec['cost'],old['cost'])
        self.assertEqual(len(m.arms()),12)
        self.assertEqual(len({a['name'] for a in m.arms()}),12)
        for axis,a,b in m.contrasts():
            aa=next(x for x in m.arms() if x['name']==a)
            bb=next(x for x in m.arms() if x['name']==b)
            different={k for k in aa if k!='name' and aa[k]!=bb[k]}
            self.assertEqual(len(different),1,(axis,a,b,different))
        # Exactly matched causal teacher/student tick exposure at these points.
        self.assertEqual((5760//64)*40,(57600//64)*4)

    def test_censored_extremes_and_interior_gradients(self):
        raw=torch.tensor([-25.,25.,.3],dtype=torch.float64,requires_grad=True)
        target=torch.tensor([1e-12,1-1e-12,.4],dtype=torch.float64)
        clipped=m.clipped_or_censored_error(torch,raw,target,False)
        censored=m.clipped_or_censored_error(torch,raw,target,True)
        self.assertGreater(float(clipped[:2].square().sum()),100.)
        self.assertEqual(float(censored[:2].abs().sum()),0.)
        self.assertEqual(float(clipped[2]),float(censored[2]))
        gradient=torch.autograd.grad(censored.square().sum(),raw)[0]
        self.assertTrue(torch.equal(gradient[:2],torch.zeros(2,dtype=torch.float64)))
        eps=1e-6
        upper=m.clipped_or_censored_error(torch,raw.detach()+torch.tensor([0.,0.,eps]),target,True).square().sum()
        lower=m.clipped_or_censored_error(torch,raw.detach()-torch.tensor([0.,0.,eps]),target,True).square().sum()
        self.assertAlmostEqual(float(gradient[2]),float((upper-lower)/(2*eps)),places=6)

    def test_zero_parent_and_untouched_coefficients(self):
        model=m.make_model(torch,self.parent,m.arms(),'cpu')
        v=torch.linspace(-95,55,73)[None].expand(36,-1);ca=torch.full_like(v,1e-4)
        with torch.no_grad():
            expected=self.parent.raw(v[:3],ca[:3]);before=model.raw(v,ca)
            for i in range(12):
                self.assertTrue(torch.equal(expected[0],before[0][i*3:i*3+3]))
            model.transfer_correction[:,0].fill_(.001)
            model.transfer_correction[:,1].fill_(.002)
            after=model.raw(v,ca)
        untouched=[k for k in range(18) if k not in m.GATES]
        self.assertTrue(torch.equal(before[0][...,untouched],after[0][...,untouched]))
        self.assertTrue(torch.equal(before[1],after[1]))
        self.assertEqual(model.transfer_correction[0].numel(),128)
        self.assertTrue(all(not p.requires_grad for name,p in model.named_parameters() if name!='transfer_correction'))

    def test_fused_old_composition_and_dtype_clone(self):
        controls=m.frozen_controls(torch,ROOT,self.spec,'cpu')
        v=torch.linspace(-100,60,641)[None].expand(3,-1);ca=torch.logspace(-7,-2,641)[None].expand(3,-1)
        with torch.no_grad():
            a=controls['D_combined'].raw(v,ca);b=controls['D_fused'].raw(v,ca)
        for aa,bb in zip(a,b):self.assertTrue(torch.equal(aa,bb))
        cloned=evaluation.clone_model(torch,controls['D_combined'])
        for child in (cloned.parent,cloned.ca_head,cloned.nap_head):child.double()
        self.assertTrue(all(torch.isfinite(v).all() for v in cloned.raw(v.double(),ca.double())))
        oracle=evaluation.GateOracle(torch,copy.deepcopy(controls['D_fused']),[0,8])
        cloned=evaluation.clone_model(torch,oracle)
        state=torch.full((3,641,18),.4)
        self.assertTrue(torch.isfinite(cloned(v,ca,state)).all())

    def test_balance_zero_gradient_and_bounded_updates(self):
        point=torch.tensor([[1.,0.],[2.,3.]],dtype=torch.float64)
        auxiliary=torch.tensor([[1e-4,1.],[0.,1e-20]],dtype=torch.float64)
        dynamic=torch.tensor([True,True])
        initial=m.balance_weights(torch,point,auxiliary,None,self.spec,dynamic)
        self.assertTrue(torch.isfinite(initial).all())
        self.assertEqual(float(initial[0,1]),0.)
        self.assertTrue(torch.equal(initial[1],torch.zeros_like(initial[1])))
        previous=torch.ones_like(point)
        changed=m.balance_weights(torch,point,auxiliary,previous,self.spec,dynamic)
        self.assertLessEqual(float(changed.max()),2.)
        self.assertEqual(float(changed[1,1]),0.)

    def test_held_tail_labels_and_finite_gradient(self):
        model=m.make_model(torch,self.parent,m.arms(),'cpu')
        held=data.held_tail_data(torch,self.parent,self.spec,'cpu',8)
        loss=train.protection_terms(torch,model,held,torch.arange(4))
        gradient=torch.autograd.grad(loss.sum(),model.transfer_correction)[0]
        self.assertEqual(tuple(gradient.shape),(36,2,64))
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_separate_and_joint_same_initial_gradient(self):
        defs=[dict(m.arms()[3],name='separate',joint=False),dict(m.arms()[3],name='joint',joint=True)]
        model=m.make_model(torch,self.parent,defs,'cpu')
        values=dict(v=np.full((4,2),-45.),ca=np.full((4,2),1e-4),g=np.full((4,2,18),.4),
                    mult=np.ones((4,2,11)),inj=np.full((4,2),.004),syn=np.zeros((4,2,4)))
        values['mult'][:,1,4]=0
        labels=data.reference_windows(torch,values,self.spec,self.cfg,self.calcium,self.base,'cpu')
        fb=data.tensor_inputs(torch,values,'cpu')
        fb['target']=torch.tensor(labels['delta'][...,list(m.CHANNELS)])
        supports={key:torch.arange(4) for key in ('broad','stratified')}
        groups=train.feedback_groups(torch,self.parent,model,defs,'cpu')
        losses=train.feedback_terms(torch,model,groups,fb,supports,torch.arange(2),self.cfg,self.calcium,self.base)
        gradient=torch.autograd.grad(losses.sum(),model.transfer_correction)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertLess(float((gradient[:3]-gradient[3:]).abs().max()),1e-6)

    def test_single_head_optimizer_isolation(self):
        model=m.make_model(torch,self.parent,m.arms(),'cpu')
        atomic,_=m.atomic_data(torch,self.parent,self.spec,'cpu',count=16)
        censored=torch.tensor([a['objective']=='censored' for a in m.arms() for _ in range(3)])
        idx=torch.arange(8)[None].expand(36,-1)
        point,_=m.atomic_terms(torch,model.transfer_correction,atomic,idx,censored)
        norm=point.detach().clamp_min(1e-12)
        opt=torch.optim.Adam([model.transfer_correction],lr=self.spec['lr'])
        train.atomic_step(torch,model,opt,atomic,idx,censored,norm)
        for member in (0,17,35):
            w=torch.nn.Parameter(torch.zeros(1,2,64));oo=torch.optim.Adam([w],lr=self.spec['lr'])
            ds={k:v[member:member+1] for k,v in atomic.items()};oo.zero_grad()
            loss,_=m.atomic_terms(torch,w,ds,idx[:1],censored[member:member+1])
            (loss/norm[member:member+1]).sum().backward();train.clip_heads(torch,w.grad);oo.step()
            self.assertLess(float((w[0]-model.transfer_correction[member]).abs().max()),1e-6)

    def test_physical_feedback_gradient_by_finite_difference(self):
        parent=copy.deepcopy(self.parent).double()
        definition=dict(m.arms()[3],joint=True)
        model=m.make_model(torch,parent,[definition],'cpu')
        values=dict(v=np.full((2,2),-45.),ca=np.full((2,2),1e-4),g=np.full((2,2,18),.4),
                    mult=np.ones((2,2,11)),inj=np.full((2,2),.004),syn=np.zeros((2,2,4)))
        values['mult'][:,1,4]=0
        labels=data.reference_windows(torch,values,self.spec,self.cfg,self.calcium,self.base,'cpu')
        fb=data.tensor_inputs(torch,values,'cpu',torch.float64)
        fb['target']=torch.tensor(labels['delta'][...,list(m.CHANNELS)])
        supports={key:torch.arange(2) for key in ('broad','stratified')}
        groups=train.feedback_groups(torch,parent,model,[definition],'cpu')
        loss=lambda:train.feedback_terms(torch,model,groups,fb,supports,torch.arange(2),self.cfg,self.calcium,self.base).sum()
        gradient=torch.autograd.grad(loss(),model.transfer_correction)[0]
        direction=torch.zeros_like(gradient);direction[:,1,21:23]=torch.tensor([.4,-.2],dtype=torch.float64)
        expected=float((gradient*direction).sum())
        eps=1e-4
        with torch.no_grad():model.transfer_correction.add_(direction,alpha=eps)
        upper=float(loss())
        with torch.no_grad():model.transfer_correction.add_(direction,alpha=-2*eps)
        lower=float(loss())
        fd=(upper-lower)/(2*eps)
        self.assertLess(abs(expected-fd),1e-7+abs(expected)*1e-3)

    def test_inference_constants_can_be_used_in_jacobian(self):
        key=('cpu',torch.float32)
        saved=m.t._CONSTANTS.pop(key,None)
        try:
            with torch.inference_mode():
                vv=torch.ones(3,1,dtype=torch.float32)
                before=tuple(v.clone() for v in m.t.constants(torch,vv))
            with torch.inference_mode(False),torch.enable_grad():
                evaluation.differentiable_constants()
                v=torch.full((3,1),-45.,requires_grad=True)
                g=torch.full((3,1,18),.4)
                _,current=m.t.readout(torch,v,g,torch.ones(3,1,11))
                derivative=torch.autograd.grad(current.sum(),v)[0]
                self.assertTrue(torch.isfinite(derivative).all())
                for a,b in zip(before,m.t._CONSTANTS[key]):
                    self.assertTrue(torch.equal(a,b));self.assertFalse(b.is_inference())
        finally:
            if saved is not None:m.t._CONSTANTS[key]=saved


if __name__=='__main__':unittest.main()
