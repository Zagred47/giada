import copy,unittest
from pathlib import Path
import torch
from src.giada_teacher import task41j_operator_reset as j


class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.parent=j.controls(torch,Path(__file__).resolve().parents[1],'cpu')['H_nominee']
    def test_parallel_scan_values_and_gradients(self):
        torch.manual_seed(7)
        u=torch.rand(3,2,17,18,dtype=torch.float64,requires_grad=True);z=torch.rand_like(u,requires_grad=True);initial=torch.rand(3,2,18,dtype=torch.float64)
        parallel=j.affine_scan(torch,u,z,initial);g=initial;seq=[]
        for k in range(17):g=g+z[:,:,k]*(u[:,:,k]-g);seq.append(g)
        sequential=torch.stack(seq,2)
        self.assertTrue(torch.allclose(parallel,sequential,atol=1e-14,rtol=1e-14))
        ga=torch.autograd.grad(parallel.sum(),(u,z),retain_graph=True);gb=torch.autograd.grad(sequential.sum(),(u,z))
        self.assertTrue(all(torch.allclose(a,b,atol=1e-13,rtol=1e-13) for a,b in zip(ga,gb)))
    def test_chain_initial_exact_and_all_trainable(self):
        m=j.model_class(torch)(self.parent,j.arms()[0]);v=torch.linspace(-100,60,33)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.raw(v,ca),self.parent.raw(v,ca))))
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
    def test_compact_same_base_different_width(self):
        a,b=[j.model_class(torch)(self.parent,x) for x in j.arms()[1:3]]
        v=torch.linspace(-90,30,17)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
        self.assertTrue(all(torch.allclose(x,y,atol=1e-5) for x,y in zip(a.raw(v,ca),b.raw(v,ca))))
        self.assertFalse(hasattr(a,'inner'))
        for m in (a,b):
            q=m(v,ca,torch.rand(3,17,18));self.assertTrue((q>=0).all() and (q<=1).all())
    def test_causal_voltage_calcium_dependencies(self):
        m=j.model_class(torch)(self.parent,j.arms()[1]);v=torch.full((3,5),-40.);c=torch.full_like(v,1e-4)
        a=m.raw(v,c);b=m.raw(v,c*2);keep=[k for k in range(18) if k!=16]
        self.assertTrue(all(torch.equal(x[...,keep],y[...,keep]) for x,y in zip(a,b)))
    def test_parameter_update_is_seed_independent(self):
        m=j.model_class(torch)(self.parent,j.arms()[1]);data,_=j.atomic_pool(torch,self.parent,691101,8,'cpu');ids=torch.arange(4)
        before=copy.deepcopy(m.state_dict());opt=torch.optim.Adam(j.parameters(m),lr=1e-4)
        loss=j.coefficient_loss(torch,m,data,ids);loss[0].backward();j.clip_members(torch,j.parameters(m));opt.step()
        self.assertTrue(all(torch.equal(v[1:],before[k][1:]) for k,v in m.state_dict().items() if v.ndim and v.shape[0]==3))
    def test_identical_chain_distillation_is_exact_zero(self):
        m=j.model_class(torch)(self.parent,j.arms()[0]);data,_=j.atomic_pool(torch,self.parent,691101,16,'cpu')
        self.assertEqual(float(j.coefficient_loss(torch,m,data,torch.arange(16),True).max()),0.)
    def test_optimizer_resume_and_gradient_addresses(self):
        a=j.model_class(torch)(self.parent,j.arms()[1]);data,_=j.atomic_pool(torch,self.parent,691101,8,'cpu');ids=torch.arange(4)
        oa=torch.optim.Adam(j.parameters(a),lr=torch.tensor(1e-4));j.step(torch,{'a':a},{'a':oa},data,ids)
        b=copy.deepcopy(a);ob=torch.optim.Adam(j.parameters(b),lr=torch.tensor(1e-4));j.step(torch,{'a':b},{'a':ob},data,ids)
        b.load_state_dict(a.state_dict());j.restore_optimizer(torch,ob,copy.deepcopy(oa.state_dict()))
        j.step(torch,{'a':a},{'a':oa},data,ids);j.step(torch,{'a':b},{'a':ob},data,ids)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
    def test_long_path_gradients_all_heads(self):
        m=j.model_class(torch)(self.parent,j.arms()[1]);spec=dict(new_seeds=dict(fit=691101,selection=691103),path_prefix_ticks=17,fit_paths=2,quality=dict(onset_offset_max_ms=.3))
        data=j.path_pool(torch,spec,'cpu',count=2);loss=j.path_loss(torch,m,data,torch.arange(2));loss.sum().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None))
        self.assertGreater(float(m.c.grad[...,17:].abs().max()),0)
    def test_selected_seed_assembly(self):
        m=j.model_class(torch)(self.parent,j.arms()[1]);template=m.state_dict()
        selected=[{k:v[s:s+1].clone() if v.ndim and v.shape[0]==3 else v.clone() for k,v in template.items()} for s in range(3)]
        merged=j.merge_selected(torch,selected,template)
        self.assertTrue(all(torch.equal(v,merged[k]) for k,v in template.items()))
    def test_restricted_control_protects_rates(self):
        m=j.model_class(torch)(self.parent,j.arms()[3]);v=torch.linspace(-90,30,17)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
        before=m.raw(v,ca)
        with torch.no_grad():m.inner.correction.fill_(.001)
        after=m.raw(v,ca)
        self.assertTrue(torch.equal(before[1],after[1]))
        keep=[k for k in range(18) if k not in (0,8)]
        self.assertTrue(torch.equal(before[0][...,keep],after[0][...,keep]))


if __name__=='__main__':unittest.main()
