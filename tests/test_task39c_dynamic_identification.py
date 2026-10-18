import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.giada_teacher import task39c_dynamic_identification as t

ROOT=Path(__file__).resolve().parents[1]


class Task39cTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data,_=t.q.windows(ROOT/'experiments/results/task37_playground_v2','fit',['steps'],8,8,99491)
        cls.anchor=t.anchor_pool(cls.data,12)
        cls.scales=[torch.ones(n) for n in (18,11,18,18)]

    def test_matrix_and_active_capacity(self):
        self.assertEqual(len(t.arms()),12)
        self.assertEqual(len(t.contrast_pairs(t.arms())),16)
        counts={a:sum(x.numel() for x in t.model_class(torch)(a,[17]).parameters()) for a in ('base','wide','separate')}
        self.assertEqual(counts,dict(base=1220,wide=2372,separate=1188))

    def test_notebook_cells_compile(self):
        notebook=json.loads((ROOT/'notebooks/giada_task39c_dynamic_identification.ipynb').read_text(encoding='utf-8'))
        for k,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell{k}','exec')

    def test_composition_gradients_and_teacher(self):
        for arch in ('base','wide','separate'):
            model=t.model_class(torch)(arch,[17]).double()
            v=torch.full((1,4),-55.,dtype=torch.float64);ca=torch.full_like(v,1e-4)
            x=torch.rand(1,4,18,dtype=torch.float64)
            u,z,ld=model.coefficients(v,ca);state=x
            for _ in range(1000):state=model(v,ca,state)
            composed=t.compose(torch,u,ld,x,torch.full((1,4,1),1000.))
            torch.testing.assert_close(state,composed,rtol=0,atol=1e-12)
            composed.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        a=torch.tensor(self.anchor)[None];ticks=torch.full((1,len(self.anchor),1),100000.)
        target=t.held_targets(torch,a,ticks)
        u,z=t.p.coefficients(self.anchor[:,0].astype(float),self.anchor[:,1].astype(float),10000.)
        expected=self.anchor[:,2:20]+z*(u-self.anchor[:,2:20])
        np.testing.assert_allclose(target[0].numpy(),expected,atol=2e-6)

    def test_independence_and_bounds(self):
        for arch in ('base','wide','separate'):
            m=t.model_class(torch)(arch,[17,29,43]);x=torch.rand(3,4,18)
            v=torch.full((3,4),-50.);ca=torch.full_like(v,.0001)
            a=m(v,ca,x);c=m(v,ca*10,x)
            torch.testing.assert_close(a[...,:16],c[...,:16],rtol=0,atol=0)
            torch.testing.assert_close(a[...,17:],c[...,17:],rtol=0,atol=0)
            xx=x.clone();xx[...,1:]=1-xx[...,1:]
            torch.testing.assert_close(a[...,0],m(v,ca,xx)[...,0],rtol=0,atol=0)
            self.assertGreaterEqual(float(a.min()),0);self.assertLessEqual(float(a.max()),1)

    def test_all_members_adam_parity(self):
        for g in t.groups(torch,t.arms(),[17,29,43],'cpu'):
            n=g['model'].a.shape[0]
            batch=torch.tensor(self.data[:2])[None].expand(n,-1,-1,-1)
            anchor=torch.tensor(self.anchor[:3])[None].expand(n,-1,-1)
            result=t.optimizer_preflight(torch,g,batch,anchor,torch.full((n,3,1),10000.),self.scales,torch.ones(2),[17,29,43])
            self.assertTrue(result['valid'])

    def test_sampler_and_frozen_head_extraction(self):
        targeted=t.anchor_pool(self.data,12,True)
        self.assertFalse(np.array_equal(targeted[:,:2],self.anchor[:,:2]))
        self.assertTrue(np.isfinite(targeted).all())
        self.assertEqual(targeted.shape,(8,107))
        spec=json.loads((ROOT/t.CONFIG).read_text())
        m,freeze=t.frozen39b(torch,ROOT,spec,'cpu',torch.float64)
        v=torch.full((36,3),-55.,dtype=torch.float64);ca=torch.full_like(v,1e-4)
        u,z=t.parent_coefficients(torch,m,v,ca);x=torch.rand(36,3,18,dtype=torch.float64)
        torch.testing.assert_close(m(v,ca,x),x+z*(u-x),rtol=0,atol=0)

    def test_selection_and_gradient_probe_preserve_grad_buffers(self):
        spec=dict(development_windows_per_family=2,recurrence_chunk=4,held_ticks=[1,10,1000])
        cohorts=t.groups(torch,[t.arms()[0],t.arms()[8],t.arms()[10]],[17,29,43],'cpu')
        normal=dict(gate=np.ones(18),current=np.ones(11))
        score,parts=t.selection(torch,cohorts,[self.data[:2]],self.anchor,normal,spec,'cpu')
        self.assertEqual(score.shape,(9,));self.assertTrue(np.isfinite(score).all())
        static=[]
        for g in cohorts:
            n=g['model'].a.shape[0]
            static.append(dict(batch=torch.tensor(self.data[:2])[None].expand(n,-1,-1,-1),
                anchor=torch.tensor(self.anchor[:3])[None].expand(n,-1,-1),ticks=torch.full((n,3,1),1000.)))
            for param in g['model'].parameters():param.grad=torch.zeros_like(param)
        buffers=[p.grad for g in cohorts for p in g['model'].parameters()]
        rows=t.gradient_probe(torch,cohorts,static,self.scales,torch.ones(2),0)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(p.grad is buf for p,buf in zip([p for g in cohorts for p in g['model'].parameters()],buffers)))


if __name__=='__main__':unittest.main()
