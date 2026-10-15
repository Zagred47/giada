import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.giada_teacher import cc_n2_unified_cell as n2

ROOT=Path(__file__).resolve().parents[1]


class N2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec,cls.cfg,cls.ca,cls.base=n2.contract(ROOT)

    def test_disjoint_reproducible_roles(self):
        a,g,u=n2.design(self.spec);b,h,w=n2.design(self.spec)
        self.assertEqual(a,b);np.testing.assert_array_equal(g,h);np.testing.assert_array_equal(u,w)
        signatures={r:set((x['preparation_voltage_mv'],x['preparation_ms'],x['tail_ms']) for x in a if x['role']==r) for r in self.spec['role_counts']}
        self.assertFalse(signatures['fit']&signatures['development']);self.assertFalse(signatures['fit']&signatures['diagnostic'])
        self.assertEqual(u.shape,(400,112));self.assertTrue(np.all((g>=0)&(g<=1)))

    def test_vectorized_independent_adam_equivalence(self):
        model,_=n2.make_model(torch,self.spec,self.ca,self.base)
        self.assertTrue(n2.equivalence(torch,model,self.spec,self.ca,self.base)['valid'])

    def test_masks_and_finite_gradient(self):
        model,choices=n2.make_model(torch,self.spec,self.ca,self.base);n=len(choices)
        v=torch.full((n,4,1),-65.);ca=torch.full_like(v,1e-4);g=torch.full((n,4,18),.3)
        z=model.initialize(v,ca,g);pv,pc,pz=model(v,ca,z,torch.zeros_like(v),torch.ones(n,4,11))
        self.assertTrue(torch.isfinite(pv).all());self.assertTrue((pc>0).all())
        self.assertEqual(float(pz[3:6].abs().max()),0.)
        (pv.square().mean()+pc.square().mean()+pz.square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_encoder_information_and_standard(self):
        model,choices=n2.make_model(torch,self.spec,self.ca,self.base);n=len(choices)
        v=torch.full((n,2,1),-65.);ca=torch.full_like(v,1e-4)
        g=torch.zeros(n,2,18);g[:,1]=1.
        z=model.initialize(v,ca,g)
        self.assertTrue(torch.equal(z[:3],g[:3]))
        self.assertEqual(float(z[21:27].abs().max()),0.)
        self.assertGreater(float((z[9:12,0]-z[9:12,1]).abs().max()),0.)

    def test_rollout_and_metrics_shapes(self):
        spec=copy.deepcopy(self.spec);spec['training']['warmup_steps']=2
        model,choices=n2.make_model(torch,spec,self.ca,self.base)
        d={'voltage':torch.full((7,3),-65.),'calcium':torch.full((7,3),1e-4),
           'gates':torch.full((7,3,18),.3),'multipliers':torch.ones(3,11),'injection':torch.zeros(6,3)}
        pred=n2.rollout(torch,model,d,[0,1],2,3,spec)
        rows=n2.metrics(torch,pred,d,[0,1],2)
        self.assertEqual(pred[0].shape,(3,27,2,1));self.assertEqual(len(rows),27)
        self.assertTrue(all(np.isfinite(r['score']) for r in rows))

    def test_fullgraph_capture(self):
        model,choices=n2.make_model(torch,self.spec,self.ca,self.base)
        n=len(choices);v=torch.full((n,2,1),-65.);ca=torch.full_like(v,1e-4)
        args=(v,ca,torch.full((n,2,18),.3),torch.zeros_like(v),torch.ones(n,2,11))
        compiled=torch.compile(model,backend=lambda gm,example_inputs:gm.forward,fullgraph=True)
        for a,b in zip(model(*args),compiled(*args)):torch.testing.assert_close(a,b)

    def test_training_stream_and_backward_smoke(self):
        spec=copy.deepcopy(self.spec);spec['training']['warmup_steps']=2
        model,choices=n2.make_model(torch,spec,self.ca,self.base)
        d={'voltage':torch.full((9,3),-65.),'calcium':torch.full((9,3),1e-4),
           'gates':torch.full((9,3,18),.3),'multipliers':torch.ones(3,11),'injection':torch.zeros(8,3)}
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        before=model.w1.detach().clone()
        for _ in range(2):
            v,ca,z=n2.initial_state(torch,model,d,[0,1],3,spec,True)
            losses=0
            for t in range(3,5):
                v,ca,z=model(v,ca,z,torch.zeros_like(v),torch.ones(27,2,11))
                losses=losses+((v+65)/5).square().mean((1,2))+((ca-1e-4)/5e-5).square().mean((1,2))
            optimizer.zero_grad();losses.sum().backward();n2.clip_members(torch,model,1.);optimizer.step()
        self.assertTrue(torch.isfinite(model.w1).all());self.assertFalse(torch.equal(before,model.w1))

    def test_notebook_syntax(self):
        nb=json.loads((ROOT/'notebooks/cc_n2_unified_cell.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'notebook','exec')


if __name__=='__main__':unittest.main()
