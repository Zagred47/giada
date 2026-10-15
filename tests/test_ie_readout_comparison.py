import unittest
import torch
from src.giada_teacher import ie_readout_comparison as p3
from src.giada_teacher import ionic_block_teacher_forced as ionic

class ReadoutTests(unittest.TestCase):
    def test_exact_readout(self):
        torch.manual_seed(6)
        g=torch.rand(71,18);mult=torch.rand(71,11)*4
        opening=[]
        for k,names in enumerate(ionic.STATE_NAMES):
            value=g[:,ionic.OFFSETS[k]]**ionic.POWERS[k]
            if len(names)==2:value=value*g[:,ionic.OFFSETS[k]+1]
            opening.append(value)
        expected=torch.stack(opening,-1)
        self.assertTrue(torch.allclose(p3.exact_apertures(torch,g),expected,atol=1e-7))
        a,b=p3.readout(torch,g,mult)
        cond=expected*torch.tensor(ionic.GBAR,dtype=g.dtype)*mult
        self.assertTrue(torch.allclose(a,cond.sum(-1),atol=1e-10))
        self.assertTrue(torch.allclose(b,(cond*torch.tensor(ionic.REVERSALS,dtype=g.dtype)).sum(-1),atol=1e-8))
        zero=p3.readout(torch,g,torch.zeros_like(mult))
        self.assertEqual(float(zero[0].abs().max()),0)

    def test_complete_core(self):
        from pathlib import Path
        import numpy as np
        from src.giada_teacher import cc_n1_contract_memory as n1
        root=Path(__file__).resolve().parents[1]
        _,cfg,calcium,_,base=n1.contract(root)
        z=np.load(root/'experiments/results/task35_native_shadow_traces.npz')
        def t(a):return torch.tensor(a,dtype=torch.float32)
        args=(t(z['voltage'][:8,10]),t(z['calcium'][:8,10]),t(z['gates'][:8,10]),t(z['multipliers'][:8]),t(z['injection'][:8,10]),t(z['base_g_us'][:8,10]),float(z['area_um2']),cfg,calcium,base)
        old=p3.core(torch,*args,legacy=True);new=p3.core(torch,*args)
        self.assertTrue(p3.p2.differences(torch,new,old,(.002,1e-8,2e-5,1e-10,1e-8))['valid'])
        compiled=torch.compile(lambda:p3.core(torch,*args),fullgraph=True,backend='eager')
        self.assertTrue(p3.p2.differences(torch,compiled(),new,(.002,1e-8,2e-5,1e-10,1e-8))['valid'])

    def test_vectorization_prediction_and_adam(self):
        model=p3.make_ensemble(torch,'cpu');x=torch.rand(6,101,4);target=torch.rand(6,101)
        optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        saved={n:p.detach().clone() for n,p in model.named_parameters()}
        loss=(model(x)-target).square().mean(-1).sum();loss.backward()
        norms=torch.stack([p.grad.flatten(1).square().sum(1) for p in model.parameters()]).sum(0).sqrt()
        for p in model.parameters():p.grad.mul_((1/norms.clamp_min(1)).reshape(6,*([1]*(p.ndim-1))))
        optimizer.step()
        for k in range(6):
            isolated=p3.make_ensemble(torch,'cpu')
            with torch.no_grad():
                for n,p in isolated.named_parameters():p.copy_(saved[n])
            opt=torch.optim.Adam(isolated.parameters(),lr=.003)
            (isolated(x)[k]-target[k]).square().mean().backward()
            norm=torch.stack([p.grad[k].square().sum() for p in isolated.parameters()]).sum().sqrt()
            for p in isolated.parameters():p.grad.mul_(1/norm.clamp_min(1))
            opt.step()
            for n,p in isolated.named_parameters():self.assertTrue(torch.allclose(p[k],dict(model.named_parameters())[n][k],atol=1e-7))
            op=p3.independent_operator(torch,model,k,16 if k<3 else 32)
            self.assertTrue(torch.allclose(op(x[k]),model(x)[k],atol=1e-7))

if __name__=='__main__':unittest.main()
