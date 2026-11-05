import os,unittest
from pathlib import Path
import numpy as np
try:import torch
except ImportError:torch=None
from src.giada_teacher import task41p_tail_repair as p

@unittest.skipIf(torch is None,'Torch not installed')
class ExistingHeadRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        source=cls.root/'artifacts/task41n_frozen.zip.bin'
        if not source.exists():raise unittest.SkipTest('Frozen local input missing')
        os.environ['GIADA_TASK41N_FROZEN']=str(source)
        cls.base=p.n.load_models(torch,cls.root,'cpu')[0]['dual']
        cls.spec=dict(fit_seed=9412101,fit_points=64)

    def test_zero_materialization_and_finite_paired_gradients(self):
        data,wv,wc=p.pool(torch,self.base,self.spec,'cpu')
        clone=p.materialize(torch,self.base,wv,wc,'joint_nap')
        self.assertEqual(p.n.weights_hash(clone),p.n.weights_hash(self.base))
        wv=torch.nn.Parameter(wv[None].repeat(2,1,1,1));wc=torch.nn.Parameter(wc[None].repeat(2,1,1,1))
        scores=p.loss(torch,wv,wc,data,torch.arange(16),torch.tensor([[0.],[1.]]))
        self.assertEqual(scores.shape,(2,3));self.assertTrue(torch.isfinite(scores).all())
        scores.sum().backward();self.assertTrue(torch.isfinite(wv.grad).all());self.assertTrue(torch.isfinite(wc.grad).all())

    def test_only_registered_raw_heads_change(self):
        _,wv,wc=p.pool(torch,self.base,self.spec,'cpu');wv=wv+.00001;wc=wc+.00001
        net=p.materialize(torch,self.base,wv,wc,'equilibrium')
        self.assertEqual(sum(v.numel() for v in net.parameters()),sum(v.numel() for v in self.base.parameters()))
        v=torch.linspace(-100,60,24)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
        with torch.no_grad():a,b=self.base.raw(v,ca);c,d=net.raw(v,ca)
        other=[k for k in range(18) if k not in p.EQ]
        self.assertTrue(torch.equal(a[...,other],c[...,other]));self.assertTrue(torch.equal(b,d))
        self.assertFalse(torch.equal(a,c))

    def test_joint_changes_nap_update_only(self):
        _,wv,wc=p.pool(torch,self.base,self.spec,'cpu');net=p.materialize(torch,self.base,wv+.00001,wc,'joint_nap')
        v=torch.linspace(-100,60,24)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
        with torch.no_grad():a,b=self.base.raw(v,ca);c,d=net.raw(v,ca)
        other=[k for k in range(18) if k not in (8,9)]
        self.assertTrue(torch.equal(b[...,other],d[...,other]));self.assertFalse(torch.equal(b[...,8:10],d[...,8:10]))

    def test_arm_gradients_independent_and_frozen_columns_stay_exact(self):
        data,a,b=p.pool(torch,self.base,self.spec,'cpu')
        wv=torch.nn.Parameter(a[None].repeat(2,1,1,1));wc=torch.nn.Parameter(b[None].repeat(2,1,1,1))
        scores=p.loss(torch,wv,wc,data,torch.arange(16),torch.tensor([[0.],[1.]]))
        single=torch.autograd.grad(scores[0].sum(),wv,retain_graph=True)[0]
        scores.sum().backward();self.assertTrue(torch.equal(single[0],wv.grad[0]));self.assertEqual(single[1].abs().max(),0)
        wv.grad[0,:,:,5:]=0;old=wv[0,:,:,5:].detach().clone()
        optimizer=torch.optim.Adam([wv,wc],lr=2e-5);optimizer.step()
        self.assertTrue(torch.equal(old,wv[0,:,:,5:]))

    def test_notebook_kernel_receipts_and_training_worker(self):
        import sys
        sys.path.insert(0,str(self.root/'scripts'))
        import task41p_kaggle as k
        doc=k.notebook('a'*40,'b'*64);code=''.join(doc['cells'][1]['source'])
        compile(code,'41p','exec');self.assertEqual(doc['metadata']['kernelspec']['name'],'python3')
        self.assertIn('scripts/run_task41p.py',code)
        self.assertNotIn('training_updates_performed=0',code)

if __name__=='__main__':unittest.main()
