import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.giada_teacher import ie_residual_repair as p6

ROOT=Path(__file__).resolve().parents[1]

class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec=json.loads((ROOT/p6.CONFIG).read_text())
        cls.parent=torch.load(ROOT/cls.spec['parent_checkpoint'],map_location='cpu',weights_only=True)

    def test_batched_adam(self):
        self.assertTrue(p6.preflight(torch,self.parent,'cpu')['valid'])

    def test_parent_equivalence(self):
        m=p6.dense(torch,self.parent,'cpu');v=torch.linspace(-100,60,71)
        for k in range(12):
            u,z=p6.p5.operator(torch,m,k)(v)
            a,b=p6.p5.operator(torch,m,k%3)(v)
            self.assertTrue(torch.equal(u,a) and torch.equal(z,b))

    def test_local_projection_and_unpadded_inference(self):
        for count in (81,161):
            m=p6.local(torch,self.parent,count,'cpu')
            with torch.no_grad():m.table[0,0,0]=-2;m.table[1,-1,-1]=3
            m.project();self.assertTrue(bool(((m.table>=0)&(m.table<=1)).all()))
            v=torch.linspace(-110,70,79);u,z=m(v[None].expand(3,-1))
            for k in range(3):
                a,b=p6.local_operator(torch,m,k)(v)
                self.assertTrue(torch.equal(a,u[k]) and torch.equal(b,z[k]))
                op=torch.compile(p6.local_operator(torch,m,k),backend='eager',fullgraph=True)
                self.assertTrue(torch.equal(op(v)[0],a))
            self.assertEqual(m.table.shape,(3,count,34))

    def test_recurrence_and_gradient(self):
        gen=torch.Generator().manual_seed(3);v,x=p6.sequences(torch,12,gen,'cpu')
        u,z=p6.p4.coefficients(torch,v);truth=p6.rollout(torch,u,z,x)
        state=x.clone()
        for i in range(8):state=(1-z[i])*state+z[i]*u[i]
        self.assertTrue(torch.equal(state,truth[-1]))
        model=p6.dense(torch,self.parent,'cpu');loss=p6.temporal_loss(torch,model,v,x,truth)
        loss.sum().backward();self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_targeting_and_probe(self):
        m=p6.dense(torch,self.parent,'cpu');v=torch.linspace(-100,60,1024)
        w,report=p6.targeting(torch,m,v)
        self.assertEqual(len(report['bin_weights']),64);self.assertTrue(bool((w>0).all()))
        probe=p6.gradient_probe(torch,self.parent,'cpu')
        self.assertEqual(len(probe['components']),3)
        self.assertFalse(probe['causal_proof'])

    def test_training_smoke_and_freeze(self):
        spec={**self.spec,'fit_count':128,'development_count':31,'batch_size':32,'training_steps':1,'checkpoints':[0,1]}
        with tempfile.TemporaryDirectory(prefix='ie-p6-test-',dir=ROOT/'artifacts') as tmp:
            models,freeze=p6.train(torch,self.parent,spec,Path(tmp),'cpu')
            self.assertFalse(freeze['fresh_accessed']);self.assertEqual(len(freeze['selected_steps']['dense']),12)
            self.assertTrue((Path(tmp)/'checkpoint_1.pt').is_file())
            self.assertEqual(set(models),{'dense','local81','local161'})

    def test_fresh_paths_distinct(self):
        v,x,f=p6.path_data(self.spec['path_seed'])
        self.assertEqual(v.shape,(400,24));self.assertEqual(len(set(f)),6)
        self.assertFalse(np.array_equal(v,p6.p5.path_data()[0]))
        self.assertTrue(np.isfinite(v).all() and v.min()>=-100 and v.max()<=60)

if __name__=='__main__':unittest.main()
