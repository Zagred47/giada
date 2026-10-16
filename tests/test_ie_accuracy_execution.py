import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.giada_teacher import ie_accuracy_execution as p5

ROOT=Path(__file__).resolve().parents[1]

class AccuracyExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((ROOT/p5.CONFIG).read_text())
        path=ROOT/cls.spec['parent_checkpoint']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cls.spec['parent_sha256']
        cls.parent=torch.load(path,map_location='cpu',weights_only=True)

    def test_warm_start_and_adam(self):
        self.assertTrue(p5.preflight(torch,self.parent,'cpu')['valid'])

    def test_compilation_uses_only_neural_outputs(self):
        def op(v):return torch.sigmoid(v[...,None]/80).expand(*v.shape,17),torch.full((*v.shape,17),.1)
        compiled,table=p5.tabulate(torch,op,.25,'cpu')
        knots=torch.linspace(-100,60,641)
        self.assertTrue(torch.equal(table,torch.cat(op(knots),-1)))
        for a,b in zip(compiled(knots),op(knots)):self.assertTrue(torch.allclose(a,b,atol=1e-7))
        for a,b in zip(compiled(torch.tensor([-110.,80.])),op(torch.tensor([-100.,60.]))):self.assertTrue(torch.allclose(a,b,atol=1e-7))
        fn=torch.compile(compiled,backend='eager',fullgraph=True)
        for a,b in zip(fn(knots),compiled(knots)):self.assertTrue(torch.equal(a,b))

    def test_endpoint_probe(self):
        u=torch.linspace(.01,.99,17,dtype=torch.float64);z=torch.linspace(.001,.7,17,dtype=torch.float64)
        values=[]
        for initial in (torch.zeros_like(u),torch.ones_like(u)):
            x=initial
            for _ in range(400):x=(1-z)*x+z*u
            values.append(x)
        self.assertTrue(torch.allclose(p5.endpoints(torch,u,z),torch.stack(values,-1),atol=1e-12))

    def test_oracle_swaps(self):
        model=p5.ensemble(torch,self.parent,'cpu');op=p5.operator(torch,model,0)
        v=torch.linspace(-100,60,47);u,z=op(v);a,b=p5.p4.coefficients(torch,v)
        uu,zz=p5.swap(torch,op,'u_all')(v);self.assertTrue(torch.equal(uu,a));self.assertTrue(torch.equal(zz,z))
        for group,ids in p5.GROUPS.items():
            uu,zz=p5.swap(torch,op,group)(v)
            self.assertTrue(torch.equal(uu[:,list(ids)],a[:,list(ids)]))
            other=[i for i in range(17) if i not in ids]
            self.assertTrue(torch.equal(uu[:,other],u[:,other]))
            self.assertTrue(torch.equal(zz[:,other],z[:,other]))

    def test_new_paths(self):
        v,x,f=p5.path_data();old=p5.p4.path_data()[0]
        self.assertEqual(v.shape,(400,24));self.assertEqual(x.shape,(24,17));self.assertEqual(len(set(f)),6)
        self.assertTrue(np.isfinite(v).all() and v.min()>=-100 and v.max()<=60)
        self.assertFalse(np.array_equal(v,old))

if __name__=='__main__':unittest.main()
