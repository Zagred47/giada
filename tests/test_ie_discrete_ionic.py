import unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import ie_discrete_ionic as p4
from src.giada_teacher import cc_n1_contract_memory as n1
from src.giada_teacher import ie_readout_comparison as p3

class DiscreteIonicTests(unittest.TestCase):
    def test_oracle(self):
        self.assertTrue(p4.oracle(torch)['valid'])

    def test_batched_adam(self):
        self.assertTrue(p4.preflight(torch,'cpu')['valid'])

    def test_lut_boundaries_and_domain(self):
        op=p4.lut_operator(torch,device='cpu')
        v=torch.linspace(-100,60,641);u,z=op(v);a,b=p4.coefficients(torch,v)
        self.assertTrue(torch.allclose(u,a,atol=1e-7))
        self.assertTrue(torch.allclose(z,b,atol=1e-7))
        x=torch.rand(641,17);ng=(1-z)*x+z*u
        self.assertTrue(bool(((ng>=0)&(ng<=1)).all()))
        for bound,outside in [(-100,-110),(60,90)]:
            low=op(torch.tensor([float(bound)]));high=op(torch.tensor([float(outside)]))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(low,high)))

    def test_phase_equivalence_and_graph(self):
        root=Path(__file__).resolve().parents[1]
        _,cfg,calcium,_,base=n1.contract(root)
        traces=np.load(root/'experiments/results/task35_native_shadow_traces.npz')
        def t(x):return torch.tensor(x,dtype=torch.float32)
        args=(t(traces['voltage'][:8,10]),t(traces['calcium'][:8,10]),t(traces['gates'][:8,10]),t(traces['multipliers'][:8]),
            t(traces['injection'][:8,10]),t(traces['base_g_us'][:8,10]),float(traces['area_um2']),cfg,calcium,base)
        golden=p3.core(torch,*args)
        out=p4.core(torch,*args,operator=lambda v:p4.coefficients(torch,v))
        self.assertTrue(p4.p2.differences(torch,out,golden,(.002,1e-8,2e-5,1e-10,1e-8))['valid'])
        for op in (None,p4.lut_operator(torch,device='cpu')):
            fn=lambda:p4.core(torch,*args,operator=op)
            compiled=torch.compile(fn,backend='eager',fullgraph=True)
            self.assertTrue(p4.p2.differences(torch,compiled(),fn(),(.002,1e-8,2e-5,1e-10,1e-8))['valid'])

    def test_path_support(self):
        paths,initial,families=p4.path_data()
        self.assertEqual(paths.shape,(400,24));self.assertEqual(initial.shape,(24,17))
        self.assertEqual(len(set(families)),6)
        self.assertTrue(np.isfinite(paths).all() and paths.min()>=-100 and paths.max()<=60)

if __name__=='__main__':unittest.main()
