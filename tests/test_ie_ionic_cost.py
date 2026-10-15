import unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import cc_n1_contract_memory as n1
from src.giada_teacher.gate_d_compute_audit import exact_gpu
from src.giada_teacher.ie_ionic_cost import core_with_gates, paired_summary


class IonicCostTests(unittest.TestCase):
    def test_cached_gates_preserve_task35_calcium_sk_and_nmda(self):
        root = Path(__file__).resolve().parents[1]
        _, cfg, calcium, _, base = n1.contract(root)
        data = np.load(root/'experiments/results/task35_native_shadow_traces.npz')
        # Independent states across all four conductance stress panels and active ticks.
        e = np.array([0,8,16,24,3,11,19,27])
        t = np.array([20,40,80,120,160,200,240,320])
        def tensor(x):
            return torch.tensor(x,dtype=torch.float32)[None].repeat(3,*([1]*x.ndim))
        v, ca, g = [tensor(data[k][e,t]) for k in ('voltage','calcium','gates')]
        mult, current, syn = tensor(data['multipliers'][e]),tensor(data['injection'][e,t]),tensor(data['base_g_us'][e,t])
        dt = torch.full_like(v,.1)
        cached = exact_gpu(v,ca,g,dt,torch)[0]
        args = (mult,current,syn,float(data['area_um2']),cfg,calcium,base)
        expected = n1.torch_step(torch,v,ca,g,*args)
        actual = core_with_gates(torch,v,ca,g,cached,*args)
        for a,b in zip(expected,actual):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        compiled = torch.compile(lambda vv,cc,gg: core_with_gates(torch,vv,cc,gg,cached,*args),
                                 backend='eager',fullgraph=True)
        for a,b in zip(expected,compiled(v,ca,g)):
            torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_paired_fraction_is_not_ratio_of_component_times(self):
        result = paired_summary([2.,4.,8.],[1.,2.,4.])
        self.assertEqual(result['median_saved_fraction'],.5)
        self.assertEqual(result['bootstrap_95pct'],[.5,.5])


if __name__ == '__main__':
    unittest.main()
