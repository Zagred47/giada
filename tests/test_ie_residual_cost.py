import unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import cc_n1_contract_memory as n1
from src.giada_teacher.ie_residual_cost import functions,differences,targeting


class ResidualCostTests(unittest.TestCase):
    def test_recomposition_and_all_cache_interventions_preserve_outputs(self):
        root=Path(__file__).resolve().parents[1]
        _,cfg,calcium,_,base=n1.contract(root)
        data=np.load(root/'experiments/results/task35_native_shadow_traces.npz')
        rng=np.random.default_rng(81503)
        ids=np.tile(np.arange(32),4);ticks=rng.integers(0,400,len(ids))
        def tensor(a):
            x=torch.tensor(a,dtype=torch.float32)
            return x[None].expand(3,*x.shape).contiguous()
        v,ca,g=[tensor(data[k][ids,ticks]) for k in ('voltage','calcium','gates')]
        funcs=functions(torch,v,ca,g,tensor(data['multipliers'][ids]),
            tensor(data['injection'][ids,ticks]),tensor(data['base_g_us'][ids,ticks]),
            float(data['area_um2']),cfg,calcium,base)
        full=funcs['full']()
        for a,b in zip(full[:3],funcs['legacy_abi']()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        for mode in ('full','cached_gates','cached_conductance','cached_prefix','cached_voltage','cached_both'):
            actual=funcs[mode]()
            self.assertTrue(differences(torch,actual,full,[0]*5)['valid'],mode)
            compiled=torch.compile(funcs[mode],backend='eager',fullgraph=True)
            self.assertTrue(differences(torch,compiled(),full,[0]*5)['valid'],mode)

    def test_nonfinite_preflight_cannot_pass(self):
        result=differences(torch,torch.tensor([float('nan')]),torch.tensor([1.]),[1.])
        self.assertFalse(result['valid'])

    def test_targeting_preserves_missing_graphs_and_abi_warning(self):
        rows=[{'mode':'compiled','batch_per_seed':642,'variant':'full','timings':{'wall_ms':[2.]*30}},
              {'mode':'compiled','batch_per_seed':642,'variant':'legacy_abi','timings':{'wall_ms':[1.]*30}},
              {'mode':'compiled','batch_per_seed':642,'variant':'cached_prefix','timings':{'wall_ms':[1.]*30},
               'saving_vs_full':{'median_saved_fraction':.5,'bootstrap_95pct':[.5,.5]}}]
        result=targeting({'rows':rows},642)
        self.assertEqual(result['execution'],[])
        group=result['block_groups'][0]
        self.assertTrue(group['abi_perturbed'])
        self.assertTrue(group['hypothesis_flags']['H_ionic'])
        self.assertIsNone(group['hypothesis_flags']['H_solver'])


if __name__=='__main__':unittest.main()
