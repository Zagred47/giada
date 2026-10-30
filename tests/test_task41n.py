"""Synthetic/interface tests only; never generate or judge native confirmation."""
import os,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task41n_independent_confirmation as n
ROOT=Path(__file__).resolve().parents[1]

class ExactConfirmationTest(unittest.TestCase):
    def test_bitwise_guard_detects_signed_zero(self):
        import torch
        self.assertFalse(n.tensor_bits_equal(torch,torch.tensor([0.]),torch.tensor([-0.])))
        self.assertFalse(n.array_bits_equal(np.array([0.]),np.array([-0.])))
    def test_contract_and_native_design(self):
        spec,*_=n.contract(ROOT);rows=n.old_native.x.design(spec)
        self.assertEqual(len(rows),176);self.assertEqual(sum(r['intervention']=='none' for r in rows),64)
        self.assertFalse(set(spec['native_seeds']+spec['stress_seeds'])&set(spec['excluded_input_seeds']))
        self.assertEqual(spec['quality'],n.m.contract(ROOT)[0]['quality'])
        for row in rows:
            inj,mult=n.case_inputs(row,spec);self.assertEqual(inj.shape,(400,));self.assertEqual(mult.shape,(11,));self.assertTrue(np.isfinite(inj).all())
    def test_cpu_exact_logits_and_state(self):
        import torch
        torch.set_num_threads(1)
        os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        nets,_=n.load_models(torch,ROOT,'cpu');report=n.parity(torch,nets,'cpu',n.contract(ROOT)[0])
        self.assertEqual(report['eligible'],list(nets));self.assertTrue(all(r['max_abs_error']==0 for r in report['rows']))
        self.assertTrue(all(not p.requires_grad for net in nets.values() for p in net.parameters()))
    def test_alignment_and_coupled_cpu(self):
        import torch
        torch.set_num_threads(1);os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        report=n.preflight(torch,ROOT,ROOT/'artifacts/task41n_cpu_preflight','cpu')
        self.assertTrue(report['valid']);self.assertFalse(report['fresh_accessed'])

if __name__=='__main__':unittest.main()
