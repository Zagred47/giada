import json,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41b_local_precision as c
ROOT=Path(__file__).resolve().parents[1]

class Task41bTests(unittest.TestCase):
    def test_factorial_and_thresholds(self):
        s,_,_,_=c.contract(ROOT)
        self.assertEqual(len(c.arms()),8)
        self.assertEqual(s['quality'],json.loads((ROOT/c.prior.CONFIG).read_text())['current_quality'])
        self.assertEqual(c.TARGETS,(('u',8),('z',7),('z',11)))
    def test_independent_supports_and_labels(self):
        a,b=c.atomic(141101,32),c.atomic(141101,32,True)
        self.assertEqual(a['v'].shape,b['v'].shape)
        self.assertFalse(np.array_equal(a['v'],b['v']))
        u,z=c.p.coefficients(a['v'].astype('float64'),a['ca'].astype('float64'),.1)
        np.testing.assert_array_equal(u,a['u']);np.testing.assert_array_equal(z,a['z'])
    def test_optimizer_and_deployed_function(self):
        torch.set_num_threads(1);spec,_,_,_=c.contract(ROOT)
        with patch.object(c.p,'save_json'):
            r=c.preflight(torch,ROOT,ROOT/'artifacts/task41b_test','cpu',spec)
        self.assertTrue(r['valid']);self.assertTrue(r['zero_initialization_exact']);self.assertTrue(r['unaffected_functions_exact'])
        self.assertLess(r['optimizer_isolation_max_error'],1e-6)
    def test_development_reproducible_and_disjoint_seed(self):
        a,meta=c.development(141107,2);b,_=c.development(141107,2)
        np.testing.assert_array_equal(a['voltage'],b['voltage'])
        self.assertEqual(a['voltage'].shape,(12000,2));self.assertEqual(len(meta),2)

if __name__=='__main__':unittest.main()
