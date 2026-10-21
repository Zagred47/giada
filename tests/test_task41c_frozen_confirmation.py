import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41c_frozen_confirmation as c
ROOT=Path(__file__).resolve().parents[1]

class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.spec,_,_,_=c.contract(ROOT)
    def test_unchanged_thresholds(self):
        old=json.loads((ROOT/c.x.CONFIG).read_text())
        self.assertEqual(self.spec['current_quality'],old['current_quality'])
        self.assertEqual(self.spec['compartment_gates'],old['compartment_gates'])
        self.assertFalse(set(old['protocols'])&set(self.spec['protocols']))
        self.assertFalse(set(old['native_seeds']+old['stress_seeds'])&set(self.spec['native_seeds']+self.spec['stress_seeds']))
    def test_factorial_and_causal_inputs(self):
        rows=c.design(self.spec);self.assertEqual(len(rows),176)
        self.assertEqual(sum(r['intervention']=='none' for r in rows),64)
        for row in rows:
            inj,mult=c.case_inputs(row,self.spec)
            self.assertEqual(inj.shape,(400,));self.assertTrue(np.isfinite(inj).all())
            if row['intervention'].startswith('block_'):self.assertEqual(mult[c.local.ionic.CHANNELS.index(row['intervention'][6:])],0)
    def test_frozen_cpu_smoke(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp,torch.inference_mode():result=c.smoke(torch,ROOT,Path(tmp),'cpu')
        self.assertTrue(result['valid']);self.assertFalse(result['fresh_confirmation_accessed'])
        model=c.load_models(torch,ROOT,'cpu')['P110']
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
    def test_adapter_restores_immutable_module(self):
        original=c.x.contract
        with c.native_adapter(ROOT):self.assertEqual(c.x.contract(ROOT)[0]['task'],'41c')
        self.assertIs(c.x.contract,original)
    def test_stress_pairs(self):
        rows=c.stress_design(self.spec);self.assertEqual(len(rows),12)
        for a,b in zip(rows[::2],rows[1::2]):
            self.assertTrue(np.array_equal(a['v'],b['v']))
            self.assertTrue(np.array_equal(a['ca'],b['ca']))
            self.assertFalse(np.array_equal(a['initial'],b['initial']))
if __name__=='__main__':unittest.main()
