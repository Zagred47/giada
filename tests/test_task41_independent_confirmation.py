import json,unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41_independent_confirmation as c
ROOT=Path(__file__).resolve().parents[1]
class Task41Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.spec,cls.cfg,cls.calcium,cls.base=c.contract(ROOT)
    def test_frozen_and_unchanged_gates(self):
        self.assertEqual(self.spec['model_seeds'],[17,29,43]);self.assertEqual(self.spec['candidate'],'B6')
        self.assertEqual(self.spec['current_quality'],json.loads((ROOT/c.t.CONFIG).read_text())['current_quality'])
        self.assertFalse(set(self.spec['native_seeds'])&{85317,85329,17,29,43,self.spec['calibration_seed']})
    def test_factorial_and_interventions(self):
        self.assertEqual(c.calibration_config(self.spec,self.cfg)['receptor_time_semantics'],'ems_fixed_step')
        rows=c.design(self.spec);self.assertEqual(len(rows),176)
        self.assertEqual(sum(r['intervention']=='none' for r in rows),64)
        for row in rows:
            inj,mult=c.case_inputs(row,self.spec);self.assertEqual(inj.shape,(400,));self.assertEqual(mult.shape,(11,))
            if row['intervention'].startswith('block_'):self.assertEqual(mult[c.ionic.CHANNELS.index(row['intervention'][6:])],0)
    def test_history_same_inputs_different_initial_states(self):
        rows=c.stress_design(self.spec);self.assertEqual(len(rows),12)
        for i in range(0,len(rows),2):
            self.assertTrue(np.array_equal(rows[i]['v'],rows[i+1]['v']))
            self.assertTrue(np.array_equal(rows[i]['ca'],rows[i+1]['ca']))
            self.assertFalse(np.array_equal(rows[i]['initial'],rows[i+1]['initial']))
    def test_old_native_phase_alignment(self):
        proof=json.loads((ROOT/'experiments/results/task41_protocol_v1/old_native_phase_preflight.json').read_text())
        self.assertEqual(proof['passed'],32);self.assertEqual(proof['unaligned_endpoint_passed'],7)
        self.assertLess(proof['one_step_phase_checks']['(1, 1)']['max_abs'],1e-12)
        self.assertFalse(proof['fresh_confirmation_accessed'])
    def test_no_future_native_input_in_autonomous_rollout(self):
        data=dict(voltage=np.zeros((2,5)),calcium=np.full((2,5),1e-4),gates=np.full((2,5,18),.5),
            multipliers=np.ones((2,11)),injection=np.zeros((2,4)),base_g_us=np.zeros((2,4,4)))
        def core(v,ca,g,m,i,s):return v+.1,ca,g,torch.zeros_like(m)
        with torch.inference_mode():
            first=c.t.rollout(torch,core,data,'cpu')
            for key in ('voltage','calcium','gates'):data[key][:,1:]=np.nan
            second=c.t.rollout(torch,core,data,'cpu')
        for key in first:self.assertTrue(np.array_equal(first[key],second[key]))
    def test_frozen_model_smoke(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as output,torch.inference_mode():
            result=c.evaluate(torch,ROOT,Path(output),device='cpu',smoke=True)
        self.assertTrue(result['valid']);self.assertTrue(result['all_parameters_finite']);self.assertFalse(result['fresh_confirmation_accessed'])
    def test_lut_non_sk_accuracy(self):
        lut=c.lut_model(torch,'cpu',.1);v=torch.linspace(-99,59,64)[None].expand(3,-1);ca=torch.full_like(v,1e-4);g=torch.full((3,64,18),.5)
        actual=lut(v,ca,g).numpy();u,z=c.p.coefficients(v.numpy().astype('float64'),ca.numpy().astype('float64'),.1)
        expected=g.numpy()+z*(u-g.numpy());self.assertLess(np.max(np.abs(actual-expected)),1e-5)
if __name__=='__main__':unittest.main()
