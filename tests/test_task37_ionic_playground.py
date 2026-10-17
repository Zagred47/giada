import json
from pathlib import Path
import unittest
import numpy as np
from src.giada_teacher import task37_ionic_playground as p

ROOT = Path(__file__).resolve().parents[1]


class IonicPlaygroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT / p.CONFIG).read_text())

    def test_controls_and_no_hidden_total_cancellation(self):
        report = p.sensitivity_controls(self.spec)
        self.assertTrue(report['valid'])
        self.assertEqual(len(report['controls_detected']), 8)
        cancellation = report['controls_detected']['opposing_current_cancellation']
        self.assertLess(cancellation['total_current_rmse_ma_cm2'], 1e-17)
        self.assertFalse(cancellation['passed'])

    def test_split_histories_and_forbidden_confirmation(self):
        a, ma = p.make_paths(self.spec, 'fit', 'steps')
        b, mb = p.make_paths(self.spec, 'development', 'steps')
        self.assertFalse(set(x['input_sha256'] for x in ma) & set(x['input_sha256'] for x in mb))
        aa, mma = p.make_paths(self.spec, 'fit', 'steps')
        self.assertTrue(np.array_equal(a['voltage'], aa['voltage']))
        self.assertEqual(ma, mma)
        with self.assertRaises(ValueError):
            p.make_paths(self.spec, 'confirmation', 'steps')

    def test_current_sign_block_reversal_and_source(self):
        v = np.array([[-60., 0.]])
        gates = np.full((1, 2, 18), .5)
        mult = np.ones((2, 11)); mult[:, 3] = 0
        rev = np.tile(p.ionic.REVERSALS, (2, 1)); rev[:, 5] = v[0]
        g, i = p.readout(v, gates, mult, rev)
        self.assertTrue((i[..., :2] < 0).all())
        self.assertTrue((i[..., 3] == 0).all())
        self.assertTrue((i[..., 5] == 0).all())
        gg, ii, source = p.recurrent_reference(np.tile(v, (20, 1)), np.full((20, 2), 1e-4), gates[0], mult, rev, .1)
        self.assertTrue(np.array_equal(source, ii[..., :2].sum(-1)))
        self.assertTrue(np.isfinite(gg).all())

    def test_no_event_is_not_passed_behavior(self):
        z = np.zeros((100, 11))
        result = p.evaluate_currents(z, z, .1, self.spec['quality'])
        self.assertTrue(result['passed'])
        self.assertTrue(all(x['event']['status'] == 'not_evaluable' for x in result['channels']))
        self.assertFalse(result['causal_behavior_validated'])
        self.assertIsNone(result['first_tolerance_exceedance_ms'])

    def test_corrupt_shapes_and_occupancy_fail(self):
        z = np.zeros((100, 11)); g = np.zeros((100, 18)); bad = g.copy(); bad[:, 0] = 1.1
        with self.assertRaises(ValueError):
            p.evaluate_currents(z, z[:-1], .1, self.spec['quality'])
        result = p.evaluate_currents(z, z, .1, self.spec['quality'], g, bad)
        self.assertFalse(result['passed'])
        self.assertEqual(result['gate']['occupancy_violations'], 100)

    def test_real_memory_reset_is_detected(self):
        v = np.full((800, 1), -40.)
        ca = np.full_like(v, 1e-4)
        initial = np.full((1, 18), .8)
        mult = np.ones((1, 11)); rev = p.ionic.REVERSALS[None]
        g, i, _ = p.recurrent_reference(v, ca, initial, mult, rev, .1)
        bg, bi, _ = p.recurrent_reference(v, ca, initial, mult, rev, .1, reset_tick=400)
        result = p.evaluate_currents(i[:, 0], bi[:, 0], .1, self.spec['quality'], g[:, 0], bg[:, 0])
        self.assertFalse(result['passed'])

    def test_common_voltage_does_not_remove_memory(self):
        v = np.full((1000, 2), -55.)
        initial = np.zeros((2, 18)); initial[1] = 1
        g, currents, _ = p.recurrent_reference(v, np.full_like(v, 1e-4), initial, np.ones((2, 11)), np.tile(p.ionic.REVERSALS, (2, 1)), .1)
        self.assertGreater(np.max(np.abs(currents[:, 0]-currents[:, 1])), 1e-5)
        self.assertGreater(abs(g[-1, 0, 9]-g[-1, 1, 9]), .1)

    def test_intervention_cannot_be_hidden_by_identical_base(self):
        r = np.full((100, 11), 1e-4)
        result = p.evaluate_intervention(r, r*2, r, r, .1, self.spec['quality'])
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
