"""CPU preflight for the original Task28 coverage and analytic interface."""
import json
import unittest
from pathlib import Path

import numpy as np

from src.giada_teacher import ionic_block_teacher_forced as task


ROOT = Path(__file__).resolve().parents[1]


class IonicTask28Tests(unittest.TestCase):
    def test_parent_and_channel_contract(self):
        cfg = task.config(ROOT)
        report, freeze, archive = task.verify_parent(ROOT, cfg)
        self.assertTrue(report['task28_preparation_authorized'])
        self.assertTrue(archive.is_file())
        self.assertEqual(len(task.CHANNELS), 11)
        self.assertEqual(task.OFFSETS[-1], 18)
        self.assertEqual(len(freeze['checkpoint_hashes']), 40)
        self.assertFalse(cfg['training_performed'])
        self.assertEqual(tuple(cfg['canonical_formula_channels']), task.CHANNELS[5:])

    def test_exact_identity_and_bounded_states(self):
        cfg = task.config(ROOT)
        voltage, calcium, state, dt = task.support(280401, 256, cfg, 'calcium_tail')
        result = task.exact_step(voltage, calcium, state, dt)
        self.assertEqual(result.shape, (256, 18))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())
        self.assertEqual(task.currents(voltage, result).shape, (256, 11))
        metric = task.measure(voltage, result, result.copy(), cfg)
        self.assertTrue(metric['finite'])
        self.assertTrue(all(v['rmse'] == 0 for v in metric['per_channel_gates'].values()))
        self.assertTrue(all(v['total_normalized_rmse'] == 0 for v in metric['current_panels'].values()))

    def test_canonical_extra_rates(self):
        voltage = np.array([-100., -65., 0., 50.])
        calcium = np.array([1e-7, 1e-5, 1e-4, 1e-2])
        for channel in task.CHANNELS:
            inf, tau = task.rates(channel, voltage, calcium)
            width = len(task.STATE_NAMES[task.CHANNELS.index(channel)])
            self.assertEqual(np.asarray(inf).shape, (4, width) if width == 2 else (4,))
            self.assertTrue(np.isfinite(inf).all())
            self.assertTrue(np.isfinite(tau).all())
            self.assertTrue((np.asarray(tau) > 0).all())
        self.assertAlmostEqual(float(task.rates('SK_E2', voltage[:1], calcium[:1])[1][0]), 1.)

    def test_current_sign_and_channel_isolation(self):
        voltage = np.array([-65.])
        state = np.ones((1, 18))
        currents = task.currents(voltage, state)[0]
        self.assertLess(currents[0], 0)  # calcium inward at -65 mV
        self.assertGreater(currents[7], 0)  # potassium outward at -65 mV
        state[0, task.OFFSETS[8]] = 0.
        changed = task.currents(voltage, state)[0]
        self.assertEqual(changed[8], 0.)
        np.testing.assert_array_equal(np.delete(changed, 8), np.delete(currents, 8))


if __name__ == '__main__':
    unittest.main()
