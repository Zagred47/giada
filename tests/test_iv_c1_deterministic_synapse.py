"""Pure local preflight for IV-C1; native test runs on Kaggle."""

from pathlib import Path
import unittest

import numpy as np

from src.giada_teacher.iv_c1_deterministic_synapse import (
    analytic_trace, load_contract, load_v2_contract, normalization)


ROOT = Path(__file__).resolve().parents[1]
CFG = load_contract(ROOT)
PARAMETERS = {'mg': 1.0,
              'tau_r_AMPA': .2, 'tau_d_AMPA': 1.7,
              'tau_r_NMDA': .29, 'tau_d_NMDA': 43.,
              'tau_r_GABAA': .2, 'tau_d_GABAA': 8.,
              'tau_r_GABAB': 3.5, 'tau_d_GABAB': 260.9}


class IVClPreflight(unittest.TestCase):
    def test_v2_state_sampling_changes_only_event_instant(self):
        v2 = load_v2_contract(ROOT)
        t = np.arange(801) * CFG['dt_ms']
        v = np.full_like(t, -75.)
        schedule = {'exc_ms': [4.0], 'inh_ms': [4.0]}
        old = analytic_trace(t, v, schedule, CFG, 0, PARAMETERS)
        new = analytic_trace(t, v, schedule, v2, 0, PARAMETERS)
        at_event = 160
        self.assertGreater(float(old['A_AMPA'][at_event]), 0.)
        self.assertEqual(float(new['A_AMPA'][at_event]), 0.)
        self.assertAlmostEqual(float(old['A_AMPA'][at_event+1]),
                               float(new['A_AMPA'][at_event+1]), places=12)
        np.testing.assert_allclose(old['g_AMPA'], new['g_AMPA'], atol=1e-15)

    def test_peak_normalization(self):
        for receptor in ('AMPA', 'NMDA', 'GABAA', 'GABAB'):
            r, d = PARAMETERS['tau_r_' + receptor], PARAMETERS['tau_d_' + receptor]
            peak = r * d / (d - r) * np.log(d / r)
            self.assertAlmostEqual(normalization(r, d) *
                (np.exp(-peak / d) - np.exp(-peak / r)), 1., places=12)

    def test_event_phase_and_reversal(self):
        t = np.arange(801) * CFG['dt_ms']
        v = np.full_like(t, -75.)
        schedule = CFG['calibration_schedule']
        zero = analytic_trace(t, v, {'exc_ms': [], 'inh_ms': []}, CFG, 0, PARAMETERS)
        self.assertEqual(float(np.max(abs(zero['g_AMPA']))), 0.)
        same = analytic_trace(t, v, schedule, CFG, 0, PARAMETERS)
        shifted = analytic_trace(t, v, schedule, CFG, 1, PARAMETERS)
        self.assertLess(float(same['i_AMPA'].min()), 0.)
        self.assertGreater(float(same['i_GABAA'].max()), 0.)
        self.assertNotEqual(float(same['g_AMPA'][81]), float(shifted['g_AMPA'][81]))

    def test_mg_block(self):
        t = np.arange(801) * CFG['dt_ms']
        schedule = CFG['calibration_schedule']
        low = analytic_trace(t, np.full_like(t, -75.), schedule, CFG, 0, PARAMETERS)
        high = analytic_trace(t, np.full_like(t, 20.), schedule, CFG, 0, PARAMETERS)
        self.assertGreater(float(high['g_NMDA'].max()), float(low['g_NMDA'].max()))


if __name__ == '__main__':
    unittest.main()
