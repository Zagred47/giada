import unittest
from pathlib import Path

import numpy as np

from src.giada_teacher.task34_privileged_current_state_probes import (
    contract, current_factorial, load_traces, teacher_voltage_model_state,
)


ROOT = Path(__file__).resolve().parents[1]


class TestTask34PrivilegedProbes(unittest.TestCase):
    def test_parent_and_trace_contract(self):
        spec, parent, cfg, t32cfg = contract(ROOT)
        native, shadows = load_traces(ROOT, spec)
        self.assertEqual(len(native), 16)
        self.assertTrue(parent['task33_passed'])
        self.assertEqual(len(shadows), 16)
        self.assertEqual(cfg['synaptic_state_phase'], 'old')
        self.assertEqual(t32cfg['model_seeds'], [17, 29, 43])
        exact_clamped = teacher_voltage_model_state(native, cfg, t32cfg)
        self.assertEqual(exact_clamped['gates'].shape, (1, 401, 16, 18))
        self.assertTrue(np.isfinite(exact_clamped['gates']).all())

    def test_current_factorial_identity(self):
        rng = np.random.default_rng(34)
        voltage = np.array([[-70., -55.]])
        states = rng.uniform(.05, .9, (1, 2, 18))
        result = current_factorial(voltage, states,
                                   voltage + np.array([[.1, -.2]]),
                                   states + .001, 1e-12)
        self.assertLess(result['identity_max_abs_error_ma_cm2'], 1e-12)
        self.assertEqual(len(result['per_channel']), 11)


if __name__ == '__main__':
    unittest.main()
