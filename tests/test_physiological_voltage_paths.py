import unittest

import numpy as np

from src.giada_teacher.physiological_voltage_paths import (
    classify_regime, formula_rollouts,
)


class FakeFormula:
    def rates(self, voltage):
        return {"m_inf": 1 / (1 + np.exp(-(voltage + 35) / 8)),
                "h_inf": 1 / (1 + np.exp((voltage + 55) / 7)),
                "m_tau_ms": 1., "h_tau_ms": 10.}


class PhysiologicalPathTests(unittest.TestCase):
    def test_regimes_and_boundary_order(self):
        self.assertEqual(classify_regime(np.full(41, -70.)), "quiet")
        self.assertEqual(classify_regime(np.linspace(-70., -40., 41)), "rising")
        self.assertEqual(classify_regime(np.linspace(-40., -70., 41)), "falling")
        self.assertEqual(classify_regime(np.linspace(-70., 20., 41)), "spike")

    def test_right_boundary_sample_used_for_fine_teacher_path(self):
        path = np.full(41, -70., dtype=np.float64)
        path[0] = 30.  # Old boundary must not be reused as the first future update.
        rows = [{"path": path, "initial": np.array([.1, .9])}]
        values = formula_rollouts(FakeFormula(), rows)
        self.assertGreater(abs(values["fine_path"][0, 0] - values["start_only"][0, 0]), .01)
        self.assertTrue(np.allclose(values["fine_path"], values["coarse_path"], atol=1e-12))


if __name__ == "__main__":
    unittest.main()
