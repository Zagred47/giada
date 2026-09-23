import unittest

import numpy as np

from src.giada_teacher.physiological_path_floor import integrate_recorded_path


class FakeFormula:
    def rates(self, voltage):
        return {"m_inf": 1 / (1 + np.exp(-(voltage + 35) / 8)),
                "h_inf": 1 / (1 + np.exp((voltage + 55) / 7)),
                "m_tau_ms": 1., "h_tau_ms": 10.}


class PhysiologicalPathFloorTests(unittest.TestCase):
    def test_constant_voltage_controls_and_refinement_identity(self):
        formula = FakeFormula()
        path = np.full(41, -45., dtype=np.float64)
        initial = np.array([.1, .9])
        values = {method: integrate_recorded_path(formula, path, initial, method)
                  for method in ("right", "right_refined_5", "left", "midpoint", "linear_5", "linear_20")}
        for value in values.values():
            self.assertTrue(np.allclose(value, values["right"], atol=1e-12))

    def test_linear_path_methods_distinguish_sampling_and_converge(self):
        formula = FakeFormula()
        path = np.linspace(-75., 15., 41)
        initial = np.array([.2, .8])
        right = integrate_recorded_path(formula, path, initial, "right")
        linear5 = integrate_recorded_path(formula, path, initial, "linear_5")
        linear20 = integrate_recorded_path(formula, path, initial, "linear_20")
        self.assertGreater(abs(right[0] - linear20[0]), 1e-4)
        self.assertLess(abs(linear5[0] - linear20[0]), abs(right[0] - linear20[0]))


if __name__ == "__main__":
    unittest.main()
