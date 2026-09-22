import unittest

import numpy as np

from src.giada_teacher.voltage_path_stress import (
    VoltagePathStressConfig, formula_targets, make_paths,
)


class FakeFormula:
    def rates(self, voltage):
        return {"m_inf": 1 / (1 + np.exp(-(voltage + 35) / 8)),
                "h_inf": 1 / (1 + np.exp((voltage + 55) / 7)),
                "m_tau_ms": 1., "h_tau_ms": 10.}


class VoltagePathStressTests(unittest.TestCase):
    def test_paths_are_deterministic_disjoint_and_bounded(self):
        config = VoltagePathStressConfig()
        a = make_paths(config, role="development")
        b = make_paths(config, role="development")
        sealed = make_paths(config, role="sealed")
        self.assertEqual(set(a), {"step_up", "step_down", "ramp", "biphasic", "near_singularity", "voltage_extremes"})
        self.assertTrue(all(np.array_equal(a[key], b[key]) for key in a))
        self.assertTrue(all((row[:, :8] >= -120).all() and (row[:, :8] <= 60).all() for row in a.values()))
        self.assertFalse({tuple(row) for rows in a.values() for row in rows} &
                         {tuple(row) for rows in sealed.values() for row in rows})

    def test_known_path_differs_from_start_voltage_when_voltage_changes(self):
        config = VoltagePathStressConfig()
        paths = {"probe": np.array([[-70., -70., -70., -70., 20., 20., 20., 20., .1, .9]])}
        truth, held = formula_targets(FakeFormula(), paths, config)
        self.assertGreater(abs(truth["probe"][0, 0] - held["probe"][0, 0]), .01)
        self.assertTrue(np.isfinite(truth["probe"]).all())


if __name__ == "__main__":
    unittest.main()
