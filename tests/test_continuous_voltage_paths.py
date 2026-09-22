import unittest

import numpy as np

from src.giada_teacher.continuous_voltage_paths import (
    ContinuousPathConfig, FAMILIES, formula_path_targets, make_continuous_paths,
)


class FakeFormula:
    def rates(self, voltage):
        return {"m_inf": 1 / (1 + np.exp(-(voltage + 35) / 8)),
                "h_inf": 1 / (1 + np.exp((voltage + 55) / 7)),
                "m_tau_ms": 1., "h_tau_ms": 10.}


class ContinuousVoltagePathTests(unittest.TestCase):
    def test_reproducible_disjoint_bounded_and_genuinely_continuous(self):
        config = ContinuousPathConfig()
        development = make_continuous_paths(config, role="development")
        sealed = make_continuous_paths(config, role="sealed")
        self.assertEqual(tuple(development), FAMILIES)
        self.assertTrue(all(np.array_equal(rows, make_continuous_paths(config, role="development")[key])
                            for key, rows in development.items()))
        self.assertFalse({tuple(row) for rows in development.values() for row in rows} &
                         {tuple(row) for rows in sealed.values() for row in rows})
        for rows in development.values():
            self.assertEqual(rows.shape, (128, 42))
            self.assertTrue(np.isfinite(rows).all())
            self.assertTrue(((rows[:, :40] >= -120) & (rows[:, :40] <= 60)).all())
            self.assertTrue(((rows[:, -2:] > 0) & (rows[:, -2:] < 1)).all())
        self.assertGreater(np.mean(np.abs(np.diff(development["high_chirp"][:, :40], axis=1))),
                           np.mean(np.abs(np.diff(development["low_chirp"][:, :40], axis=1))))

    def test_path_information_controls_are_distinct(self):
        config = ContinuousPathConfig()
        paths = make_continuous_paths(config, role="development")
        truth, coarse, start = formula_path_targets(FakeFormula(), paths, config)
        for family in FAMILIES:
            self.assertTrue(np.isfinite(truth[family]).all())
            self.assertGreater(float(np.max(np.abs(truth[family] - start[family]))), 1e-3)
            self.assertGreater(float(np.max(np.abs(truth[family] - coarse[family]))), 1e-5)


if __name__ == "__main__":
    unittest.main()
