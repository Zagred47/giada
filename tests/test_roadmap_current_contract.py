import unittest

import numpy as np

from src.giada_teacher.roadmap_current_contract import (
    analytical_conductance,
    analytical_current,
)


class CurrentContractTests(unittest.TestCase):
    def test_analytic_current_and_inward_sign(self):
        gbar = np.array([1e-5, 2e-5])
        voltage = np.array([-60.0, -40.0])
        m = np.array([0.2, 0.3])
        h = np.array([0.8, 0.5])
        expected_g = gbar * m**2 * h
        np.testing.assert_allclose(analytical_conductance(gbar, m, h), expected_g)
        np.testing.assert_allclose(
            analytical_current(gbar, voltage, m, h), expected_g * (voltage-120.0)
        )
        self.assertTrue(np.all(analytical_current(gbar, voltage, m, h) < 0))

    def test_zero_gbar_cannot_identify_timing(self):
        pre = analytical_current(0, -70, 0.1, 0.9)
        post = analytical_current(0, 20, 0.9, 0.1)
        self.assertEqual(float(pre), float(post))


if __name__ == "__main__":
    unittest.main()
