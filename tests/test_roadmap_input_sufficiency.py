import unittest

import numpy as np

from src.giada_teacher.roadmap_input_sufficiency import (
    VIEWS, SUBSTEPS, integrate_view, reconstruct_voltage_path,
)
from src.giada_teacher.physiological_path_floor import integrate_recorded_path


class DummyFormula:
    @staticmethod
    def rates(voltage):
        return {"m_inf": 1 / (1 + np.exp(-voltage / 20)),
                "h_inf": 1 / (1 + np.exp(voltage / 20)),
                "m_tau_ms": .2, "h_tau_ms": 1.5}


class RoadmapInputSufficiencyTests(unittest.TestCase):
    def setUp(self):
        self.path = np.array([-70] * 10 + [20] * 20 + [-70] * 11, dtype=float)
        self.initial = np.array([.2, .8])

    def test_all_views_have_registered_shape(self):
        for view in VIEWS:
            values = reconstruct_voltage_path(self.path, view)
            self.assertEqual(values.shape, (41,))
            self.assertTrue(np.isfinite(values).all())
        self.assertEqual(SUBSTEPS, (1, 2, 4, 8, 40))

    def test_full_path_40_reproduces_midpoint(self):
        actual = integrate_view(DummyFormula(), self.path, self.initial, "full41_oracle", 40)
        expected = integrate_recorded_path(DummyFormula(), self.path, self.initial, "midpoint")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)

    def test_same_endpoints_do_not_imply_same_gate(self):
        other = np.array([-70] * 25 + [20] * 5 + [-70] * 11, dtype=float)
        self.assertEqual(self.path[0], other[0])
        self.assertEqual(self.path[-1], other[-1])
        for view in ("start_only", "endpoints_oracle"):
            a = integrate_view(DummyFormula(), self.path, self.initial, view, 40)
            b = integrate_view(DummyFormula(), other, self.initial, view, 40)
            np.testing.assert_array_equal(a, b)
        a = integrate_view(DummyFormula(), self.path, self.initial, "full41_oracle", 40)
        b = integrate_view(DummyFormula(), other, self.initial, "full41_oracle", 40)
        self.assertGreater(float(np.max(np.abs(a - b))), .001)


if __name__ == "__main__":
    unittest.main()
