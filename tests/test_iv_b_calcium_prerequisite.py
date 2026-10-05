import math
import unittest

from src.giada_teacher.iv_b_calcium_prerequisite import calcium_exact, sk_inf


class CalciumPrerequisiteTests(unittest.TestCase):
    def test_zero_current_relaxes_to_minimum(self):
        start = 5e-4
        result = calcium_exact(start, 0.0, 80.0, .05, .1, 80.0)
        self.assertAlmostEqual(result, 1e-4 + (start-1e-4)/math.e)

    def test_inward_current_raises_calcium(self):
        rest = calcium_exact(1e-4, 0.0, .1, .05, .1, 80.0)
        inward = calcium_exact(1e-4, -.005, .1, .05, .1, 80.0)
        self.assertGreater(inward, rest)

    def test_depth_and_gamma_change_source_predictably(self):
        base = calcium_exact(1e-4, -.005, .1, .05, .1, 80.0) - 1e-4
        doubled_gamma = calcium_exact(1e-4, -.005, .1, .1, .1, 80.0) - 1e-4
        doubled_depth = calcium_exact(1e-4, -.005, .1, .05, .2, 80.0) - 1e-4
        self.assertAlmostEqual(doubled_gamma/base, 2.0)
        self.assertAlmostEqual(doubled_depth/base, .5)

    def test_sk_activation_monotonic_and_bounded(self):
        values = [sk_inf(ca) for ca in (1e-7, 1e-4, 4.3e-4, 1e-3)]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertAlmostEqual(values[2], .5)


if __name__ == '__main__':
    unittest.main()
