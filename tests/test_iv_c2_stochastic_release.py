import math
import unittest

from src.giada_teacher.iv_c2_stochastic_release import shadow_step


class TestIVC2Shadow(unittest.TestCase):
    def test_zero_facilitation_release_then_depression(self):
        state = {'u': 0., 'Rstate': 1, 'tsyn_fac': 0., 'tsyn': 0.}
        arm = {'Use': .2, 'Fac': 0., 'Dep': 2.}
        state, draws = shadow_step(state, 2., arm, lambda: .1)
        self.assertEqual(draws, 1)
        self.assertTrue(state['released'])
        self.assertEqual(state['Rstate'], 0)
        state, draws = shadow_step(state, 2.25, arm, lambda: 0.)
        self.assertEqual(draws, 1)
        self.assertFalse(state['released'])
        self.assertEqual(state['tsyn'], 2.25)

    def test_recovery_consumes_second_draw_in_same_event(self):
        draws = iter([1., .01])
        state = {'u': .2, 'Rstate': 0, 'tsyn_fac': 2., 'tsyn': 2.}
        state, count = shadow_step(state, 4., {'Use': .2, 'Fac': 8., 'Dep': 2.}, lambda: next(draws))
        self.assertEqual(count, 2)
        self.assertTrue(state['released'])
        self.assertEqual(state['Rstate'], 0)
        self.assertEqual(state['tsyn'], 4.)
        self.assertAlmostEqual(state['u'], .2 * math.exp(-.25) + .2 * (1 - .2 * math.exp(-.25)))

    def test_facilitation_without_release_still_updates(self):
        state = {'u': .2, 'Rstate': 1, 'tsyn_fac': 2., 'tsyn': 0.}
        state, count = shadow_step(state, 2.25, {'Use': .2, 'Fac': 8., 'Dep': 2.}, lambda: 9.)
        self.assertEqual(count, 1)
        self.assertFalse(state['released'])
        self.assertGreater(state['u'], .2)


if __name__ == '__main__':
    unittest.main()
