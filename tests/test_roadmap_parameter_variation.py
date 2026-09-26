import unittest

import numpy as np

from src.giada_teacher.roadmap_causal_operator import CausalOperatorConfig, flatten_role, generate_role
from src.giada_teacher.roadmap_parameter_variation import (
    _eca_one_step_twins, _reference_role, _scenario_definitions,
    ParameterVariationConfig,
)


class ParameterVariationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = generate_role(14059, 3, CausalOperatorConfig())

    def test_parameterized_baseline_reproduces_task11(self):
        reference = _reference_role(self.base)
        np.testing.assert_array_equal(reference["states"], self.base["states"])

    def test_eca_is_missing_from_numeric_input_but_changes_target(self):
        low = _eca_one_step_twins(self.base, eca=100.)
        high = _eca_one_step_twins(self.base, eca=140.)
        np.testing.assert_array_equal(flatten_role(low)["x_full"], flatten_role(high)["x_full"])
        self.assertGreater(np.max(np.abs(low["states"]-high["states"])), 0.)

    def test_visible_off_uses_zero_effective_gbar(self):
        off = _reference_role(self.base, mask=0.)
        self.assertTrue(np.all(off["gbar"] == 0.))
        self.assertTrue(np.all(flatten_role(off)["x_full"][:, 3] == 0.))

    def test_matrix_is_registered_size(self):
        self.assertEqual(len(_scenario_definitions(ParameterVariationConfig())), 9)


if __name__ == "__main__":
    unittest.main()
