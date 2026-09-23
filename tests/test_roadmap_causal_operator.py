import unittest

import numpy as np

from src.giada_teacher.roadmap_causal_operator import (
    CausalOperatorConfig, _rates, flatten_role, generate_role,
    integrate_predicted_path, missing_input_counterfactual, numerical_matrix,
)


class CausalOperatorTests(unittest.TestCase):
    def test_reference_coefficients_and_oracle_path(self):
        config = CausalOperatorConfig()
        role = generate_role(11011, 3, config)
        flat = flatten_role(role)
        coefficient = flat["coefficients"]
        np.testing.assert_allclose(
            coefficient[:, 0] * flat["initial"][:, 1:] + coefficient[:, 1],
            flat["end"][:, 1:], rtol=0, atol=1e-12)
        knots = role["states"][:, :, [10, 20, 30, 40], 0].reshape(-1, 4)
        path = integrate_predicted_path(flat["initial"], knots)
        self.assertLess(np.max(np.abs(path[:, 1:] - flat["end"][:, 1:])), .01)
        self.assertTrue(np.isfinite(_rates(np.array([-27., -27.0001]))).all())

    def test_missing_future_current_is_a_true_information_contrast(self):
        role = generate_role(11029, 3, CausalOperatorConfig())
        contrast = missing_input_counterfactual(role)
        self.assertTrue(contrast["same_reduced_features"])
        self.assertGreater(contrast["minimum_voltage_difference_mv"], 0)
        matrix = numerical_matrix(role, CausalOperatorConfig())
        self.assertEqual(matrix["40"]["voltage_rmse_mv"], 0)
        self.assertGreater(matrix["1"]["voltage_rmse_mv"], matrix["8"]["voltage_rmse_mv"])


if __name__ == "__main__":
    unittest.main()
