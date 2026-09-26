import json
import unittest
from pathlib import Path

import numpy as np

from src.giada_teacher.roadmap_current_architecture_comparison import (
    CurrentArchitectureConfig, _counterfactual_roles, _parameterized_role,
    common_tensor, prepare_current_role,
)


class CurrentArchitectureContractTests(unittest.TestCase):
    def test_preregistered_config(self):
        CurrentArchitectureConfig().validate()
        with self.assertRaises(ValueError):
            CurrentArchitectureConfig(steps=401).validate()

    def test_causal_input_and_mask_contract(self):
        config = CurrentArchitectureConfig()
        role = _parameterized_role(config.train_seed, 3, config)
        prepared = prepare_current_role(role)
        x = common_tensor(role)
        self.assertEqual(x.shape, (3 * config.duration_ms, 10))
        np.testing.assert_array_equal(x, prepared["x"])
        np.testing.assert_allclose(x[:, 4], prepared["mask"])
        np.testing.assert_allclose(x[:, 5] * 40 + 120, prepared["eca"])
        self.assertTrue(np.isfinite(prepared["current"]).all())

    def test_paired_counterfactuals_have_visible_mask(self):
        config = CurrentArchitectureConfig()
        roles = _counterfactual_roles(config)
        self.assertIn("baseline", roles)
        self.assertIn("mechanism_off_visible", roles)
        self.assertNotIn("mechanism_off_hidden", roles)
        off = prepare_current_role(roles["mechanism_off_visible"])
        self.assertTrue(np.all(off["x"][:, 4] == 0))
        self.assertTrue(np.all(off["current"] == 0))
        self.assertEqual(off["x"].shape[1], 10)

    def test_notebook_is_valid_json(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/15_roadmap_current_architecture_comparison.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
