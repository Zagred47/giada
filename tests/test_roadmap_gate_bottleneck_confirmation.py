import ast
import json
import unittest
from pathlib import Path

import numpy as np

from src.giada_teacher.roadmap_gate_bottleneck_confirmation import (
    ARMS, GateBottleneckConfig, verified_task15_result,
)
from src.giada_teacher.roadmap_current_architecture_comparison import (
    _parameterized_role, prepare_current_role,
)


class GateBottleneckContractTests(unittest.TestCase):
    def test_preregistered_matrix_and_config(self):
        self.assertEqual(len(ARMS), 4)
        self.assertEqual(set(ARMS.values()), {(4.0, 0.0), (16.0, 0.0),
                                              (4.0, 1.0), (16.0, 1.0)})
        GateBottleneckConfig().validate()
        with self.assertRaises(ValueError):
            GateBottleneckConfig(steps=801).validate()

    def test_new_roles_have_finite_causal_input(self):
        config = GateBottleneckConfig()
        role = _parameterized_role(config.train_seed, 3, config)
        prepared = prepare_current_role(role)
        self.assertEqual(prepared["x"].shape, (48, 10))
        self.assertTrue(np.isfinite(prepared["current"]).all())
        self.assertNotIn(config.sealed_seed, (15059, 15159))

    def test_original_task15_artifact_exact(self):
        path = Path.home() / "Downloads/giada_roadmap_task15_current_architecture_comparison.zip"
        if not path.is_file():
            self.skipTest("Task15 user artifact unavailable")
        result = verified_task15_result(path)
        self.assertEqual(result["code_revision"],
                         "c4a91c79f853225840fe548e1a4aab170539dab3")

    def test_notebook_code_parses(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/15b_roadmap_gate_bottleneck_confirmation.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
