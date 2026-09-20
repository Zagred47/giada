from pathlib import Path
import unittest

import numpy as np

from src.giada_teacher.atomic_gate_playground import (
    AtomicGateTaskConfig,
    build_atomic_gate_models,
    materialize_atomic_gate_dataset,
)
from src.giada_teacher.double_oracle import ExtractedGateFormula


TEACHER_MOD = (
    Path(__file__).resolve().parents[2]
    / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod"
)


try:
    import torch
except ImportError:
    torch = None


class AtomicGatePlaygroundTests(unittest.TestCase):
    def test_task1_dataset_matches_registered_split_and_teacher(self) -> None:
        dataset = materialize_atomic_gate_dataset(ExtractedGateFormula.from_mod(TEACHER_MOD))
        self.assertEqual(dataset["gate"], "m")
        self.assertEqual(dataset["strata"]["train"]["inputs"].shape, (4048, 3))
        self.assertEqual(sum(len(row["inputs"]) for row in dataset["strata"].values()), 6291)
        self.assertTrue(all(np.isfinite(row["targets"]).all() for row in dataset["strata"].values()))

    @unittest.skipUnless(torch is not None, "PyTorch is only required on the Kaggle execution host")
    def test_structured_models_preserve_gate_domain(self) -> None:
        values = torch.tensor([[-120.0, 0.0, 0.025], [-27.0, 0.5, 1.0], [60.0, 1.0, 2.0]])
        models = build_atomic_gate_models(torch_module=torch)
        for name in ("physical_tau", "direct_z"):
            prediction = models[name](values)
            self.assertTrue(bool(torch.all(prediction >= 0.0)))
            self.assertTrue(bool(torch.all(prediction <= 1.0)))

    def test_task1_config_rejects_wrong_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "restricted"):
            AtomicGateTaskConfig(gate="h").validate()
