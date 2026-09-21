from pathlib import Path
import unittest

from src.giada_teacher.double_oracle import ExtractedGateFormula
from src.giada_teacher.joint_gate_symmetric_confirmation import (
    JointGateSymmetricConfirmationConfig,
    prepare_joint_gate_symmetric_confirmation,
)


TEACHER_MOD = Path(__file__).resolve().parents[2] / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod"


class JointGateSymmetricConfirmationTests(unittest.TestCase):
    def test_registered_vectorized_design_is_valid(self):
        config = JointGateSymmetricConfirmationConfig()
        config.validate()
        self.assertEqual(config.seeds, (17, 29, 43))
        self.assertEqual(config.checkpoints[-1], 50000)
        self.assertTrue(config.enable_torch_compile)

    def test_new_fresh_is_disjoint_from_all_prior_rows(self):
        bundle = prepare_joint_gate_symmetric_confirmation(ExtractedGateFormula.from_mod(TEACHER_MOD))
        contract = bundle["confirmation_contract"]
        self.assertEqual(contract["overlap_with_task3_or_development"], 0)
        self.assertFalse(contract["fresh_used_for_training_or_freeze"])
        self.assertTrue(all(value > 0 for value in contract["new_fresh_counts"].values()))


if __name__ == "__main__":
    unittest.main()
