from pathlib import Path
import unittest

from src.giada_teacher.double_oracle import ExtractedGateFormula
from src.giada_teacher.joint_gate_cell_playground import JointGateCellConfig, prepare_joint_gate_dataset


TEACHER_MOD = Path(__file__).resolve().parents[2] / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod"


class JointGateCellTests(unittest.TestCase):
    def test_registered_design_is_valid(self):
        JointGateCellConfig().validate()

    def test_joint_fresh_contract_is_disjoint(self):
        bundle = prepare_joint_gate_dataset(ExtractedGateFormula.from_mod(TEACHER_MOD))
        self.assertEqual(bundle["contract"]["fresh_overlap"], 0)
        self.assertFalse(bundle["contract"]["fresh_used_for_selection"])
        self.assertTrue(all(v > 0 for v in bundle["contract"]["fresh_counts"].values()))


if __name__ == "__main__":
    unittest.main()
