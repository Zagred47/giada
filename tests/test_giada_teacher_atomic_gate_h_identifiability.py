from pathlib import Path
import unittest

from src.giada_teacher.atomic_gate_h_identifiability import (
    GateHIdentifiabilityConfig,
    prepare_gate_h_identifiability,
)
from src.giada_teacher.double_oracle import ExtractedGateFormula


TEACHER_MOD = (
    Path(__file__).resolve().parents[2]
    / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod"
)


class GateHIdentifiabilityTests(unittest.TestCase):
    def test_registered_factorial_is_valid(self) -> None:
        GateHIdentifiabilityConfig().validate()

    def test_fresh_confirmation_is_disjoint(self) -> None:
        bundle = prepare_gate_h_identifiability(ExtractedGateFormula.from_mod(TEACHER_MOD))
        self.assertFalse(bundle["contract"]["task2_rows_reused_for_fresh"])
        self.assertFalse(bundle["contract"]["fresh_used_for_selection"])
        self.assertTrue(all(count > 0 for count in bundle["contract"]["fresh_counts"].values()))

