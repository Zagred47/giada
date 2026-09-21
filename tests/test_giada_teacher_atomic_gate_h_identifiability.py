from pathlib import Path
import unittest

from src.giada_teacher.atomic_gate_h_identifiability import (
    GateHIdentifiabilityConfig,
    _physical_tau_decision,
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

    def test_registered_decision_uses_canonical_occupancy_metric(self) -> None:
        metrics = {"rate_supervision": {
            "fresh": {"rmse": 1e-4, "occupancy_violation_count": 0}
        }}
        rollout = {"rate_supervision": {"fresh": {"1000": 1e-3}}}
        rates = {"rate_supervision": {
            "fresh": {"inf_rmse": 1e-3, "log_tau_rmse": 1e-2}
        }}
        decision = _physical_tau_decision(
            metrics, rollout, rates, "rate_supervision", ("fresh",)
        )
        self.assertTrue(decision["physical_tau_repaired"])
        self.assertEqual(decision["winner_occupancy_violations"], 0)
