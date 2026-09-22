import unittest

import numpy as np

from src.giada_teacher.joint_gate_full_repair_confirmation import (
    JointGateFullRepairConfirmationConfig,
    _sealed_rows,
)


class _Formula:
    def rates(self, voltage):
        return {
            "m_inf": 1.0 / (1.0 + np.exp(-voltage / 20.0)),
            "m_tau_ms": 1.0 + abs(voltage) / 100.0,
            "h_inf": 1.0 / (1.0 + np.exp(voltage / 20.0)),
            "h_tau_ms": 5.0 + abs(voltage) / 50.0,
        }

    def step(self, gate, state, voltage, dt):
        rates = self.rates(voltage); target = rates[f"{gate}_inf"]; tau = rates[f"{gate}_tau_ms"]
        return target + (state - target) * np.exp(-dt / tau)


class JointGateFullRepairConfirmationTests(unittest.TestCase):
    def test_sealed_contract_is_deterministic_and_stratified(self):
        config = JointGateFullRepairConfirmationConfig(); config.validate()
        first = _sealed_rows(_Formula()); second = _sealed_rows(_Formula())
        self.assertEqual({name: len(row["inputs"]) for name, row in first.items()}, {
            "sealed_central": 4096, "sealed_voltage_tail": 2048, "sealed_long_horizon": 2048,
        })
        for name in first:
            np.testing.assert_array_equal(first[name]["inputs"], second[name]["inputs"])
        all_rows = [tuple(row) for group in first.values() for row in group["inputs"]]
        self.assertEqual(len(all_rows), len(set(all_rows)))


if __name__ == "__main__":
    unittest.main()
