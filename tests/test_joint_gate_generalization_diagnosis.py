import unittest

import numpy as np

from src.giada_teacher.joint_gate_cell_playground import _joint_row
from src.giada_teacher.joint_gate_generalization_diagnosis import (
    JointGateGeneralizationDiagnosisConfig,
    prepare_joint_gate_generalization_diagnosis,
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
        rates = self.rates(voltage)
        target = rates[f"{gate}_inf"]
        tau = rates[f"{gate}_tau_ms"]
        return target + (state - target) * np.exp(-dt / tau)


class JointGateGeneralizationDiagnosisTests(unittest.TestCase):
    def test_development_contract_excludes_fresh_and_materializes_all_arms(self):
        formula = _Formula()
        values = np.asarray([[-80.0, 0.2, 0.8, 0.1], [-20.0, 0.5, 0.5, 0.5], [20.0, 0.8, 0.2, 1.0]])
        fit = _joint_row(formula, values, "fit", "train")
        config = JointGateGeneralizationDiagnosisConfig(pool_size=32, probe_count=8)
        bundle = prepare_joint_gate_generalization_diagnosis(
            formula, {"formula": formula, "fit": fit, "development": {}, "fresh": {}}, config
        )
        self.assertEqual(set(bundle["pools"]), set(config.shared_arms))
        self.assertFalse(bundle["contract"]["fresh_task3c_accessed"])
        self.assertFalse(bundle["contract"]["fresh_used_for_selection"])
        self.assertEqual(bundle["contract"]["fit_counts"]["baseline_current"], len(values))
        self.assertIn("voltage_tail_stress", bundle["development"])


if __name__ == "__main__":
    unittest.main()
