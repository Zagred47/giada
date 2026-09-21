from pathlib import Path
import unittest

from src.giada_teacher.double_oracle import ExtractedGateFormula
from src.giada_teacher.joint_gate_cell_playground import prepare_joint_gate_dataset
from src.giada_teacher.joint_gate_optimization_diagnosis import (
    JointGateOptimizationDiagnosisConfig,
    _arm_properties,
    _finalize_diagnosis,
    augment_joint_gate_rate_targets,
)


TEACHER_MOD = Path(__file__).resolve().parents[2] / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod"


class JointGateOptimizationDiagnosisTests(unittest.TestCase):
    def test_registered_factorial_is_valid(self):
        config = JointGateOptimizationDiagnosisConfig()
        config.validate()
        self.assertTrue(_arm_properties("symmetric_rates_pcgrad", 10, config)["pcgrad"])
        self.assertTrue(_arm_properties("rate_pretrain", 10, config)["rate_only"])
        self.assertFalse(_arm_properties("rate_pretrain", config.pretrain_steps, config)["rate_only"])

    def test_m_rate_targets_are_added_without_changing_rows(self):
        bundle = prepare_joint_gate_dataset(ExtractedGateFormula.from_mod(TEACHER_MOD))
        inputs = bundle["fit"]["inputs"].copy()
        augment_joint_gate_rate_targets(bundle)
        self.assertEqual(len(bundle["fit"]["m_inf"]), len(inputs))
        self.assertEqual(len(bundle["fit"]["m_tau_ms"]), len(inputs))
        self.assertTrue((bundle["fit"]["inputs"] == inputs).all())

    def test_registered_rules_localize_budget(self):
        config = JointGateOptimizationDiagnosisConfig()
        runs = []
        for arm in config.arms:
            for seed in config.seeds:
                points = [{"step": 50000, "score": 1.0, "gradient_geometry": {"conflict_fraction": 0.0}}]
                if arm == "baseline_extended":
                    points += [
                        {"step": 1000, "score": 2.0, "gradient_geometry": {"conflict_fraction": 0.0}},
                        {"step": 3000, "score": 1.8, "gradient_geometry": {"conflict_fraction": 0.0}},
                        {"step": 10000, "score": 1.4, "gradient_geometry": {"conflict_fraction": 0.0}},
                        {"step": 100000, "score": 0.7, "gradient_geometry": {"conflict_fraction": 0.0}},
                    ]
                runs.append({"arm": arm, "seed": seed, "checkpoints": points})
        report = _finalize_diagnosis(runs, config)
        self.assertEqual(report["diagnosis"], "TRAINING_BUDGET_LIMIT")
        self.assertTrue(report["supported_causes"]["extended_budget"])


if __name__ == "__main__":
    unittest.main()
