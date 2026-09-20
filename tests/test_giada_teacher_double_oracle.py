from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.giada_teacher import ExtractedGateFormula, run_double_oracle


def teacher_mod() -> Path:
    return Path(__file__).resolve().parents[2] / "neuron_as_deep_net" / "L5PC_NEURON_simulation" / "mods" / "Ca_HVA.mod"


class _IndependentFakeNeuron:
    def __init__(self, formula: ExtractedGateFormula, perturbation: float = 0.0) -> None:
        self.formula = formula
        self.perturbation = perturbation
        self.h = type("H", (), {"nrnversion": staticmethod(lambda: "fake-independent-test")})()

    def step(self, gate: str, state: float, voltage_mv: float, dt_ms: float) -> float:
        return self.formula.step(gate, state, voltage_mv, dt_ms) + self.perturbation


class DoubleOracleTests(unittest.TestCase):
    def test_extracts_cahva_rates_without_hardcoding_outputs(self) -> None:
        formula = ExtractedGateFormula.from_mod(teacher_mod())
        self.assertTrue(formula.source_sha256)
        self.assertEqual(len(formula.assignments), 8)
        rates = formula.rates(-40.0)
        self.assertTrue(0.0 < rates["m_inf"] < 1.0)
        self.assertTrue(0.0 < rates["h_inf"] < 1.0)
        self.assertGreater(rates["m_tau_ms"], 0.0)
        self.assertGreater(rates["h_tau_ms"], 0.0)

    def test_singular_voltage_rule_is_extracted(self) -> None:
        formula = ExtractedGateFormula.from_mod(teacher_mod())
        self.assertEqual(formula.singular_voltage_mv, -27.0)
        self.assertEqual(formula.singular_perturbation_mv, 0.0001)
        self.assertAlmostEqual(formula.rates(-27.0)["effective_voltage_mv"], -26.9999)

    def test_exponential_step_has_expected_invariants(self) -> None:
        formula = ExtractedGateFormula.from_mod(teacher_mod())
        for gate in ("m", "h"):
            for state in (0.0, 0.5, 1.0):
                updated = formula.step(gate, state, -40.0, 1.0)
                self.assertTrue(0.0 <= updated <= 1.0)
                inf = formula.rates(-40.0)[f"{gate}_inf"]
                self.assertLessEqual(abs(updated - inf), abs(state - inf))

    def test_double_oracle_report_detects_agreement_and_disagreement(self) -> None:
        formula = ExtractedGateFormula.from_mod(teacher_mod())
        valid = run_double_oracle(
            formula, _IndependentFakeNeuron(formula), voltages_mv=(-80.0, -27.0, 20.0),
            states=(0.0, 0.5, 1.0), dt_ms=(0.025, 1.0),
        )
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["grid"]["case_count"], 36)
        invalid = run_double_oracle(
            formula, _IndependentFakeNeuron(formula, 1e-4), voltages_mv=(-40.0,),
            states=(0.5,), dt_ms=(1.0,),
        )
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["failure_count"], 2)
        self.assertTrue(math.isclose(invalid["maximum_absolute_error"], 1e-4, rel_tol=0, abs_tol=1e-15))
