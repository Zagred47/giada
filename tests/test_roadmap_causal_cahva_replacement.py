"""Offline contract tests; native NEURON execution runs in the Kaggle notebook."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.giada_teacher.roadmap_causal_cahva_replacement import (
    CausalReplacementConfig, SITES, _explicit_lut_rates, _paired_effect, candidate_protocols,
    generate_candidate_mods,
)


class Task17ContractTests(unittest.TestCase):
    def test_config_and_protocols_are_frozen(self):
        CausalReplacementConfig().validate()
        self.assertEqual(len(candidate_protocols()), 3)
        self.assertEqual(CausalReplacementConfig().seeds, (170017, 170029, 170043))

    def test_candidate_mods_preserve_ionic_contract(self):
        native = (Path(__file__).resolve().parents[2]
                  / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod")
        if not native.is_file():
            self.skipTest("canonical teacher checkout not present beside elmneuron")
        with tempfile.TemporaryDirectory() as temporary:
            records = generate_candidate_mods(native, Path(temporary) / "mods")
            self.assertEqual(set(records), {"formula", "lut"})
            formula = Path(records["formula"]["path"]).read_text()
            lut = Path(records["lut"]["path"]).read_text()
            self.assertIn("USEION ca READ eca WRITE ica", formula)
            self.assertIn("SOLVE states METHOD cnexp", formula)
            self.assertNotIn("TABLE mInf", formula)
            self.assertNotIn("TABLE mInf", lut)
            self.assertIn("LOCAL frac", lut)
            self.assertIn("if (v < -30)", lut)
            self.assertIn("RANGE gCa_HVAbar, gCa_HVA, ica, mInf, hInf, mTau, hTau", lut)

    def test_explicit_interpolator_has_every_frozen_interval(self):
        rates = np.column_stack([np.linspace(0, 1, 513)] * 4).astype(np.float32)
        rendered = _explicit_lut_rates(rates)
        self.assertEqual(rendered.count("frac ="), 512)
        self.assertEqual(rendered.count("mInf ="), 514)
        self.assertNotIn("TABLE", rendered)

    def test_paired_effect_keeps_weak_effect_unidentifiable(self):
        def row(value):
            return {"traces": {str(site): {"v": [value, value]}
                               for site in SITES}}
        result = _paired_effect(row(0), row(0.01), row(0), row(0), 0.05)
        self.assertFalse(result["0"]["effect_identifiable"])
        self.assertAlmostEqual(result["0"]["relative_error"], 0.2)


if __name__ == "__main__":
    unittest.main()
