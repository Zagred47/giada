"""Offline contract tests; native NEURON execution runs in the Kaggle notebook."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.giada_teacher.roadmap_causal_cahva_replacement import (
    CausalReplacementConfig, SITES, _paired_effect, candidate_protocols,
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
            self.assertIn("TABLE mInf, hInf, mTau, hTau FROM -135 TO 75 WITH 512", lut)

    def test_paired_effect_keeps_weak_effect_unidentifiable(self):
        def row(value):
            return {"traces": {str(site): {"v": [value, value]}
                               for site in SITES}}
        result = _paired_effect(row(0), row(0.01), row(0), row(0), 0.05)
        self.assertFalse(result["0"]["effect_identifiable"])
        self.assertAlmostEqual(result["0"]["relative_error"], 0.2)


if __name__ == "__main__":
    unittest.main()
