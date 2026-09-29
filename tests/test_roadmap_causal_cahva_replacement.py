"""Offline contract tests; native NEURON execution runs in the Kaggle notebook."""

import tempfile
import unittest
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.giada_teacher.roadmap_causal_cahva_replacement import (
    CausalReplacementConfig, SITES, _canonical_gbar,
    _canonical_synapse_weights, _changed_synapse_weights, _explicit_lut_rates,
    _paired_effect, _restore_canonical_gbar, candidate_protocols,
    generate_candidate_mods, _trial,
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

    def test_synapse_records_are_positional_list(self):
        records = [{"binding": SimpleNamespace(base_weight=0.7),
                    "netcon": SimpleNamespace(weight=[0.7])}]
        weights = _canonical_synapse_weights(records)
        self.assertEqual(weights, (0.7,))
        self.assertEqual(_changed_synapse_weights(records, weights), [])
        records[0]["netcon"].weight[0] = 0.6
        self.assertEqual(_changed_synapse_weights(records, weights), [0])

    def test_incompatible_snapshot_is_rejected_before_native_restore(self):
        session = SimpleNamespace(task17_snapshot_suffix="Ca_HVA")
        with self.assertRaisesRegex(RuntimeError, "across a mechanism change"):
            _trial(session, None, None, 17, 1.0, "lut", CausalReplacementConfig(), {})

    def test_gbar_is_reset_before_each_paired_arm(self):
        segment = SimpleNamespace(x=0.5, gCa_HVAbar_Ca_HVA=0.002)
        class Section:
            def name(self):
                return "soma[0]"

            def __iter__(self):
                return iter([segment])

        section = Section()
        hoc = SimpleNamespace(allsec=lambda: [section],
                              ismembrane=lambda suffix, sec: suffix == "Ca_HVA",
                              fcurrent=lambda: None)
        session = SimpleNamespace(h=hoc, cvode=SimpleNamespace(re_init=lambda: None))
        baseline = _canonical_gbar(session)
        segment.gCa_HVAbar_Ca_HVA *= 0.5
        _restore_canonical_gbar(session, baseline)
        self.assertEqual(segment.gCa_HVAbar_Ca_HVA, 0.002)


@unittest.skipUnless(os.environ.get("GIADA_TASK17_NATIVE_TEST") == "1",
                     "Opt-in: requires compiled pinned Linux teacher and verified Task15c input")
class Task17NativeIntegrationTests(unittest.TestCase):
    def test_complete_matrix_and_snapshot_roundtrip(self):
        repo = Path(__file__).resolve().parents[1]
        # Keep native artifacts for audit; no implicit deletion of successful evidence.
        root = Path(os.environ["GIADA_TASK17_TEST_OUTPUT"])
        command = [sys.executable, str(repo / "scripts/run_roadmap_task17.py"),
                   "--repo", str(repo), "--teacher", os.environ["GIADA_NATIVE_TEACHER"],
                   "--task15c", os.environ["GIADA_TASK15C_ARTIFACT"], "--output", str(root)]
        completed = subprocess.run(command, check=False)
        self.assertEqual(completed.returncode, 0)
        report = json.loads((root / "final_report.json").read_text())
        status = json.loads((root / "process_status.json").read_text())
        self.assertEqual(status["returncode"], 0)
        self.assertTrue(report["valid"])
        self.assertTrue(report["formula_control_valid"])
        self.assertFalse(report["gate_c_authorized"])
        self.assertEqual(report["episode_count"], 27)
        self.assertEqual(len(list((root / "completed_trials").glob("*.json"))), 81)
        self.assertLessEqual(max(x["voltage_max_error_mv"] for x in
                                 report["native_after_swap_preflight"].values()), 1e-5)
        self.assertEqual(report["snapshot_policy"],
                         "fresh_equilibrium_snapshot_per_mechanism_generation")
        # Do not demand a LUT GO: integration success is not candidate promotion.


if __name__ == "__main__":
    unittest.main()
