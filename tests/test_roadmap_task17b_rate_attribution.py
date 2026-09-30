"""Offline contracts for the diagnostic Task 17b matrix."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.giada_teacher.roadmap_task17b_rate_attribution import (
    ARMS, HYBRID_FIELDS, Task17bConfig, generate_hybrid_mods,
)
from src.giada_teacher.roadmap_causal_cahva_replacement import compile_candidate_mods


class Task17bTests(unittest.TestCase):
    def test_preregistration_frozen(self):
        Task17bConfig().validate()
        self.assertEqual(Task17bConfig().seeds, (170029, 170083, 170097))
        self.assertEqual(len(ARMS), 8)

    def test_each_hybrid_replaces_one_rate_only(self):
        teacher = (Path(__file__).resolve().parents[2]
                   / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod")
        if not teacher.is_file():
            self.skipTest("canonical teacher checkout unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            records, table = generate_hybrid_mods(teacher, Path(tmp) / "mods")
            self.assertEqual(table.shape, (513, 4))
            self.assertEqual(set(records), set(ARMS) - {"native"})
            for arm, fields in HYBRID_FIELDS.items():
                rendered = Path(records[arm]["path"]).read_text(encoding="utf-8")
                self.assertIn("USEION ca READ eca WRITE ica", rendered)
                self.assertIn("SOLVE states METHOD cnexp", rendered)
                self.assertIn("PROCEDURE exact_formula_rates()", rendered)
                self.assertIn("PROCEDURE frozen_lut_rates()", rendered)
                for field in fields:
                    self.assertIn(f"frozen_{field} = {field}", rendered)
                    self.assertIn(f"{field} = frozen_{field}", rendered)
                self.assertNotIn("TABLE mInf", rendered)

    @unittest.skipUnless(os.environ.get("GIADA_TASK17B_NATIVE_SMOKE") == "1",
                         "Opt-in: requires pinned Linux NEURON")
    def test_all_hybrids_compile_and_expose_rates(self):
        from neuron import h, load_mechanisms
        teacher = Path(os.environ["GIADA_NATIVE_TEACHER"]) / "L5PC_NEURON_simulation/mods/Ca_HVA.mod"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "mods"
            generate_hybrid_mods(teacher, destination)
            compiler = shutil.which("nrnivmodl")
            self.assertIsNotNone(compiler)
            compile_candidate_mods(destination, compiler)
            self.assertTrue(load_mechanisms(str(destination)))
            for arm in HYBRID_FIELDS:
                from src.giada_teacher.roadmap_task17b_rate_attribution import HYBRID_SUFFIXES
                suffix = HYBRID_SUFFIXES[arm]
                section = h.Section(name=f"task17b_{arm}")
                section.insert(suffix)
                h.finitialize(-20)
                segment = section(0.5)
                self.assertTrue(hasattr(segment, f"mInf_{suffix}"))
                h.delete_section(sec=section)


if __name__ == "__main__":
    unittest.main()
