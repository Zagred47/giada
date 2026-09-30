"""Offline and opt-in native tests for Task 17c NMODL precision controls."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.giada_teacher.roadmap_causal_cahva_replacement import compile_candidate_mods
from src.giada_teacher.roadmap_task17c_m_precision_solver import (
    ARMS, GRID_ARMS, Task17cConfig, _expected_linear, _render_m_lut,
    generate_precision_mods, probe_compiled_m_tables,
)


class Task17cTests(unittest.TestCase):
    def test_preregistered_matrix_and_interpolator(self):
        Task17cConfig().validate()
        self.assertEqual(len(ARMS), 7)
        self.assertEqual(len(Task17cConfig().conditions), 3)
        for knots in (513, 1025, 2049):
            table = np.column_stack((np.linspace(0, 1, knots),
                                     np.linspace(1, 2, knots)))
            source = _render_m_lut(table)
            self.assertEqual(source.count("frac ="), knots - 1)
            self.assertNotIn("TABLE", source)
            self.assertAlmostEqual(_expected_linear(table, -30.25)[0],
                                   (-30.25 + 135) / 210)

    def test_precision_mods_preserve_ionic_contract(self):
        teacher = (Path(__file__).resolve().parents[2]
                   / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod")
        if not teacher.is_file():
            self.skipTest("canonical teacher checkout unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            records, tables = generate_precision_mods(teacher, Path(tmp) / "mods")
            self.assertEqual(set(records), set(ARMS) - {"native"})
            for arm, (knots, precision) in GRID_ARMS.items():
                self.assertEqual(tables[arm].shape, (knots, 2))
                self.assertEqual(str(tables[arm].dtype), precision)
                rendered = Path(records[arm]["path"]).read_text(encoding="utf-8")
                self.assertIn("USEION ca READ eca WRITE ica", rendered)
                self.assertIn("SOLVE states METHOD cnexp", rendered)
                self.assertIn("PROCEDURE exact_formula_rates()", rendered)
                self.assertIn("PROCEDURE m_lut_rates()", rendered)
                self.assertIn("mInf = frozen_mInf", rendered)
                self.assertIn("mTau = frozen_mTau", rendered)

    @unittest.skipUnless(os.environ.get("GIADA_TASK17C_NATIVE_SMOKE") == "1",
                         "Opt-in: requires pinned Linux NEURON")
    def test_native_compile_and_rate_probe(self):
        from neuron import h, load_mechanisms
        teacher = Path(os.environ["GIADA_NATIVE_TEACHER"]) / "L5PC_NEURON_simulation/mods/Ca_HVA.mod"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "mods"
            _, tables = generate_precision_mods(teacher, destination)
            compiler = shutil.which("nrnivmodl")
            self.assertIsNotNone(compiler)
            compile_candidate_mods(destination, compiler)
            self.assertTrue(load_mechanisms(str(destination)))
            probe = probe_compiled_m_tables(h, tables)
            self.assertEqual(set(probe), set(GRID_ARMS))


if __name__ == "__main__":
    unittest.main()
