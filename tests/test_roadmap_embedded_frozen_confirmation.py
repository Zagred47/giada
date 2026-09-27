import ast
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src.giada_teacher.roadmap_embedded_frozen_confirmation import (
    FrozenEmbeddedConfig, _ShadowReplaySession, _lut_step, _lut_table,
    stage_snapshots, verified_15c,
)


class FrozenEmbeddedConfirmationTests(unittest.TestCase):
    def test_replay_session_constructs_without_calibration_artifact(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _ShadowReplaySession(
                repository, root,
                calibration_source=root / "base_dataset",
                dataset_config_path=repository / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
                output_dir=root / "replay", site_ids=(1,),
                lut_table=np.zeros((2, 4), dtype=np.float32),
            )
            self.assertEqual(session.shadow_site_ids, (1,))
            with self.assertRaisesRegex(RuntimeError, "calibration loading is not permitted"):
                session._prepare_calibration_source()

    def test_online_lut_is_bounded_and_uses_midpoint_step(self):
        class ConstantFormula:
            def rates(self, voltage):
                return {"m_inf": .8, "h_inf": .2, "m_tau_ms": .1,
                        "h_tau_ms": 10.}

        table = _lut_table(ConstantFormula())
        initial = np.asarray([.1, .9], dtype=np.float32)
        predicted = _lut_step(table, -55., initial)
        exact = np.asarray([.8, .2]) + (initial - np.asarray([.8, .2])) * np.exp(-.025 / np.asarray([.1, 10.]))
        self.assertTrue(np.allclose(predicted, exact, atol=1e-7))
        self.assertTrue(np.all((predicted >= 0) & (predicted <= 1)))

    def test_config_is_frozen_to_independent_test(self):
        FrozenEmbeddedConfig().validate()
        with self.assertRaises(ValueError):
            FrozenEmbeddedConfig(split="validation").validate()

    def test_task15c_selection_exact_if_available(self):
        source = Path.home() / "Downloads/giada_roadmap_task15c_embedded_interface_bridge.zip"
        if not source.is_file():
            self.skipTest("Task15c artifact unavailable")
        self.assertEqual(verified_15c(source)["selected_candidate"], "lut_fine_path")
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory) / "archive.zip"
            with zipfile.ZipFile(outer, "w") as archive:
                archive.write(source, source.name)
            self.assertEqual(verified_15c(outer)["selected_candidate"], "lut_fine_path")

    def test_stages_only_requested_snapshot_from_directory_and_zip(self):
        rows = [{"native_snapshot_ref": "snapshots/a.neuron.bin"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "snapshots").mkdir()
            (root / "snapshots/a.neuron.bin").write_bytes(b"snapshot-a")
            (root / "snapshots/b.neuron.bin").write_bytes(b"snapshot-b")
            staged = stage_snapshots(root, rows, {"root": str(root)}, root / "out")
            self.assertEqual(staged["staged_snapshot_count"], 1)
            self.assertEqual((root / "out/snapshots/a.neuron.bin").read_bytes(), b"snapshot-a")
            self.assertFalse((root / "out/snapshots/b.neuron.bin").exists())
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.write(root / "snapshots/a.neuron.bin", "nested/snapshots/a.neuron.bin")
            staged = stage_snapshots(archive, rows, {"root": str(root / "missing")}, root / "zip_out")
            self.assertEqual(staged["staged_snapshot_count"], 1)
            self.assertEqual((root / "zip_out/snapshots/a.neuron.bin").read_bytes(), b"snapshot-a")

    def test_notebook_code_and_compact_blob_download(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/16_roadmap_embedded_frozen_confirmation.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                         if cell["cell_type"] == "code")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        for token in ("shutil.make_archive", "base64.b64encode", "Javascript", "new Blob", "l.click()"):
            self.assertIn(token, code)
        for token in ("neuron==8.2.7", "nrnivmodl", "subprocess.run([nrnivmodl,'mods']"):
            self.assertIn(token, code)
        self.assertIn("neuron.__version__.split('+',1)[0]=='8.2.7'", code)
        self.assertLess(code.index("neuron==8.2.7"), code.index("run_frozen_embedded_confirmation("))
        self.assertIn("for name in ('final_report.json','selected_test_paths.json')", code)


if __name__ == "__main__":
    unittest.main()
