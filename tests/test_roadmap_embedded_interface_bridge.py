import ast
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src.giada_teacher.roadmap_embedded_interface_bridge import (
    EmbeddedInterfaceConfig, verified_task15b_result, verified_task5_any,
    voltage_view,
)


class EmbeddedInterfaceBridgeTests(unittest.TestCase):
    def test_preregistered_contract_and_voltage_alignment(self):
        EmbeddedInterfaceConfig().validate()
        with self.assertRaises(ValueError):
            EmbeddedInterfaceConfig(source_split="deterministic_test").validate()
        path = np.arange(41, dtype=float)
        self.assertEqual(voltage_view(path, "left_available")[1], 0)
        self.assertEqual(voltage_view(path, "midpoint_oracle")[1], 0.5)
        self.assertEqual(voltage_view(path, "right_oracle")[1], 1)
        with self.assertRaises(ValueError):
            voltage_view(path[:-1], "left_available")

    def test_exact_historical_artifacts_if_available(self):
        downloads = Path.home() / "Downloads"
        task15b = downloads / "giada_roadmap_task15b_gate_bottleneck_confirmation.zip"
        task5 = downloads / "giada_primitive_scaling_laws.zip"
        if not task15b.is_file() or not task5.is_file():
            self.skipTest("historical user artifacts unavailable")
        self.assertEqual(verified_task15b_result(task15b)["code_revision"],
                         "4262849a7fd7c321734eacef61f8d03b89817f5d")
        with tempfile.TemporaryDirectory() as directory:
            root = verified_task5_any(task5, Path(directory) / "task5")
            self.assertTrue((root / "frozen_scaling_checkpoints.pt").is_file())

    def test_repacked_task5_is_verified_by_inner_freeze(self):
        task5 = Path.home() / "Downloads/giada_primitive_scaling_laws.zip"
        if not task5.is_file():
            self.skipTest("historical Task5 artifact unavailable")
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory) / "archive.zip"
            with zipfile.ZipFile(outer, "w") as archive:
                archive.write(task5, "giada_primitive_scaling_laws.zip")
            root = verified_task5_any(outer, Path(directory) / "unpack")
            self.assertTrue((root / "final_report.json").is_file())

    def test_notebook_code_parses_and_uses_blob_download(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/15c_roadmap_embedded_interface_bridge.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        for token in ("shutil.make_archive", "base64.b64encode", "Javascript", "new Blob", "l.click()"):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
