import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.frozen_voltage_generalization_forensic import (
    EXPECTED_06BG_ARCHIVE_SHA256,
    EXPECTED_06BG_FINAL_SHA256,
    EXPECTED_06BG_INDEX_SHA256,
    FROZEN_MODEL_ARMS,
    FrozenVoltageForensicConfig,
    FrozenVoltageGeneralizationForensic,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    ROOT
    / "experiments"
    / "hayflow"
    / "06b_h_frozen_voltage_generalization_forensic"
)
NOTEBOOK = ROOT / "notebooks" / "06b_h_frozen_voltage_generalization_forensic.ipynb"
CONFIG = ROOT / "configs" / "hayflow" / "hayflow_frozen_voltage_generalization_forensic.yml"


def test_06bh_exactly_registers_the_returned_06bg_artifact():
    assert EXPECTED_06BG_ARCHIVE_SHA256 == (
        "5a09397433d941f6d80adf12d6ac7936be9404189bdd2952d88ebd43ef45c4eb"
    )
    assert EXPECTED_06BG_INDEX_SHA256 == (
        "94896c60d767fe1d38cd9f848d3d4cdf6df3dcabd38a3afe7c3e1ce675dc2567"
    )
    assert EXPECTED_06BG_FINAL_SHA256 == (
        "e78531d7ec6e810e268ffb76ac18548fefeb3836d5481eefb093cac126d490b2"
    )


def test_06bh_preregisters_frozen_matrix_before_audit():
    prereg = json.loads((EXPERIMENT / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["frozen_matrix"]["candidate_count"] == 40
    assert not prereg["frozen_matrix"]["neural_training_performed"]
    assert prereg["roles"]["calibration"]["used_for_candidate_selection"]
    assert not prereg["roles"]["audit"]["used_for_candidate_selection"]
    assert prereg["roles"]["calibration_audit_disjoint"]


def test_06bh_configuration_crosses_four_by_five_by_two():
    config = FrozenVoltageForensicConfig()
    config.validate()
    assert len(FROZEN_MODEL_ARMS) == 4
    assert config.voltage_shrinkage_grid == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert len(FROZEN_MODEL_ARMS) * len(config.voltage_shrinkage_grid) * 2 == 40
    assert "forensic_audit_components_per_regime: 1" in CONFIG.read_text()


def test_06bh_uses_seventh_component_for_calibration_and_eighth_for_audit():
    source = inspect.getsource(FrozenVoltageGeneralizationForensic._build_forensic_roles)
    assert '"voltage_calibration"' in source
    assert '"voltage_audit"' in source
    assert "upstream + 1" in source
    assert "upstream + 2" in source
    assert "previous & calibration" in source
    assert "calibration & audit" in source
    assert 'row.get("split")' in source


def test_06bh_calibration_is_frozen_and_never_reads_audit():
    source = inspect.getsource(FrozenVoltageGeneralizationForensic.calibrate_frozen_voltage)
    assert '"voltage_calibration"' in source
    assert '"voltage_audit"' not in source
    assert "optimizer" not in source.lower()
    assert "backward" not in source
    assert "candidate_count" in source
    assert '"audit_accessed": False' in source


def test_06bh_audit_stratifies_voltage_activity():
    source = inspect.getsource(FrozenVoltageGeneralizationForensic._activity_metrics)
    for name in (
        "quiescent_lt_1mV",
        "moderate_1_to_5mV",
        "active_5_to_20mV",
        "regenerative_ge_20mV",
        "active_ge_5mV",
    ):
        assert name in source
    region_source = inspect.getsource(
        FrozenVoltageGeneralizationForensic._region_metrics
    )
    assert "self.layout.region_names" in region_source
    assert "self.layout.segment_region_ids" in region_source
    final = inspect.getsource(
        FrozenVoltageGeneralizationForensic.finalize_frozen_voltage_forensic
    )
    assert '"coupled_06c_canary_authorized": False' in final
    assert '"full_training_authorized": False' in final


def test_06bh_notebook_is_compact_and_uses_stable_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 12
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    assert "EXPECTED_06BG_INDEX_SHA256" in code
    assert "calibrate_frozen_voltage" in code
    assert "audit_frozen_voltage" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    assert "display(audit_report)" not in code
