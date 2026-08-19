import ast
import json
from pathlib import Path

import pytest

from src.hayflow_model.hines_architecture_failure_reassessment import (
    EXPECTED_05KC_ARCHIVE_SHA256,
    EXPECTED_05KC_FINAL_SHA256,
    EXPECTED_05KC_INDEX_SHA256,
    HinesArchitectureFailureReassessmentConfig,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05k_d_architecture_failure_reassessment.ipynb"


def test_architecture_decision_contract_is_exact():
    config = HinesArchitectureFailureReassessmentConfig()
    config.validate()
    assert config.proposed_canary_families == (
        "morphology_graph_gru",
        "ordered_convgru_control",
    )
    with pytest.raises(ValueError):
        HinesArchitectureFailureReassessmentConfig(horizon_ms=4).validate()


def test_registered_05kc_hashes_are_exact():
    assert EXPECTED_05KC_ARCHIVE_SHA256 == "ec6cbdb0a015c8a61720bf7713feb20cbffb34b7142c69165030e7673c06f4b4"
    assert EXPECTED_05KC_INDEX_SHA256 == "822bde9a5917d68d056a1c7e85a517248e5bb55d7c0427d137c37761f1de7a1c"
    assert EXPECTED_05KC_FINAL_SHA256 == "bfc793e13a6184e56f9081ed161b40c9273ff4a5b28b08cbbe6e4d7fecfa861a"


def test_notebook_is_small_valid_and_uses_exact_artifact_and_blob():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert len(notebook["cells"]) == 7
    assert "EXPECTED_05KC_INDEX_SHA256" in code
    assert "torch" not in code
    assert "fresh" not in code.lower()
    assert "base64.b64encode" in code and "new Blob" in code and "a.click()" in code
    assert "FileLink" not in code
