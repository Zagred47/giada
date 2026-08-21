import ast
import json
from pathlib import Path

from src.hayflow_model.temporal_representation_reassessment import (
    EXPECTED_05Q_ARCHIVE_SHA256,
    EXPECTED_05Q_FINAL_SHA256,
    EXPECTED_05Q_INDEX_SHA256,
    TemporalRepresentationReassessmentConfig,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05r_temporal_representation_reassessment.ipynb"


def test_registered_05q_hashes_are_exact():
    assert EXPECTED_05Q_ARCHIVE_SHA256 == "73ed6fc3e244f49fe5172e5e917b253519e0bd7b614ab88584f1f8a06b41e0ec"
    assert EXPECTED_05Q_INDEX_SHA256 == "26ed1c29cded7a53f2d2dfce0519b19f6dfa5b9f4945a2d0867f046a51ac5d9b"
    assert EXPECTED_05Q_FINAL_SHA256 == "51a29770690b9fd52306d5946bb5611649da03a702f06c72be27b392166cd94e"


def test_config_is_frozen_and_valid():
    config = TemporalRepresentationReassessmentConfig.from_mapping(
        {"horizons_ms": [2, 4, 8], "seeds": [17, 29, 43]}
    )
    assert config.category_ablation_materiality_fraction == 0.02
    assert config.temporal_shift_materiality_fraction == 0.02
    assert config.sketch_reconstruction_atol == 1e-5
    config.validate()


def test_05r_notebook_is_compact_frozen_and_uses_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 11
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    for constant in (
        "EXPECTED_05Q_INDEX_SHA256",
        "EXPECTED_05P_INDEX_SHA256",
        "EXPECTED_05O_INDEX_SHA256",
    ):
        assert constant in code
    assert "discover_indexed_artifact_source" in code
    assert "TemporalRepresentationReassessment" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
