import ast
import json
from pathlib import Path

from src.hayflow_model.consolidated_autoregressive_go_no_go import (
    EXPECTED_05S_ARCHIVE_SHA256,
    EXPECTED_05S_FINAL_SHA256,
    EXPECTED_05S_INDEX_SHA256,
    ConsolidatedAutoregressiveGoNoGoConfig,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05t_consolidated_autoregressive_go_no_go.ipynb"


def test_registered_05s_hashes_are_exact():
    assert EXPECTED_05S_ARCHIVE_SHA256 == (
        "cd879a8cf8fe25b42e9b139544424d7674144b983d3de126a34a882c4a1fd090"
    )
    assert EXPECTED_05S_INDEX_SHA256 == (
        "eac72ae71568dec45f3e454d6729733b8ebb0afbc38d3b6d447d980b2fb5ccda"
    )
    assert EXPECTED_05S_FINAL_SHA256 == (
        "500cdd8a7920d47412b727af915bc44d2392a5f674397aed3f06037a546068bb"
    )


def test_config_is_frozen_validation_only_and_valid():
    config = ConsolidatedAutoregressiveGoNoGoConfig.from_mapping(
        {
            "horizons_ms": [2, 4, 8],
            "seeds": [17, 29, 43],
            "validation_horizons_ms": [8, 16, 32],
        }
    )
    assert config.validation_horizons_ms == (8, 16, 32)
    assert config.validation_windows_per_episode == 2
    assert config.minimum_gain_vs_legacy_fraction == 0.05
    assert config.source_metric_reproduction_atol_mv == 1e-4
    config.validate()


def test_05t_notebook_is_compact_frozen_and_uses_blob_download():
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
        "EXPECTED_05S_INDEX_SHA256",
        "EXPECTED_05R_INDEX_SHA256",
        "EXPECTED_05Q_INDEX_SHA256",
        "EXPECTED_05P_INDEX_SHA256",
        "EXPECTED_05O_INDEX_SHA256",
    ):
        assert constant in code
    assert "ConsolidatedAutoregressiveGoNoGo" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
