import ast
import json
from pathlib import Path

import pytest

from src.hayflow_model.temporal_observability_reassessment import (
    EXPECTED_05P_ARCHIVE_SHA256,
    EXPECTED_05P_FINAL_SHA256,
    EXPECTED_05P_INDEX_SHA256,
    TemporalObservabilityContractReassessment,
    TemporalObservabilityReassessmentConfig,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05q_temporal_observability_contract_reassessment.ipynb"


def test_registered_05p_hashes_are_exact():
    assert EXPECTED_05P_ARCHIVE_SHA256 == "f133ed92009464a6bc120a13860669230605245b245d6adcb2da946c58f3020e"
    assert EXPECTED_05P_INDEX_SHA256 == "8e5e81a6af86fef15e07da589a4ca2d03c23c8d8caeb37b0545992e4d27b63f6"
    assert EXPECTED_05P_FINAL_SHA256 == "feb189283547a2143069ac8f74de54284b2f9383e7778e66fa1e8479a4bc3f3a"


def test_config_is_frozen_and_valid():
    config = TemporalObservabilityReassessmentConfig.from_mapping(
        {"horizons_ms": [2, 4, 8], "seeds": [17, 29, 43]}
    )
    assert config.permutation_seed == 510073
    assert config.checkpoint_reproduction_atol_mv == 1e-4
    config.validate()


def test_comparison_reports_candidate_gain_and_regenerative_gain():
    results = {
        "candidate": {
            "17": {"8": {"endpoint_rmse_mv": 8.0, "regenerative_endpoint_rmse_mv": 9.0}},
            "29": {"8": {"endpoint_rmse_mv": 7.0, "regenerative_endpoint_rmse_mv": 8.0}},
            "43": {"8": {"endpoint_rmse_mv": 9.0, "regenerative_endpoint_rmse_mv": 9.5}},
        },
        "baseline": {
            seed: {"8": {"endpoint_rmse_mv": 10.0, "regenerative_endpoint_rmse_mv": 10.0}}
            for seed in ("17", "29", "43")
        },
    }
    row = TemporalObservabilityContractReassessment._comparison(
        results, "candidate", "baseline", (17, 29, 43)
    )
    assert row["positive_win_count"] == 3
    assert row["median_rmse_gain_fraction"] == pytest.approx(0.2)
    assert row["median_regenerative_gain_fraction"] == pytest.approx(0.1)


def test_05q_notebook_is_compact_frozen_and_uses_blob_download():
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
    assert "EXPECTED_05P_INDEX_SHA256" in code
    assert "EXPECTED_05O_INDEX_SHA256" in code
    assert "discover_indexed_artifact_source" in code
    assert "TemporalObservabilityContractReassessment" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
