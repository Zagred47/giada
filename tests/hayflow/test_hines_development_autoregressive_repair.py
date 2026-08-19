import ast
import inspect
import json
from pathlib import Path

import pytest

from src.hayflow_model.hines_development_autoregressive_repair import (
    EXPECTED_05KB_ARCHIVE_SHA256,
    EXPECTED_05KB_FINAL_SHA256,
    EXPECTED_05KB_INDEX_SHA256,
    HinesDevelopmentAutoregressiveRepair,
    HinesDevelopmentAutoregressiveRepairConfig,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05k_c_development_autoregressive_repair.ipynb"


def _notebook_code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_development_repair_contract_is_fixed():
    config = HinesDevelopmentAutoregressiveRepairConfig()
    config.validate()
    assert config.seeds == (17, 29, 43)
    assert config.horizons_ms == (2, 4, 8)
    assert config.gate_horizons_ms == (4, 8)
    assert config.development_pair_count == 24
    with pytest.raises(ValueError):
        HinesDevelopmentAutoregressiveRepairConfig(development_pair_count=23).validate()


def test_registered_05kb_hashes_and_result_are_exact():
    assert EXPECTED_05KB_ARCHIVE_SHA256 == "ec0b721f3828a8428b1347bcd1126d783104d5abf512e44a31991c56a94e4700"
    assert EXPECTED_05KB_INDEX_SHA256 == "9cdf563becbbb26775155836460625e2f0746f1e02cc3e469c1067e79918f4c4"
    assert EXPECTED_05KB_FINAL_SHA256 == "d53b41923bc13b1b34aa5100c110c11f67e0714f3f24ef27fe40cfe0493ed8e7"
    result = json.loads(
        (
            ROOT
            / "experiments"
            / "hayflow"
            / "05k_b_autoregressive_failure_reassessment"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["diagnosis"] == "CLOSED_LOOP_STATE_DISTRIBUTION_SHIFT"
    assert result["dominant_intervention"] == "teacher_boundary_reset"
    assert not result["training_authorized"]


def test_provenance_is_verified_before_parent_preparation():
    source = inspect.getsource(
        HinesDevelopmentAutoregressiveRepair.prepare_development_autoregressive_repair
    )
    assert source.index("verified_failure_reassessment_artifact_root") < source.index(
        "prepare_regenerative_decoder_refit"
    )
    assert "fresh_test_is_now_consumed_for_diagnosis" in source
    assert "fresh_05jo_state_or_outcomes_loaded" in source


def test_recommit_uses_corrected_voltage_for_both_recurrent_commits():
    source = inspect.getsource(
        HinesDevelopmentAutoregressiveRepair._state_consistent_recommit
    )
    assert '(corrected_voltage / 100.0).unsqueeze(-1)' in source
    assert "corrected_voltage.mean(1)" in source
    assert '"voltage": corrected_voltage' in source
    assert "local_commit" in source
    assert "global_commit" in source


def test_evaluation_has_no_training_or_fresh_test_access():
    source = inspect.getsource(
        HinesDevelopmentAutoregressiveRepair.evaluate_development_repair
    )
    assert "confirmation_store" in source
    assert "fresh_store" not in source
    assert "optimizer" not in source
    assert ".backward(" not in source
    assert "state_consistent_recommit" in source


def test_finalize_writes_index_and_never_reinstates_candidate(tmp_path):
    session = object.__new__(HinesDevelopmentAutoregressiveRepair)
    session.output_dir = tmp_path
    session.code_revision = "test-revision"
    session.artifact_05kb_contract = {"artifact_index_sha256": EXPECTED_05KB_INDEX_SHA256}
    (tmp_path / "diagnostic.json").write_text('{"valid": true}\n', encoding="utf-8")
    final = session.finalize_development_repair(
        {"valid": True, "state_consistent_repair_supported": True}
    )
    index = json.loads((tmp_path / "artifact_index.json").read_text(encoding="utf-8"))
    assert final["rollout_aware_training_canary_authorized"]
    assert not final["candidate_reinstated"]
    assert not final["full_training_authorized"]
    assert index["schema_version"] == "05k-c-artifact-index-v1"


def test_notebook_is_valid_compact_exact_and_has_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    source = _notebook_code()
    assert "HinesDevelopmentAutoregressiveRepair" in source
    assert "EXPECTED_05KB_INDEX_SHA256" in source
    assert "globals()[variable]=exact_artifact" in source
    assert "ARTIFACT_05JO_SOURCE" not in source
    assert "fresh_05jo_loaded" in source
    assert "display(repair)" not in source
    assert "display(final_report)" not in source
    assert "base64.b64encode" in source
    assert "new Blob" in source
    assert "a.click()" in source
    assert "FileLink" not in source


def test_notebook_constructor_keywords_are_unique():
    tree = ast.parse(_notebook_code())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "HinesDevelopmentAutoregressiveRepair"
    ]
    assert len(calls) == 1
    names = [keyword.arg for keyword in calls[0].keywords]
    assert len(names) == len(set(names))
