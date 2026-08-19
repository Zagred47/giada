import ast
import inspect
import json
from pathlib import Path

import pytest

from src.hayflow_model.hines_autoregressive_failure_reassessment import (
    EXPECTED_05K_ARCHIVE_SHA256,
    EXPECTED_05K_FINAL_SHA256,
    EXPECTED_05K_INDEX_SHA256,
    EXPECTED_05K_MICRO_SHA256,
    HinesAutoregressiveFailureReassessment,
    HinesAutoregressiveFailureReassessmentConfig,
    classify_autoregressive_failure,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05k_b_autoregressive_failure_reassessment.ipynb"


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_preregistered_reassessment_config_is_exact():
    config = HinesAutoregressiveFailureReassessmentConfig()
    config.validate()
    assert config.seeds == (17, 29, 43)
    assert config.horizons_ms == (2, 4, 8)
    assert config.intervention_modes == (
        "teacher_boundary_reset",
        "teacher_voltage_clamp",
        "teacher_latent_reset",
        "decoder_no_feedback",
        "residual_first_step_only",
    )
    with pytest.raises(ValueError):
        HinesAutoregressiveFailureReassessmentConfig(seeds=(17,)).validate()


def test_registered_05k_hashes_and_result_are_exact():
    assert EXPECTED_05K_ARCHIVE_SHA256 == "0e28e16e9f4b7e14830495ae74382d2981ba6b402e95fd200b537dafaabe4ceb"
    assert EXPECTED_05K_INDEX_SHA256 == "a471bac2fbb643018aea760269087082408063245028a319cceb58e05ff25a95"
    assert EXPECTED_05K_FINAL_SHA256 == "2136c05fd459f529953ea0fe5e4e0ad7c1c428125905267f6b94beab594c39a5"
    assert EXPECTED_05K_MICRO_SHA256 == "e0af4b102123d4bdcae8cf6830867ca01ca61dcf66ad2e8a1a4504f091586a4b"
    result = json.loads(
        (
            ROOT
            / "experiments"
            / "hayflow"
            / "05k_frozen_candidate_micro_rollout"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["diagnosis"] == "FROZEN_CANDIDATE_FAILS_AUTOREGRESSIVE_MICRO_ROLLOUT"
    assert result["passing_seed_count"] == 0
    assert not result["limited_rollout_aware_training_canary_authorized"]


@pytest.mark.parametrize(
    ("reductions", "diagnosis"),
    [
        ({"teacher_voltage_clamp": 0.7, "teacher_latent_reset": 0.2}, "VOLTAGE_FEEDBACK_DOMINANT_INSTABILITY"),
        ({"teacher_voltage_clamp": 0.2, "teacher_latent_reset": 0.7}, "LATENT_STATE_FEEDBACK_DOMINANT_INSTABILITY"),
        ({"decoder_no_feedback": 0.6, "teacher_boundary_reset": 0.4}, "DECODER_FEEDBACK_COMPOUNDING"),
        ({"residual_first_step_only": 0.6, "teacher_boundary_reset": 0.4}, "REPEATED_RESIDUAL_APPLICATION_INSTABILITY"),
        ({"teacher_boundary_reset": 0.2, "teacher_voltage_clamp": 0.1}, "COUPLED_AUTOREGRESSIVE_INSTABILITY_NOT_ISOLATED"),
    ],
)
def test_failure_classification_is_fixed_and_descriptive(reductions, diagnosis):
    report = classify_autoregressive_failure(
        reductions, material_reduction=0.25, strong_reduction=0.5
    )
    assert report["diagnosis"] == diagnosis


def test_05k_is_verified_before_parent_chain_or_fresh_store_is_opened():
    source = inspect.getsource(
        HinesAutoregressiveFailureReassessment.prepare_autoregressive_failure_reassessment
    )
    assert source.index("verified_micro_rollout_artifact_root") < source.index(
        "prepare_frozen_candidate_micro_rollout"
    )
    assert "passing_seed_count" in source
    assert "future_teacher_membrane_or_ion_states_injected" in source


def test_interventions_do_not_train_or_select_checkpoints():
    source = inspect.getsource(
        HinesAutoregressiveFailureReassessment.evaluate_failure_interventions
    )
    assert "optimizer" not in source
    assert ".backward(" not in source
    assert "model_or_training_authorized" in source
    rollout = inspect.getsource(
        HinesAutoregressiveFailureReassessment._intervention_rollout
    )
    for mode in HinesAutoregressiveFailureReassessmentConfig().intervention_modes:
        assert mode in rollout
    assert 'mode == "teacher_boundary_reset"' in rollout
    assert 'mode == "decoder_no_feedback"' in rollout


def test_notebook_is_valid_compact_and_uses_browser_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    source = _code()
    assert "HinesAutoregressiveFailureReassessment" in source
    assert "failure_reassessment_config=autoregressive_config" in source
    assert "artifact_05k_source=ARTIFACT_05K_SOURCE" in source
    assert "def index_matches(path,expected)" in source
    assert "def exact_artifact(env,name,marker,expected)" in source
    assert "EXPECTED_05JO_INDEX_SHA256" in source
    assert "EXPECTED_05K_INDEX_SHA256" in source
    assert "globals()[variable]=exact_artifact" in source
    assert "display(report)" not in source
    assert "display(final_report)" not in source
    assert "base64.b64encode" in source
    assert "new Blob" in source
    assert "a.click()" in source
    assert "FileLink" not in source


def test_notebook_constructor_has_no_duplicate_keywords():
    tree = ast.parse(_code())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "HinesAutoregressiveFailureReassessment"
    ]
    assert len(calls) == 1
    names = [keyword.arg for keyword in calls[0].keywords]
    assert len(names) == len(set(names))
