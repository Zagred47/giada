import ast
import inspect
import json
from pathlib import Path

import numpy as np

from src.hayflow_model.voltage_error_model_revision import (
    CAUSAL_GATE_SCHEMES,
    EXPECTED_06BK_ARCHIVE_SHA256,
    EXPECTED_06BK_FINAL_SHA256,
    EXPECTED_06BK_INDEX_SHA256,
    FALLBACK_SCHEME,
    ORACLE_SCHEMES,
    PRIMARY_SCHEME,
    STATIC_REFERENCE,
    TEACHER_OPTIMAL_BLEND_ORACLE,
    TEACHER_REGIME_ORACLE,
    VoltageErrorModelRevision,
    VoltageErrorModelRevisionConfig,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_K = (
    ROOT / "experiments" / "hayflow" / "06b_k_temporal_voltage_correction_state"
)
EXPERIMENT_L = (
    ROOT / "experiments" / "hayflow" / "06b_l_voltage_error_model_revision"
)
NOTEBOOK = ROOT / "notebooks" / "06b_l_voltage_error_model_revision.ipynb"


def test_06bl_registers_exact_verified_06bk_result():
    result = json.loads((EXPERIMENT_K / "result.json").read_text())
    assert result["archive_sha256"] == EXPECTED_06BK_ARCHIVE_SHA256
    assert result["artifact_index_sha256"] == EXPECTED_06BK_INDEX_SHA256
    assert result["final_report_sha256"] == EXPECTED_06BK_FINAL_SHA256
    assert result["formal_diagnosis"] == "TEMPORAL_CORRECTION_STATE_NOT_IDENTIFIED"
    assert result["absolute_low_activity_errors"]["persistence_quiescent_rmse_mv"] < 0.5
    assert min(result["absolute_low_activity_errors"]["static_quiescent_rmse_mv_by_seed"].values()) > 4


def test_06bl_preregisters_terminal_mixture_decision():
    prereg = json.loads((EXPERIMENT_L / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["primary_scheme"] == PRIMARY_SCHEME
    assert prereg["fallback_scheme"] == FALLBACK_SCHEME
    assert not prereg["oracles"]["eligible_for_selection"]
    assert prereg["scope_limitations"]["terminal_diagnostic_before_architecture_revision"]
    assert set(ORACLE_SCHEMES) == {
        TEACHER_REGIME_ORACLE,
        TEACHER_OPTIMAL_BLEND_ORACLE,
    }


def test_06bl_configuration_and_aligned_gate_matrix():
    config = VoltageErrorModelRevisionConfig()
    config.validate()
    assert config.hurdle_probability_thresholds == (0.25, 0.5, 0.75)
    assert config.teacher_quiescent_threshold_mv == 1.0
    assert len(CAUSAL_GATE_SCHEMES) == 4
    assert STATIC_REFERENCE not in CAUSAL_GATE_SCHEMES
    assert VoltageErrorModelRevision._gate_feature_names(
        "hard_hurdle_instantaneous"
    ) == VoltageErrorModelRevision._gate_feature_names("soft_blend_instantaneous")
    assert VoltageErrorModelRevision._gate_feature_names(
        "hard_hurdle_temporal"
    ) == VoltageErrorModelRevision._gate_feature_names("soft_blend_temporal")


def test_06bl_soft_and_hard_targets_are_distinct_and_bounded():
    instance = object.__new__(VoltageErrorModelRevision)
    instance.config = VoltageErrorModelRevisionConfig()
    observations = {
        "baseline_delta": np.asarray([2.0, 2.0, -2.0, -2.0]),
        "target_correction": np.asarray([-2.0, 0.0, 1.0, 3.0]),
        "teacher_current_error": np.asarray([0.0, 0.0, 0.0, 0.0]),
    }
    soft = instance._gate_targets(observations, "soft_blend_instantaneous")
    hard = instance._gate_targets(observations, "hard_hurdle_instantaneous")
    assert np.all((0 <= soft) & (soft <= 1))
    assert set(np.unique(hard)).issubset({0.0, 1.0})
    assert not np.array_equal(soft, hard)


def test_06bl_closed_form_gate_fit_has_no_optimizer():
    fit = inspect.getsource(VoltageErrorModelRevision._fit_gate_model)
    calibration = inspect.getsource(
        VoltageErrorModelRevision.fit_and_calibrate_gate_models
    )
    assert "np.linalg.solve" in fit and "np.linalg.pinv" in fit
    assert "optimizer" not in calibration.lower()
    assert "backward" not in calibration.lower()
    assert '"calibration"' in calibration


def test_06bl_oracles_are_explicit_and_terminal_decision_always_revises_architecture():
    rollout = inspect.getsource(VoltageErrorModelRevision._recursive_gate_evaluation)
    final = inspect.getsource(
        VoltageErrorModelRevision.finalize_voltage_error_model_revision
    )
    assert "batch[\"voltage_t1\"]" in rollout
    assert "TEACHER_REGIME_ORACLE" in rollout
    assert "TEACHER_OPTIMAL_BLEND_ORACLE" in rollout
    assert '"oracles_eligible_for_selection": False' in final
    assert '"terminal_diagnostic_before_architecture_revision": True' in final
    assert final.count('next_step = "architecture_revision_') == 4
    assert '"coupled_06c_canary_authorized": False' in final


def test_06bl_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "EXPECTED_06BK_INDEX_SHA256" in code
    assert "materialize_nested_indexed_artifact_source" in code
    assert "fit_and_calibrate_gate_models" in code
    assert "evaluate_voltage_error_models" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
