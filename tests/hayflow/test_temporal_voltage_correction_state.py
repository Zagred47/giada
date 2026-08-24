import ast
import inspect
import json
from pathlib import Path

import numpy as np

from src.hayflow_model.temporal_voltage_correction_state import (
    EXPECTED_06BJ_ARCHIVE_SHA256,
    EXPECTED_06BJ_FINAL_SHA256,
    EXPECTED_06BJ_INDEX_SHA256,
    EXPOSURE_CONTROL_SCHEME,
    FALLBACK_SCHEME,
    ORACLE_SCHEME,
    PRIMARY_SCHEME,
    STATIC_REFERENCE,
    TEMPORAL_SCHEMES,
    TemporalVoltageCorrectionConfig,
    TemporalVoltageCorrectionState,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_J = (
    ROOT / "experiments" / "hayflow" / "06b_j_analytic_causal_gain_identifiability"
)
EXPERIMENT_K = (
    ROOT / "experiments" / "hayflow" / "06b_k_temporal_voltage_correction_state"
)
NOTEBOOK = ROOT / "notebooks" / "06b_k_temporal_voltage_correction_state.ipynb"


def test_06bk_registers_exact_verified_06bj_result():
    result = json.loads((EXPERIMENT_J / "result.json").read_text())
    assert result["archive_sha256"] == EXPECTED_06BJ_ARCHIVE_SHA256
    assert result["artifact_index_sha256"] == EXPECTED_06BJ_INDEX_SHA256
    assert result["final_report_sha256"] == EXPECTED_06BJ_FINAL_SHA256
    assert result["formal_diagnosis"] == (
        "STATIC_GAIN_IDENTIFIED_BUT_TEMPORAL_COMPOSITION_FAILS"
    )
    assert result["primary"]["median_direct_improvement_over_alpha075"] > 0.05
    assert result["primary"]["median_quiescent_gain"] < 0


def test_06bk_preregisters_aligned_matrix_and_sealed_oracle():
    prereg = json.loads((EXPERIMENT_K / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["primary_scheme"] == PRIMARY_SCHEME
    assert prereg["fallback_scheme"] == FALLBACK_SCHEME
    assert prereg["exposure_control_scheme"] == EXPOSURE_CONTROL_SCHEME
    assert prereg["oracle"]["scheme"] == ORACLE_SCHEME
    assert not prereg["oracle"]["eligible_for_selection"]
    assert not prereg["scope_limitations"]["new_independent_confirmation_claimed"]


def test_06bk_configuration_is_bounded_and_ordered():
    config = TemporalVoltageCorrectionConfig()
    config.validate()
    assert 0 < config.ema_fast_decay < config.ema_slow_decay < 1
    assert config.temporal_correction_clip_mv == 5.0
    assert config.minimum_recursive_gain_over_static_fraction == 0.02
    assert config.minimum_temporal_specificity_over_exposure_control_fraction == 0.01
    assert STATIC_REFERENCE not in TEMPORAL_SCHEMES


def test_06bk_feature_ablation_is_causal_except_explicit_oracle():
    instantaneous = TemporalVoltageCorrectionState._feature_names(
        EXPOSURE_CONTROL_SCHEME
    )
    primary = TemporalVoltageCorrectionState._feature_names(PRIMARY_SCHEME)
    oracle = TemporalVoltageCorrectionState._feature_names(ORACLE_SCHEME)
    assert "teacher_current_error" not in instantaneous
    assert "teacher_current_error" not in primary
    assert "teacher_current_error" in oracle
    assert "ema_fast_signed" in primary
    assert "ema_slow_abs" in primary
    assert "predicted_displacement" in primary


def test_06bk_closed_form_fit_and_recursive_exposure_control():
    fit = inspect.getsource(TemporalVoltageCorrectionState._fit_ridge_model)
    calibration = inspect.getsource(
        TemporalVoltageCorrectionState.fit_and_calibrate_temporal_models
    )
    observations = inspect.getsource(
        TemporalVoltageCorrectionState._recursive_exposure_observations
    )
    assert "np.linalg.solve" in fit and "np.linalg.pinv" in fit
    assert "optimizer" not in calibration.lower()
    assert "backward" not in calibration.lower()
    assert '"fit"' in calibration and '"calibration"' in calibration
    assert "current_state, current_voltage = next_state, next_voltage" in observations


def test_06bk_closed_form_ridge_recovers_region_specific_correction():
    instance = object.__new__(TemporalVoltageCorrectionState)
    instance.config = TemporalVoltageCorrectionConfig()
    instance.layout = type("Layout", (), {"region_names": ("a", "b")})()
    raw = np.linspace(-2.0, 2.0, 40)
    region = np.repeat([0, 1], 20)
    observations = {
        "raw_delta": raw,
        "current_voltage": -65.0 + raw,
        "baseline_delta": 0.8 * raw,
        "target_correction": np.where(region == 0, 0.5 * raw, -0.25 * raw),
        "region": region,
    }
    model = instance._fit_ridge_model(
        observations, EXPOSURE_CONTROL_SCHEME, 0.01
    )
    metrics = instance._direct_metrics(observations, model)
    assert model["parameter_count"] == 8
    assert metrics["improvement_over_static_fraction"] > 0.9


def test_06bk_finalization_requires_temporal_specificity_and_blocks_06c():
    source = inspect.getsource(
        TemporalVoltageCorrectionState.finalize_temporal_correction_state
    )
    assert "minimum_temporal_specificity_over_exposure_control_fraction" in source
    assert source.index("if primary_pass") < source.index("elif fallback_pass")
    assert '"oracle_eligible_for_selection": False' in source
    assert '"coupled_06c_canary_authorized": False' in source
    assert '"full_training_authorized": False' in source


def test_06bk_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "EXPECTED_06BJ_INDEX_SHA256" in code
    assert "materialize_nested_indexed_artifact_source" in code
    assert "fit_and_calibrate_temporal_models" in code
    assert "evaluate_temporal_models" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
