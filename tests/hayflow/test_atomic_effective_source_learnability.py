import ast
import inspect
import json
from pathlib import Path

import pytest

from src.hayflow_model import atomic_state_dynamics_playground as atomic
from src.hayflow_model.atomic_effective_source_learnability import (
    ENDPOINT_ONLY,
    GLOBAL_P99,
    HYBRID,
    NATIVE_ONLY,
    RAW_SOURCE,
    REGION_P99,
    AtomicEffectiveSourceConfig,
    AtomicEffectiveSourceLearnability,
    EXPECTED_06BO_ARCHIVE_SHA256,
    EXPECTED_06BO_FINAL_SHA256,
    EXPECTED_06BO_INDEX_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_O = ROOT / "experiments" / "hayflow" / "06b_o_effective_membrane_source_playground"
EXPERIMENT_P = ROOT / "experiments" / "hayflow" / "06b_p_atomic_effective_source_learnability"
NOTEBOOK = ROOT / "notebooks" / "06b_p_atomic_effective_source_learnability.ipynb"


def test_06bp_is_authorized_by_independently_registered_06bo_result():
    result = json.loads((EXPERIMENT_O / "result.json").read_text())
    assert result["archive_sha256"] == EXPECTED_06BO_ARCHIVE_SHA256
    assert result["artifact_index_sha256"] == EXPECTED_06BO_INDEX_SHA256
    assert result["final_report_sha256"] == EXPECTED_06BO_FINAL_SHA256
    assert result["formal_diagnosis"] == (
        "PASSIVE_HINES_PRIOR_HELPS_LONG_ROLLOUT_BUT_EFFECTIVE_SOURCE_IS_NOT_LEARNED"
    )


def test_06bp_preregisters_exact_paired_3x3_matrix_and_boundary_test():
    prereg = json.loads((EXPERIMENT_P / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    matrix = prereg["stage_2_atomic_matrix"]
    assert matrix["factorial_design"] == "3x3"
    assert matrix["factor_count"] == 9
    assert matrix["same_numeric_input_tensor"]
    assert matrix["same_initialization_within_seed"]
    assert matrix["same_minibatch_stream_within_seed"]
    boundary = prereg["stage_3_frozen_boundary_test"]
    assert not boundary["models_retrained"]
    assert boundary["predicted_state_trained"] is False
    assert boundary["temporal_memory_trained"] is False


def test_06bp_config_covers_scaling_and_objective_axes_without_hidden_sweep():
    config = AtomicEffectiveSourceConfig()
    config.validate()
    session = object.__new__(AtomicEffectiveSourceLearnability)
    session.config = config
    specs = session._atomic_specs()
    assert len(specs) == 9 and len(set(specs)) == 9
    assert {spec[0] for spec in specs} == {RAW_SOURCE, GLOBAL_P99, REGION_P99}
    assert {spec[1] for spec in specs} == {NATIVE_ONLY, ENDPOINT_ONLY, HYBRID}
    assert config.atomic_checkpoints == (0, 50, 100, 200, 300)


def test_06bp_training_is_one_step_paired_and_records_initial_gradients():
    source = inspect.getsource(
        AtomicEffectiveSourceLearnability.train_atomic_source_matrix
    )
    assert 'self._flat_tensors("fit", rows, device)' in source
    assert "batch_stream_sha256" in source
    assert "initial_gradients" in source
    assert "selected_calibration_endpoint_rmse_mv" in source
    assert '"development"' not in source


def test_06bp_boundary_test_freezes_models_and_keeps_teacher_state_explicit():
    evaluation = inspect.getsource(
        AtomicEffectiveSourceLearnability.evaluate_atomic_and_recursive_boundaries
    )
    recursive = inspect.getsource(AtomicEffectiveSourceLearnability._recursive_metrics)
    assert "models_retrained_for_boundary_test" in evaluation
    assert "recursive_voltage_teacher_STATE" in evaluation
    assert 'values["state_t"]' in recursive
    assert "optimizer" not in recursive.lower()


@pytest.mark.skipif(atomic.torch is None, reason="PyTorch is optional locally")
def test_06bp_scale_decode_preserves_raw_source_identity():
    config = AtomicEffectiveSourceConfig()
    session = object.__new__(AtomicEffectiveSourceLearnability)
    session.config = config
    session.source_scales = {RAW_SOURCE: atomic.np.ones(3, dtype=atomic.np.float64)}
    reference = atomic.torch.tensor([[1.0, -2.0, 0.5]])
    decoded = reference * session._scale_tensor(RAW_SOURCE, reference)
    assert atomic.torch.equal(decoded, reference)


def test_06bp_notebook_is_compact_parseable_and_uses_blob_download():
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
    assert "EXPECTED_06BO_INDEX_SHA256" in code
    assert "prepare_atomic_source_learnability" in code
    assert "train_atomic_source_matrix" in code
    assert "evaluate_atomic_and_recursive_boundaries" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
