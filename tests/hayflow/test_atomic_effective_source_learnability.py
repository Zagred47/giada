import ast
import inspect
import json
from pathlib import Path

import pytest

from src.hayflow_model import atomic_state_dynamics_playground as atomic
from src.hayflow_model.atomic_effective_source_learnability import (
    BOUNDARY_COMPLETE,
    COMPACT_MOMENTS,
    ENDPOINT_ONLY,
    EXACT_EVENTS,
    GLOBAL_P99,
    HYBRID,
    INTRINSIC_RESIDUAL,
    NATIVE_ONLY,
    NET_EFFECTIVE_SOURCE,
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
    matrix = prereg["stage_1_atomic_matrix"]
    assert matrix["factorial_design"] == "3x3"
    assert matrix["factor_count"] == 9
    assert matrix["same_numeric_input_tensor"]
    assert matrix["same_initialization_within_seed"]
    assert matrix["same_minibatch_stream_within_seed"]
    adaptive = prereg["stage_2_adaptive_fragment_matrix"]
    assert adaptive["factorial_design"] == "3x2"
    assert adaptive["factor_count"] == 6
    assert adaptive["fixed_width_numeric_input"]
    assert adaptive["equal_parameter_count_within_stage"]
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
    fragments = session._fragment_specs()
    assert len(fragments) == 6 and len(set(fragments)) == 6
    assert {spec[0] for spec in fragments} == {
        COMPACT_MOMENTS,
        EXACT_EVENTS,
        BOUNDARY_COMPLETE,
    }
    assert {spec[1] for spec in fragments} == {
        NET_EFFECTIVE_SOURCE,
        INTRINSIC_RESIDUAL,
    }


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
    fragment_recursive = inspect.getsource(
        AtomicEffectiveSourceLearnability._recursive_fragment_metrics
    )
    assert "encode_realized_synaptic_drive" in fragment_recursive
    assert 'raw_state_t=raw_state' in fragment_recursive
    assert '"teacher_endpoint_used_as_input": False' in fragment_recursive


def test_06bp_adaptive_stage_uses_disjoint_calibration_and_paired_streams():
    source = inspect.getsource(
        AtomicEffectiveSourceLearnability.train_adaptive_fragment_matrix
    )
    assert 'self._calibration_rows("fragment")' in source
    assert "same_fixed_width_numeric_tensor" in source
    assert "same_minibatch_stream_within_seed" in source
    assert "regime_gradient_probes_at_initialization" in source
    assert "selected_fragment_arm" in source
    assert "median_calibration_endpoint_rmse_by_arm" in source
    assert '"development"' not in source

    finalization = inspect.getsource(
        AtomicEffectiveSourceLearnability.finalize_atomic_source_learnability
    )
    assert 'fragment_training["selected_fragment_arm"]' in finalization
    assert '"development_used_for_checkpoint_or_arm_selection": False' in finalization


def test_06bp_substep_audit_is_diagnostic_not_selectable():
    prereg = json.loads((EXPERIMENT_P / "preregistration.json").read_text())
    audit = prereg["stage_0_nonselective_audits"]
    assert audit["substep_dt_ms"] == [1.0, 0.5, 0.25]
    assert audit["substep_selection_eligible"] is False
    source = inspect.getsource(
        AtomicEffectiveSourceLearnability._substep_source_support_audit
    )
    assert '"selection_eligible": False' in source
    assert '"trainable_substep_arm_built": False' in source


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
    assert "IMPLEMENTATION_COMMIT_PLACEHOLDER" not in code
    assert "prepare_atomic_source_learnability" in code
    assert "train_atomic_source_matrix" in code
    assert "train_adaptive_fragment_matrix" in code
    assert "evaluate_atomic_and_recursive_boundaries" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    EXACT_EVENTS,
    INTRINSIC_RESIDUAL,
    NET_EFFECTIVE_SOURCE,
