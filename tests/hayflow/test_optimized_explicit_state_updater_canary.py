import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.optimized_explicit_state_updater_canary import (
    CAUSAL_CANARY_ARMS,
    EXPECTED_06AB_FINAL_SHA256,
    EXPECTED_06AB_INDEX_SHA256,
    OptimizedExplicitStateCanaryConfig,
    OptimizedExplicitStateUpdaterCanary,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "06b_optimized_explicit_state_updater_canary.ipynb"
PREREGISTRATION = (
    ROOT
    / "experiments"
    / "hayflow"
    / "06b_optimized_explicit_state_updater_canary"
    / "preregistration.json"
)
RESULT = PREREGISTRATION.with_name("result.json")


def test_registered_06ab_authority_is_exact():
    assert EXPECTED_06AB_INDEX_SHA256 == (
        "95d348e1bc7d4a7709592f32fc41354544993ca715e284fbb07df498469f52f5"
    )
    assert EXPECTED_06AB_FINAL_SHA256 == (
        "4a4bdfa7660fe8f128c7e15a9be148ebd1a8876fa836a86b53a7ee471461ff22"
    )


def test_causal_canary_configuration_is_multi_seed_and_bounded():
    config = OptimizedExplicitStateCanaryConfig()
    config.validate()
    assert config.pilot_seeds == (61017, 61029, 61043)
    assert config.training_steps == 1200
    assert config.maximum_parameter_count == 7238
    assert config.minimum_median_causal_gain_fraction == 0.10
    assert config.minimum_positive_semantic_group_fraction == 0.70
    assert CAUSAL_CANARY_ARMS == (
        "causal_start_voltage",
        "linear_endpoint_path",
    )


def test_preregistration_keeps_primary_arm_causal_and_heldout_sealed():
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    primary, reference = registered["paired_arms"]
    assert primary["name"] == "causal_start_voltage"
    assert not primary["teacher_endpoint_used"]
    assert reference["teacher_endpoint_used"]
    assert not registered["alignment"]["future_microtraces_read"]
    assert registered["roles"]["source_split"] == "train_only"
    assert "test_access" in registered["not_authorized"]
    assert "full_neuron_training" in registered["not_authorized"]


def test_registered_result_preserves_partial_success_and_failed_full_gate():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["archive_sha256"] == (
        "0d44d0f6aeb90c7df67a65cd2f92ffbdad9c9163acc11af92ab90f1c52d785ec"
    )
    assert result["artifact_index_sha256"] == (
        "0fe4985566f7276c333bd288280b3756751e82f02353d2e43c791374f126a612"
    )
    assert result["integrity_valid"] and result["valid"]
    assert result["component_decision_grade"]
    assert result["diagnosis"] == "ATOMIC_STATE_REQUIRES_EXPLICIT_VOLTAGE_COUPLING"
    assert not result["causal_updater_confirmed"]
    assert result["endpoint_reference_robust"]
    assert result["gate_checks"]["every_seed_one_step_at_least_2_percent"]
    assert result["gate_checks"]["median_semantic_macro_at_least_3_percent"]
    assert not result["gate_checks"]["median_one_step_at_least_10_percent"]
    assert not result["gate_checks"]["median_active_at_least_10_percent"]
    assert not result["gate_checks"]["median_retention_at_least_70_percent"]
    assert not result["full_training_authorized"]


def test_canary_materialization_explicitly_bypasses_future_microtrace_reader():
    source = inspect.getsource(OptimizedExplicitStateUpdaterCanary._materialize_role)
    assert "AtomicStateDynamicsPlayground._materialize_role" in source
    assert "_read_teacher_path" not in source
    path_source = inspect.getsource(OptimizedExplicitStateUpdaterCanary._path_for_values)
    assert 'arm == "causal_start_voltage"' in path_source
    assert "np.zeros" in path_source


def test_06b_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "OptimizedExplicitStateUpdaterCanary" in code
    assert "EXPECTED_06AB_INDEX_SHA256" in code
    assert "future_microtraces_read" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
    assert "display(preflight)" not in code
