import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.causal_voltage_state_coupling_forensic import (
    COUPLING_MODES,
    EXPECTED_06B_ARCHIVE_SHA256,
    EXPECTED_06B_FINAL_SHA256,
    EXPECTED_06B_INDEX_SHA256,
    CausalVoltageStateCouplingConfig,
    CausalVoltageStateCouplingForensic,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    ROOT / "notebooks" / "06b_b_causal_voltage_state_coupling_forensic.ipynb"
)
PREREGISTRATION = (
    ROOT
    / "experiments"
    / "hayflow"
    / "06b_b_causal_voltage_state_coupling_forensic"
    / "preregistration.json"
)
RESULT = PREREGISTRATION.with_name("result.json")


def test_registered_06b_authority_is_exact():
    assert EXPECTED_06B_ARCHIVE_SHA256 == (
        "0d44d0f6aeb90c7df67a65cd2f92ffbdad9c9163acc11af92ab90f1c52d785ec"
    )
    assert EXPECTED_06B_INDEX_SHA256 == (
        "0fe4985566f7276c333bd288280b3756751e82f02353d2e43c791374f126a612"
    )
    assert EXPECTED_06B_FINAL_SHA256 == (
        "89512fc5cd37a06c21d59d9d5d74d6418f40afa0d09a3bbaf7a8bf2ff1e4ccc7"
    )


def test_coupling_configuration_is_bounded_and_multi_seed():
    config = CausalVoltageStateCouplingConfig()
    config.validate()
    assert config.pilot_seeds == (61017, 61029, 61043)
    assert config.bridge_training_steps == 800
    assert config.maximum_bridge_parameter_count == 12000
    assert config.minimum_median_voltage_gain_fraction == 0.10
    assert config.minimum_median_state_gain_over_causal_fraction == 0.02
    assert config.minimum_median_state_gain_over_shuffled_fraction == 0.02
    assert COUPLING_MODES == (
        "frozen_causal",
        "predicted_endpoint",
        "shuffled_predicted_endpoint",
        "teacher_endpoint_oracle",
    )


def test_preregistration_freezes_state_updaters_and_requires_causal_control():
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert registered["trainable_component"].endswith("voltage-delta bridge only")
    assert len(registered["frozen_components"]) == 2
    assert registered["paired_modes"] == list(COUPLING_MODES)
    assert "predicted delta-V tensor" in registered[
        "causal_specificity_control"
    ]
    assert registered["selection_role"].startswith("train-derived")
    assert registered["evaluation_role"].startswith("disjoint train-derived")
    assert registered["rollout_contract"]["recursive_quantity"] == "mechanism STATE"
    assert not registered["rollout_contract"]["autonomous_voltage_rollout_claimed"]
    assert "reading validation or test state/outcomes" in registered["prohibited"]
    assert "training the full neuron" in registered["prohibited"]


def test_registered_result_preserves_causal_signal_and_recursive_no_go():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["archive_sha256"] == (
        "d652c3fdf088569b212c6fc710185ab4f870857e5e84cc2947459b7f456bb349"
    )
    assert result["artifact_index_sha256"] == (
        "824cc0fdfb977c69fed7bbf3dfcea691f6c1346c84fadedd988aa20c12c56986"
    )
    assert result["integrity_valid"] and result["valid"]
    assert result["component_decision_grade"]
    assert result["diagnosis"] == "CAUSAL_VOLTAGE_BRIDGE_NOT_PREDICTIVE"
    assert not result["coupling_identified"]
    assert result["gate_checks"]["predicted_beats_frozen_causal"]
    assert result["gate_checks"]["predicted_beats_shuffled_control"]
    assert result["gate_checks"]["oracle_gap_recovered"]
    assert not result["gate_checks"]["bridge_predictive"]
    assert not result["gate_checks"]["eight_ms_rollout_recovered"]
    assert result["interpretation"]["causal_alignment_signal_identified"]
    assert not result["full_training_authorized"]


def test_coupling_forensic_bypasses_microtraces_and_freezes_state_checkpoints():
    materialization = inspect.getsource(
        CausalVoltageStateCouplingForensic._materialize_role
    )
    assert "AtomicStateDynamicsPlayground._materialize_role" in materialization
    assert "_read_teacher_path" not in materialization
    loading = inspect.getsource(
        CausalVoltageStateCouplingForensic._load_frozen_state_models
    )
    assert "requires_grad_(False)" in loading
    assert "model.eval()" in loading
    training = inspect.getsource(CausalVoltageStateCouplingForensic._train_bridge)
    assert "self._new_bridge" in training
    assert "frozen_state_models" not in training


def test_primary_state_path_uses_predicted_voltage_and_same_prediction_shuffle():
    source = inspect.getsource(
        CausalVoltageStateCouplingForensic.evaluate_one_step_coupling
    )
    assert '"predicted_endpoint"' in source
    assert '"shuffled_predicted_endpoint"' in source
    assert "predicted[permutation]" in source
    assert '"teacher_endpoint_oracle"' in source


def test_rollout_is_nested_but_does_not_claim_autonomous_voltage():
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    rollout = registered["rollout_contract"]
    assert rollout["nested_common_windows"]
    assert rollout["voltage_boundary_condition"] == "teacher V_t at each millisecond"
    assert not rollout["autonomous_voltage_rollout_claimed"]
    source = inspect.getsource(
        CausalVoltageStateCouplingForensic.evaluate_coupled_nested_rollouts
    )
    assert "_nested_development_windows" in source
    assert 'categories=("voltage",)' in source


def test_06bb_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "CausalVoltageStateCouplingForensic" in code
    assert "EXPECTED_06B_INDEX_SHA256" in code
    assert "frozen_state_checkpoint_count" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
    assert "display(preflight)" not in code
    assert "display(final_report)" not in code
