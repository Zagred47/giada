import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.branch_elm_enriched_benchmark import (
    ELM_INPUT_VIEWS,
    ORIGINAL_ELM_PARAMETER_COUNT,
    ORIGINAL_ELM_TEST_AUC,
    ORIGINAL_ELM_TEST_SOMA_RMSE_MV,
    BranchELMEnrichedBenchmark,
    BranchELMEnrichedBenchmarkConfig,
)
from src.hayflow_model.branch_elm_matched_comparison import (
    MatchedFrozenHayFlowComparison,
    restore_registered_branch_elm_checkpoints,
)
from src.hayflow_model.branch_elm_information_matched_transition import (
    InformationMatchedTransitionConfig,
    InformationMatchedVoltageTransitionBenchmark,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "06b_c_supplement_branch_elm_enriched_benchmark.ipynb"
PREREGISTRATION = ROOT / "experiments" / "hayflow" / "06b_c_branch_elm_enriched_benchmark" / "preregistration.json"
EXECUTION_AMENDMENT = PREREGISTRATION.with_name("execution_amendment.json")
REGISTERED_RESULT = PREREGISTRATION.with_name("result.json")
MATCHED_AMENDMENT = PREREGISTRATION.with_name("matched_comparison_amendment.json")
INFORMATION_MATCHED_AMENDMENT = PREREGISTRATION.with_name(
    "information_matched_transition_amendment.json"
)


def test_branch_elm_contract_is_exactly_the_published_small_model():
    config = BranchELMEnrichedBenchmarkConfig()
    config.validate()
    assert ORIGINAL_ELM_PARAMETER_COUNT == 8002
    assert ELM_INPUT_VIEWS == ("U_scheduled", "U_realized")
    assert config.expected_dendritic_segment_count == 639
    assert (
        config.fit_episode_count,
        config.calibration_episode_count,
        config.development_episode_count,
    ) == (28, 10, 10)
    assert config.training_steps == 300
    assert ORIGINAL_ELM_TEST_SOMA_RMSE_MV == 0.6375671602714604
    assert ORIGINAL_ELM_TEST_AUC == 0.9921568089858758


def test_sidecar_preregistration_refuses_false_scalar_leaderboard():
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert registered["role"].startswith("supplementary professor-requested")
    assert registered["architecture_contract"]["trainable_parameter_count"] == 8002
    assert registered["architecture_contract"]["num_input"] == 1278
    assert registered["input_views"]["U_scheduled"].startswith("closest")
    assert "all-642-segment" in registered["comparability_warning"]
    assert any("claiming that 0.40 beats 0.60" in row for row in registered["prohibited"])


def test_fresh_test_is_not_used_during_checkpoint_selection():
    source = inspect.getsource(BranchELMEnrichedBenchmark.run_benchmark)
    training_position = source.index("self._train(view, seed, device)")
    fresh_position = source.index("zero_shot =")
    assert training_position < fresh_position
    train_source = inspect.getsource(BranchELMEnrichedBenchmark._train)
    assert 'self.roles["calibration"]' in train_source
    assert "fresh_store" not in train_source


def test_undefined_auc_is_json_safe_and_completed_checkpoints_are_reused():
    evaluation = inspect.getsource(BranchELMEnrichedBenchmark._evaluate)
    recovery = inspect.getsource(BranchELMEnrichedBenchmark._recover_completed_run)
    benchmark = inspect.getsource(BranchELMEnrichedBenchmark.run_benchmark)
    assert "auc if math.isfinite(auc) else None" in evaluation
    assert '"spike_auc_defined"' in evaluation
    assert "recovered_after_report_serialization_interruption" in recovery
    assert "_recover_completed_run(view, seed, device)" in benchmark
    amendment = json.loads(EXECUTION_AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["model_effect"] == "none"
    assert amendment["selection_effect"] == "none"
    assert amendment["recovery_contract"]["expected_checkpoint_count"] == 6
    assert not amendment["recovery_contract"]["retrain_recovered_checkpoints"]


def test_original_input_mapping_excludes_unrepresentable_somatic_current():
    mapping = inspect.getsource(BranchELMEnrichedBenchmark._episode)
    selection = inspect.getsource(BranchELMEnrichedBenchmark._select_roles)
    assert "segment_id - 1" in inspect.getsource(BranchELMEnrichedBenchmark._validate_morphology)
    assert "channel += 639" in mapping
    assert "value = -1.0" in mapping
    assert "somatic_current_not_representable" in selection


def test_elm_sidecar_notebook_uses_compact_outputs_and_stable_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 12
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    assert "InformationMatchedVoltageTransitionBenchmark" in code
    assert "EXPECTED_05T_INDEX_SHA256" in code
    assert "EXPECTED_06B_INDEX_SHA256" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    assert "display(results)" not in code
    assert "run_information_matched_benchmark" in code
    assert "finalize_information_matched_benchmark" in code
    assert "EXPECTED_05JO_INDEX_SHA256" not in code
    assert "EXPECTED_05JN_INDEX_SHA256" not in code
    assert "H2_SOURCE" not in code


def test_registered_branch_elm_result_preserves_integrity_and_scope():
    result = json.loads(REGISTERED_RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "retracted_wrong_hayflow_comparator_and_input_contract"
    assert result["retraction"]["artifact_integrity_valid"]
    assert not result["retraction"]["scientific_comparison_valid"]
    assert result["integrity"]["valid"]
    assert result["integrity"]["archive_sha256"] == "2a8e10ade7ed5cd82556a017ecfb2fee28e342c6ca49f60bb787f500e6b17045"
    assert result["integrity"]["artifact_index_sha256"] == "2abadf5e4172113d1d80f0946ae37eb62fdb519f4959ec7537f4b0a83d4637ca"
    assert result["integrity"]["final_report_sha256"] == "d0b98688208d9dd5cc0fa45f4be85f5db38e3c4c89e967592a9e4e7631120d0c"
    assert result["integrity"]["matched_hayflow_comparison_sha256"] == "d4a00f64cb6f2b9c69c9237d2ef9eef2081172a5127dfdc3e8a88fc78d689a7b"
    assert result["integrity"]["indexed_member_failures"] == []
    assert result["integrity"]["checkpoint_file_count"] == 6
    assert result["execution"]["recovered_completed_checkpoint_count"] == 6
    assert result["execution"]["retraining_avoided_for_recovered_checkpoints"]
    assert result["contract"]["trainable_parameter_count"] == 8002
    assert result["contract"]["roles"] == {
        "fit": 28,
        "calibration": 10,
        "development": 10,
    }
    assert result["contract"]["fresh_test_compatible_episode_count"] == 64
    assert result["retrained_exact_architecture"]["fresh_test_clipped_soma_rmse_mv"]["U_realized"]["median"] == 2.709690563381758
    assert result["retrained_exact_architecture"]["fresh_test_clipped_soma_rmse_mv"]["U_scheduled"]["median"] == 3.3010539723936594
    assert result["spike_metrics"]["positive_count_in_every_reported_subset"] == 0
    assert not result["spike_metrics"]["valid_for_interpretation"]
    assert result["spike_metrics"]["auc"] is None
    assert not result["comparability"]["same_target_scope"]
    assert not result["comparability"][
        "same_fresh_transitions_as_hayflow_approximately_0_40_mv"
    ]
    assert not result["comparability"]["direct_scalar_ranking_against_original_hayflow_0_40_authorized"]
    assert not result["comparability"]["matched_voltage_ranking_authorized"]
    assert not result["comparability"]["matched_voltage_same_input_contract"]
    assert result["matched_frozen_hayflow"]["fresh_test_clipped_soma_rmse_mv"]["median"] == 1.349443744937548
    assert result["matched_frozen_hayflow"]["paired_error_reduction_vs_branch_elm_U_realized_fraction"]["median"] == 0.5019934146084155
    assert not result["matched_frozen_hayflow"]["scientific_comparison_valid"]
    assert result["matched_frozen_hayflow"]["wrong_comparator_for_requested_question"]
    assert result["next_steps"]["professor_sidecar"] == "requires_corrected_current_compact_model_and_information_matched_evaluation"
    assert not result["primary_experiment_replaced"]


def test_matched_completion_is_same_sidecar_and_frozen_metric_only():
    amendment = json.loads(MATCHED_AMENDMENT.read_text(encoding="utf-8"))
    assert not amendment["completion"]["new_experiment_created"]
    assert not amendment["completion"]["hayflow_retraining"]
    assert not amendment["completion"]["checkpoint_selection"]
    assert amendment["completion"]["shared_burn_in_ms"] == 4
    assert "512" in amendment["completion"]["shared_support"]
    assert not amendment["sidecar_closes_after_this_completion"]
    assert not amendment["retraction"]["scientific_comparison_valid"]


def test_corrective_transition_contract_is_same_input_and_authentic_target():
    amendment = json.loads(
        INFORMATION_MATCHED_AMENDMENT.read_text(encoding="utf-8")
    )
    assert not amendment["new_experiment_created"]
    assert amendment["common_contract"]["target"] == "raw NEURON V_t_plus_1 minus V_t"
    assert amendment["common_contract"]["target_clipping"] is None
    assert not amendment["common_contract"]["teacher_endpoint_used_as_input"]
    assert not amendment["common_contract"]["autoregressive_rollout"]
    assert amendment["arms"]["branch_elm_core"]["total_parameter_count"] == 8002
    assert amendment["arms"]["hayflow_voltage_bridge"]["voltage_path_parameter_count"] == 8985
    assert amendment["arms"]["hayflow_voltage_bridge"]["complete_compact_transition_system_parameter_count"] == 16197


def test_information_matched_session_passes_one_tensor_to_both_arms():
    config = InformationMatchedTransitionConfig()
    config.validate()
    assert config.seeds == (61017, 61029, 61043)
    assert config.training_steps == 800
    train = inspect.getsource(InformationMatchedVoltageTransitionBenchmark._train_seed)
    batch = inspect.getsource(
        InformationMatchedVoltageTransitionBenchmark._common_numpy_batch
    )
    final = inspect.getsource(
        InformationMatchedVoltageTransitionBenchmark.finalize_information_matched_benchmark
    )
    assert "prediction = model(features)" in train
    assert "for name in MATCHED_MODEL_NAMES" in train
    assert 'values["voltage_t1"]' in batch and 'values["voltage_t"]' in batch
    assert "np.minimum" not in batch
    assert '"same_numeric_input_tensor": True' in final
    assert '"teacher_endpoint_used_as_input": False' in final
    assert '"autoregressive_rollout_performed": False' in final


def test_matched_hayflow_path_cannot_train_or_change_metric_support():
    source = inspect.getsource(MatchedFrozenHayFlowComparison.evaluate)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "expected_count != 512" in source
    assert "np.minimum(target, -55.0)" in inspect.getsource(
        MatchedFrozenHayFlowComparison._soma_metrics
    )
    assert '"same_target": True' in source
    assert '"same_transitions": True' in source
    assert '"same_input_contract": False' in source
    assert '"retraining_performed": False' in source
    recovery = inspect.getsource(restore_registered_branch_elm_checkpoints)
    assert "len(checkpoints) != 6" in recovery
    assert '"reports_imported": False' in recovery
