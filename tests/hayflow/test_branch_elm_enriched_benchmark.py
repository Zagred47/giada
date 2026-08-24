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


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "06b_c_supplement_branch_elm_enriched_benchmark.ipynb"
PREREGISTRATION = ROOT / "experiments" / "hayflow" / "06b_c_branch_elm_enriched_benchmark" / "preregistration.json"
EXECUTION_AMENDMENT = PREREGISTRATION.with_name("execution_amendment.json")


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
    assert "BranchELMEnrichedBenchmark" in code
    assert "EXPECTED_05JO_INDEX_SHA256" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    assert "display(results)" not in code
