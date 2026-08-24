import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.nested_coupling_optimization_scaling_forensic import (
    ARM_OBJECTIVE,
    ARM_SCHEDULE,
    SCALING_ARMS,
    NestedCouplingOptimizationScalingConfig,
    NestedCouplingOptimizationScalingForensic,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    ROOT
    / "experiments"
    / "hayflow"
    / "06b_d_nested_coupling_optimization_scaling_forensic"
)
NOTEBOOK = ROOT / "notebooks" / "06b_d_nested_coupling_optimization_scaling_forensic.ipynb"
CONFIG = ROOT / "configs" / "hayflow" / "hayflow_nested_coupling_optimization_scaling_forensic.yml"


def test_06bd_preregisters_a_small_synchronized_factorial_matrix():
    prereg = json.loads((EXPERIMENT / "preregistration.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["factorial_matrix"]["arms"] == list(SCALING_ARMS)
    assert prereg["factorial_matrix"]["fixed_budget_checkpoints"] == [0, 250, 500, 1000, 1500]
    assert prereg["alignment_contract"]["same_initialization_within_seed"]
    assert prereg["alignment_contract"]["same_minibatch_stream_within_seed"]
    assert prereg["alignment_contract"]["state_updater_frozen"]
    assert not prereg["alignment_contract"]["development_used_for_selection"]
    assert len(prereg["simultaneous_questions"]) == 6


def test_06bd_configuration_has_fixed_budget_mini_scaling_law():
    config = NestedCouplingOptimizationScalingConfig()
    config.validate()
    config_text = CONFIG.read_text(encoding="utf-8")
    assert config.scaling_training_steps == 1500
    assert config.scaling_checkpoints == (0, 250, 500, 1000, 1500)
    assert config.pilot_seeds == (61017, 61029, 61043)
    assert config.coupling_coordinates_per_batch == 1024
    assert "scaling_checkpoints: [0, 250, 500, 1000, 1500]" in config_text
    assert set(ARM_OBJECTIVE.values()) == {"voltage", "joint", "joint_shuffled"}
    assert set(ARM_SCHEDULE.values()) == {"constant", "cosine"}


def test_06bd_joint_loss_updates_only_bridge_through_frozen_state_model():
    state_loss = inspect.getsource(
        NestedCouplingOptimizationScalingForensic._differentiable_state_loss
    )
    training = inspect.getsource(
        NestedCouplingOptimizationScalingForensic.train_synchronized_scaling_matrix
    )
    assert '("linear_endpoint_path", seed)' in state_loss
    assert "predicted[:, None]" in state_loss
    assert "torch.roll" in state_loss
    assert "loss = voltage_loss + float(scale) * state_loss" in training
    assert "same_minibatch_stream_within_seed" in training
    assert "state_updater_retraining_performed" in training
    assert "self.frozen_state_models" not in inspect.getsource(
        NestedCouplingOptimizationScalingForensic._new_continuation_model
    ).split("load_state_dict", 1)[1]


def test_06bd_reports_all_registered_orthogonal_contrasts():
    final = inspect.getsource(
        NestedCouplingOptimizationScalingForensic.finalize_scaling_forensic
    )
    for contrast in (
        "scaling_voltage_gain",
        "joint_state_effect",
        "causal_specificity_state_effect",
        "cosine_voltage_objective_effect",
        "cosine_joint_objective_effect",
        "joint_recursive_effect",
    ):
        assert contrast in final
    assert '"development_used_for_selection": False' in final
    assert '"autonomous_voltage_rollout_claimed": False' in final
    assert '"full_training_authorized": False' in final


def test_06bd_notebook_is_compact_safe_and_uses_stable_zip_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 14
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    assert "train_synchronized_scaling_matrix" in code
    assert "evaluate_fixed_budget_matrix" in code
    assert "evaluate_final_nested_rollouts" in code
    assert "EXPECTED_06BC_INDEX_SHA256" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    assert "display(training)" not in code
