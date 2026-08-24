import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.recursive_joint_repair_matrix import (
    ACCEPTED_06BE_ARTIFACTS,
    REPAIR_ARMS,
    RecursiveJointRepairConfig,
    RecursiveJointRepairMatrix,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "hayflow" / "06b_f_recursive_joint_repair_matrix"
NOTEBOOK = ROOT / "notebooks" / "06b_f_recursive_joint_repair_matrix.ipynb"
CONFIG = ROOT / "configs" / "hayflow" / "hayflow_recursive_joint_repair_matrix.yml"


def test_06bf_preregisters_six_aligned_arms_and_one_primary_arm():
    prereg = json.loads((EXPERIMENT / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["fixed_matrix"]["arms"] == list(REPAIR_ARMS)
    assert prereg["fixed_matrix"]["primary_arm"] == (
        "full_feedback_voltage_protected"
    )
    assert prereg["alignment_contract"]["same_minibatch_stream_within_seed"]
    assert prereg["alignment_contract"]["both_models_trainable_in_every_arm"]
    assert len(prereg["simultaneous_questions"]) == 5


def test_06bf_accepts_both_registered_exact_06be_executions():
    assert set(ACCEPTED_06BE_ARTIFACTS) == {
        "acc8b29f4eacd9c209e6ca4e622da5fa5527d500b30ccaefdfc2580f721fa2ad",
        "dc43155bb0065768f473b41cfd0f7fc3bbe40d00052953a2786e27bf5da0a3ac",
    }
    assert {value["role"] for value in ACCEPTED_06BE_ARTIFACTS.values()} == {
        "canonical",
        "confirmatory_exact_replication",
    }


def test_06bf_configuration_registers_fixed_budget_scaling_and_joint_gates():
    config = RecursiveJointRepairConfig()
    config.validate()
    text = CONFIG.read_text(encoding="utf-8")
    assert config.repair_checkpoints == (0, 200, 400, 600)
    assert config.repair_unroll_horizon_ms == 4
    assert config.minimum_state_error_reduction_fraction == 0.02
    assert config.minimum_voltage_error_reduction_fraction == 0.10
    assert config.minimum_scaling_error_reduction_fraction == 0.02
    assert "repair_checkpoints: [0, 200, 400, 600]" in text


def test_06bf_uses_synchronized_stream_and_voltage_protected_gradients():
    training = inspect.getsource(
        RecursiveJointRepairMatrix.train_synchronized_repair_matrix
    )
    objective = inspect.getsource(RecursiveJointRepairMatrix._unroll_objectives)
    assert "rows = rng.choice" in training
    assert "replace=False" in training
    row_position = training.index("rows = rng.choice")
    assert row_position < training.index("for arm, pair in pairs.items()", row_position)
    assert 'REPAIR_ARMS[arm][2] == "voltage_protected"' in training
    assert "state_loss.backward(retain_graph=True)" in training
    assert "parameter.grad = None" in training
    assert "voltage_loss.backward()" in training
    assert "torch.roll" in objective


def test_06bf_final_gate_requires_joint_safety_specificity_and_scaling():
    finalization = inspect.getsource(
        RecursiveJointRepairMatrix.finalize_recursive_joint_repair
    )
    for name in (
        "state_repaired",
        "voltage_repaired",
        "all_seed_safe",
        "one_step_retained",
        "causal_specificity",
        "scaling_continues",
    ):
        assert name in finalization
    assert 'next_step = "06c_coupled_voltage_state_micro_canary"' in finalization
    assert '"full_training_authorized": False' in finalization


def test_06bf_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "train_synchronized_repair_matrix" in code
    assert "evaluate_final_repair_matrix" in code
    assert "EXPECTED_06BE_INDEX_SHA256" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    assert "display(training_report)" not in code
