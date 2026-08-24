import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.recursive_voltage_state_contract_forensic import (
    BOUNDARY_CONTRACTS,
    RecursiveVoltageStateContractConfig,
    RecursiveVoltageStateContractForensic,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    ROOT
    / "experiments"
    / "hayflow"
    / "06b_e_recursive_voltage_state_contract_forensic"
)
NOTEBOOK = ROOT / "notebooks" / "06b_e_recursive_voltage_state_contract_forensic.ipynb"
CONFIG = ROOT / "configs" / "hayflow" / "hayflow_recursive_voltage_state_contract_forensic.yml"


def test_06be_preregisters_complete_frozen_boundary_factorial():
    prereg = json.loads((EXPERIMENT / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert len(BOUNDARY_CONTRACTS) == 8
    assert set(BOUNDARY_CONTRACTS.values()) == {
        (teacher_v, teacher_s, teacher_i)
        for teacher_v in (False, True)
        for teacher_s in (False, True)
        for teacher_i in (False, True)
    }
    assert prereg["alignment_contract"]["training_performed"] is False
    assert prereg["alignment_contract"]["same_windows_for_every_cell"]
    assert len(prereg["simultaneous_questions"]) == 6


def test_06be_configuration_uses_exact_final_06bd_checkpoints():
    config = RecursiveVoltageStateContractConfig()
    config.validate()
    text = CONFIG.read_text(encoding="utf-8")
    assert config.frozen_checkpoint_budget == 1500
    assert config.minimum_material_feedback_effect_fraction == 0.02
    assert config.minimum_positive_seed_count == 2
    assert "frozen_checkpoint_budget: 1500" in text


def test_06be_crosses_feedback_without_training_or_selection():
    evaluation = inspect.getsource(
        RecursiveVoltageStateContractForensic.evaluate_recursive_contract_matrix
    )
    preparation = inspect.getsource(
        RecursiveVoltageStateContractForensic.prepare_recursive_contract_forensic
    )
    finalization = inspect.getsource(
        RecursiveVoltageStateContractForensic.finalize_recursive_contract_forensic
    )
    assert "teacher_voltage" in evaluation
    assert "teacher_state" in evaluation
    assert "teacher_ions" in evaluation
    assert "held_ions" in evaluation
    assert "optimizer" not in evaluation.lower()
    assert '"training_performed": False' in preparation
    assert '"candidate_selection_performed": False' in finalization
    assert '"coupled_06c_canary_authorized": False' in finalization


def test_06be_reports_main_effects_interactions_and_negative_control():
    summary = inspect.getsource(
        RecursiveVoltageStateContractForensic.summarize_recursive_contract_effects
    )
    for name in (
        "state_feedback_penalty",
        "voltage_feedback_penalty",
        "ion_feedback_penalty",
        "voltage_state_interaction",
        "state_ion_interaction",
        "constant_over_cosine_full_state_gain",
        "causal_specificity_full_state_gain",
    ):
        assert name in summary


def test_06be_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "evaluate_recursive_contract_matrix" in code
    assert "summarize_recursive_contract_effects" in code
    assert "EXPECTED_06BD_INDEX_SHA256" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    assert "display(matrix)" not in code
