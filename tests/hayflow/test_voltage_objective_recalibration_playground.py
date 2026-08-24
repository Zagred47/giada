import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.voltage_objective_recalibration_playground import (
    EXPECTED_06BH_ARCHIVE_SHA256,
    EXPECTED_06BH_FINAL_SHA256,
    EXPECTED_06BH_INDEX_SHA256,
    FALLBACK_ARM,
    FROZEN_REFERENCE_ARM,
    PRIMARY_ARM,
    TRAINABLE_OBJECTIVE_ARMS,
    VoltageObjectiveRecalibrationConfig,
    VoltageObjectiveRecalibrationPlayground,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_H = (
    ROOT / "experiments" / "hayflow" / "06b_h_frozen_voltage_generalization_forensic"
)
EXPERIMENT_I = (
    ROOT / "experiments" / "hayflow" / "06b_i_voltage_objective_recalibration_playground"
)
NOTEBOOK = ROOT / "notebooks" / "06b_i_voltage_objective_recalibration_playground.ipynb"


def test_06bi_registers_exact_verified_06bh_result():
    assert EXPECTED_06BH_ARCHIVE_SHA256 == (
        "8c2c255a234c7fb25673bafcec0e7521ca084fd98d7d9c0c9cc86211bab1b948"
    )
    assert EXPECTED_06BH_INDEX_SHA256 == (
        "4ab21e154a1972cdbd9e059c7bcc7094814810eb4b0b0cac8a8a2c49f69a1c2d"
    )
    assert EXPECTED_06BH_FINAL_SHA256 == (
        "bfc1a664d3549a2aecd28a6c2df200e188e0a05498c90c2afde353f75ea2f813"
    )
    result = json.loads((EXPERIMENT_H / "result.json").read_text())
    assert result["artifact_integrity"]["indexed_member_failures"] == []
    assert result["formal_diagnosis"] == (
        "FROZEN_VOLTAGE_CALIBRATION_RESCUES_GENERALIZATION"
    )
    assert result["sealed_audit"]["median_voltage_gain_vs_persistence"] > 0
    assert result["activity_breakdown"]["median_quiescent_lt_1mV_gain"] < 0


def test_06bi_preregisters_parallel_atomic_matrix_and_role_reuse():
    prereg = json.loads((EXPERIMENT_I / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert len(prereg["synchronized_matrix"]["arms"]) == 5
    assert prereg["synchronized_matrix"]["primary_arm"] == PRIMARY_ARM
    assert prereg["synchronized_matrix"]["fallback_arm"] == FALLBACK_ARM
    assert prereg["synchronized_matrix"]["mechanism_STATE_updater_frozen"]
    assert not prereg["roles"]["new_independent_confirmation_claimed"]


def test_06bi_configuration_has_fixed_endpoint_and_bounded_gain():
    config = VoltageObjectiveRecalibrationConfig()
    config.validate()
    assert config.objective_checkpoints == (0, 100, 200, 400)
    assert config.objective_unroll_horizon_ms == 8
    assert config.gain_minimum < config.gain_initial < config.gain_maximum
    assert FROZEN_REFERENCE_ARM == "frozen_alpha_075"
    assert len(TRAINABLE_OBJECTIVE_ARMS) == 5


def test_06bi_state_updater_is_frozen_and_stream_is_synchronized():
    new_arm = inspect.getsource(VoltageObjectiveRecalibrationPlayground._new_objective_arm)
    assert "parameter.requires_grad_(False)" in new_arm
    training = inspect.getsource(
        VoltageObjectiveRecalibrationPlayground.train_synchronized_objective_matrix
    )
    row_position = training.index("rows = rng.choice")
    arm_position = training.index("for arm, model in models.items()", row_position)
    assert row_position < arm_position
    assert 'self._batch_tensors("fit"' in training
    assert 'self._batch_tensors("development"' not in training


def test_06bi_future_targets_only_define_training_weights():
    loss = inspect.getsource(VoltageObjectiveRecalibrationPlayground._objective_unroll)
    assert 'batch["voltage_t1"]' in loss
    gain = inspect.getsource(VoltageObjectiveRecalibrationPlayground._gain)
    assert "target" not in gain
    assert "raw_delta" in gain and "voltage" in gain


def test_06bi_finalization_preserves_primary_fallback_and_blocks_06c():
    source = inspect.getsource(
        VoltageObjectiveRecalibrationPlayground.finalize_voltage_objective_recalibration
    )
    assert source.index("if primary_pass") < source.index("elif fallback_pass")
    assert '"coupled_06c_canary_authorized": False' in source
    assert '"new_independent_confirmation_claimed": False' in source
    assert '"full_training_authorized": False' in source


def test_06bi_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "EXPECTED_06BH_INDEX_SHA256" in code
    assert "train_synchronized_objective_matrix" in code
    assert "evaluate_objective_matrix" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    assert "display(evaluation_report)" not in code
