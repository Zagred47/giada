import ast
import inspect
import json
from pathlib import Path

import numpy as np

import pytest

from src.hayflow_model import atomic_state_dynamics_playground as atomic
from src.hayflow_model.effective_membrane_source_playground import (
    DIRECT_VOLTAGE,
    FROZEN_BOUNDARY_STATE,
    HINES_SOURCE,
    INSTANTANEOUS,
    LOCAL_RECURRENT,
    PREDICTED_DYNAMIC_STATE,
    EXPECTED_06BN_ARCHIVE_SHA256,
    EXPECTED_06BN_FINAL_SHA256,
    EXPECTED_06BN_INDEX_SHA256,
    CausalMembraneSourceCell,
    EffectiveMembraneSourceConfig,
    EffectiveMembraneSourcePlayground,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_N = ROOT / "experiments" / "hayflow" / "06b_n_structure_preserving_coupling_forensic"
EXPERIMENT_O = ROOT / "experiments" / "hayflow" / "06b_o_effective_membrane_source_playground"
NOTEBOOK = ROOT / "notebooks" / "06b_o_effective_membrane_source_playground.ipynb"


def test_06bo_is_authorized_by_exact_terminal_06bn_result():
    result = json.loads((EXPERIMENT_N / "result.json").read_text())
    assert result["archive_sha256"] == EXPECTED_06BN_ARCHIVE_SHA256
    assert result["artifact_index_sha256"] == EXPECTED_06BN_INDEX_SHA256
    assert result["final_report_sha256"] == EXPECTED_06BN_FINAL_SHA256
    assert result["formal_diagnosis"] == (
        "OBJECTIVE_COUPLING_AND_RELAXATION_DO_NOT_CLOSE_ROLLOUT_GAP"
    )


def test_06bo_preregisters_aligned_three_axis_matrix_and_controls():
    prereg = json.loads((EXPERIMENT_O / "preregistration.json").read_text())
    matrix = prereg["stage_2_factorial_matrix"]
    assert prereg["status"] == "preregistered_before_execution"
    assert matrix["arm_count"] == 8
    assert matrix["same_numeric_input_tensor"]
    assert matrix["same_minibatches_within_seed"]
    assert matrix["same_initialization_within_seed"]
    assert not prereg["stage_1_exact_reconstruction"]["selection_eligible"]
    assert not prereg["stage_3_frozen_counterfactuals"]["selection_eligible"]


def test_06bo_config_and_specs_cover_exact_2x2x2_matrix():
    config = EffectiveMembraneSourceConfig()
    config.validate()
    session = object.__new__(EffectiveMembraneSourcePlayground)
    session.config = config
    specs = session._specs()
    assert len(specs) == 8 and len(set(specs)) == 8
    assert {row[0] for row in specs} == {DIRECT_VOLTAGE, HINES_SOURCE}
    assert {row[1] for row in specs} == {
        FROZEN_BOUNDARY_STATE,
        PREDICTED_DYNAMIC_STATE,
    }
    assert {row[2] for row in specs} == {INSTANTANEOUS, LOCAL_RECURRENT}
    assert config.matrix_checkpoints == (0, 100, 200, 400)


def test_06bo_source_target_is_defined_by_cable_equation_residual():
    source = inspect.getsource(
        EffectiveMembraneSourcePlayground._normalized_source_target
    )
    assert "_matrix_apply(target_voltage" in source
    assert "base_rhs" in source and "source_scale" in source
    apply_source = inspect.getsource(EffectiveMembraneSourcePlayground._apply_output)
    assert "solver(diagonal, coupling, base_rhs + source)" in apply_source


def test_06bo_exact_identity_audit_separates_float64_from_operational_float32():
    source = inspect.getsource(
        EffectiveMembraneSourcePlayground.run_exact_source_reconstruction_audit
    )
    assert ".to(dtype=atomic.torch.float64)" in source
    assert '"identity_audit_dtype": "float64"' in source
    assert '"operational_training_dtype": "float32"' in source
    assert '"maximum_float32_authentic_reconstruction_error_mv"' in source


def test_06bo_empty_metric_support_is_json_null_not_nan():
    session = object.__new__(EffectiveMembraneSourcePlayground)
    metrics = session._masked_voltage_metrics(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        {"empty": np.zeros((1, 2), dtype=bool)},
    )["empty"]
    assert metrics == {
        "coordinate_count": 0,
        "voltage_rmse_mv": None,
        "persistence_rmse_mv": None,
        "voltage_gain_vs_persistence_fraction": None,
    }
    json.dumps(metrics, allow_nan=False)
    assert session._available_median([None, float("nan"), 1.0, 3.0]) == 2.0
    assert session._available_median([None, float("nan")]) is None


def test_06bo_training_keeps_streams_paired_and_selects_on_calibration():
    source = inspect.getsource(
        EffectiveMembraneSourcePlayground.train_synchronized_source_matrix
    )
    assert "batch_stream_sha256" in source
    assert "self._batch_tensors(\"fit\"" in source
    assert "selected_calibration_rmse_mv" in source
    assert "self._batch_tensors(\"development\"" not in source
    assert "median_calibration_rmse_by_step" in inspect.getsource(
        EffectiveMembraneSourcePlayground._checkpoint_scaling
    )


def test_06bo_frozen_controls_cover_state_topology_axial_and_identity():
    source = inspect.getsource(EffectiveMembraneSourcePlayground.evaluate_source_matrix)
    for token in (
        "teacher_state_refresh_upper_bound",
        "relabelled_topology",
        "no_axial",
        "spatial_output_shuffle",
    ):
        assert token in source
    assert "counterfactuals_retrained" in source


@pytest.mark.skipif(atomic.torch is None, reason="PyTorch is optional locally")
def test_06bo_parameter_matched_cell_is_zero_output_at_initialization():
    model = CausalMembraneSourceCell(13, 4, 3, 8, 6.0)
    features = atomic.torch.randn(2, 5, 13)
    regions = atomic.torch.zeros(5, dtype=atomic.torch.long)
    hidden = atomic.torch.zeros(2, 5, 8)
    output, next_hidden = model(features, regions, hidden, recurrent=True)
    assert output.shape == (2, 5)
    assert next_hidden.shape == (2, 5, 8)
    assert atomic.torch.count_nonzero(output) == 0


def test_06bo_notebook_is_compact_parseable_and_uses_blob_download():
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
    assert "EXPECTED_06BN_INDEX_SHA256" in code
    assert "abe09f40a737f5df183bd2c3801c3beefe02c323" in code
    assert "MODULE_PATH.is_file()" in code
    assert "name.startswith('src.')" in code
    assert (
        "from src.hayflow_model.effective_membrane_source_playground import "
        "EffectiveMembraneSourceConfig,EffectiveMembraneSourcePlayground"
    ) in code
    assert "run_exact_source_reconstruction_audit" in code
    assert "train_synchronized_source_matrix" in code
    assert "evaluate_source_matrix" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
