import ast
import inspect
import json
from pathlib import Path

import pytest

from src.hayflow_model import atomic_state_dynamics_playground as atomic
from src.hayflow_model.recursive_event_exposure_playground import (
    AUXILIARY_MODES,
    CAUSAL_CONTROLS,
    EXPOSURE_MODES,
    STABILITY_MODES,
    RecursiveEventExposureConfig,
    RecursiveEventExposurePlayground,
    RecursiveEventSourceCell,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "hayflow" / "06b_r_recursive_event_exposure_playground"
NOTEBOOK = ROOT / "notebooks" / "06b_r_recursive_event_exposure_playground.ipynb"


def test_06br_preregisters_atomic_control_and_paired_2x2x2_matrix():
    prereg = json.loads((EXPERIMENT / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["stage_A_atomic_auxiliary_probe"]["arms"] == [
        "causal_target",
        "permuted_target",
    ]
    assert prereg["stage_B_factorial_matrix"]["design"] == "2x2x2"
    assert prereg["stage_B_factorial_matrix"]["factor_count"] == 8
    assert prereg["stage_B_factorial_matrix"]["development_used_for_selection"] is False
    assert prereg["registered_gates"]["maximum_per_seed_8ms_regression_fraction"] == 0.0


def test_06br_configuration_is_fixed_and_spans_registered_budgets():
    config = RecursiveEventExposureConfig()
    config.validate()
    assert len(AUXILIARY_MODES) * len(EXPOSURE_MODES) * len(STABILITY_MODES) == 8
    assert config.recursive_checkpoints[0] == 0
    assert config.recursive_checkpoints[-1] == config.recursive_training_steps
    session = object.__new__(RecursiveEventExposurePlayground)
    session.config = config
    assert len(session._specs()) == 8
    assert len(CAUSAL_CONTROLS) == 4


@pytest.mark.skipif(atomic.torch is None, reason="PyTorch is optional locally")
def test_06br_all_factorial_arms_share_one_parameterization():
    model = RecursiveEventSourceCell(
        base_feature_width=20,
        event_embedding_width=8,
        region_count=4,
        region_width=3,
        hidden_width=12,
        output_limit=8.0,
        auxiliary_width=3,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count > 0
    assert all(
        sum(parameter.numel() for parameter in model.parameters()) == count
        for _ in AUXILIARY_MODES
        for _ in EXPOSURE_MODES
        for _ in STABILITY_MODES
    )


def test_06br_pushforward_is_detached_and_selection_is_calibration_only():
    loss_source = inspect.getsource(
        RecursiveEventExposurePlayground._recursive_training_loss
    )
    train_source = inspect.getsource(
        RecursiveEventExposurePlayground.train_recursive_factorial_matrix
    )
    assert "prediction.detach()" in loss_source
    assert "passive_relative_directional" in inspect.getsource(
        RecursiveEventExposurePlayground._passive_relative_directional_penalty
    ) or "passive_gain" in inspect.getsource(
        RecursiveEventExposurePlayground._passive_relative_directional_penalty
    )
    assert "calibration_half_A" in train_source
    assert "calibration_half_B" in train_source
    assert '"development"' not in train_source


def test_06br_auxiliary_target_is_causal_and_permuted_control_is_real():
    prepare_source = inspect.getsource(
        RecursiveEventExposurePlayground._materialize_causal_auxiliary_targets
    )
    probe_source = inspect.getsource(
        RecursiveEventExposurePlayground.run_causal_auxiliary_probe
    )
    assert "encode_realized_synaptic_drive" in prepare_source
    assert 'self.store.read_state(flat, "t")' in prepare_source
    assert "t_plus_1" not in prepare_source
    assert "PERMUTED_TARGET" in probe_source
    assert "torch.roll" in probe_source


def test_06br_notebook_is_compact_parseable_and_uses_blob_download():
    if not NOTEBOOK.is_file():
        pytest.skip("notebook is added after the implementation commit is pinned")
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
    assert "IMPLEMENTATION_COMMIT_PLACEHOLDER" not in code
    assert "cbd7be51e87650b7790bf6514309eb065617effa" in code
    assert "prepare_recursive_event_exposure_playground" in code
    assert "run_causal_auxiliary_probe" in code
    assert "train_recursive_factorial_matrix" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
