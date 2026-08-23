import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.hayflow_model.atomic_voltage_path_identifiability import (
    EXPECTED_06A_FINAL_SHA256,
    EXPECTED_06A_INDEX_SHA256,
    AtomicVoltagePathConfig,
    VoltagePathMechanismStateUpdater,
    voltage_path_features,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "06a_b_atomic_voltage_path_identifiability.ipynb"
PREREGISTRATION = (
    ROOT
    / "experiments"
    / "hayflow"
    / "06a_b_atomic_voltage_path_identifiability"
    / "preregistration.json"
)


def test_registered_06a_authority_is_exact():
    assert EXPECTED_06A_INDEX_SHA256 == (
        "ad28ed4666e8bd99fb0be5f5d2230e7b731868e20740e94fdd7537aa56e96cb5"
    )
    assert EXPECTED_06A_FINAL_SHA256 == (
        "fa72141a9ca50ceb2582d6eec1852dcd5af7d83e6e34b7e408750a48bf518fa2"
    )


def test_factorial_configuration_is_bounded_and_nested():
    config = AtomicVoltagePathConfig()
    config.validate()
    assert config.short_training_steps == 300
    assert config.training_steps == 1200
    assert config.maximum_parameter_count == 7238
    assert config.voltage_path_sample_indices[-1] == 40
    assert config.rollout_horizons_ms == (1, 2, 4, 8)


def test_voltage_path_control_and_teacher_features_have_equal_width():
    trace = np.zeros((2, 41, 3), dtype=np.float32)
    voltage_t = np.asarray([[-70.0, -60.0, -50.0], [-65.0, -55.0, -45.0]])
    voltage_t1 = voltage_t + 8.0
    for sample in range(41):
        fraction = sample / 40.0
        trace[:, sample, :] = voltage_t + (fraction * fraction * 8.0)
    indices = (5, 10, 15, 20, 25, 30, 35, 40)
    endpoint = voltage_path_features(
        trace, voltage_t, voltage_t1, indices, "linear_endpoint_path"
    )
    teacher = voltage_path_features(
        trace, voltage_t, voltage_t1, indices, "teacher_microtrace_path"
    )
    assert endpoint.shape == teacher.shape == (2, 3, 8)
    np.testing.assert_allclose(endpoint[:, :, -1], 8.0)
    np.testing.assert_allclose(teacher[:, :, -1], 8.0)
    assert not np.allclose(endpoint[:, :, :-1], teacher[:, :, :-1])


def test_path_updater_starts_as_exact_persistence():
    torch = pytest.importorskip("torch")
    model = VoltagePathMechanismStateUpdater(
        mechanism_count=3,
        variable_count=4,
        kind_count=2,
        region_count=2,
        static_width=3,
        drive_width=5,
        path_width=8,
        hidden_width=12,
        embedding_width=4,
        normalized_delta_limit=8.0,
    )
    count = 6
    prediction = model(
        torch.zeros(count),
        torch.zeros(count),
        torch.zeros(count, 8),
        torch.zeros(count, 5),
        torch.zeros(count, 3),
        torch.zeros(count, dtype=torch.long),
        torch.zeros(count, dtype=torch.long),
        torch.zeros(count, dtype=torch.long),
        torch.zeros(count, dtype=torch.long),
    )
    assert torch.equal(prediction, torch.zeros_like(prediction))


def test_preregistration_forbids_capacity_and_heldout_sweeps():
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert registered["roles"]["source_split"] == "train_only"
    assert registered["factorial_design"]["maximum_parameter_count"] == 7238
    assert not registered["factorial_design"]["model_capacity_growth"]
    assert registered["evaluation"]["rollout_window_policy"].startswith(
        "all horizons are prefixes"
    )
    assert "validation_model_selection" in registered["not_authorized"]


def test_06ab_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "AtomicVoltagePathIdentifiability" in code
    assert "EXPECTED_06A_INDEX_SHA256" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
    assert "display(preflight)" not in code
