import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.hayflow_model.atomic_state_dynamics_playground import (
    EXPECTED_05T_FINAL_SHA256,
    EXPECTED_05T_INDEX_SHA256,
    AtomicStateDynamicsConfig,
    SemanticMechanismStateUpdater,
    fit_mechanism_statistics,
    inverse_mechanism_logit,
    mechanism_logit,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "06a_atomic_state_dynamics_playground.ipynb"
PREREGISTRATION = (
    ROOT
    / "experiments"
    / "hayflow"
    / "06a_atomic_state_dynamics_playground"
    / "preregistration.json"
)


def test_registered_05t_authority_is_exact():
    assert EXPECTED_05T_INDEX_SHA256 == (
        "84221c3c4dde34e909ab81024c6ab3d44606db00ca44b76edd1374138c95fd34"
    )
    assert EXPECTED_05T_FINAL_SHA256 == (
        "7f061b3b58f9d0d8654873ae607084c62e2ebc26b4ccc17de6dc4d8d7f9a4ffe"
    )


def test_config_is_train_only_bounded_pilot():
    config = AtomicStateDynamicsConfig.from_mapping(
        {"rollout_horizons_ms": [1, 2, 4, 8]}
    )
    assert config.training_steps == 300
    assert config.fit_transition_limit == 1024
    assert config.minimum_pilot_improvement_fraction == 0.02
    config.validate()


def test_preregistration_forbids_heldout_access_and_candidate_selection():
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert registered["roles"]["source_split"] == "train_only"
    assert registered["roles"]["validation_state_access_forbidden"]
    assert registered["roles"]["test_state_access_forbidden"]
    assert not registered["alignment"]["candidate_selection_performed"]
    assert registered["technical_gate"]["decision_grade"] is False


def test_mechanism_logit_roundtrip_and_semantic_scale_repair():
    values = np.asarray([[0.1, 0.2, 0.5], [0.2, 0.2, 0.7]], dtype=np.float64)
    restored = inverse_mechanism_logit(mechanism_logit(values))
    np.testing.assert_allclose(restored, values, atol=1e-12)
    next_values = values.copy()
    next_values[:, 0] += 0.01
    stats = fit_mechanism_statistics(
        values,
        next_values,
        np.asarray([0, 0, 1]),
        minimum_delta_scale=1e-5,
    )
    assert np.all(stats["delta_scale"] >= 1e-5)
    assert int(stats["repaired_coordinate_count"][0]) >= 1


def test_updater_is_exact_persistence_at_initialization():
    torch = pytest.importorskip("torch")
    model = SemanticMechanismStateUpdater(
        mechanism_count=2,
        variable_count=3,
        kind_count=2,
        region_count=2,
        static_width=2,
        drive_width=3,
        hidden_width=8,
        embedding_width=4,
        normalized_delta_limit=8.0,
    )
    count = 5
    prediction = model(
        torch.zeros(count),
        torch.zeros(count),
        torch.zeros(count),
        torch.zeros(count, 3),
        torch.zeros(count, 2),
        torch.zeros(count, dtype=torch.long),
        torch.zeros(count, dtype=torch.long),
        torch.zeros(count, dtype=torch.long),
        torch.zeros(count, dtype=torch.long),
    )
    assert torch.equal(prediction, torch.zeros_like(prediction))


def test_06a_notebook_is_compact_train_only_and_uses_blob_download():
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
    assert "AtomicStateDynamicsPlayground" in code
    assert "EXPECTED_05T_INDEX_SHA256" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
    assert "validation_state_accessed" in code
