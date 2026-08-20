import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.hayflow_model.graph_state_contract_reassessment import (
    EXPECTED_05M_ARCHIVE_SHA256,
    EXPECTED_05M_FINAL_SHA256,
    EXPECTED_05M_INDEX_SHA256,
    GraphStateContractReassessmentConfig,
    _metrics,
    _ridge_fit,
    _ridge_predict,
    axial_voltage_features,
    semantic_state_projection,
    sketch_normalized_state,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05n_graph_operator_and_state_contract_reassessment.ipynb"


def _layout():
    records = [
        {"category": "voltage"},
        {"category": "voltage"},
        {"category": "voltage"},
        {"category": "mechanism_states"},
        {"category": "mechanism_states"},
        {"category": "calcium_ions"},
    ]
    return SimpleNamespace(
        state_width=6,
        segment_count=3,
        core_records=records,
        core_segment_ids=np.asarray([0, 1, 2, 0, 1, 2]),
        parent_ids=np.asarray([0, 0, 1]),
        segments=[
            {"axial_conductance_to_parent_us": 0.0},
            {"axial_conductance_to_parent_us": 2.0},
            {"axial_conductance_to_parent_us": 4.0},
        ],
    )


def test_registered_05m_hashes_are_exact():
    assert EXPECTED_05M_ARCHIVE_SHA256 == "155d7234e6ce27e8bd7eaa4378f6b90eb240d350cc6befa454cf8a5d3b8eccc6"
    assert EXPECTED_05M_INDEX_SHA256 == "d7e0d88a9e90fdf97f00041157beba54fd948c2b6d146a46fdda58182aae91ba"
    assert EXPECTED_05M_FINAL_SHA256 == "6c100e4fe6983dfb0477afe2938567590cd523b640e9864dbb5940bbaaf5bd98"


def test_config_contract_is_fixed_and_valid():
    config = GraphStateContractReassessmentConfig.from_mapping(
        {"ridge_lambdas": [0.001, 0.1, 1.0]}
    )
    assert config.ridge_lambdas == (0.001, 0.1, 1.0)
    assert config.state_sketch_dim == 16
    config.validate()


def test_semantic_projection_excludes_voltage_and_is_deterministic():
    layout = _layout()
    first = semantic_state_projection(layout, dimension=4, seed=17)
    second = semantic_state_projection(layout, dimension=4, seed=17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:3], 0.0)
    assert np.any(first[3:] != 0.0)
    normalized = np.asarray([[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]])
    sketch = sketch_normalized_state(
        normalized, layout, first, clip=8.0
    )
    assert sketch.shape == (1, 3, 4)
    assert np.isfinite(sketch).all()


def test_axial_features_use_authentic_parent_and_conductance():
    layout = _layout()
    voltage = np.asarray([[-70.0, -60.0, -40.0]], dtype=np.float32)
    features = axial_voltage_features(voltage, layout)
    assert features.shape == (1, 3, 4)
    np.testing.assert_allclose(features[0, :, 0], [0.0, -0.1, -0.2])
    assert features[0, 2, 2] < features[0, 1, 2]
    assert np.isfinite(features).all()


def test_ridge_probe_recovers_linear_signal_and_metrics():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(1000, 4))
    target = 2.0 * features[:, 0] - 0.5 * features[:, 2] + 0.2
    model = _ridge_fit(features, target, 1e-4)
    prediction = _ridge_predict(model, features)
    metric = _metrics(prediction, target, np.abs(target) > 1.0)
    assert metric["rmse_mv"] < 1e-5
    assert metric["regenerative_rmse_mv"] < 1e-5
    assert metric["finite"]


def test_05n_notebook_is_compact_causal_and_uses_blob_download():
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
    assert "EXPECTED_05M_INDEX_SHA256" in code
    assert "discover_indexed_artifact_source" in code
    assert "prepare_composite_flowmap_bundle" in code
    assert "GraphStateContractReassessment" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
    assert "fresh_test" not in code.lower()
