import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.hayflow_model.nonlinear_observability_reassessment import (
    EXPECTED_05N_ARCHIVE_SHA256,
    EXPECTED_05N_FINAL_SHA256,
    EXPECTED_05N_INDEX_SHA256,
    FixedWidthObservabilityProbe,
    NonlinearObservabilityReassessment,
    NonlinearObservabilityReassessmentConfig,
    torch,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05o_nonlinear_observability_reassessment.ipynb"


def test_registered_05n_hashes_are_exact():
    assert EXPECTED_05N_ARCHIVE_SHA256 == "495c4419d2205c9a3e4c58fb0cf67da534d9ada38e9e28b393ba8dbc97b80316"
    assert EXPECTED_05N_INDEX_SHA256 == "975c8b0741d8639d19095447d1b06c06795020e478b21a036e5e3aadc60bd627"
    assert EXPECTED_05N_FINAL_SHA256 == "2460f61579d972464159dfe60f7d73e02e9ce0c9245138dc750253655db9ab30"


def test_config_is_registered_and_valid():
    config = NonlinearObservabilityReassessmentConfig.from_mapping(
        {"ridge_lambdas": [0.001, 0.1], "seeds": [17, 29, 43]}
    )
    assert config.state_sketch_dim == 64
    assert config.hidden_width == 64
    config.validate()


def test_fixed_matrix_zero_fills_absent_contract_blocks():
    arrays = {
        "base": np.ones((2, 3), dtype=np.float32),
        "graph": np.full((2, 2), 2.0, dtype=np.float32),
        "state": np.full((2, 4), 3.0, dtype=np.float32),
        "target": np.zeros(2),
    }
    stats = {
        name: (np.zeros(values.shape[1]), np.ones(values.shape[1]))
        for name, values in arrays.items()
        if name != "target"
    }
    matrix = NonlinearObservabilityReassessment._fixed_matrix(
        None, arrays, ("base",), stats
    )
    assert matrix.shape == (2, 9)
    np.testing.assert_array_equal(matrix[:, :3], 1.0)
    np.testing.assert_array_equal(matrix[:, 3:], 0.0)


@pytest.mark.skipif(torch is None, reason="PyTorch is optional locally")
def test_probe_has_fixed_shape_and_finite_forward():
    probe = FixedWidthObservabilityProbe(88, 64, 2)
    output = probe(torch.zeros((7, 88)))
    assert tuple(output.shape) == (7,)
    assert bool(torch.isfinite(output).all())


def test_05o_notebook_is_compact_causal_and_uses_blob_download():
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
    assert "EXPECTED_05N_INDEX_SHA256" in code
    assert "discover_indexed_artifact_source" in code
    assert "prepare_composite_flowmap_bundle" in code
    assert "NonlinearObservabilityReassessment" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
    assert "fresh_test" not in code.lower()
