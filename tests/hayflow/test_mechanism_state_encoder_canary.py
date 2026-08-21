import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.hayflow_model.mechanism_state_encoder_canary import (
    EXPECTED_05R_ARCHIVE_SHA256,
    EXPECTED_05R_FINAL_SHA256,
    EXPECTED_05R_INDEX_SHA256,
    MechanismStateEncoderCanaryConfig,
    shared_semantic_state_projection,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05s_mechanism_state_encoder_canary.ipynb"


def test_registered_05r_hashes_are_exact():
    assert EXPECTED_05R_ARCHIVE_SHA256 == (
        "7126ab8540934a60483bb4bcf3d64a9e5e6cc90364322a4df0b3ad70b84dff64"
    )
    assert EXPECTED_05R_INDEX_SHA256 == (
        "2b2e731006373ba1ec53a0cbd8099e9a269b1e095e75039d35c043bf8de88912"
    )
    assert EXPECTED_05R_FINAL_SHA256 == (
        "5c876547680eaee2bd8c12e73f63bf381294c3a9cf7e181fcc34468182c019c5"
    )


def test_shared_semantic_projection_is_stable_and_category_masked():
    records = [
        {
            "category": "mechanism_states",
            "mechanism": "NaTa_t",
            "variable": "m",
            "kind": "state",
        },
        {
            "category": "mechanism_states",
            "mechanism": "NaTa_t",
            "variable": "m",
            "kind": "state",
        },
        {
            "category": "calcium_ions",
            "mechanism": "cadynamics",
            "variable": "cai",
            "kind": "concentration",
        },
        {
            "category": "voltage",
            "mechanism": "membrane",
            "variable": "v",
            "kind": "voltage",
        },
    ]
    layout = SimpleNamespace(state_width=len(records), core_records=records)
    first = shared_semantic_state_projection(
        layout, dimension=64, seed=510091, category="mechanism_states"
    )
    second = shared_semantic_state_projection(
        layout, dimension=64, seed=510091, category="mechanism_states"
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], first[1])
    assert np.any(first[0])
    assert not np.any(first[2:])


def test_config_is_bounded_and_valid():
    config = MechanismStateEncoderCanaryConfig.from_mapping(
        {"horizons_ms": [2, 4, 8], "seeds": [17, 29, 43]}
    )
    assert config.epochs == 30
    assert config.progress_interval == 15
    assert config.candidate_material_gain_fraction == 0.02
    config.validate()


def test_05s_notebook_is_compact_paired_and_uses_blob_download():
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
    for constant in (
        "EXPECTED_05R_INDEX_SHA256",
        "EXPECTED_05Q_INDEX_SHA256",
        "EXPECTED_05P_INDEX_SHA256",
        "EXPECTED_05O_INDEX_SHA256",
    ):
        assert constant in code
    assert "MechanismStateEncoderCanary" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
