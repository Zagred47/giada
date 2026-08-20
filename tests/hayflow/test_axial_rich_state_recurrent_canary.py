import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.hayflow_model.axial_rich_state_recurrent_canary import (
    EXPECTED_05O_ARCHIVE_SHA256,
    EXPECTED_05O_FINAL_SHA256,
    EXPECTED_05O_INDEX_SHA256,
    AxialRichStateGraphGRU,
    AxialRichStateRecurrentCanaryConfig,
    torch,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05p_axial_graphgru_rich_state_micro_canary.ipynb"


def test_registered_05o_hashes_are_exact():
    assert EXPECTED_05O_ARCHIVE_SHA256 == "a51fbdab2b7d14c33c112c17c7cc2f50a26cb86e67f6aa2c655cc52d22e29477"
    assert EXPECTED_05O_INDEX_SHA256 == "e8f83a57ba51ef6db5a6fdd4a10c393a934fda493df071d898e0fe299d622ea3"
    assert EXPECTED_05O_FINAL_SHA256 == "80842fe8e56f25a6c8735da06f5cec860cb22d924f4798ca9a973ced058fa984"


def test_config_preserves_verified_state_contract():
    config = AxialRichStateRecurrentCanaryConfig.from_mapping(
        {"horizons_ms": [2, 4, 8], "seeds": [17, 29, 43]}
    )
    assert config.state_sketch_dim == 64
    assert config.state_clip == 8.0
    assert config.epochs == 30
    config.validate()


@pytest.mark.skipif(torch is None, reason="PyTorch is optional locally")
def test_contract_models_are_parameter_identical_and_finite():
    kwargs = {
        "segment_static": np.zeros((3, 2), dtype=np.float32),
        "parent_ids": np.asarray([0, 0, 1]),
        "axial_conductance": np.asarray([0.0, 2.0, 4.0]),
        "state_width": 4,
        "hidden_width": 8,
        "voltage_delta_limit_mv": 20.0,
    }
    models = [
        AxialRichStateGraphGRU(
            **kwargs, use_axial=axial, use_rich_state=state
        )
        for axial, state in ((False, False), (True, False), (False, True), (True, True))
    ]
    counts = [sum(value.numel() for value in model.parameters()) for model in models]
    assert len(set(counts)) == 1
    voltage = torch.full((2, 3), -70.0)
    drive = torch.zeros((2, 2, 3, 12))
    state = torch.zeros((2, 3, 4))
    for model in models:
        output = model(voltage, drive, state)["voltage"]
        assert tuple(output.shape) == (2, 2, 3)
        assert bool(torch.isfinite(output).all())


def test_05p_notebook_is_compact_causal_and_uses_blob_download():
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
    assert "EXPECTED_05O_INDEX_SHA256" in code
    assert "discover_indexed_artifact_source" in code
    assert "prepare_composite_flowmap_bundle" in code
    assert "AxialRichStateRecurrentCanary" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
