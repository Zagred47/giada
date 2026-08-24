import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.causal_voltage_bridge_representation_forensic import (
    EXPECTED_06BB_ARCHIVE_SHA256,
    EXPECTED_06BB_FINAL_SHA256,
    EXPECTED_06BB_INDEX_SHA256,
    REPRESENTATION_ARMS,
    CausalVoltageBridgeRepresentationConfig,
    CausalVoltageBridgeRepresentationForensic,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "06b_c_voltage_bridge_representation_forensic.ipynb"
PREREGISTRATION = ROOT / "experiments" / "hayflow" / "06b_c_voltage_bridge_representation_forensic" / "preregistration.json"


def test_06bb_authority_is_exact_and_registered():
    assert EXPECTED_06BB_ARCHIVE_SHA256 == "d652c3fdf088569b212c6fc710185ab4f870857e5e84cc2947459b7f456bb349"
    assert EXPECTED_06BB_INDEX_SHA256 == "824cc0fdfb977c69fed7bbf3dfcea691f6c1346c84fadedd988aa20c12c56986"
    assert EXPECTED_06BB_FINAL_SHA256 == "41a1fc65c2bc1b812e06758f171983e8201b2627905fdd6a7b0813c442bacc2e"


def test_representation_controls_are_bounded_and_paired():
    config = CausalVoltageBridgeRepresentationConfig()
    config.validate()
    assert config.continuation_training_steps == config.residual_training_steps == 500
    assert config.maximum_residual_parameter_count == 4000
    assert REPRESENTATION_ARMS == (
        "frozen_local_bridge",
        "continued_local_bridge",
        "authentic_tree_residual",
        "relabelled_tree_residual",
    )


def test_preregistration_separates_optimization_and_authentic_topology():
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert registered["paired_arms"] == list(REPRESENTATION_ARMS)
    assert "extra optimizer budget" in registered["causal_isolation"]["continued_local_bridge"]
    assert "real NEURON segment tree" in registered["causal_isolation"]["authentic_tree_residual"]
    assert "parameter-, initialization-, sample-, and optimizer-matched" in registered["causal_isolation"]["relabelled_tree_residual"]
    assert "Branch-ELM sidecar" in registered["prohibited"][-1]


def test_true_and_relabelled_heads_share_seed_and_sample_stream():
    source = inspect.getsource(CausalVoltageBridgeRepresentationForensic._train_tree_residual)
    assert "manual_seed(seed + 631000)" in source
    assert "default_rng(seed + 632000)" in source
    assert 'arm == "relabelled_tree_residual"' in source
    assert "topology_relabelled_parent_ids" in source


def test_06bc_notebook_is_compact_and_uses_stable_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 12
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    assert "CausalVoltageBridgeRepresentationForensic" in code
    assert "EXPECTED_06BB_INDEX_SHA256" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
