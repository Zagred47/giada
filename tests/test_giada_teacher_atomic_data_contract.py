import json
from pathlib import Path

from src.giada_teacher import build_atomic_data_contract


def test_canonical_atomic_data_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "experiments/teacher_mechanism_inventory_v1.json").read_text(encoding="utf-8"))
    classification = json.loads((root / "experiments/teacher_causal_mechanism_classification_v1.json").read_text(encoding="utf-8"))
    contract = build_atomic_data_contract(inventory, classification)
    assert contract["validation"]["valid"]
    assert contract["validation"]["family_count"] == 7
    assert contract["validation"]["mechanism_binding_count"] == 20
    gate = contract["family_contracts"]["family:voltage_dependent_gate"]
    assert "only" in gate["sufficiency"]
    assert "path-aware" in gate["solver"]
    synapse = contract["family_contracts"]["family:event_driven_synapse"]
    assert any("RNG" in item for item in synapse["state"])
    assert "unordered event counts" in synapse["forbidden_inputs"]
    axial = contract["family_contracts"]["family:axial_coupling"]
    assert "complete instantiated segment tree" in axial["state"][0]
