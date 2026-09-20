import json
from pathlib import Path

from src.giada_teacher import build_causal_classification


def test_canonical_causal_classification() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "experiments/teacher_mechanism_inventory_v1.json").read_text(encoding="utf-8"))
    result = build_causal_classification(inventory)
    assert result["validation"]["valid"]
    assert result["summary"]["mechanism_count"] == 20
    by_name = {row["name"]: row for row in result["mechanisms"]}
    assert by_name["Ca_HVA"]["primary_family"] == "voltage_dependent_gate"
    assert by_name["SK_E2"]["primary_family"] == "calcium_dependent_gate"
    assert by_name["CaDynamics_E2"]["primary_family"] == "ion_concentration_dynamics"
    assert "stochastic_release" in by_name["ProbAMPANMDA2"]["secondary_families"]
    assert by_name["Axial coupling"]["source_kind"] == "neuron_runtime"
    assert by_name["epsp"]["primary_family"] == "prescribed_time_dependent_point_process"
