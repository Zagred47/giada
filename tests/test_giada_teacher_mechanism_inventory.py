from pathlib import Path

from src.giada_teacher.mechanism_inventory import build_inventory, parse_mod_file


def test_parse_mod_contract(tmp_path: Path) -> None:
    source = tmp_path / "Toy.mod"
    source.write_text(
        """NEURON {
 SUFFIX Toy
 USEION ca READ eca WRITE ica
 RANGE g
 NONSPECIFIC_CURRENT ix
}
PARAMETER { gbar = 0.1 (S/cm2) }
ASSIGNED { v (mV) eca (mV) ica (mA/cm2) }
STATE { m h }
BREAKPOINT { SOLVE states METHOD cnexp ica = gbar*m*h*(v-eca) }
DERIVATIVE states { m' = (1-m)/tau h' = (hinf-h)/tauh }
""",
        encoding="utf-8",
    )
    row = parse_mod_file(source)
    assert row["mechanism"]["name"] == "Toy"
    assert [item["name"] for item in row["declarations"]["state"]] == ["m", "h"]
    assert row["mechanism"]["ions"] == [{"ion": "ca", "read": ["eca"], "write": ["ica"]}]
    assert row["integration"]["solve_statements"] == [{"target": "states", "method": "cnexp"}]


def test_canonical_teacher_inventory_is_complete() -> None:
    teacher = Path(__file__).resolve().parents[2] / "neuron_as_deep_net"
    if not teacher.is_dir():
        return
    inventory = build_inventory(teacher / "L5PC_NEURON_simulation" / "mods", teacher_root=teacher)
    assert inventory["summary"]["mod_file_count"] == 18
    assert inventory["summary"]["state_variable_count"] == 37
    names = {row["mechanism"]["name"] for row in inventory["mechanisms"]}
    assert {"Ca_HVA", "ProbAMPANMDA2", "ProbGABAAB_EMS"} <= names
    assert inventory["teacher_commit"] == "074c4666300a8ad246601dab179a97a6942f0f29"
