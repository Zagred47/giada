"""Independently verify the archived Task28 result against its preregistration."""
import hashlib
import json
import math
import subprocess
import zipfile
from collections import Counter
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "experiments/results/task28_kaggle_4c3cba7"


def read(name):
    return json.loads((FOLDER / name).read_text(encoding="utf-8"))


def audit():
    cfg = json.loads((ROOT / "experiments/task28_ionic_block_teacher_forced.json").read_text(encoding="utf-8"))
    report, native, contract, provenance = map(read, (
        "final_report.json", "native_audit.json", "run_contract.json", "code_provenance.json"))
    assert cfg == contract
    assert read("process_status.json") == {"phase": "gpu", "returncode": 0}
    assert provenance["code_revision"] == "4c3cba797da73aa48e7a7a5b68f9536e89010b14"
    assert not provenance["dirty_runtime"]
    for name, expected in provenance["sources"].items():
        local = (ROOT / name).read_bytes()
        if hashlib.sha256(local).hexdigest() != expected:
            # Windows checkouts can transform a tracked CRLF file. Compare the
            # immutable Git blob that Kaggle checked out on Linux instead.
            blob = subprocess.run(["git", "show", provenance["code_revision"] + ":" + name],
                                  cwd=ROOT, check=True, capture_output=True).stdout
            assert hashlib.sha256(blob).hexdigest() == expected, name
    for name, expected in provenance["parent_hashes"].items():
        assert hashlib.sha256((ROOT / "experiments/results/task27_kaggle_c072e69" / name).read_bytes()).hexdigest() == expected, name
    with zipfile.ZipFile(FOLDER / "artifact_bundle.zip") as archive:
        assert archive.testzip() is None
        for name in ("final_report.json", "native_audit.json", "run_contract.json", "code_provenance.json", "process_status.json"):
            assert archive.read(name) == (FOLDER / name).read_bytes(), name
    assert native["valid"] and {r["channel"] for r in native["rows"]} == set(cfg["canonical_formula_channels"])
    assert all(r["passed"] and r["maximum_step_error"] <= cfg["oracle_max_gate_error"]
               and r["maximum_current_error_ma_cm2"] <= cfg["oracle_max_current_error_ma_cm2"] for r in native["rows"])
    assert report["channels"] == cfg["channels"]
    assert report["learned_channels"] == cfg["learned_channels"]
    assert report["formula_channels"] == cfg["canonical_formula_channels"]
    assert len(report["channels"]) == 11 and len(report["learned_channels"]) == 5 and len(report["formula_channels"]) == 6
    assert not report["training_performed"] and not report["fresh_used_for_selection"]
    assert not report["compute_gate_measured"] and not report["gate_d_completed"] and not report["task29_authorized"]
    families, arms, seeds = cfg["frozen_families"], cfg["frozen_arms"], (17, 29, 43)
    domains = ("in_support", "activation_boundary", "state_extrema", "calcium_tail")
    expected_fresh = set(product(domains, cfg["fresh_seeds"], families, arms, seeds))
    expected_paths = set(product(range(cfg["path_count"]), families, arms, seeds))
    fresh_keys = [(r["domain"], r["support_seed"], r["family"], r["arm"], r["model_seed"]) for r in report["fresh_rows"]]
    path_keys = [(r["path"], r["family"], r["arm"], r["model_seed"]) for r in report["path_rows"]]
    assert len(fresh_keys) == len(expected_fresh) and set(fresh_keys) == expected_fresh
    assert len(path_keys) == len(expected_paths) and set(path_keys) == expected_paths
    assert all(n == 1 for n in Counter(fresh_keys).values()) and all(n == 1 for n in Counter(path_keys).values())
    all_rows = report["fresh_rows"] + report["path_rows"]
    for row in report["fresh_rows"]:
        metrics = row["metrics"]
        gates = metrics["per_channel_gates"]
        panels = metrics["current_panels"]
        assert set(gates) == set(cfg["channels"])
        assert set(panels) == set(cfg["panel_gbar_multipliers"])
        gate = max(v["rmse"] for v in gates.values())
        individual = max(v["worst_individual_normalized_rmse"] for v in panels.values())
        total = max(v["total_normalized_rmse"] for v in panels.values())
        assert math.isclose(gate, row["worst_gate_rmse"], abs_tol=1e-15)
        assert math.isclose(individual, row["worst_individual_current_normalized_rmse"], abs_tol=1e-15)
        assert math.isclose(total, row["worst_total_current_normalized_rmse"], abs_tol=1e-15)
        passed = (metrics["finite"] and gate <= cfg["learned_gate_rmse_limit"]
                  and individual <= cfg["learned_current_normalized_rmse_limit"]
                  and total <= cfg["learned_total_current_normalized_rmse_limit"]
                  and all(v["occupancy_violations"] == 0 for v in gates.values()))
        assert row["passed"] == passed
    for row in report["path_rows"]:
        gate = max(v["rmse"] for v in row["metrics"]["per_channel_gates"].values())
        current = max(v["worst_individual_normalized_rmse"] for v in row["metrics"]["current_panels"].values())
        assert math.isclose(gate, row["worst_gate_rmse"], abs_tol=1e-15)
        assert math.isclose(current, row["worst_individual_current_normalized_rmse"], abs_tol=1e-15)
        assert row["passed"] == (gate <= cfg["learned_path_gate_rmse_limit"]
                                  and current <= cfg["learned_path_current_normalized_rmse_limit"])
    per_arm = {family: {arm: all(r["passed"] for r in all_rows if r["family"] == family and r["arm"] == arm)
                        for arm in arms} for family in families}
    assert report["per_arm_passed"] == per_arm
    assert report["teacher_forced_ionic_block_passed"] == (native["valid"] and all(per_arm[f]["both"] for f in families))
    assert report["valid"] and report["teacher_forced_ionic_block_passed"]
    return {"valid": True, "fresh_rows": len(fresh_keys), "path_rows": len(path_keys),
            "all_arms_passed": all(all(arms.values()) for arms in per_arm.values()),
            "worst_fresh_gate_rmse": max(r["worst_gate_rmse"] for r in report["fresh_rows"]),
            "worst_fresh_individual_current_normalized_rmse": max(r["worst_individual_current_normalized_rmse"] for r in report["fresh_rows"]),
            "worst_fresh_total_current_normalized_rmse": max(r["worst_total_current_normalized_rmse"] for r in report["fresh_rows"]),
            "worst_path_gate_rmse": max(r["worst_gate_rmse"] for r in report["path_rows"]),
            "worst_path_individual_current_normalized_rmse": max(r["worst_individual_current_normalized_rmse"] for r in report["path_rows"]),
            "gate_d_completed": False, "task29_authorized": False}


if __name__ == "__main__":
    result = audit()
    (FOLDER / "result_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
