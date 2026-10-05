"""Retrospective Task 31 exposure diagnostic from immutable Task 30b/30c reports.

This is not a scheduled-sampling training experiment and cannot select a model.
"""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "formula": (
        ROOT / "experiments/results/task30b_kaggle_ddbffb0/final_report.json",
        "8dd5b2abed6ce3f7b9ca3d130d8fb0c3b7b5b0f5df16b0eb67d41f64ea5e2eb5",
    ),
    "native": (
        ROOT / "experiments/results/task30c_kaggle_df82f42/final_report.json",
        "16dd1825ade7993a614a73118f62811824ba92c6d0ebfe992987723bb5f95665",
    ),
}


def rms(rows, key):
    return math.sqrt(sum(float(row[key]) ** 2 for row in rows) / len(rows))


def run():
    source = {}
    for name, (path, expected) in SOURCES.items():
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name} source hash mismatch: {actual}")
        source[name] = json.loads(raw)
    formula, native = source["formula"], source["native"]
    if not (formula["scientific_primary_passed"] and native["scientific_primary_passed"]
            and native["native_floor_admissible"]):
        raise RuntimeError("Task30b/30c parent scientific gates did not pass")
    horizons = [8.0, 20.0, 40.0, 80.0]
    results = []
    for family in ("independent", "shared_heads"):
        for seed in (17, 29, 43):
            for horizon in horizons:
                f = [r for r in formula["rows"] if r["family"] == family
                     and r["arm"] == "both" and r["seed"] == seed
                     and r["horizon_ms"] == horizon]
                n = [r for r in native["rows"] if r.get("family") == family
                     and r["arm"] == "both" and r.get("seed") == seed
                     and r["horizon_ms"] == horizon]
                if len(f) != 32 or len(n) != 32:
                    raise RuntimeError(f"Missing matched episodes: {family}/{seed}/{horizon}")
                if {r["episode_index"] for r in f} != {r["episode_index"] for r in n}:
                    raise RuntimeError("Episode pairing failed")
                forced = rms(f, "teacher_voltage_gate_rmse")
                autonomous = rms(f, "gate_rmse")
                voltage = rms(n, "voltage_rmse_mv")
                results.append({
                    "family": family, "seed": seed, "horizon_ms": horizon,
                    "teacher_voltage_gate_rmse": forced,
                    "autonomous_gate_rmse": autonomous,
                    "gate_feedback_excess_rmse": autonomous - forced,
                    "gate_feedback_ratio": autonomous / forced if forced else None,
                    "native_voltage_rmse_mv": voltage,
                    "native_worst_episode_voltage_rmse_mv": max(r["voltage_rmse_mv"] for r in n),
                })
    result = {
        "schema_version": "giada-task31-retrospective-exposure-diagnostic-v1",
        "valid": True,
        "analysis_kind": "retrospective_frozen_checkpoint_diagnostic",
        "scheduled_sampling_training_performed": False,
        "new_test_executed": False,
        "model_selected": False,
        "source_sha256": {name: digest for name, (_, digest) in SOURCES.items()},
        "matched_family_seed_horizon_count": len(results),
        "rows": results,
        "decision": "DEFER_SCHEDULED_SAMPLING_INTERVENTION_AT_THIS_SCALE",
        "rationale": (
            "The frozen autonomous active single-compartment candidate already has a very low "
            "native voltage error through 80 ms. Gate-feedback excess exists but is small in absolute "
            "units; training a scheduled-sampling continuation here would not target an observed "
            "voltage bottleneck. Reassess on a future held-out feedback regime if one emerges."
        ),
        "limits": (
            "Exploratory analysis of previously exposed Task30b/30c outputs, not a preregistered "
            "causal intervention or evidence that scheduled sampling never helps. Fixed calcium, "
            "single compartment, no synapses, morphology or acceleration claim."
        ),
        "gate_d_performance_status": "NO_GO_UNCHANGED",
    }
    output = ROOT / "experiments/results/task31_retrospective_exposure_diagnostic.json"
    output.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({k: result[k] for k in ("valid", "decision", "matched_family_seed_horizon_count")}, indent=2))


if __name__ == "__main__":
    run()
