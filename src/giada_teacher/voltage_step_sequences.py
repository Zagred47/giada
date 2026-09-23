"""Roadmap Task 7: paired exogenous Ca_HVA voltage-step histories.

The historical 07/07b/07c notebooks are unrelated closed-loop experiments.
This module completes the step-sequence contrasts not covered by Task 6.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .primitive_matrix_playground import _file_sha, _sha
from .voltage_path_stress import (
    VoltagePathStressConfig,
    authentic_path_pilot,
    evaluate_path_role,
    formula_targets,
    verified_task5_root,
)


FAMILIES = ("rise_timing", "pulse_timing", "pulse_duration", "order_reversal")


def make_paired_step_paths(config: VoltagePathStressConfig, *, role: str):
    """Return paired paths; rows (2j, 2j+1) have identical endpoints/state."""
    config.validate()
    if role not in {"development", "sealed"}:
        raise ValueError(role)
    count = config.development_per_family if role == "development" else config.sealed_per_family
    if count % 2 or config.segments_per_ms != 8:
        raise ValueError("Task 7 requires an even paired count and 8 voltage bins")
    rng = np.random.default_rng(config.development_seed if role == "development"
                                else config.sealed_seed)
    result = {}
    for family in FAMILIES:
        rows = np.empty((count, 10), dtype=np.float64)
        for pair in range(count // 2):
            low = rng.uniform(-90, -45)
            high = min(45.0, low + rng.uniform(35, 85))
            middle = low + .5 * (high - low)
            gate = rng.uniform(.01, .99, size=2)
            if family == "rise_timing":
                a = np.array([low, high, high, high, high, high, high, high])
                b = np.array([low, low, low, low, low, low, high, high])
            elif family == "pulse_timing":
                a = np.array([low, high, high, low, low, low, low, low])
                b = np.array([low, low, low, low, low, high, high, low])
            elif family == "pulse_duration":
                a = np.array([low, low, high, low, low, low, low, low])
                b = np.array([low, low, high, high, high, high, low, low])
            else:
                a = np.array([low, high, high, middle, middle, low, low, low])
                b = np.array([low, middle, middle, high, high, low, low, low])
            rows[2 * pair] = np.r_[a, gate]
            rows[2 * pair + 1] = np.r_[b, gate]
        result[family] = rows
    return result


def paired_teacher_differences(formula, paths, config):
    targets, _ = formula_targets(formula, paths, config)
    return {
        family: {
            "pair_count": int(len(values) // 2),
            "median_max_gate_difference": float(np.median(np.max(
                np.abs(values[0::2] - values[1::2]), axis=1))),
            "maximum_max_gate_difference": float(np.max(np.max(
                np.abs(values[0::2] - values[1::2]), axis=1))),
        }
        for family, values in targets.items()
    }


def _fingerprint(paths):
    return hashlib.sha256(b"".join(np.asarray(paths[k], dtype="<f8").tobytes()
                                    for k in sorted(paths))).hexdigest()


def run_roadmap_task7(formula, task5_source, mechanism_root, output_dir,
                      config=None, *, code_revision="unknown"):
    config = config or VoltagePathStressConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    root = verified_task5_root(task5_source, output_dir / ".verified_task5")
    development = make_paired_step_paths(config, role="development")
    pilot = authentic_path_pilot(formula, development, config, mechanism_root)
    print(f"[GIADA roadmap Task 7] pilot NEURON {pilot['case_count']} casi", flush=True)
    development_report = evaluate_path_role(formula, root, development, config,
                                            role="development")
    dev_pairs = paired_teacher_differences(formula, development, config)
    freeze = {"schema_version": "giada-roadmap-task7-freeze-v1",
              "code_revision": code_revision, "config": asdict(config),
              "families": FAMILIES, "development_path_sha256": _fingerprint(development),
              "development_target_sha256": development_report["target_sha256"],
              "task5_report_sha256": _file_sha(root / "final_report.json"),
              "candidates_retrained": False, "sealed_accessed": False}
    freeze["freeze_sha256"] = _sha(freeze)
    (output_dir / "development_report.json").write_text(json.dumps(
        {"pilot": pilot, "pairs": dev_pairs, "metrics": development_report}, indent=2))
    (output_dir / "selection_freeze.json").write_text(json.dumps(freeze, indent=2))
    print("[GIADA roadmap Task 7] development concluso; freeze scritto", flush=True)
    sealed = make_paired_step_paths(config, role="sealed")
    old = {tuple(row) for values in development.values() for row in values}
    new = {tuple(row) for values in sealed.values() for row in values}
    overlap = len(old & new)
    if overlap:
        raise RuntimeError("Roadmap Task 7 development/sealed path overlap")
    sealed_report = evaluate_path_role(formula, root, sealed, config, role="sealed")
    sealed_pairs = paired_teacher_differences(formula, sealed, config)
    lut_scores = {family: row["normalized_score"] for family, row in
                  sealed_report["metrics"]["lut_known_path"].items()}
    all_finite = bool(np.isfinite(list(lut_scores.values())).all() and
                      np.isfinite([v["maximum_max_gate_difference"] for v in
                                   sealed_pairs.values()]).all())
    report = {"schema_version": "giada-roadmap-task7-final-v1",
              "valid": bool(pilot["valid"] and all_finite and overlap == 0),
              "registered_gate_passed": bool(all_finite and
                   all(score <= .02 for score in lut_scores.values()) and
                   any(row["maximum_max_gate_difference"] >= .001 for row in sealed_pairs.values())),
              "code_revision": code_revision, "historical_07_is_not_roadmap_task7": True,
              "teacher_voltage_path_is_exogenous": True,
              "endogenous_membrane_coupling_tested": False,
              "development": development_report, "development_pairs": dev_pairs,
              "sealed": sealed_report, "sealed_pairs": sealed_pairs,
              "lut_known_path_scores": lut_scores, "sealed_overlap_count": overlap,
              "selection_used_sealed": False, "oracle_pilot": pilot,
              "freeze_sha256": freeze["freeze_sha256"],
              "interpretation_policy": "Known future voltage is an exogenous-clamp oracle, "
                  "not a causal input to an autonomous neuron. This covers step sequence "
                  "history, not the roadmap Task 10 input-sufficiency matrix."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    print("[GIADA roadmap Task 7] sealed concluso", flush=True)
    return report
