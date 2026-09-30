"""Task 17d-b: release decisions, native convergence and matched Ca_HVA arms.

This is a diagnostic run on previously studied conditions. It cannot promote
the frozen causal replacement or authorize Gate C.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.hayflow_teacher.dendritic_calibration import DendriticProtocolCalibrator
from src.hayflow_teacher.diagnostic_dataset_v1_1 import TargetedDiagnosticDatasetSession

from .double_oracle import ExtractedGateFormula
from .native_process import native_phase
from .roadmap_causal_cahva_replacement import (
    CausalReplacementConfig, _activate_arm, _canonical_gbar, _compare,
    _trial, candidate_protocols, compile_candidate_mods,
    verify_frozen_rate_table,
)
from .roadmap_embedded_frozen_confirmation import _lut_table
from .roadmap_task17c_m_precision_solver import (
    ALL_SUFFIXES, _grid_table, generate_precision_mods,
    probe_compiled_m_tables,
)
from .roadmap_task17d_native_solver_mechanism import (
    _extended_sample, _select_calcium_state, _set_policy, _state_names,
    _state_metrics, _timing_metrics, Task17dConfig,
)


DECISION_FIELDS = (
    "transition_id", "event_index", "synapse_id", "scheduled_time_ms",
    "random123_seed", "random123_stream_id", "random123_global_index",
    "rng_sequence_before", "rng_sequence_after", "rng_preview_value",
    "release_success",
)


@dataclass(frozen=True)
class Task17dbConfig:
    conditions: tuple[tuple[int, float], ...] = (
        (170029, 1.0), (170083, 1.5), (170029, 0.5))
    protocol_index: int = 1
    duration_ms: int = 60
    sample_interval_ms: float = 0.025
    super_ultra_atol: float = 1e-8
    super_ultra_rtol: float = 1e-9
    reference_voltage_rmse_limit_mv: float = 0.05
    reference_gate_max_error_limit: float = 0.002
    formula_voltage_rmse_limit_mv: float = 0.05
    formula_gate_max_error_limit: float = 0.002

    def validate(self):
        if asdict(self) != asdict(Task17dbConfig()):
            raise ValueError("Task 17d-b config differs from preregistration")


def _release_audit(trial):
    rows = trial["release_rows"]
    if len(rows) != 60 or any(not r["verification"].get("valid", False) for r in rows):
        raise RuntimeError("Task 17d-b incomplete causal release verification")
    events = []
    for row in rows:
        for outcome in row["outcomes"]:
            events.append({"step": int(row["step"]),
                           **{field: outcome[field] for field in DECISION_FIELDS},
                           "release_probability": outcome["release_probability"],
                           "released_quantity": outcome["released_quantity"]})
    decisions = [{field: event[field] for field in ("step", *DECISION_FIELDS)}
                 for event in events]
    return {"event_count": len(events),
            "success_count": sum(bool(e["release_success"]) for e in events),
            "discrete_sha256": hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest(),
            "events": events}


def _release_pair(reference, candidate):
    left, right = reference["events"], candidate["events"]
    same_length = len(left) == len(right)
    fields = ("step", *DECISION_FIELDS)
    mismatch_counts = {name: 0 for name in fields}
    for a, b in zip(left, right):
        for name in fields:
            mismatch_counts[name] += int(a[name] != b[name])
    return {"event_count_reference": len(left), "event_count_candidate": len(right),
            "discrete_match": same_length and not any(mismatch_counts.values()),
            "mismatch_counts": mismatch_counts,
            "max_release_probability_difference": max(
                (abs(a["release_probability"] - b["release_probability"])
                 for a, b in zip(left, right)), default=0.0),
            "max_released_quantity_difference": max(
                (abs(a["released_quantity"] - b["released_quantity"])
                 for a, b in zip(left, right)), default=0.0)}


def _set_super_ultra(session, calcium_state, config):
    cvode = session.cvode
    cvode.atolscale(calcium_state, 1.0)
    cvode.atolscale("v", 1.0)
    cvode.atol(config.super_ultra_atol)
    cvode.rtol(config.super_ultra_rtol)
    cvode.re_init()
    observed = (float(cvode.atol()), float(cvode.rtol()),
                float(cvode.atolscale(calcium_state)), float(cvode.atolscale("v")))
    expected = (config.super_ultra_atol, config.super_ultra_rtol, 1.0, 1.0)
    if not np.allclose(observed, expected, rtol=1e-6, atol=1e-15):
        raise RuntimeError(f"Task 17d-b super-ultra policy failed: {observed}")
    return dict(zip(("atol", "rtol", "calcium_scale", "voltage_scale"), observed))


def _maximum(metrics, key):
    return max(site[key] for site in metrics.values())


def run_task17db(elm_repo, teacher_repo, native_mod, output_dir,
                 config=Task17dbConfig(), *, code_revision="unknown"):
    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    formula = ExtractedGateFormula.from_mod(native_mod)
    frozen_table = _lut_table(formula)
    frozen_hash = hashlib.sha256(frozen_table.tobytes()).hexdigest()
    table_2049 = _grid_table(formula, 2049, np.dtype("float64"))
    table_2049_hash = hashlib.sha256(table_2049.tobytes()).hexdigest()
    mods, tables = generate_precision_mods(native_mod, output_dir / "candidate_mods")
    if hashlib.sha256(tables["m_pair_2049_f64"].tobytes()).hexdigest() != table_2049_hash:
        raise RuntimeError("Task 17d-b 2049-knot generated MOD/table mismatch")
    compiler = shutil.which("nrnivmodl")
    if not compiler:
        raise RuntimeError("nrnivmodl unavailable")
    native_phase("task17db_compile")
    compilation = compile_candidate_mods(output_dir / "candidate_mods", compiler)
    from neuron import h, load_mechanisms
    if not load_mechanisms(str(output_dir / "candidate_mods")):
        raise RuntimeError("Task 17d-b compiled mechanisms did not load")
    frozen_probe = verify_frozen_rate_table(h, frozen_table, CausalReplacementConfig())
    grid_probe = probe_compiled_m_tables(h, tables)
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, calibration_source=Path(native_mod),
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        output_dir=output_dir / "teacher_workspace")
    teacher = session.prepare_teacher()
    if teacher["segment_count"] != 642:
        raise RuntimeError("Task 17d-b canonical teacher missing")
    session.run_burn_in()
    session.task17_snapshot_suffix = "Ca_HVA"
    calibrator = DendriticProtocolCalibrator(session, output_dir=output_dir / "protocol_workspace")
    gbar = _canonical_gbar(session)
    calcium_state = _select_calcium_state(_state_names(session))
    base_config = Task17dConfig()
    if (float(session.cvode.atol()), float(session.cvode.rtol())) != (base_config.default_atol, base_config.default_rtol):
        raise RuntimeError("Task 17d-b canonical default solver changed")
    protocol = candidate_protocols()[config.protocol_index]
    cache, releases, policy_audit = {}, {}, {}
    # Fail fast on all policy setters before the first full trajectory.
    for policy in ("default", "voltage_scaled", "ultra"):
        policy_audit[policy] = _set_policy(session, policy, calcium_state, base_config)
    policy_audit["super_ultra"] = _set_super_ultra(session, calcium_state, config)
    for policy in ("default", "voltage_scaled", "ultra", "super_ultra"):
        for seed, multiplier in config.conditions:
            if policy == "super_ultra":
                _set_super_ultra(session, calcium_state, config)
            else:
                _set_policy(session, policy, calcium_state, base_config)
            native_phase("task17db_native_trial", policy=policy, seed=seed, multiplier=multiplier)
            row = _trial(session, calibrator, protocol, seed, multiplier, "native",
                         config, gbar, ALL_SUFFIXES, sample_fn=_extended_sample,
                         capture_release=True)
            key = (policy, seed, multiplier)
            cache[key] = row
            releases[key] = _release_audit(row)
    native_pairs = []
    for seed, multiplier in config.conditions:
        reference = cache[("ultra", seed, multiplier)]
        for policy in ("default", "voltage_scaled", "super_ultra"):
            key = (policy, seed, multiplier)
            metrics = _compare(reference, cache[key])
            native_pairs.append({"policy": policy, "seed": seed,
                                 "gbar_multiplier": multiplier, "metrics": metrics,
                                 "state_metrics": _state_metrics(reference, cache[key]),
                                 "timing": _timing_metrics(reference, cache[key], config.sample_interval_ms),
                                 "release": _release_pair(releases[("ultra", seed, multiplier)], releases[key])})
    reference_stable = all(
        _maximum(row["metrics"], "voltage_rmse_mv") <= config.reference_voltage_rmse_limit_mv
        and max(max(site["m_max_error"], site["h_max_error"])
                for site in row["metrics"].values()) <= config.reference_gate_max_error_limit
        for row in native_pairs if row["policy"] == "super_ultra")
    arm_pairs = []
    for arm in ("formula", "lut", "m_pair_2049_f64"):
        _activate_arm(session, gbar, arm, ALL_SUFFIXES)
        for seed, multiplier in config.conditions:
            _set_policy(session, "ultra", calcium_state, base_config)
            native_phase("task17db_candidate_trial", arm=arm, seed=seed, multiplier=multiplier)
            row = _trial(session, calibrator, protocol, seed, multiplier, arm,
                         config, gbar, ALL_SUFFIXES, sample_fn=_extended_sample,
                         capture_release=True)
            release = _release_audit(row)
            reference = cache[("ultra", seed, multiplier)]
            arm_pairs.append({"arm": arm, "seed": seed, "gbar_multiplier": multiplier,
                              "metrics": _compare(reference, row),
                              "state_metrics": _state_metrics(reference, row),
                              "timing": _timing_metrics(reference, row, config.sample_interval_ms),
                              "release": _release_pair(releases[("ultra", seed, multiplier)], release)})
            releases[(arm, seed, multiplier)] = release
    formula_control_valid = all(
        _maximum(row["metrics"], "voltage_rmse_mv") <= config.formula_voltage_rmse_limit_mv
        and max(max(site["m_max_error"], site["h_max_error"])
                for site in row["metrics"].values()) <= config.formula_gate_max_error_limit
        for row in arm_pairs if row["arm"] == "formula")
    compact_releases = {f"{policy}/{seed}/{multiplier}": audit
                        for (policy, seed, multiplier), audit in releases.items()}
    (output_dir / "release_decisions.json").write_text(json.dumps(compact_releases), encoding="utf-8")
    (output_dir / "native_pairs.json").write_text(json.dumps(native_pairs, indent=2), encoding="utf-8")
    (output_dir / "candidate_pairs.json").write_text(json.dumps(arm_pairs, indent=2), encoding="utf-8")
    report = {"schema_version": "giada-task17db-solver-release-confirmation-v1",
              "valid": True, "decision": "DIAGNOSTIC_ONLY", "gate_c_authorized": False,
              "code_revision": code_revision, "config": asdict(config),
              "teacher_segment_count": teacher["segment_count"],
              "rederived_513_table_sha256": frozen_hash,
              "rederived_2049_m_table_sha256": table_2049_hash,
              "candidate_mods": mods,
              "compilation": compilation, "frozen_probe": frozen_probe,
              "grid_probe": grid_probe, "policy_audit": policy_audit,
              "native_pair_count": len(native_pairs), "candidate_pair_count": len(arm_pairs),
              "native_discrete_release_match_ultra": all(r["release"]["discrete_match"] for r in native_pairs),
              "candidate_discrete_release_match_native": all(r["release"]["discrete_match"] for r in arm_pairs),
              "reference_stable_at_registered_limits": reference_stable,
              "formula_control_valid": formula_control_valid,
              "warning": "Known development cases only; no candidate selection, no sealed confirmation."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
