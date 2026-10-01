"""Task 17e: independent, frozen Ca_HVA causal-replacement confirmation."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from src.hayflow_teacher.dendritic_calibration import DendriticProtocolCalibrator
from src.hayflow_teacher.diagnostic_dataset_v1_1 import TargetedDiagnosticDatasetSession
from src.hayflow_teacher.event_extractor import default_event_definitions, extract_events

from .double_oracle import ExtractedGateFormula
from .native_process import native_phase
from .roadmap_causal_cahva_replacement import (
    SITES, _activate_arm, _canonical_gbar, _compare, _paired_effect,
    _trial, candidate_protocols, compile_candidate_mods,
)
from .roadmap_task17c_m_precision_solver import (
    ALL_SUFFIXES, generate_precision_mods, probe_compiled_m_tables,
)
from .roadmap_task17d_native_solver_mechanism import (
    _extended_sample, _select_calcium_state, _set_policy, _state_names,
    Task17dConfig,
)
from .roadmap_task17db_solver_release_confirmation import (
    _release_audit, _release_pair,
)


FROZEN_2049_TABLE_SHA256 = "e15f712faccc68dafbc7d861a14e23262388b9715ae5b564c51c06f261251765"
ARMS = ("native", "formula", "m_pair_2049_f64")
REQUIRED_EVENT_KINDS = ("somatic_spike", "calcium_spike", "nmda_spike")


@dataclass(frozen=True)
class Task17eConfig:
    seeds: tuple[int, ...] = (171001, 171019, 171043)
    gbar_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)
    duration_ms: int = 60
    sample_interval_ms: float = 0.025
    nmda_variant_synapse_count: int = 8
    nmda_variant_event_window_ms: float = 0.6
    calcium_variant_synapse_count: int = 10
    calcium_variant_event_window_ms: float = 0.6
    native_repeat_voltage_max_atol_mv: float = 1e-5
    formula_voltage_rmse_limit_mv: float = 0.05
    formula_gate_max_error_limit: float = 0.002
    candidate_voltage_rmse_limit_mv: float = 2.0
    candidate_gate_max_error_limit: float = 0.01
    candidate_current_rmse_limit_ma_cm2: float = 1e-4
    paired_effect_relative_error_limit: float = 0.2
    paired_effect_floor_mv: float = 0.05
    spike_onset_limit_ms: float = 0.2
    dendritic_onset_limit_ms: float = 0.5
    release_quantity_atol: float = 1e-10

    def validate(self):
        if asdict(self) != asdict(Task17eConfig()):
            raise ValueError("Task 17e config differs from preregistration")


def _protocols(config):
    base = candidate_protocols()
    return (
        None, base[1], base[2],
        replace(base[1], synapse_count=config.nmda_variant_synapse_count,
                event_window_ms=config.nmda_variant_event_window_ms),
        replace(base[2], synapse_count=config.calcium_variant_synapse_count,
                event_window_ms=config.calcium_variant_event_window_ms),
    )


def _recording_trial(session, calibrator, protocol, seed, multiplier, arm,
                     config, canonical_gbar):
    representatives = dict(session.audit.representatives)
    labels = ("soma", "ais", "trunk", "hot_zone", "nexus", "tuft")
    if any(label not in representatives for label in labels):
        raise RuntimeError("Task 17e representative event sites missing")
    samples = []

    def observer(active, suffix):
        row = _extended_sample(active, suffix)
        samples.append({label: float(active.audit.live_segments[representatives[label]].v)
                        for label in labels})
        return row

    trial = _trial(session, calibrator, protocol, seed, multiplier, arm,
                   config, canonical_gbar, ALL_SUFFIXES, sample_fn=observer,
                   capture_release=True)
    steps = int(round(1 / config.sample_interval_ms))
    expected = 1 + config.duration_ms * (steps + 1)
    if len(samples) != expected:
        raise RuntimeError(f"Task 17e event trace length {len(samples)} != {expected}")
    selected = [0] + [1 + step * (steps + 1) + offset
                      for step in range(config.duration_ms)
                      for offset in range(1, steps + 1)]
    event_traces = {label: [samples[index][label] for index in selected]
                    for label in labels}
    time_grid = np.arange(len(selected), dtype=float) * config.sample_interval_ms
    definitions = [definition for definition in default_event_definitions(representatives)
                   if definition.signal in event_traces]
    trial["events"] = extract_events(time_grid, event_traces, definitions)
    trial["event_traces"] = event_traces
    trial["release_audit"] = _release_audit(trial)
    return trial


def _event_comparison(reference, candidate, config):
    result = {}
    kinds = sorted(set(e["kind"] for e in reference["events"])
                   | set(e["kind"] for e in candidate["events"]))
    for kind in kinds:
        left = sorted((e for e in reference["events"] if e["kind"] == kind),
                      key=lambda e: e["onset_ms"])
        right = sorted((e for e in candidate["events"] if e["kind"] == kind),
                       key=lambda e: e["onset_ms"])
        limit = config.spike_onset_limit_ms if kind in ("axonal_spike", "somatic_spike") else config.dendritic_onset_limit_ms
        onset = max((abs(a["onset_ms"] - b["onset_ms"])
                     for a, b in zip(left, right)), default=0.0)
        censored_match = len(left) == len(right) and all(
            a["right_censored"] == b["right_censored"] for a, b in zip(left, right))
        result[kind] = {"reference_count": len(left), "candidate_count": len(right),
                        "maximum_onset_error_ms": onset,
                        "right_censored_match": censored_match,
                        "limit_ms": limit,
                        "passed": len(left) == len(right) and censored_match and onset <= limit + 1e-12}
    return result


def _max_metric(metrics, key):
    return max(site[key] for site in metrics.values())


def run_task17e(elm_repo, teacher_repo, native_mod, output_dir,
                config=Task17eConfig(), *, code_revision="unknown"):
    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    native_mod = Path(native_mod)
    formula = ExtractedGateFormula.from_mod(native_mod)
    mods, tables = generate_precision_mods(native_mod, output_dir / "candidate_mods")
    table_hash = hashlib.sha256(tables["m_pair_2049_f64"].tobytes()).hexdigest()
    if table_hash != FROZEN_2049_TABLE_SHA256:
        raise RuntimeError(f"Task 17e frozen 2049 table mismatch: {table_hash}")
    compiler = shutil.which("nrnivmodl")
    if not compiler:
        raise RuntimeError("nrnivmodl unavailable")
    native_phase("task17e_compile")
    compilation = compile_candidate_mods(output_dir / "candidate_mods", compiler)
    from neuron import h, load_mechanisms
    if not load_mechanisms(str(output_dir / "candidate_mods")):
        raise RuntimeError("Task 17e compiled mechanisms did not load")
    grid_probe = probe_compiled_m_tables(h, tables)
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, calibration_source=native_mod,
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        output_dir=output_dir / "teacher_workspace")
    teacher = session.prepare_teacher()
    if teacher["segment_count"] != 642:
        raise RuntimeError("Task 17e canonical teacher missing")
    session.run_burn_in()
    session.task17_snapshot_suffix = "Ca_HVA"
    calibrator = DendriticProtocolCalibrator(session, output_dir=output_dir / "protocol_workspace")
    canonical_gbar = _canonical_gbar(session)
    calcium_state = _select_calcium_state(_state_names(session))
    base_config = Task17dConfig()
    _set_policy(session, "ultra", calcium_state, base_config)
    protocols = _protocols(config)
    names = ["quiescent", protocols[1].candidate_id, protocols[2].candidate_id,
             "nmda_variant_n8_w0p6", "calcium_variant_n10_w0p6"]
    if len(set(names)) != len(names):
        raise RuntimeError("Task 17e duplicate protocol schedule")
    # Repeated native preflight before any sealed episode.
    anchor = _recording_trial(session, calibrator, None, config.seeds[0], 1.0,
                              "native", config, canonical_gbar)
    repeat = _recording_trial(session, calibrator, None, config.seeds[0], 1.0,
                              "native", config, canonical_gbar)
    repeat_metrics = _compare(anchor, repeat)
    if (_max_metric(repeat_metrics, "voltage_max_error_mv") > config.native_repeat_voltage_max_atol_mv
            or not _release_pair(anchor["release_audit"], repeat["release_audit"])["discrete_match"]):
        raise RuntimeError("Task 17e native repeat preflight failed")
    cache, runtime = {}, {}
    total = len(ARMS) * len(protocols) * len(config.seeds) * len(config.gbar_multipliers)
    completed = 0
    for arm in ARMS:
        _activate_arm(session, canonical_gbar, arm, ALL_SUFFIXES)
        for protocol_index, protocol in enumerate(protocols):
            for seed in config.seeds:
                for multiplier in config.gbar_multipliers:
                    _set_policy(session, "ultra", calcium_state, base_config)
                    native_phase("task17e_trial", arm=arm, protocol=protocol_index,
                                 seed=seed, multiplier=multiplier)
                    started = time.perf_counter()
                    trial = _recording_trial(session, calibrator, protocol, seed,
                                             multiplier, arm, config, canonical_gbar)
                    elapsed = time.perf_counter() - started
                    key = (protocol_index, seed, multiplier)
                    cache[(arm, *key)] = trial
                    runtime[f"{arm}/{protocol_index}/{seed}/{multiplier}"] = elapsed
                    if arm == "formula":
                        native = cache[("native", *key)]
                        metrics = _compare(native, trial)
                        event_control = _event_comparison(native, trial, config)
                        release_control = _release_pair(native["release_audit"], trial["release_audit"])
                        if (_max_metric(metrics, "voltage_rmse_mv") > config.formula_voltage_rmse_limit_mv
                                or max(max(v["m_max_error"], v["h_max_error"])
                                       for v in metrics.values()) > config.formula_gate_max_error_limit
                                or _max_metric(metrics, "current_rmse_ma_cm2") > config.candidate_current_rmse_limit_ma_cm2
                                or not all(detail["passed"] for detail in event_control.values())
                                or not release_control["discrete_match"]):
                            raise RuntimeError(f"Task 17e formula control failed at {key}")
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == total:
                        print(f"[GIADA Task 17e] {completed}/{total} arm={arm} protocol={protocol_index}", flush=True)
    comparisons, effects = [], []
    event_coverage = {kind: 0 for kind in REQUIRED_EVENT_KINDS}
    for protocol_index, protocol in enumerate(protocols):
        for seed in config.seeds:
            by_multiplier = {}
            for multiplier in config.gbar_multipliers:
                key = (protocol_index, seed, multiplier)
                reference = cache[("native", *key)]
                for event in reference["events"]:
                    if event["kind"] in event_coverage:
                        event_coverage[event["kind"]] += 1
                by_multiplier[multiplier] = {arm: cache[(arm, *key)] for arm in ARMS}
                for arm in ("formula", "m_pair_2049_f64"):
                    candidate = cache[(arm, *key)]
                    comparisons.append({"protocol_index": protocol_index,
                                        "protocol": names[protocol_index], "seed": seed,
                                        "gbar_multiplier": multiplier, "arm": arm,
                                        "metrics": _compare(reference, candidate),
                                        "events": _event_comparison(reference, candidate, config),
                                        "release": _release_pair(reference["release_audit"],
                                                                 candidate["release_audit"])})
            low, high = min(config.gbar_multipliers), max(config.gbar_multipliers)
            effects.append({"protocol_index": protocol_index, "protocol": names[protocol_index],
                            "seed": seed,
                            "formula": _paired_effect(by_multiplier[low]["native"],
                                                      by_multiplier[high]["native"],
                                                      by_multiplier[low]["formula"],
                                                      by_multiplier[high]["formula"],
                                                      config.paired_effect_floor_mv),
                            "candidate": _paired_effect(by_multiplier[low]["native"],
                                                        by_multiplier[high]["native"],
                                                        by_multiplier[low]["m_pair_2049_f64"],
                                                        by_multiplier[high]["m_pair_2049_f64"],
                                                        config.paired_effect_floor_mv)})
    candidate_rows = [row for row in comparisons if row["arm"] == "m_pair_2049_f64"]
    formula_rows = [row for row in comparisons if row["arm"] == "formula"]
    identifiable = [site for effect in effects for site in effect["candidate"].values()
                    if site["effect_identifiable"]]
    gates = {
        "formula_control": all(
            _max_metric(row["metrics"], "voltage_rmse_mv") <= config.formula_voltage_rmse_limit_mv
            and max(max(v["m_max_error"], v["h_max_error"])
                    for v in row["metrics"].values()) <= config.formula_gate_max_error_limit
            and _max_metric(row["metrics"], "current_rmse_ma_cm2") <= config.candidate_current_rmse_limit_ma_cm2
            and all(detail["passed"] for detail in row["events"].values())
            and row["release"]["discrete_match"]
            for row in formula_rows),
        "voltage": all(_max_metric(row["metrics"], "voltage_rmse_mv") <= config.candidate_voltage_rmse_limit_mv
                       for row in candidate_rows),
        "gates": all(max(max(v["m_max_error"], v["h_max_error"])
                         for v in row["metrics"].values()) <= config.candidate_gate_max_error_limit
                     for row in candidate_rows),
        "current": all(_max_metric(row["metrics"], "current_rmse_ma_cm2") <= config.candidate_current_rmse_limit_ma_cm2
                       for row in candidate_rows),
        "events": all(all(detail["passed"] for detail in row["events"].values())
                      for row in candidate_rows),
        "release": all(row["release"]["discrete_match"]
                       and row["release"]["max_released_quantity_difference"] <= config.release_quantity_atol
                       for row in candidate_rows),
        "paired_effect": bool(identifiable) and all(
            site["relative_error"] <= config.paired_effect_relative_error_limit
            for site in identifiable),
        "event_support": all(event_coverage.values()),
    }
    # Compact traces support independent inspection without flooding Kaggle.
    stride = int(round(1 / config.sample_interval_ms))
    reduced = {f"{arm}/{protocol_index}/{seed}/{multiplier}": {
        "traces_1ms": {site: {name: values[::stride] for name, values in signals.items()}
                       for site, signals in trial["traces"].items()},
        "events": trial["events"],
        "release_discrete_sha256": trial["release_audit"]["discrete_sha256"],
    } for (arm, protocol_index, seed, multiplier), trial in cache.items()}
    (output_dir / "paired_metrics.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (output_dir / "paired_effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    (output_dir / "runtime_seconds.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (output_dir / "diagnostic_traces_1ms.json").write_text(json.dumps(reduced), encoding="utf-8")
    report = {"schema_version": "giada-task17e-independent-causal-confirmation-v1",
              "valid": True, "code_revision": code_revision,
              "decision": "GATE_C_PASS_BOUNDED_CAHVA" if all(gates.values()) else "GATE_C_NO_GO",
              "gate_c_authorized": bool(all(gates.values())),
              "scope": "Ca_HVA in canonical 642-segment teacher; five fixed protocols and three new seeds",
              "task18_authorized": bool(all(gates.values())),
              "config": asdict(config), "protocols": names, "teacher_segment_count": 642,
              "candidate": "m_pair_2049_f64", "frozen_table_sha256": table_hash,
              "candidate_mods": mods, "compilation": compilation,
              "grid_probe": grid_probe, "native_repeat_preflight": repeat_metrics,
              "episode_count": total, "comparison_count": len(comparisons),
              "effect_pair_count": len(effects), "identifiable_effect_count": len(identifiable),
              "event_coverage": event_coverage, "gates": gates,
              "worst_candidate_voltage_rmse_mv": max(_max_metric(row["metrics"], "voltage_rmse_mv") for row in candidate_rows),
              "worst_candidate_gate_error": max(max(max(v["m_max_error"], v["h_max_error"])
                                                    for v in row["metrics"].values()) for row in candidate_rows),
              "worst_candidate_current_rmse_ma_cm2": max(_max_metric(row["metrics"], "current_rmse_ma_cm2") for row in candidate_rows),
              "worst_identifiable_effect_relative_error": max((site["relative_error"] for site in identifiable), default=None),
              "runtime_total_seconds_by_arm": {arm: sum(value for key, value in runtime.items()
                                                         if key.startswith(arm + "/")) for arm in ARMS},
              "warning": "Bounded Ca_HVA gate only; no speedup or cross-morphology claim."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
