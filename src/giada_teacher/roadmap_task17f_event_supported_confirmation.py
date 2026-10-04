"""Task 17f: native support qualification, then frozen causal confirmation.

Pilot selection sees only native teacher outcomes. Confirmation uses disjoint
seeds; missing support is an inconclusive experiment, never a candidate failure.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from src.hayflow_data import InputAction
from src.hayflow_teacher.dendritic_calibration import (
    DendriticCandidate, DendriticProtocolCalibrator, build_candidate_actions,
)
from src.hayflow_teacher.diagnostic_dataset_v1_1 import TargetedDiagnosticDatasetSession
from src.hayflow_teacher.event_extractor import default_event_definitions, extract_events
from .native_process import native_phase
from .roadmap_causal_cahva_replacement import (
    _activate_arm, _canonical_gbar, _compare, _paired_effect, _trial,
    compile_candidate_mods,
)
from .roadmap_task17c_m_precision_solver import (
    ALL_SUFFIXES, generate_precision_mods, probe_compiled_m_tables,
)
from .roadmap_task17d_native_solver_mechanism import (
    Task17dConfig, _extended_sample, _select_calcium_state, _set_policy, _state_names,
)
from .roadmap_task17db_solver_release_confirmation import (
    Task17dbConfig, _release_audit, _release_pair, _set_super_ultra,
)
from .roadmap_task17e_independent_causal_confirmation import (
    FROZEN_2049_TABLE_SHA256, REQUIRED_EVENT_KINDS, Task17eConfig,
    _event_comparison, _max_metric,
)


@dataclass(frozen=True)
class Task17fConfig(Task17eConfig):
    seeds: tuple[int, ...] = (171601, 171619, 171643)
    pilot_seeds: tuple[int, ...] = (171501, 171519, 171543)
    somatic_amplitudes_na: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 6.0)
    repeats_per_burst: tuple[int, ...] = (1, 2, 4)
    minimum_positive_seeds: int = 3

    def validate(self):
        if asdict(self) != asdict(Task17fConfig()):
            raise ValueError("Task 17f config differs from preregistration")
        if set(self.seeds) & set(self.pilot_seeds):
            raise ValueError("pilot and confirmation seeds overlap")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def make_spec(name, target_kind=None, candidate=None, somatic_current_na=0.0):
    return {"name": name, "target_kind": target_kind,
            "candidate": asdict(candidate) if candidate else None,
            "somatic_current_na": float(somatic_current_na)}


def pilot_specs(config, family, somatic_current=0.0):
    if family == "somatic_spike":
        return [make_spec(f"soma_{amp:g}nA", family, somatic_current_na=amp)
                for amp in config.somatic_amplitudes_na]
    nmda = family == "nmda_spike"
    if family not in ("nmda_spike", "calcium_spike"):
        raise ValueError(family)
    base = DendriticCandidate(
        family="task17f_nmda" if nmda else "task17f_calcium",
        target="tuft" if nmda else "hot_zone", required_event_kinds=(family,),
        synapse_count=10 if nmda else 12, burst_count=2 if nmda else 3,
        burst_start_ms=3, event_window_ms=0.4 if nmda else 0.8,
        selection_mode="branch_cluster" if nmda else "target_nearest",
        event_probe_mode="cluster_center" if nmda else "target_representative",
        event_probe_kinds=("nmda_spike",) if nmda else (),
    )
    specs = []
    # Fixed order: canonical-weight unassisted ladder, then somatic assistance.
    for amp in ((0.0,) if nmda else (0.0, somatic_current)):
        for repeats in config.repeats_per_burst:
            candidate = replace(base, events_per_synapse_per_burst=repeats)
            specs.append(make_spec(f"{family}_r{repeats}_soma{amp:g}", family, candidate, amp))
    return specs


def resolve_spec(spec, calibrator, representatives, config):
    """Freeze synapse IDs, probe identity, definitions and actual input actions."""
    candidate = DendriticCandidate(**spec["candidate"]) if spec["candidate"] else None
    selection = None
    actions = {}
    if candidate:
        selection = calibrator.select_synapse_cluster(
            candidate.target, candidate.synapse_count,
            candidate.maximum_tree_distance_um, candidate.selection_mode)
        actions = build_candidate_actions(candidate, selection.synapse_ids,
                                          duration_ms=config.duration_ms)
    amp = spec["somatic_current_na"]
    if amp:
        for step in (2, 3):
            actions[step] = tuple(actions.get(step, ())) + (
                InputAction("somatic_current", 0.05, duration_ms=0.9, amplitude_na=amp),)
    actions = {step: tuple(sorted(rows, key=lambda a: (
        a.offset_ms, a.kind, -1 if a.synapse_id is None else a.synapse_id)))
        for step, rows in actions.items()}
    signals = {label: int(representatives[label]) for label in
               ("soma", "ais", "trunk", "hot_zone", "nexus", "tuft")}
    definitions = default_event_definitions(representatives)
    if candidate and candidate.event_probe_mode == "cluster_center":
        signals["event_probe"] = int(selection.center_segment_id)
        definitions = [replace(d, signal="event_probe", segment_id=signals["event_probe"],
                               region=f"{candidate.target}_stimulus_cluster")
                       if d.kind in candidate.event_probe_kinds else d for d in definitions]
    contract = {"spec": spec, "signals": signals,
                "synapse_ids": list(map(int, selection.synapse_ids)) if selection else [],
                "definitions": [asdict(d) for d in definitions],
                "actions": {str(k): [asdict(a) for a in rows] for k, rows in actions.items()}}
    return candidate, actions, signals, definitions, contract


def _record(session, calibrator, spec, seed, multiplier, arm, config, gbar):
    protocol, actions, signals, definitions, contract = resolve_spec(
        spec, calibrator, session.audit.representatives, config)
    samples = []

    def observer(active, suffix):
        samples.append({label: float(active.audit.live_segments[site].v)
                        for label, site in signals.items()})
        return _extended_sample(active, suffix)

    trial = _trial(session, calibrator, protocol, seed, multiplier, arm, config,
                   gbar, ALL_SUFFIXES, sample_fn=observer, capture_release=True,
                   actions_by_step=actions)
    steps = int(round(1 / config.sample_interval_ms))
    if len(samples) != 1 + config.duration_ms * (steps + 1):
        raise RuntimeError("Task17f sample boundary contract changed")
    indices = [0] + [1 + step * (steps + 1) + offset
                     for step in range(config.duration_ms) for offset in range(1, steps + 1)]
    traces = {label: [samples[i][label] for i in indices] for label in signals}
    times = np.arange(len(indices)) * config.sample_interval_ms
    trial.update(events=extract_events(times, traces, definitions), event_traces=traces,
                 event_contract=contract, schedule_sha256=_hash(contract))
    # Same trajectory, old fixed probe: attribution only, never a selectable label.
    trial["legacy_probe_events"] = extract_events(
        times, traces, default_event_definitions(session.audit.representatives))
    trial["release_audit"] = _release_audit(trial)
    trial.pop("release_rows")
    return trial


def _save_trial(root, key, trial):
    root.mkdir(exist_ok=True)
    arrays = {f"site_{site}_{name}": np.asarray(values, dtype=np.float64)
              for site, fields in trial["traces"].items() for name, values in fields.items()}
    arrays.update({f"event_{name}": np.asarray(values, dtype=np.float64)
                   for name, values in trial["event_traces"].items()})
    np.savez_compressed(root / f"{key}.npz", **arrays)
    _write(root / f"{key}.json", {k: v for k, v in trial.items()
                                 if k not in ("traces", "event_traces")})


def support_report(rows, seeds):
    """Require completed (not censored) positives for each class on every seed."""
    coverage = {kind: {str(seed): 0 for seed in seeds} for kind in REQUIRED_EVENT_KINDS}
    for row in rows:
        if row["gbar_multiplier"] != 1.0:
            continue
        target = row["event_contract"]["spec"]["target_kind"]
        for event in row["events"]:
            if event["kind"] == target and target in coverage and not event["right_censored"]:
                coverage[event["kind"]][str(row["seed"])] += 1
    return {"valid": all(all(counts.values()) for counts in coverage.values()),
            "uncensored_events_at_gbar1_by_seed": coverage}


def compare_trials(reference, candidate, config, *, control=False):
    if reference["schedule_sha256"] != candidate["schedule_sha256"]:
        raise RuntimeError("paired input/probe contract differs")
    metrics = _compare(reference, candidate)
    events = _event_comparison(reference, candidate, config)
    releases = _release_pair(reference["release_audit"], candidate["release_audit"])
    vlimit = config.formula_voltage_rmse_limit_mv if control else config.candidate_voltage_rmse_limit_mv
    glimit = config.formula_gate_max_error_limit if control else config.candidate_gate_max_error_limit
    # Include dynamic event probe and AIS/trunk/nexus in voltage fidelity checks.
    event_voltage = {label: float(np.sqrt(np.mean((np.asarray(values) -
                      np.asarray(candidate["event_traces"][label])) ** 2)))
                     for label, values in reference["event_traces"].items()}
    gates = {"voltage": max(_max_metric(metrics, "voltage_rmse_mv"), max(event_voltage.values())) <= vlimit,
             "gates": max(max(v["m_max_error"], v["h_max_error"]) for v in metrics.values()) <= glimit,
             "current": _max_metric(metrics, "current_rmse_ma_cm2") <= config.candidate_current_rmse_limit_ma_cm2,
             "events": all(d["passed"] for d in events.values()),
             "release": releases["discrete_match"] and
                        releases["max_released_quantity_difference"] <= config.release_quantity_atol}
    return {"passed": all(gates.values()), "gates": gates, "metrics": metrics,
            "event_probe_voltage_rmse_mv": event_voltage, "events": events, "release": releases}


def run_task17f(elm_repo, teacher_repo, native_mod, output_dir,
                config=Task17fConfig(), *, code_revision="unknown", pilot_only=False,
                development_smoke=False):
    config.validate()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "giada-task17f-event-supported-confirmation-v1",
              "valid": True, "code_revision": code_revision, "config": asdict(config),
              "gate_c_authorized": False, "task18_authorized": False,
              "candidate": "m_pair_2049_f64", "pilot_only": pilot_only,
              "development_smoke": development_smoke,
              "confirmation_accessed": False}

    def finish(decision, **details):
        report.update(decision=decision, **details)
        _write(root / "final_report.json", report)
        return report

    # The preregistration is preserved in the result even after native failure.
    _write(root / "run_contract.json", report)
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, calibration_source=Path(native_mod),
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        output_dir=root / "teacher_workspace")
    teacher = session.prepare_teacher()
    if teacher["segment_count"] != 642:
        raise RuntimeError("canonical 642-segment teacher required")
    representatives = dict(session.audit.representatives)
    expected_sites = {"soma": 0, "ais": 640, "trunk": 361, "hot_zone": 387, "nexus": 437, "tuft": 460}
    if any(representatives.get(k) != v for k, v in expected_sites.items()):
        raise RuntimeError(f"canonical event site contract changed: {representatives}")
    report["representatives"] = representatives
    session.run_burn_in()
    session.task17_snapshot_suffix = "Ca_HVA"
    calibrator = DendriticProtocolCalibrator(session, output_dir=root / "protocol_workspace")
    gbar = _canonical_gbar(session)
    calcium_state = _select_calcium_state(_state_names(session))
    base_config = Task17dConfig()
    _set_policy(session, "ultra", calcium_state, base_config)
    selected = [make_spec("quiet")]
    pilot_log = []
    somatic_current = 0.0
    for kind in REQUIRED_EVENT_KINDS:
        eligible = None
        for spec in pilot_specs(config, kind, somatic_current):
            hits = []
            for seed in config.pilot_seeds:
                trial = _record(session, calibrator, spec, seed, 1.0, "native", config, gbar)
                _save_trial(root / "pilot_traces", f"{spec['name']}_{seed}", trial)
                hits.append(any(e["kind"] == kind and not e["right_censored"] for e in trial["events"]))
                pilot_log.append({"spec": spec["name"], "kind": kind, "seed": seed,
                                  "hit": hits[-1], "events": trial["events"],
                                  "legacy_probe_events": trial["legacy_probe_events"],
                                  "peak_voltage_mv": {k: max(v) for k, v in trial["event_traces"].items()}})
                _write(root / "pilot_progress.json", pilot_log)
                print(f"[GIADA Task 17f pilot] {spec['name']} seed={seed} {kind}={hits[-1]}", flush=True)
            if sum(hits) >= config.minimum_positive_seeds:
                eligible = spec
                break
        if eligible is None:
            return finish("PILOT_SUPPORT_INSUFFICIENT", missing_event_class=kind,
                          pilot_trial_count=len(pilot_log), selected_protocols=selected)
        selected.append(eligible)
        if kind == "somatic_spike":
            somatic_current = eligible["somatic_current_na"]
    contracts = [resolve_spec(s, calibrator, representatives, config)[-1] for s in selected]
    freeze = {"protocols": selected, "resolved_contracts": contracts,
              "pilot_seeds": config.pilot_seeds, "confirmation_seeds": config.seeds,
              "candidate_table_sha256": FROZEN_2049_TABLE_SHA256,
              "selection": "first native protocol with uncensored positive on all three pilot seeds",
              "confirmation_accessed": False, "candidate_outcomes_accessed": False}
    freeze["sha256"] = _hash(freeze)
    _write(root / "protocol_freeze.json", freeze)
    report.update(protocol_freeze_sha256=freeze["sha256"], selected_protocols=selected,
                  pilot_trial_count=len(pilot_log))
    if pilot_only:
        return finish("PILOT_COMPLETE_CONFIRMATION_NOT_RUN")
    # Verify frozen contracts before opening confirmation or compiling candidates.
    if _hash({k: v for k, v in freeze.items() if k != "sha256"}) != freeze["sha256"]:
        raise RuntimeError("protocol freeze hash changed")
    for spec, contract in zip(selected, contracts):
        if resolve_spec(spec, calibrator, representatives, config)[-1] != contract:
            raise RuntimeError("resolved pilot schedule changed before confirmation")
    cache, comparisons, runtime = {}, [], {}
    # Integration check may use only the already observed first pilot seed.
    # It is explicitly incapable of opening/authorizing the independent test.
    evaluation_seeds = config.pilot_seeds[:1] if development_smoke else config.seeds
    evaluation_gbars = (1.0,) if development_smoke else config.gbar_multipliers
    report["confirmation_accessed"] = not development_smoke

    def run_matrix(label, arm, policy):
        for index, spec in enumerate(selected):
            for seed in evaluation_seeds:
                for multiplier in evaluation_gbars:
                    if policy == "super_ultra":
                        _set_super_ultra(session, calcium_state, Task17dbConfig())
                    else:
                        _set_policy(session, "ultra", calcium_state, base_config)
                    key = (index, seed, multiplier)
                    started = time.perf_counter()
                    trial = _record(session, calibrator, spec, seed, multiplier, arm, config, gbar)
                    runtime[f"{label}/{index}/{seed}/{multiplier}"] = time.perf_counter() - started
                    cache[(label, *key)] = trial
                    _save_trial(root / "confirmation_traces", f"{label}_{index}_{seed}_{multiplier}", trial)
                    if label != "native":
                        contrast = compare_trials(cache[("native", *key)], trial, config,
                                                  control=label in ("native_super", "formula"))
                        comparisons.append({"arm": label, "protocol_index": index, "seed": seed,
                                            "gbar_multiplier": multiplier, **contrast})
                        _write(root / "paired_metrics.json", comparisons)
                    print(f"[GIADA Task 17f] {label} {index + 1}/{len(selected)} seed={seed} gbar={multiplier}", flush=True)
        _write(root / "runtime_seconds.json", runtime)

    run_matrix("native", "native", "ultra")
    native_rows = list(cache.values())
    support = support_report(native_rows, evaluation_seeds)
    support["quiet_control_silent"] = not any(
        row["events"] for row in native_rows if row["event_contract"]["spec"]["name"] == "quiet")
    support["valid"] &= support["quiet_control_silent"]
    _write(root / "confirmation_support.json", support)
    if not support["valid"]:
        return finish("CONFIRMATION_SUPPORT_INSUFFICIENT", support=support,
                      completed_trial_count=len(cache))
    # Repeat an ACTIVE native case; repeat and convergence are different tests.
    _set_policy(session, "ultra", calcium_state, base_config)
    anchor = cache[("native", 1, evaluation_seeds[0], 1.0)]
    repeat = _record(session, calibrator, selected[1], evaluation_seeds[0], 1.0, "native", config, gbar)
    repeat_check = compare_trials(anchor, repeat, config, control=True)
    repeat_check["passed"] &= _max_metric(repeat_check["metrics"], "voltage_max_error_mv") <= config.native_repeat_voltage_max_atol_mv
    _write(root / "native_repeat.json", repeat_check)
    if not repeat_check["passed"]:
        return finish("NATIVE_REPEAT_FAILED", support=support)
    run_matrix("native_super", "native", "super_ultra")
    if not all(r["passed"] for r in comparisons):
        return finish("NATIVE_SOLVER_FLOOR_FAILED", support=support,
                      completed_trial_count=len(cache))
    mods, tables = generate_precision_mods(native_mod, root / "candidate_mods")
    table_hash = hashlib.sha256(tables["m_pair_2049_f64"].tobytes()).hexdigest()
    if table_hash != FROZEN_2049_TABLE_SHA256:
        raise RuntimeError("frozen candidate table changed")
    compiler = shutil.which("nrnivmodl")
    if not compiler:
        raise RuntimeError("nrnivmodl unavailable")
    native_phase("task17f_compile_candidates")
    compilation = compile_candidate_mods(root / "candidate_mods", compiler)
    from neuron import h, load_mechanisms
    if not load_mechanisms(str(root / "candidate_mods")):
        raise RuntimeError("candidate mechanisms failed to load")
    # Probe calls finitialize: restore equilibrium before any subsequent trial.
    probe = probe_compiled_m_tables(h, tables)
    report.update(frozen_table_sha256=table_hash, compilation=compilation, grid_probe=probe)
    for arm in ("formula", "m_pair_2049_f64"):
        _activate_arm(session, gbar, arm, ALL_SUFFIXES)
        run_matrix(arm, arm, "ultra")
        if arm == "formula" and not all(r["passed"] for r in comparisons if r["arm"] == arm):
            return finish("FORMULA_CONTROL_FAILED", support=support,
                          completed_trial_count=len(cache))
    if development_smoke:
        return finish("DEVELOPMENT_SMOKE_COMPLETE", support=support,
                      completed_trial_count=len(cache),
                      candidate_smoke_passed=all(r["passed"] for r in comparisons))
    effects = []
    lo, hi = min(config.gbar_multipliers), max(config.gbar_multipliers)
    for index in range(len(selected)):
        for seed in config.seeds:
            effects.append({"protocol_index": index, "seed": seed,
                            "candidate": _paired_effect(cache[("native", index, seed, lo)],
                                cache[("native", index, seed, hi)],
                                cache[("m_pair_2049_f64", index, seed, lo)],
                                cache[("m_pair_2049_f64", index, seed, hi)], config.paired_effect_floor_mv)})
    _write(root / "paired_effects.json", effects)
    identifiable = [v for row in effects for v in row["candidate"].values() if v["effect_identifiable"]]
    candidate_rows = [r for r in comparisons if r["arm"] == "m_pair_2049_f64"]
    gates = {name: all(r["gates"][name] for r in candidate_rows)
             for name in ("voltage", "gates", "current", "events", "release")}
    gates.update(event_support=support["valid"], formula_control=True, native_solver_floor=True,
                 native_repeat=True, paired_effect=bool(identifiable) and all(
                     v["relative_error"] <= config.paired_effect_relative_error_limit for v in identifiable))
    passed = all(gates.values())
    return finish("GATE_C_PASS_BOUNDED_CAHVA" if passed else "CANDIDATE_FIDELITY_FAILED",
                  support=support, gates=gates, completed_trial_count=len(cache),
                  comparison_count=len(comparisons), identifiable_effect_count=len(identifiable),
                  gate_c_authorized=passed, task18_authorized=passed,
                  scope="Ca_HVA canonical 642 segments; qualified protocols and registered seeds only")
