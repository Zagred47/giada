"""Task 17: preregistered, paired causal Ca_HVA replacement in the full teacher.

The cloned NMODL mechanisms retain USEION ca WRITE ica.  No post-step
voltage, teacher gate, or teacher current is supplied to the replacement.
This module deliberately fails closed if the formula clone or replay control
does not reproduce the native teacher before a LUT result is interpreted.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.hayflow_teacher.dendritic_calibration import (
    DendriticCandidate, DendriticProtocolCalibrator, build_candidate_actions,
)
from src.hayflow_teacher.diagnostic_dataset_v1_1 import TargetedDiagnosticDatasetSession

from .roadmap_embedded_frozen_confirmation import _lut_table, verified_15c
from .native_process import native_phase


SUFFIXES = {"formula": "GIADA_CaHVA_formula", "lut": "GIADA_CaHVA_lut"}
SITES = (0, 387, 460, 469)


@dataclass(frozen=True)
class CausalReplacementConfig:
    seeds: tuple[int, ...] = (170017, 170029, 170043)
    duration_ms: int = 40
    gbar_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)
    sample_interval_ms: float = 0.025
    source_grid_min_mv: float = -135.0
    source_grid_max_mv: float = 75.0
    source_grid_intervals: int = 512
    lut_rate_atol: float = 2e-5
    repeated_native_voltage_atol_mv: float = 1e-5
    formula_voltage_rmse_limit_mv: float = 0.05
    formula_gate_max_error: float = 0.002
    lut_voltage_rmse_limit_mv: float = 2.0
    lut_gate_max_error: float = 0.01
    paired_effect_relative_error_limit: float = 0.2
    paired_effect_floor_mv: float = 0.05

    def validate(self):
        if asdict(self) != asdict(CausalReplacementConfig()):
            raise ValueError("Task 17 config differs from preregistration")


def candidate_protocols():
    """Fixed schedules; no event or voltage outcomes are used for selection."""
    return (
        None,
        DendriticCandidate(
            family="task17_nmda", target="tuft", required_event_kinds=("nmda_spike",),
            synapse_count=10, burst_count=2, burst_start_ms=3,
            selection_mode="branch_cluster", event_probe_mode="cluster_center",
            event_probe_kinds=("nmda_spike",), event_window_ms=0.4,
        ),
        DendriticCandidate(
            family="task17_calcium", target="hot_zone",
            required_event_kinds=("calcium_spike",), synapse_count=12,
            burst_count=3, burst_start_ms=3, event_window_ms=0.8,
        ),
    )


def _explicit_lut_rates(table):
    """Render balanced, O(log 512) NMODL interpolation without TABLE translator.

    Values are the exact float32 knots selected in Task 15c/used in Task 16.
    The generated arithmetic is checked against that frozen tensor before
    any full-teacher trial.  This is a compiler-workaround, not a new model.
    """
    values = np.asarray(table, dtype=np.float32)
    if values.shape != (513, 4) or not np.isfinite(values).all():
        raise ValueError("frozen LUT must contain 513 finite four-rate rows")
    grid = np.linspace(-135.0, 75.0, 513)
    names = ("mInf", "hInf", "mTau", "hTau")
    lines = ["PROCEDURE rates() {", " LOCAL frac", " UNITSOFF"]

    def literal(number):
        return format(float(number), ".17g")

    def emit_assignments(index, indentation):
        for column, name in enumerate(names):
            lines.append(f"{indentation}{name} = {literal(values[index, column])}")

    def emit_interval(index, indentation):
        left = grid[index]
        lines.append(f"{indentation}frac = (v - {literal(left)}) * {literal(512 / 210)}")
        for column, name in enumerate(names):
            lower = literal(values[index, column])
            upper = literal(values[index + 1, column])
            lines.append(f"{indentation}{name} = {lower} + frac * ({upper} - {lower})")

    def emit_tree(low, high, indentation):
        if high - low == 1:
            emit_interval(low, indentation)
            return
        middle = (low + high) // 2
        lines.append(f"{indentation}if (v < {literal(grid[middle])}) {{")
        emit_tree(low, middle, indentation + " ")
        lines.append(f"{indentation}}} else {{")
        emit_tree(middle, high, indentation + " ")
        lines.append(f"{indentation}}}")

    lines.append(f" if (v <= {literal(grid[0])}) {{")
    emit_assignments(0, "  ")
    lines.append(f" }} else if (v >= {literal(grid[-1])}) {{")
    emit_assignments(512, "  ")
    lines.append(" } else {")
    emit_tree(0, 512, "  ")
    lines.extend([" }", " UNITSON", "}"])
    return "\n".join(lines) + "\n"


def generate_candidate_mods(native_mod, destination):
    """Make separately compiled NMODL clones without touching the teacher repo."""
    native_mod, destination = Path(native_mod), Path(destination)
    source = native_mod.read_text(encoding="utf-8")
    if len(re.findall(r"\bSUFFIX\s+Ca_HVA\b", source)) != 1:
        raise ValueError("canonical Ca_HVA SUFFIX contract changed")
    if "USEION ca READ eca WRITE ica" not in source or "SOLVE states METHOD cnexp" not in source:
        raise ValueError("canonical ion/solver contract changed")
    if len(re.findall(r"PROCEDURE\s+rates\s*\(\s*\)\s*\{", source)) != 1:
        raise ValueError("canonical rates procedure changed")
    from .double_oracle import ExtractedGateFormula

    frozen_table = _lut_table(ExtractedGateFormula.from_mod(native_mod))
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for arm, suffix in SUFFIXES.items():
        modified = re.sub(r"\bSUFFIX\s+Ca_HVA\b", f"SUFFIX {suffix}", source)
        if arm == "lut":
            # NEURON 8.2.7 NMODL translator segfaults on TABLE here.  Emit a
            # balanced decision tree over the *same* frozen knots instead.
            match = re.search(r"PROCEDURE\s+rates\s*\(\s*\)\s*\{", modified)
            if match is None or not modified.rstrip().endswith("}"):
                raise ValueError("canonical rates tail changed")
            modified = modified[:match.start()] + _explicit_lut_rates(frozen_table)
            modified = modified.replace(
                "RANGE gCa_HVAbar, gCa_HVA, ica",
                "RANGE gCa_HVAbar, gCa_HVA, ica, mInf, hInf, mTau, hTau",
                1,
            )
        path = destination / f"{suffix}.mod"
        path.write_text(modified, encoding="utf-8")
        records[arm] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return records


def compile_candidate_mods(mod_directory, nrnivmodl):
    mod_directory = Path(mod_directory)
    command = [str(nrnivmodl)]
    completed = subprocess.run(command, cwd=mod_directory, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Task 17 NMODL compilation failed:\n{completed.stdout[-6000:]}")
    if not list(mod_directory.rglob("libnrnmech.so")):
        raise RuntimeError("Task 17 compiled library missing")
    return {"command": command, "library": str(next(mod_directory.rglob("libnrnmech.so")))}


def _present_sections(h, suffix):
    return [sec for sec in h.allsec() if bool(h.ismembrane(suffix, sec=sec))]


def _canonical_gbar(session):
    return {
        (sec.name(), float(seg.x)): float(seg.gCa_HVAbar_Ca_HVA)
        for sec in _present_sections(session.h, "Ca_HVA")
        for seg in sec
    }


def _restore_canonical_gbar(session, canonical_gbar, suffix="Ca_HVA"):
    observed = set()
    for sec in _present_sections(session.h, suffix):
        for seg in sec:
            key = (sec.name(), float(seg.x))
            if key not in canonical_gbar:
                raise RuntimeError(f"Task 17 unknown Ca_HVA compartment: {key}")
            setattr(seg, _segment_field(suffix, "gCa_HVAbar"), canonical_gbar[key])
            observed.add(key)
    if observed != set(canonical_gbar):
        raise RuntimeError("Task 17 Ca_HVA-bearing morphology changed")
    session.cvode.re_init()
    session.h.fcurrent()


def _canonical_synapse_weights(records):
    # TeacherAuditSession exposes a positional list whose index is synapse_id.
    return tuple(float(record["binding"].base_weight) for record in records)


def _changed_synapse_weights(records, canonical_weights):
    if len(records) != len(canonical_weights):
        raise RuntimeError("Task 17 synapse count changed")
    return [index for index, record in enumerate(records)
            if float(record["netcon"].weight[0]) != canonical_weights[index]]


def verify_frozen_rate_table(h, frozen_table, config):
    """Check compiled NMODL interpolation against Task 16's frozen tensor."""
    section = h.Section(name="giada_task17_lut_table_probe")
    try:
        section.insert(SUFFIXES["lut"])
        segment = section(0.5)
        checks = []
        for index in (0, 1, 0.5, 64, 128, 127.5, 255, 256, 263,
                      263.5, 384, 511, 511.5, 512):
            voltage = config.source_grid_min_mv + index * (
                config.source_grid_max_mv - config.source_grid_min_mv
            ) / config.source_grid_intervals
            h.finitialize(voltage)
            observed = np.asarray([
                float(getattr(segment, _segment_field(SUFFIXES["lut"], name)))
                for name in ("mInf", "hInf", "mTau", "hTau")
            ])
            low = min(int(np.floor(index)), 511)
            fraction = index - low
            expected = ((1.0 - fraction) * frozen_table[low].astype(float)
                        + fraction * frozen_table[low + 1].astype(float))
            error = float(np.max(np.abs(observed - expected)))
            checks.append({"index": index, "voltage_mv": voltage,
                           "maximum_absolute_rate_error": error})
        maximum = max(row["maximum_absolute_rate_error"] for row in checks)
        if maximum > config.lut_rate_atol:
            raise RuntimeError(f"Task 17 NMODL interpolation differs from frozen Task 16 LUT: {maximum}")
        return {"valid": True, "maximum_absolute_error": maximum,
                "atol": config.lut_rate_atol, "checks": checks}
    finally:
        h.delete_section(sec=section)


def _segment_field(suffix, name):
    return f"{name}_{suffix}"


def swap_cahva(session, source_suffix, target_suffix):
    """Replace density mechanism everywhere, preserving local gbar and m/h."""
    h = session.h
    sections = _present_sections(h, source_suffix)
    if not sections:
        raise RuntimeError(f"no sections have {source_suffix}")
    before = len(sections)
    transferred = 0
    for sec in sections:
        segments = list(sec)
        state = [tuple(float(getattr(seg, _segment_field(source_suffix, key)))
                       for key in ("gCa_HVAbar", "m", "h")) for seg in segments]
        sec.uninsert(source_suffix)
        sec.insert(target_suffix)
        if len(list(sec)) != len(state):
            raise RuntimeError("segment topology changed during Ca_HVA swap")
        for seg, (gbar, m, gate_h) in zip(sec, state):
            for key, value in (("gCa_HVAbar", gbar), ("m", m), ("h", gate_h)):
                setattr(seg, _segment_field(target_suffix, key), value)
            transferred += 1
    if len(_present_sections(h, target_suffix)) != before or _present_sections(h, source_suffix):
        raise RuntimeError("Ca_HVA swap did not cover precisely the native sections")
    session.cvode.re_init()
    session.h.fcurrent()
    return {"section_count": before, "segment_count": transferred,
            "from": source_suffix, "to": target_suffix}


def _sample(session, suffix):
    rows = {}
    for site in SITES:
        seg = session.audit.live_segments[site]
        if not hasattr(seg, _segment_field(suffix, "m")):
            raise RuntimeError(f"Ca_HVA mechanism absent at registered site {site}")
        m = float(getattr(seg, _segment_field(suffix, "m")))
        gate_h = float(getattr(seg, _segment_field(suffix, "h")))
        gbar = float(getattr(seg, _segment_field(suffix, "gCa_HVAbar")))
        v, eca = float(seg.v), float(seg.eca)
        rows[str(site)] = {"v": v, "m": m, "h": gate_h, "eca": eca,
                           "gbar": gbar, "ica_hva": gbar * m * m * gate_h * (v - eca)}
    return rows


def _trial(session, calibrator, protocol, seed, multiplier, arm, config,
           canonical_gbar, suffixes=None, *, sample_fn=None,
           capture_release=False, actions_by_step=None):
    suffix = "Ca_HVA" if arm == "native" else (suffixes or SUFFIXES)[arm]
    if session.task17_snapshot_suffix != suffix:
        raise RuntimeError("Refusing SaveState restore across a mechanism change")
    native_phase("trial_restore", arm=arm, seed=seed, multiplier=multiplier,
                 protocol=("explicit_schedule" if actions_by_step is not None else
                           "quiescent" if protocol is None else protocol.candidate_id))
    rng = json.loads(session.equilibrium_rng_path.read_text(encoding="utf-8"))
    session._restore_native_snapshot(session.equilibrium_snapshot_path,
                                     rng["sequences"], rng.get("random123_seed", session.seed))
    # NEURON SaveState does not guarantee restoration of density parameters.
    # Reset them explicitly before every paired arm to prevent multiplication
    # from compounding across trials.
    _restore_canonical_gbar(session, canonical_gbar, suffix)
    session._rekey_rngs(seed)
    session.active_random123_seed = int(seed)
    if actions_by_step is not None:
        # New protocols may supply a frozen, explicitly audited action schedule.
        # Existing experiments keep their original path when this is omitted.
        actions = actions_by_step
        label = "explicit_schedule" if protocol is None else protocol.candidate_id
    elif protocol is None:
        actions = {}
        label = "quiescent"
    else:
        selection = calibrator.select_synapse_cluster(
            protocol.target, protocol.synapse_count,
            protocol.maximum_tree_distance_um, protocol.selection_mode)
        actions = build_candidate_actions(protocol, selection.synapse_ids,
                                          duration_ms=config.duration_ms)
        label = protocol.candidate_id
    canonical_weights = _canonical_synapse_weights(session.audit.synapse_records)
    transfer = getattr(session, "task17_transfer", None)
    # Perturbation is applied equally to every Ca_HVA-bearing compartment.
    for sec in _present_sections(session.h, suffix):
        for seg in sec:
            name = _segment_field(suffix, "gCa_HVAbar")
            setattr(seg, name, float(getattr(seg, name)) * multiplier)
    session.cvode.re_init()
    session.h.fcurrent()
    sample = sample_fn or _sample
    initial = sample(session, suffix)
    traces = {str(site): {key: [value] for key, value in initial[str(site)].items()}
              for site in SITES}
    release_rows = []
    for step in range(config.duration_ms):
        start = float(session.h.t)
        session._active_transition_id = int(seed * 1000 + step)
        _, _, observations = session._drive_one_ms(
            start, tuple(actions.get(step, ())), lambda: sample(session, suffix),
            sample_interval_ms=config.sample_interval_ms)
        if capture_release:
            release_rows.append({"step": step,
                                 "outcomes": [item.to_dict() for item in session._last_release_outcomes],
                                 "verification": session._last_release_verification})
        for observed in observations[1:]:
            for site in SITES:
                for key, value in observed[str(site)].items():
                    traces[str(site)][key].append(value)
    expected = config.duration_ms * int(round(1 / config.sample_interval_ms)) + 1
    if any(len(traces[str(site)]["v"]) != expected for site in SITES):
        raise RuntimeError("Task 17 trace length mismatch")
    if any(not np.isfinite(np.asarray(values, dtype=float)).all()
           for site in traces.values() for values in site.values()):
        raise RuntimeError("Task 17 NaN/Inf in causal rollout")
    if any(min(traces[str(site)][gate]) < -1e-7 or max(traces[str(site)][gate]) > 1 + 1e-7
           for site in SITES for gate in ("m", "h")):
        raise RuntimeError("Task 17 gate occupancy violation")
    changed_weights = _changed_synapse_weights(
        session.audit.synapse_records, canonical_weights)
    if changed_weights:
        raise RuntimeError(f"Task 17 changed canonical NetCon weights: {changed_weights[:5]}")
    return {"arm": arm, "protocol": label, "seed": seed,
            "gbar_multiplier": multiplier, "transfer": transfer,
            "samples": expected, "traces": traces,
            **({"release_rows": release_rows} if capture_release else {})}


def _activate_arm(session, canonical_gbar, arm, suffixes=None):
    """Transfer equilibrium once, then save a NEW snapshot for this structure.

    NEURON SaveState is tied to mechanism insertion order, not just suffixes.
    Never restore a snapshot made before uninsert/insert, even after switching
    back to the same named mechanism. All trials of one arm run contiguously.
    """
    source_suffix = session.task17_snapshot_suffix
    target_suffix = "Ca_HVA" if arm == "native" else (suffixes or SUFFIXES)[arm]
    if source_suffix == target_suffix:
        return
    native_phase("arm_equilibrium_restore", source=source_suffix, target=target_suffix)
    rng = json.loads(session.equilibrium_rng_path.read_text(encoding="utf-8"))
    session._restore_native_snapshot(session.equilibrium_snapshot_path, rng["sequences"],
                                     rng.get("random123_seed", session.seed))
    _restore_canonical_gbar(session, canonical_gbar, source_suffix)
    before = _sample(session, source_suffix)
    native_phase("arm_mechanism_transfer", source=source_suffix, target=target_suffix)
    session.task17_transfer = swap_cahva(session, source_suffix, target_suffix)
    after = _sample(session, target_suffix)
    if before != after:
        raise RuntimeError("Mechanism transfer changed the equilibrium boundary")
    # No path reuse: the former native snapshot is invalid after reinsertion.
    generation = getattr(session, "task17_snapshot_generation", 0) + 1
    path = session.snapshots_dir / f"task17_equilibrium_{generation}_{arm}.bin"
    native_phase("arm_snapshot_write", arm=arm, generation=generation)
    session._write_native_snapshot(path)
    session.equilibrium_snapshot_path = path
    session.task17_snapshot_suffix = target_suffix
    session.task17_snapshot_generation = generation


def _rmse(left, right):
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


def _compare(reference, candidate):
    return {str(site): {
        "voltage_rmse_mv": _rmse(reference["traces"][str(site)]["v"], candidate["traces"][str(site)]["v"]),
        "voltage_max_error_mv": float(np.max(np.abs(np.asarray(reference["traces"][str(site)]["v"]) - np.asarray(candidate["traces"][str(site)]["v"])))),
        "m_max_error": float(np.max(np.abs(np.asarray(reference["traces"][str(site)]["m"]) - np.asarray(candidate["traces"][str(site)]["m"])))),
        "h_max_error": float(np.max(np.abs(np.asarray(reference["traces"][str(site)]["h"]) - np.asarray(candidate["traces"][str(site)]["h"])))),
        "current_rmse_ma_cm2": _rmse(reference["traces"][str(site)]["ica_hva"], candidate["traces"][str(site)]["ica_hva"]),
    } for site in SITES}


def _paired_effect(native_low, native_high, candidate_low, candidate_high, floor_mv):
    """Contrast matched gbar interventions, not just absolute prediction error."""
    result = {}
    for site in SITES:
        key = str(site)
        native_delta = (np.asarray(native_high["traces"][key]["v"])
                        - np.asarray(native_low["traces"][key]["v"]))
        candidate_delta = (np.asarray(candidate_high["traces"][key]["v"])
                           - np.asarray(candidate_low["traces"][key]["v"]))
        native_magnitude = float(np.sqrt(np.mean(native_delta ** 2)))
        result[key] = {"native_effect_rmse_mv": native_magnitude,
                       "candidate_effect_error_rmse_mv": _rmse(native_delta, candidate_delta),
                       "relative_error": _rmse(native_delta, candidate_delta)
                       / max(native_magnitude, floor_mv),
                       "effect_identifiable": native_magnitude >= floor_mv}
    return result


def run_causal_replacement(elm_repo, teacher_repo, native_mod, task15c_source,
                           output_dir, config=CausalReplacementConfig(), *, code_revision="unknown"):
    config.validate()
    native_phase("verify_input")
    verified_15c(task15c_source)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    source_hash = hashlib.sha256(Path(native_mod).read_bytes()).hexdigest()
    from .double_oracle import ExtractedGateFormula
    formula = ExtractedGateFormula.from_mod(native_mod)
    frozen_table = _lut_table(formula)
    mods = generate_candidate_mods(native_mod, output_dir / "candidate_mods")
    from neuron import h, load_mechanisms
    import shutil as _shutil
    compiler = _shutil.which("nrnivmodl")
    if not compiler:
        raise RuntimeError("nrnivmodl unavailable")
    native_phase("compile_candidates")
    compilation = compile_candidate_mods(output_dir / "candidate_mods", compiler)
    if not load_mechanisms(str(output_dir / "candidate_mods")):
        raise RuntimeError("compiled Task 17 mechanisms did not load")
    # Probe the compiled interpolation before building the teacher or SaveState.
    native_phase("frozen_table_probe")
    table_contract = verify_frozen_rate_table(h, frozen_table, config)
    (output_dir / "frozen_table_probe.json").write_text(
        json.dumps(table_contract, indent=2), encoding="utf-8")
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, calibration_source=Path(native_mod),
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        output_dir=output_dir / "teacher_workspace")
    native_phase("prepare_teacher")
    teacher = session.prepare_teacher()
    if teacher["segment_count"] != 642:
        raise RuntimeError("Task 17 teacher morphology is not 642 segments")
    native_phase("burn_in")
    session.run_burn_in()
    session.task17_snapshot_suffix = "Ca_HVA"
    calibrator = DendriticProtocolCalibrator(session, output_dir=output_dir / "protocol_workspace")
    canonical_gbar = _canonical_gbar(session)
    if not canonical_gbar:
        raise RuntimeError("Task 17 no native Ca_HVA conductances found")
    rows = []
    comparisons = []
    effect_rows = []
    protocols = candidate_protocols()
    # Preflight first: same seed, quiescent protocol, native twice and formula.
    anchor = _trial(session, calibrator, None, config.seeds[0], 1.0, "native", config, canonical_gbar)
    repeat = _trial(session, calibrator, None, config.seeds[0], 1.0, "native", config, canonical_gbar)
    repetition = _compare(anchor, repeat)
    if max(x["voltage_max_error_mv"] for x in repetition.values()) > config.repeated_native_voltage_atol_mv:
        raise RuntimeError("Task 17 native replay preflight failed")
    # Complete one arm before changing mechanism structure. SaveState files
    # may only be reused within that generation of the mechanism topology.
    trial_cache = {}
    def run_arm(arm):
        _activate_arm(session, canonical_gbar, arm)
        for protocol_index, protocol in enumerate(protocols):
            for seed in config.seeds:
                for multiplier in config.gbar_multipliers:
                    row = _trial(session, calibrator, protocol, seed, multiplier,
                                 arm, config, canonical_gbar)
                    key = (protocol_index, seed, multiplier)
                    if arm == "formula":
                        metrics = _compare(trial_cache[("native", *key)], row)
                        if (max(x["voltage_rmse_mv"] for x in metrics.values()) > config.formula_voltage_rmse_limit_mv
                                or max(max(x["m_max_error"], x["h_max_error"]) for x in metrics.values()) > config.formula_gate_max_error):
                            raise RuntimeError(f"Task 17 formula control failed at {key}; LUT not evaluated")
                    trial_cache[(arm, *key)] = row
                    # Persist each completed episode even if a later C call aborts.
                    trial_dir = output_dir / "completed_trials"
                    trial_dir.mkdir(exist_ok=True)
                    (trial_dir / f"{arm}_{protocol_index}_{seed}_{multiplier}.json").write_text(
                        json.dumps(row), encoding="utf-8")
                    print(f"[GIADA Task 17] completed {arm} {len([k for k in trial_cache if k[0] == arm])}/27", flush=True)

    run_arm("native")
    _activate_arm(session, canonical_gbar, "formula")
    formula_control = _trial(session, calibrator, None, config.seeds[0], 1.0, "formula", config, canonical_gbar)
    control = _compare(anchor, formula_control)
    if (max(x["voltage_rmse_mv"] for x in control.values()) > config.formula_voltage_rmse_limit_mv
            or max(max(x["m_max_error"], x["h_max_error"]) for x in control.values()) > config.formula_gate_max_error):
        raise RuntimeError("Task 17 formula clone preflight failed; LUT not evaluated")
    run_arm("formula")
    _activate_arm(session, canonical_gbar, "native")
    restored_native = _trial(session, calibrator, None, config.seeds[0], 1.0, "native", config, canonical_gbar)
    restoration = _compare(anchor, restored_native)
    if max(x["voltage_max_error_mv"] for x in restoration.values()) > config.repeated_native_voltage_atol_mv:
        raise RuntimeError("Task 17 native SaveState restoration after formula swap failed")
    run_arm("lut")
    native_phase("aggregate_reports")
    for protocol_index, protocol in enumerate(protocols):
        for seed in config.seeds:
            by_multiplier = {}
            for multiplier in config.gbar_multipliers:
                arm_rows = {arm: trial_cache[(arm, protocol_index, seed, multiplier)]
                            for arm in ("native", "formula", "lut")}
                by_multiplier[multiplier] = arm_rows
                key = {"protocol": "quiescent" if protocol is None else protocol.candidate_id,
                       "seed": seed, "gbar_multiplier": multiplier}
                pair = {arm: _compare(arm_rows["native"], arm_rows[arm]) for arm in ("formula", "lut")}
                comparisons.append({**key, "metrics": pair})
                rows.append({**key, "arms": arm_rows})
                with (output_dir / "paired_metrics.partial.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(comparisons[-1]) + "\n")
                print(f"[GIADA Task 17] {len(comparisons)}/{len(protocols)*len(config.seeds)*len(config.gbar_multipliers)} paired trials", flush=True)
            low, high = min(config.gbar_multipliers), max(config.gbar_multipliers)
            effect_rows.append({"protocol": "quiescent" if protocol is None else protocol.candidate_id,
                                "seed": seed,
                                "formula": _paired_effect(by_multiplier[low]["native"],
                                                          by_multiplier[high]["native"],
                                                          by_multiplier[low]["formula"],
                                                          by_multiplier[high]["formula"],
                                                          config.paired_effect_floor_mv),
                                "lut": _paired_effect(by_multiplier[low]["native"],
                                                      by_multiplier[high]["native"],
                                                      by_multiplier[low]["lut"],
                                                      by_multiplier[high]["lut"],
                                                      config.paired_effect_floor_mv)})
    # Persist full traces on Kaggle, but download only a 1-ms diagnostic view.
    (output_dir / "paired_traces.json").write_text(json.dumps(rows), encoding="utf-8")
    stride = int(round(1 / config.sample_interval_ms))
    reduced = []
    for row in rows:
        reduced.append({"protocol": row["protocol"], "seed": row["seed"],
                        "gbar_multiplier": row["gbar_multiplier"],
                        "arms": {arm: {"traces": {
                            site: {name: values[::stride] for name, values in channels.items()}
                            for site, channels in trial["traces"].items()}}
                            for arm, trial in row["arms"].items()}})
    (output_dir / "paired_traces_1ms.json").write_text(
        json.dumps(reduced), encoding="utf-8")
    (output_dir / "paired_metrics.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (output_dir / "paired_effects.json").write_text(json.dumps(effect_rows, indent=2), encoding="utf-8")
    worst_formula_v = max(site["voltage_rmse_mv"] for row in comparisons for site in row["metrics"]["formula"].values())
    worst_formula_gate = max(max(site["m_max_error"], site["h_max_error"]) for row in comparisons for site in row["metrics"]["formula"].values())
    worst_lut_v = max(site["voltage_rmse_mv"] for row in comparisons for site in row["metrics"]["lut"].values())
    worst_lut_gate = max(max(site["m_max_error"], site["h_max_error"]) for row in comparisons for site in row["metrics"]["lut"].values())
    formula_valid = worst_formula_v <= config.formula_voltage_rmse_limit_mv and worst_formula_gate <= config.formula_gate_max_error
    lut_valid = worst_lut_v <= config.lut_voltage_rmse_limit_mv and worst_lut_gate <= config.lut_gate_max_error
    identifiable = [site for row in effect_rows for site in row["lut"].values()
                    if site["effect_identifiable"]]
    effect_valid = bool(identifiable) and max(
        site["relative_error"] for site in identifiable
    ) <= config.paired_effect_relative_error_limit
    report = {"schema_version": "giada-task17-causal-replacement-v1",
              "valid": True, "gate_c_authorized": False,
              "decision": "CAUSAL_MICROCANARY_PASS" if formula_valid and lut_valid and effect_valid
                          else "CAUSAL_MICROCANARY_NO_GO",
              "reason": "full independent morphology/regime coverage and sealed validation still required for Gate C",
              "code_revision": code_revision, "native_mod_sha256": source_hash,
              "selected_candidate": "lut_fine_path", "source_table_knots": len(frozen_table),
              "frozen_table_equivalence": table_contract,
              "new_independent_seeds": list(config.seeds), "teacher": teacher,
              "candidate_mods": mods, "compilation": compilation,
              "native_repeat_preflight": repetition, "formula_clone_preflight": control,
              "native_after_swap_preflight": restoration,
              "snapshot_policy": "fresh_equilibrium_snapshot_per_mechanism_generation",
              "execution_order": ["native_matrix", "formula_matrix", "native_roundtrip", "lut_matrix"],
              "formula_control_valid": formula_valid, "lut_absolute_valid": lut_valid,
              "paired_effect_valid": effect_valid,
              "identifiable_effect_count": len(identifiable),
              "worst_formula_voltage_rmse_mv": worst_formula_v,
              "worst_formula_gate_error": worst_formula_gate,
              "worst_lut_voltage_rmse_mv": worst_lut_v,
              "worst_lut_gate_error": worst_lut_gate,
              "episode_count": len(comparisons), "config": asdict(config)}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
