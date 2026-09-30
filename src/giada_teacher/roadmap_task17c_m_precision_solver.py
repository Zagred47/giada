"""Task 17c: m-rate grid precision crossed with CVode tolerance.

All arms are diagnostic interventions in the authentic 642-segment teacher.
The original Task 15c LUT and Task 17a gate criterion are never modified.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.hayflow_teacher.dendritic_calibration import DendriticProtocolCalibrator
from src.hayflow_teacher.diagnostic_dataset_v1_1 import TargetedDiagnosticDatasetSession

from .double_oracle import ExtractedGateFormula
from .native_process import native_phase
from .roadmap_embedded_frozen_confirmation import verified_15c
from .roadmap_causal_cahva_replacement import (
    CausalReplacementConfig, SUFFIXES, _activate_arm, _canonical_gbar,
    _compare, _trial, candidate_protocols, compile_candidate_mods,
    generate_candidate_mods, verify_frozen_rate_table,
)


GRID_ARMS = {
    "m_pair_513_f32": (513, "float32"),
    "m_pair_513_f64": (513, "float64"),
    "m_pair_1025_f64": (1025, "float64"),
    "m_pair_2049_f64": (2049, "float64"),
}
GRID_SUFFIXES = {arm: f"GIADA_CaHVA_17c_{arm}" for arm in GRID_ARMS}
ALL_SUFFIXES = {**SUFFIXES, **GRID_SUFFIXES}
ARMS = ("native", "formula", "lut", *GRID_ARMS)
RATE_NAMES = ("mInf", "mTau")
RATE_KEYS = ("m_inf", "m_tau_ms")


@dataclass(frozen=True)
class Task17cConfig:
    conditions: tuple[tuple[int, float], ...] = ((170029, 1.0), (170083, 1.5), (170029, 0.5))
    protocol_index: int = 1
    duration_ms: int = 40
    sample_interval_ms: float = 0.025
    grid_min_mv: float = -135.0
    grid_max_mv: float = 75.0
    tight_atol: float = 1e-5
    tight_rtol: float = 1e-6
    stricter_atol: float = 1e-6
    stricter_rtol: float = 1e-7
    ultra_atol: float = 1e-7
    ultra_rtol: float = 1e-8
    repeat_voltage_max_atol_mv: float = 1e-5
    formula_voltage_rmse_limit_mv: float = 0.05
    formula_gate_max_error: float = 0.002

    def validate(self):
        if asdict(self) != asdict(Task17cConfig()):
            raise ValueError("Task 17c config differs from preregistration")


def _grid_table(formula, knots: int, dtype, low=-135.0, high=75.0):
    values = np.asarray([[formula.rates(float(voltage))[key] for key in RATE_KEYS]
                         for voltage in np.linspace(low, high, knots)], dtype=dtype)
    if values.shape != (knots, 2) or not np.isfinite(values).all():
        raise RuntimeError("Task 17c nonfinite m-rate grid")
    return values


def _render_m_lut(table, *, low=-135.0, high=75.0):
    """Balanced NMODL tree, exactly linear between typed, frozen knots."""
    table = np.asarray(table)
    knots = len(table)
    if table.shape != (knots, 2) or knots < 2 or not np.isfinite(table).all():
        raise ValueError("invalid two-rate m table")
    grid = np.linspace(low, high, knots)
    scale = (knots - 1) / (high - low)
    lines = ["PROCEDURE m_lut_rates() {", " LOCAL frac", " UNITSOFF"]

    def lit(value):
        return format(float(value), ".17g")

    def assign(index, pad):
        for column, name in enumerate(RATE_NAMES):
            lines.append(f"{pad}{name} = {lit(table[index, column])}")

    def leaf(index, pad):
        lines.append(f"{pad}frac = (v - {lit(grid[index])}) * {lit(scale)}")
        for column, name in enumerate(RATE_NAMES):
            a, b = lit(table[index, column]), lit(table[index + 1, column])
            lines.append(f"{pad}{name} = {a} + frac * ({b} - {a})")

    def split(start, stop, pad):
        if stop - start == 1:
            leaf(start, pad)
            return
        mid = (start + stop) // 2
        lines.append(f"{pad}if (v < {lit(grid[mid])}) {{")
        split(start, mid, pad + " ")
        lines.append(f"{pad}}} else {{")
        split(mid, stop, pad + " ")
        lines.append(f"{pad}}}")

    lines.append(f" if (v <= {lit(low)}) {{")
    assign(0, "  ")
    lines.append(f" }} else if (v >= {lit(high)}) {{")
    assign(knots - 1, "  ")
    lines.append(" } else {")
    split(0, knots - 1, "  ")
    lines.extend([" }", " UNITSON", "}"])
    return "\n".join(lines) + "\n"


def generate_precision_mods(native_mod, destination):
    native_mod, destination = Path(native_mod), Path(destination)
    records = generate_candidate_mods(native_mod, destination)
    source = native_mod.read_text(encoding="utf-8")
    match = re.search(r"PROCEDURE\s+rates\s*\(\s*\)\s*\{", source)
    if match is None or not source.rstrip().endswith("}"):
        raise ValueError("canonical rates tail changed")
    formula = ExtractedGateFormula.from_mod(native_mod)
    tables = {}
    for arm, (knots, precision) in GRID_ARMS.items():
        suffix = GRID_SUFFIXES[arm]
        table = _grid_table(formula, knots, np.dtype(precision))
        header = re.sub(r"\bSUFFIX\s+Ca_HVA\b", f"SUFFIX {suffix}", source[:match.start()])
        body = source[match.start():].replace("PROCEDURE rates()", "PROCEDURE exact_formula_rates()", 1)
        wrapper = ("PROCEDURE rates() {\n LOCAL frozen_mInf, frozen_mTau\n"
                   " m_lut_rates()\n frozen_mInf = mInf\n frozen_mTau = mTau\n"
                   " exact_formula_rates()\n mInf = frozen_mInf\n mTau = frozen_mTau\n}\n")
        modified = header + body + "\n" + _render_m_lut(table) + "\n" + wrapper
        modified = modified.replace("RANGE gCa_HVAbar, gCa_HVA, ica",
                                    "RANGE gCa_HVAbar, gCa_HVA, ica, mInf, hInf, mTau, hTau", 1)
        path = destination / f"{suffix}.mod"
        path.write_text(modified, encoding="utf-8")
        records[arm] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "knots": knots, "table_precision": precision,
                        "table_sha256": hashlib.sha256(table.tobytes()).hexdigest()}
        tables[arm] = table
    return records, tables


def _expected_linear(table, voltage, *, low=-135.0, high=75.0):
    coordinate = np.clip((voltage - low) * (len(table) - 1) / (high - low), 0, len(table) - 1)
    i = min(int(np.floor(coordinate)), len(table) - 2)
    frac = coordinate - i
    return (1 - frac) * table[i].astype(float) + frac * table[i + 1].astype(float)


def probe_compiled_m_tables(h, tables):
    checks = {}
    for arm, table in tables.items():
        suffix = GRID_SUFFIXES[arm]
        section = h.Section(name=f"giada_task17c_probe_{arm}")
        try:
            section.insert(suffix)
            segment = section(0.5)
            errors = []
            for voltage in (-80.123, -55.25, -40.0, -27.125, -20.35,
                            -14.7, 0.0, 24.125, 54.2):
                h.finitialize(voltage)
                observed = np.asarray([float(getattr(segment, f"{name}_{suffix}"))
                                       for name in RATE_NAMES])
                errors.append(float(np.max(np.abs(observed - _expected_linear(table, voltage)))))
            maximum = max(errors)
            if maximum > 1e-8:
                raise RuntimeError(f"Task 17c compiled {arm} table mismatch: {maximum}")
            checks[arm] = maximum
        finally:
            h.delete_section(sec=section)
    return checks


def _static_rate_errors(formula, tables):
    voltages = np.linspace(-80.0, 55.0, 5401)
    exact = np.asarray([[formula.rates(float(v))[key] for key in RATE_KEYS]
                        for v in voltages])
    result = {}
    for arm, table in tables.items():
        approx = np.asarray([_expected_linear(table, float(v)) for v in voltages])
        result[arm] = {key: float(np.max(np.abs(approx[:, index] - exact[:, index])))
                       for index, key in enumerate(RATE_KEYS)}
    return result


def _set_solver_tolerance(session, atol, rtol):
    session.cvode.atol(float(atol))
    session.cvode.rtol(float(rtol))
    session.cvode.re_init()
    actual = (float(session.cvode.atol()), float(session.cvode.rtol()))
    if abs(actual[0] - atol) > 1e-15 or abs(actual[1] - rtol) > 1e-15:
        raise RuntimeError(f"Task 17c CVode tolerance not applied: {actual}")


def run_task17c(elm_repo, teacher_repo, native_mod, task15c_source, output_dir,
                config=Task17cConfig(), *, code_revision="unknown"):
    config.validate()
    verified_15c(task15c_source)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    mods, tables = generate_precision_mods(native_mod, output_dir / "candidate_mods")
    compiler = shutil.which("nrnivmodl")
    if not compiler:
        raise RuntimeError("nrnivmodl unavailable")
    native_phase("task17c_compile")
    compilation = compile_candidate_mods(output_dir / "candidate_mods", compiler)
    from neuron import h, load_mechanisms
    if not load_mechanisms(str(output_dir / "candidate_mods")):
        raise RuntimeError("Task 17c compiled mechanisms did not load")
    table_probe = probe_compiled_m_tables(h, tables)
    formula = ExtractedGateFormula.from_mod(native_mod)
    frozen_table = np.asarray([[formula.rates(float(v))[key]
                                for key in ("m_inf", "h_inf", "m_tau_ms", "h_tau_ms")]
                               for v in np.linspace(-135.0, 75.0, 513)], dtype=np.float32)
    frozen_probe = verify_frozen_rate_table(h, frozen_table, CausalReplacementConfig())
    static_errors = _static_rate_errors(formula, tables)
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, calibration_source=Path(native_mod),
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        output_dir=output_dir / "teacher_workspace")
    teacher = session.prepare_teacher()
    if teacher["segment_count"] != 642:
        raise RuntimeError("Task 17c teacher is not 642 segments")
    session.run_burn_in()
    session.task17_snapshot_suffix = "Ca_HVA"
    calibrator = DendriticProtocolCalibrator(session, output_dir=output_dir / "protocol_workspace")
    gbar = _canonical_gbar(session)
    default = (float(session.cvode.atol()), float(session.cvode.rtol()))
    tight = (config.tight_atol, config.tight_rtol)
    if default[0] <= tight[0] or default == tight:
        raise RuntimeError(f"Task 17c solver contrast missing: default={default}, tight={tight}")
    solver_levels = {"default": default, "tight": tight}
    protocol = candidate_protocols()[config.protocol_index]
    cache, repeat_controls = {}, {}
    for arm in ARMS:
        _activate_arm(session, gbar, arm, ALL_SUFFIXES)
        for solver, tolerance in solver_levels.items():
            _set_solver_tolerance(session, *tolerance)
            if arm == "native":
                anchor = _trial(session, calibrator, protocol, config.conditions[0][0],
                                config.conditions[0][1], arm, config, gbar, ALL_SUFFIXES)
                repeat = _trial(session, calibrator, protocol, config.conditions[0][0],
                                config.conditions[0][1], arm, config, gbar, ALL_SUFFIXES)
                repeat_metrics = _compare(anchor, repeat)
                if max(site["voltage_max_error_mv"] for site in repeat_metrics.values()) > config.repeat_voltage_max_atol_mv:
                    raise RuntimeError(f"Task 17c native replay failed for {solver}")
                repeat_controls[solver] = repeat_metrics
            for seed, multiplier in config.conditions:
                native_phase("task17c_trial", arm=arm, solver=solver,
                             seed=seed, multiplier=multiplier)
                row = _trial(session, calibrator, protocol, seed, multiplier,
                             arm, config, gbar, ALL_SUFFIXES)
                actual = (float(session.cvode.atol()), float(session.cvode.rtol()))
                if actual != tolerance:
                    raise RuntimeError(f"Task 17c solver settings changed during trial: {actual}")
                cache[(arm, solver, seed, multiplier)] = row
                if arm == "formula":
                    check = _compare(cache[("native", solver, seed, multiplier)], row)
                    if (max(site["voltage_rmse_mv"] for site in check.values()) > config.formula_voltage_rmse_limit_mv
                        or max(max(site["m_max_error"], site["h_max_error"]) for site in check.values()) > config.formula_gate_max_error):
                        raise RuntimeError(f"Task 17c formula control failed: {solver}, {seed}, {multiplier}")
                print(f"[GIADA Task 17c] {arm} {solver} seed={seed} gbar={multiplier}", flush=True)
    comparisons = []
    for solver in solver_levels:
        for seed, multiplier in config.conditions:
            native = cache[("native", solver, seed, multiplier)]
            comparisons.append({"solver": solver, "seed": seed, "gbar_multiplier": multiplier,
                                "metrics": {arm: _compare(native, cache[(arm, solver, seed, multiplier)])
                                            for arm in ARMS if arm != "native"}})
    cross_solver_native = [{"seed": seed, "gbar_multiplier": multiplier,
                            "metrics": _compare(cache[("native", "default", seed, multiplier)],
                                                cache[("native", "tight", seed, multiplier)])}
                           for seed, multiplier in config.conditions]
    # Amendment: a native-only convergence ladder, triggered by the large
    # default-vs-tight native divergence. No candidate is assessed on this
    # extra ladder, so it cannot be used for model selection.
    _activate_arm(session, gbar, "native", ALL_SUFFIXES)
    extra_solver_levels = {
        "stricter": (config.stricter_atol, config.stricter_rtol),
        "ultra": (config.ultra_atol, config.ultra_rtol),
    }
    for solver, tolerance in extra_solver_levels.items():
        _set_solver_tolerance(session, *tolerance)
        anchor = _trial(session, calibrator, protocol, config.conditions[0][0],
                        config.conditions[0][1], "native", config, gbar, ALL_SUFFIXES)
        repeat = _trial(session, calibrator, protocol, config.conditions[0][0],
                        config.conditions[0][1], "native", config, gbar, ALL_SUFFIXES)
        repeat_metrics = _compare(anchor, repeat)
        if max(site["voltage_max_error_mv"] for site in repeat_metrics.values()) > config.repeat_voltage_max_atol_mv:
            raise RuntimeError(f"Task 17c native replay failed for {solver}")
        repeat_controls[solver] = repeat_metrics
        for seed, multiplier in config.conditions:
            native_phase("task17c_native_ladder", solver=solver,
                         seed=seed, multiplier=multiplier)
            cache[("native", solver, seed, multiplier)] = _trial(
                session, calibrator, protocol, seed, multiplier,
                "native", config, gbar, ALL_SUFFIXES)
            if (float(session.cvode.atol()), float(session.cvode.rtol())) != tolerance:
                raise RuntimeError(f"Task 17c ladder tolerance changed during {solver}")
    ladder = []
    for earlier, later in (("default", "tight"), ("tight", "stricter"),
                           ("stricter", "ultra")):
        for seed, multiplier in config.conditions:
            ladder.append({"earlier": earlier, "later": later, "seed": seed,
                           "gbar_multiplier": multiplier,
                           "metrics": _compare(cache[("native", earlier, seed, multiplier)],
                                               cache[("native", later, seed, multiplier)])})
    (output_dir / "paired_metrics.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (output_dir / "native_solver_contrast.json").write_text(
        json.dumps(cross_solver_native, indent=2), encoding="utf-8")
    (output_dir / "native_solver_convergence_ladder.json").write_text(
        json.dumps(ladder, indent=2), encoding="utf-8")
    traces = {f"{arm}_{solver}_{seed}_{multiplier}": cache[(arm, solver, seed, multiplier)]["traces"]
              for arm in ("native", "lut", "m_pair_513_f32", "m_pair_2049_f64")
              for solver in solver_levels for seed, multiplier in config.conditions[:2]}
    (output_dir / "diagnostic_traces.json").write_text(json.dumps(traces), encoding="utf-8")
    summary = {solver: {arm: {
        "maximum_gate_error": max(max(v["m_max_error"], v["h_max_error"])
                                  for row in comparisons if row["solver"] == solver
                                  for v in row["metrics"][arm].values()),
        "maximum_voltage_rmse_mv": max(v["voltage_rmse_mv"]
                                       for row in comparisons if row["solver"] == solver
                                       for v in row["metrics"][arm].values()),
    } for arm in ARMS if arm != "native"} for solver in solver_levels}
    report = {
        "schema_version": "giada-roadmap-task17c-m-precision-solver-v1",
        "valid": True, "decision": "DIAGNOSTIC_ONLY", "gate_c_authorized": False,
        "task17a_result_preserved": "CAUSAL_MICROCANARY_NO_GO",
        "code_revision": code_revision, "config": asdict(config),
        "teacher_segment_count": teacher["segment_count"],
        "solver_levels": solver_levels, "native_repeat_preflight": repeat_controls,
        "native_only_extra_solver_levels": extra_solver_levels,
        "frozen_table_probe": frozen_probe, "compiled_m_table_probe": table_probe,
        "static_m_rate_errors": static_errors,
        "candidate_mods": mods, "compilation": compilation,
        "summary": summary, "paired_condition_count": len(config.conditions),
        "warning": "Known stress conditions only; no candidate selected or Gate C promotion.",
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
