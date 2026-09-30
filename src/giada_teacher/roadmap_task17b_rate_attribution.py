"""Task 17b: paired, single-rate causal attribution after the Task 17a NO-GO.

The failed 17a seed is diagnostic, not a new sealed validation set. Four
hybrids each replace exactly one frozen rate function; a fifth replaces both
m rates to expose nonlinear interaction. New seeds test whether attribution
is seed-specific. No arm is promoted to Gate C using this experiment.
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
from .roadmap_embedded_frozen_confirmation import _lut_table, verified_15c
from .roadmap_causal_cahva_replacement import (
    CausalReplacementConfig, SUFFIXES, _activate_arm, _canonical_gbar,
    _compare, _explicit_lut_rates, _trial, candidate_protocols,
    compile_candidate_mods, generate_candidate_mods, verify_frozen_rate_table,
)


HYBRID_FIELDS = {
    "m_inf_lut": ("mInf",), "m_tau_lut": ("mTau",),
    "m_pair_lut": ("mInf", "mTau"),
    "h_inf_lut": ("hInf",), "h_tau_lut": ("hTau",),
}
HYBRID_SUFFIXES = {arm: f"GIADA_CaHVA_17b_{arm}" for arm in HYBRID_FIELDS}
ALL_SUFFIXES = {**SUFFIXES, **HYBRID_SUFFIXES}
ARMS = ("native", "formula", "lut", *HYBRID_FIELDS)


@dataclass(frozen=True)
class Task17bConfig:
    # 170029 is the known 17a failure. 170059/170071 informed the m-pair
    # amendment; use entirely new 170083/170097 for this amended matrix.
    seeds: tuple[int, ...] = (170029, 170083, 170097)
    gbar_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)
    protocol_index: int = 1  # The NMDA schedule exposed the 17a failure.
    duration_ms: int = 40
    sample_interval_ms: float = 0.025
    formula_voltage_rmse_limit_mv: float = 0.05
    formula_gate_max_error: float = 0.002
    repeated_native_voltage_atol_mv: float = 1e-5
    attribution_error_floor: float = 0.002

    def validate(self):
        if asdict(self) != asdict(Task17bConfig()):
            raise ValueError("Task 17b config differs from preregistration")


def generate_hybrid_mods(native_mod: Path, destination: Path):
    """Render single-rate and m-pair interventions, preserving current/solver."""
    native_mod, destination = Path(native_mod), Path(destination)
    records = generate_candidate_mods(native_mod, destination)
    source = native_mod.read_text(encoding="utf-8")
    original = re.search(r"PROCEDURE\s+rates\s*\(\s*\)\s*\{", source)
    if original is None or not source.rstrip().endswith("}"):
        raise ValueError("canonical Ca_HVA rates tail changed")
    table = _lut_table(ExtractedGateFormula.from_mod(native_mod))
    frozen_lut = _explicit_lut_rates(table).replace(
        "PROCEDURE rates()", "PROCEDURE frozen_lut_rates()", 1)
    for arm, fields in HYBRID_FIELDS.items():
        suffix = HYBRID_SUFFIXES[arm]
        header = re.sub(r"\bSUFFIX\s+Ca_HVA\b", f"SUFFIX {suffix}", source[:original.start()])
        formula = source[original.start():].replace(
            "PROCEDURE rates()", "PROCEDURE exact_formula_rates()", 1)
        # Store only registered LUT outputs, then recompute all four using
        # the canonical formula before restoring the selected LUT outputs.
        locals_ = ", ".join(f"frozen_{field}" for field in fields)
        wrapper = (f"PROCEDURE rates() {{\n LOCAL {locals_}\n"
                   " frozen_lut_rates()\n"
                   + "".join(f" frozen_{field} = {field}\n" for field in fields)
                   + " exact_formula_rates()\n"
                   + "".join(f" {field} = frozen_{field}\n" for field in fields)
                   + "}\n")
        modified = header + formula + "\n" + frozen_lut + "\n" + wrapper
        modified = modified.replace(
            "RANGE gCa_HVAbar, gCa_HVA, ica",
            "RANGE gCa_HVAbar, gCa_HVA, ica, mInf, hInf, mTau, hTau", 1)
        path = destination / f"{suffix}.mod"
        path.write_text(modified, encoding="utf-8")
        records[arm] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "only_replaced_rates": list(fields)}
    return records, table


def _diagnostic_attribution(native, lut, hybrid_metrics, floor):
    """Rate isolation is causal; residual interaction is explicitly reported."""
    result = {}
    for site in native["traces"]:
        baseline = lut[site]
        endpoint = max(baseline["m_max_error"], baseline["h_max_error"])
        individual = {
            arm: max(metrics[site]["m_max_error"], metrics[site]["h_max_error"])
            for arm, metrics in hybrid_metrics.items()
        }
        rank = sorted(individual, key=lambda arm: individual[arm], reverse=True)
        result[site] = {
            "full_lut_gate_max_error": endpoint,
            "single_rate_gate_max_error": individual,
            "ranked_single_rate_arms": rank,
            "largest_single_to_full_fraction": individual[rank[0]] / max(endpoint, floor),
            "nonadditivity_warning": (
                "Individual closed-loop interventions need not sum to the full LUT effect."
            ),
        }
    return result


def run_task17b(elm_repo, teacher_repo, native_mod, task15c_source, output_dir,
                config=Task17bConfig(), *, code_revision="unknown"):
    config.validate()
    verified_15c(task15c_source)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    mods, table = generate_hybrid_mods(native_mod, output_dir / "candidate_mods")
    compiler = shutil.which("nrnivmodl")
    if not compiler:
        raise RuntimeError("nrnivmodl unavailable")
    native_phase("compile_17b_hybrids")
    compilation = compile_candidate_mods(output_dir / "candidate_mods", compiler)
    from neuron import h, load_mechanisms
    if not load_mechanisms(str(output_dir / "candidate_mods")):
        raise RuntimeError("compiled Task 17b mechanisms did not load")
    probe = verify_frozen_rate_table(h, table, CausalReplacementConfig())
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, calibration_source=Path(native_mod),
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        output_dir=output_dir / "teacher_workspace")
    teacher = session.prepare_teacher()
    if teacher["segment_count"] != 642:
        raise RuntimeError("Task 17b requires the 642-segment native teacher")
    session.run_burn_in()
    session.task17_snapshot_suffix = "Ca_HVA"
    calibrator = DendriticProtocolCalibrator(session, output_dir=output_dir / "protocol_workspace")
    gbar = _canonical_gbar(session)
    if not gbar:
        raise RuntimeError("No native Ca_HVA conductances")
    protocol = candidate_protocols()[config.protocol_index]
    cache = {}
    native_anchor = _trial(session, calibrator, protocol, config.seeds[0], 1.0,
                           "native", config, gbar, ALL_SUFFIXES)
    native_repeat = _trial(session, calibrator, protocol, config.seeds[0], 1.0,
                           "native", config, gbar, ALL_SUFFIXES)
    native_repeat_metrics = _compare(native_anchor, native_repeat)
    if max(site["voltage_max_error_mv"] for site in native_repeat_metrics.values()) > config.repeated_native_voltage_atol_mv:
        raise RuntimeError("Task 17b native repeated replay preflight failed")
    for arm in ARMS:
        _activate_arm(session, gbar, arm, ALL_SUFFIXES)
        for seed in config.seeds:
            for multiplier in config.gbar_multipliers:
                native_phase("task17b_trial", arm=arm, seed=seed, multiplier=multiplier)
                row = _trial(session, calibrator, protocol, seed, multiplier,
                             arm, config, gbar, ALL_SUFFIXES)
                cache[(arm, seed, multiplier)] = row
                if arm == "formula":
                    check = _compare(cache[("native", seed, multiplier)], row)
                    if (max(x["voltage_rmse_mv"] for x in check.values()) > config.formula_voltage_rmse_limit_mv
                        or max(max(x["m_max_error"], x["h_max_error"]) for x in check.values()) > config.formula_gate_max_error):
                        raise RuntimeError(f"Task 17b formula control failed at {seed}, {multiplier}")
                print(f"[GIADA Task 17b] {arm} seed={seed} gbar={multiplier}", flush=True)
    comparisons = []
    for seed in config.seeds:
        for multiplier in config.gbar_multipliers:
            reference = cache[("native", seed, multiplier)]
            metrics = {arm: _compare(reference, cache[(arm, seed, multiplier)])
                       for arm in ARMS if arm != "native"}
            attribution = _diagnostic_attribution(reference, metrics["lut"],
                {arm: metrics[arm] for arm in HYBRID_FIELDS}, config.attribution_error_floor)
            comparisons.append({"seed": seed, "gbar_multiplier": multiplier,
                                "metrics": metrics, "attribution": attribution})
    (output_dir / "paired_metrics.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8")
    # Keep detailed traces only for the registered difficult case. Avoid huge
    # Kaggle output or browser flooding; metrics cover every arm/seed/gbar.
    difficult = {arm: cache[(arm, config.seeds[0], 1.0)]["traces"] for arm in ARMS}
    (output_dir / "diagnostic_traces.json").write_text(
        json.dumps(difficult), encoding="utf-8")
    worst = max(max(x["m_max_error"], x["h_max_error"])
                for row in comparisons for x in row["metrics"]["lut"].values())
    report = {
        "schema_version": "giada-roadmap-task17b-rate-attribution-v1", "valid": True,
        "decision": "DIAGNOSTIC_ONLY", "gate_c_authorized": False,
        "task17a_result_preserved": "CAUSAL_MICROCANARY_NO_GO",
        "known_failure_seed_used_for_diagnosis": config.seeds[0],
        "new_seeds_not_sealed_confirmation": list(config.seeds[1:]),
        "full_lut_worst_gate_error": worst,
        "single_rate_interventions": HYBRID_FIELDS,
        "hybrid_mods": mods, "compilation": compilation,
        "frozen_table_probe": probe, "teacher_segment_count": teacher["segment_count"],
        "native_repeat_preflight": native_repeat_metrics,
        "code_revision": code_revision, "config": asdict(config),
        "warning": "Rate attribution only; no candidate selected or threshold changed. Gate C needs independent preregistered validation.",
    }
    (output_dir / "final_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report
